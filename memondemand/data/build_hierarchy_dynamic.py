"""V5 Dynamic Hierarchy Builder"""

from __future__ import annotations

from memondemand.core import dns_patch  # noqa: F401

import argparse
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
EMERGENCY_DEPTH_CAP = 8       # v5_checklist §2.4 hard safety net
MIN_NODES_FOR_LEVEL = 4       # v5_checklist §2.4: don't create a level above ≤3 nodes
MAX_RECLUSTER_ATTEMPTS = 2    # max RECLUSTER before forcing STOP
DISTILLED_PREFIX_CHARS = 150  # chars of content used for L0 distilled

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT = "gpt-5.4-mini"
AZURE_API_VERSION = "2024-12-01-preview"

BEDROCK_ENDPOINT = "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
BEDROCK_MODEL = "openai.gpt-5.4"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class DynamicNode:
    node_id: str
    level: int                        # 0 = L0, 1 = L1, ...
    tenant_id: str                    # source_type from L0
    distilled_text: str               # short summary (20-300 tokens)
    detailed_text: str                # full content or child summaries
    distilled_tokens: int = 0
    detailed_tokens: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)  # L0 doc_ids
    state: str = "LIGHT"              # LIGHT | PROMOTED
    cluster_coherence: float = 0.0    # intra-cluster cosine mean (L1+)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class TokenLedger:
    total_in: int = 0
    total_out: int = 0
    total_cost_usd: float = 0.0
    n_calls: int = 0

    # gpt-5.4-mini pricing (Azure, approximate)
    MINI_IN_PER_1M = 0.40
    MINI_OUT_PER_1M = 1.60
    # gpt-5.4 pricing (Bedrock, approximate)
    GPT54_IN_PER_1M = 5.0
    GPT54_OUT_PER_1M = 15.0

    def add_mini(self, n_in: int, n_out: int) -> None:
        self.total_in += n_in
        self.total_out += n_out
        self.total_cost_usd += (n_in * self.MINI_IN_PER_1M + n_out * self.MINI_OUT_PER_1M) / 1_000_000
        self.n_calls += 1

    def add_gpt54(self, n_in: int, n_out: int) -> None:
        self.total_in += n_in
        self.total_out += n_out
        self.total_cost_usd += (n_in * self.GPT54_IN_PER_1M + n_out * self.GPT54_OUT_PER_1M) / 1_000_000
        self.n_calls += 1


# ─── LLM Clients ──────────────────────────────────────────────────────────────

def _azure_client():
    from openai import AzureOpenAI
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_LLM_API_KEY", "")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT not set")
    if not api_key:
        raise RuntimeError("AZURE_LLM_API_KEY not set")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=AZURE_API_VERSION,
    )


def call_mini(prompt: str, ledger: TokenLedger, dry_run: bool = False,
              max_tokens: int = 200) -> str:
    """Distillation call. v5 Step C v3 fix 1: route to Bedrock gpt-5.4 (full)
    instead of Azure gpt-5.4-mini for higher-quality cluster summaries."""
    if dry_run:
        return "[DRY-RUN summary]"
    import requests
    api_key = (os.environ.get("AWS_BEDROCK_API_KEY") or
               os.environ.get("AWS_BEDROCK_EXPERIMENT_KEY") or
               os.environ.get("BEDROCK_API_KEY") or "")
    if not api_key:
        # Fallback: Azure gpt_5_4_mini when Bedrock unavailable
        import os as _os
        _az_key = (_os.environ.get("AZURE_LLM_API_KEY") or
                   _os.environ.get("AZURE_OPENAI_API_KEY") or "")
        _az_dep = _os.environ.get("AZURE_GPT54_MINI_DEPLOYMENT", "gpt-5.4-mini")
        if not _az_key:
            raise RuntimeError("No Bedrock or Azure key for distillation")
        log.warning("No Bedrock key — falling back to Azure %s for distillation", _az_dep)
        try:
            cli = _azure_client()
            resp = cli.chat.completions.create(
                model=_az_dep,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            n_in = resp.usage.prompt_tokens if resp.usage else 0
            n_out = resp.usage.completion_tokens if resp.usage else 0
            ledger.add_mini(n_in, n_out)
            return text.strip()
        except Exception as _e:
            log.warning("Azure distill fallback failed: %s", _e)
            return ""
    t0 = time.time()
    flat_input = f"User: {prompt}"
    payload = {
        "model": BEDROCK_MODEL,
        "input": flat_input,
        "max_output_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(BEDROCK_ENDPOINT, json=payload, headers=headers, timeout=120)
    if not r.ok:
        log.warning("Bedrock distill %d: %s", r.status_code, r.text[:200])
        return ""
    data = r.json()
    text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                text += c.get("text", "")
    usage = data.get("usage", {})
    n_in = usage.get("input_tokens", 0)
    n_out = usage.get("output_tokens", 0)
    ledger.add_gpt54(n_in, n_out)
    log.debug("distill(gpt54) %.1fs in=%d out=%d", time.time()-t0, n_in, n_out)
    return text.strip()


def call_gpt54_depth(prompt: str, ledger: TokenLedger,
                     dry_run: bool = False) -> Dict[str, Any]:
    """Call gpt-5.4 for depth decision. Returns parsed JSON dict.

    Bedrock /responses API takes `input` as a flat string (not message list).
    """
    if dry_run:
        return {"decision": "STOP", "reason": "dry_run", "suggested_k": 0}
    import requests
    # Try multiple env var names matching what run_erag scripts set
    api_key = (os.environ.get("AWS_BEDROCK_API_KEY") or
               os.environ.get("AWS_BEDROCK_EXPERIMENT_KEY") or
               os.environ.get("BEDROCK_API_KEY") or "")
    if not api_key:
        # Fallback: use Azure gpt_5_4_mini for depth decision when Bedrock unavailable
        log.warning("No Bedrock API key — falling back to call_mini for depth decision")
        raw = call_mini(prompt, ledger, dry_run=dry_run)
        import re as _re, json as _json
        m = _re.search(r"\{[^}]+\}", raw or "")
        if m:
            try:
                return _json.loads(m.group())
            except Exception:
                pass
        return {"decision": "CREATE_NEXT_LEVEL", "reason": "mini_fallback", "suggested_k": 8}
    t0 = time.time()
    # Bedrock /responses requires flat input string, not message list
    flat_input = f"User: {prompt}"
    payload = {
        "model": BEDROCK_MODEL,
        "input": flat_input,
        "max_output_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(BEDROCK_ENDPOINT, json=payload, headers=headers, timeout=90)
    if not r.ok:
        log.warning("Bedrock %d error: %s — defaulting to STOP", r.status_code, r.text[:300])
        return {"decision": "STOP", "reason": f"bedrock_http_{r.status_code}", "suggested_k": 0}
    data = r.json()
    # parse output
    text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                text += c.get("text", "")
    # token counts
    usage = data.get("usage", {})
    n_in = usage.get("input_tokens", 0)
    n_out = usage.get("output_tokens", 0)
    ledger.add_gpt54(n_in, n_out)
    log.debug("gpt54 depth call %.1fs  in=%d out=%d", time.time()-t0, n_in, n_out)
    # parse JSON from text
    text = text.strip()
    # try to extract JSON block
    import re
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    # fallback: try direct parse
    try:
        return json.loads(text)
    except Exception:
        log.warning("depth decision parse failed, defaulting STOP. text=%r", text[:200])
        return {"decision": "STOP", "reason": "parse_error", "suggested_k": 0}


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Embed list of texts, return L2-normalized float32 array. Backend chosen via env MEMONDEMAND_EMBED_BACKEND."""
    import os as _os
    backend = _os.environ.get("MEMONDEMAND_EMBED_BACKEND", "minilm")
    if backend in ("azure_large", "azure_small"):
        from memondemand.data.azure_embedder import embed_batch
        log.info("Encoding %d texts via Azure text-embedding-3-large ...", len(texts))
        t0 = time.time()
        vecs = embed_batch(list(texts), batch_size=128)
        log.info("Embedded %d texts in %.1fs (azure)", len(texts), time.time() - t0)
        return vecs.astype(np.float32)
    from sentence_transformers import SentenceTransformer
    log.info("Loading embedding model %s ...", model_name)
    model = SentenceTransformer(model_name)
    log.info("Encoding %d texts ...", len(texts))
    t0 = time.time()
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    log.info("Embedded %d texts in %.1fs", len(texts), time.time() - t0)
    return vecs.astype(np.float32)


# ─── Cluster Coherence ────────────────────────────────────────────────────────

def cluster_coherence(vecs: np.ndarray) -> float:
    """Mean pairwise cosine similarity within a cluster (using centroid approx)."""
    if len(vecs) <= 1:
        return 1.0
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-9:
        return 0.0
    centroid /= norm
    sims = vecs @ centroid
    return float(sims.mean())


def noise_ratio(labels: np.ndarray, min_cluster_size: int = 2) -> float:
    """Fraction of points in clusters smaller than min_cluster_size."""
    from collections import Counter
    counts = Counter(labels)
    small = sum(cnt for cnt in counts.values() if cnt < min_cluster_size)
    return small / len(labels)


# ─── Depth Decision Prompt ────────────────────────────────────────────────────

def build_depth_prompt(
    current_depth: int,
    n_current_nodes: int,
    n_candidate_clusters: int,
    cluster_sizes: List[int],
    coherence_scores: List[float],
    noise_r: float,
    budget_remaining_usd: float,
    recluster_attempts: int,
) -> str:
    avg_coh = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    min_coh = float(np.min(coherence_scores)) if coherence_scores else 0.0
    return f"""You are managing a dynamic memory hierarchy for enterprise document retrieval.

Current state:
- Current level depth: L{current_depth} ({n_current_nodes} nodes)
- Proposed next level: L{current_depth+1} ({n_candidate_clusters} candidate clusters)
- Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}
- Cluster coherence: mean={avg_coh:.3f}, min={min_coh:.3f} (1.0=perfect, 0.0=random)
- Noise ratio (tiny clusters): {noise_r:.3f}
- Recluster attempts at this level: {recluster_attempts}/{MAX_RECLUSTER_ATTEMPTS}
- Budget remaining: ${budget_remaining_usd:.2f}
- Emergency depth cap: {EMERGENCY_DEPTH_CAP} (current depth {current_depth})

Guidelines:
- CREATE_NEXT_LEVEL if: coherence > 0.3, clusters are meaningful, budget allows, depth < cap
- RECLUSTER if: coherence < 0.2 or noise_ratio > 0.3, and recluster_attempts < max
- STOP if: coherence is reasonable but further abstraction won't help, or budget/depth limits near

Respond with ONLY valid JSON (no markdown):
{{"decision": "CREATE_NEXT_LEVEL|RECLUSTER|STOP", "reason": "...", "suggested_k": <int or 0>}}"""


# ─── Summary Prompt ──────────────────────────────────────────────────────────

def build_summary_prompt(child_texts: List[str], level: int) -> str:
    sample = "\n\n---\n\n".join(t[:400] for t in child_texts[:5])
    return f"""Summarize the following {len(child_texts)} related documents into a concise L{level} memory node (50-200 words). Focus on key topics, entities, and relationships that would help answer future queries. Include the most important factual content.

Documents:
{sample}

Write ONLY the summary, no preamble:"""


# ─── Core Builder ─────────────────────────────────────────────────────────────

def build_dynamic_hierarchy(
    l0_df: pd.DataFrame,
    budget_usd: float = 5.0,
    dry_run: bool = False,
    out_dir: Optional[Path] = None,
) -> tuple[List[DynamicNode], Dict[str, Any]]:
    """
    Build a dynamic multi-level hierarchy from L0 nodes.

    Returns:
        (all_nodes, build_report)
    """
    t_start = time.time()
    ledger = TokenLedger()
    decision_log = []

    # ── Step 1: Create L0 DynamicNodes ──────────────────────────────────────
    log.info("Creating L0 nodes from %d docs ...", len(l0_df))
    l0_nodes: List[DynamicNode] = []
    for _, row in l0_df.iterrows():
        distilled = f"{row['title']}: {str(row['content'])[:DISTILLED_PREFIX_CHARS]}"
        detailed = f"{row['title']}\n{row['text']}"
        node = DynamicNode(
            node_id=str(row['doc_id']),
            level=0,
            tenant_id=str(row['source_type']),
            distilled_text=distilled,
            detailed_text=detailed,
            distilled_tokens=len(distilled.split()),
            detailed_tokens=len(detailed.split()),
            source_evidence_ids=[str(row['doc_id'])],
        )
        l0_nodes.append(node)

    all_nodes: List[DynamicNode] = list(l0_nodes)
    nodes_per_level = [len(l0_nodes)]

    # ── Step 2: Embed L0 ────────────────────────────────────────────────────
    t_embed_start = time.time()
    l0_texts = [n.distilled_text for n in l0_nodes]
    l0_vecs = embed_texts(l0_texts)
    t_embed_seconds = time.time() - t_embed_start

    # ── Step 3: Dynamic Hierarchy Loop ──────────────────────────────────────
    current_nodes = l0_nodes
    current_vecs = l0_vecs
    depth = 0
    t_llm_depth_total = 0.0
    t_llm_summary_total = 0.0
    n_llm_depth_calls = 0

    while True:
        n_current = len(current_nodes)
        log.info("=== Level L%d → L%d (n=%d nodes) ===", depth, depth+1, n_current)

        # Stop conditions
        if n_current < MIN_NODES_FOR_LEVEL:
            log.info("STOP: only %d nodes at L%d, below minimum %d", n_current, depth, MIN_NODES_FOR_LEVEL)
            decision_log.append({"depth": depth, "decision": "STOP", "reason": f"n={n_current} < MIN={MIN_NODES_FOR_LEVEL}"})
            break
        if depth >= EMERGENCY_DEPTH_CAP - 1:
            log.info("STOP: emergency depth cap %d reached", EMERGENCY_DEPTH_CAP)
            decision_log.append({"depth": depth, "decision": "STOP", "reason": "emergency_depth_cap"})
            break

        # Determine initial k
        k = max(4, int(np.sqrt(n_current)))
        k = min(k, n_current // 2)

        recluster_attempts = 0
        cluster_nodes: List[DynamicNode] = []

        while True:
            # ── Cluster ─────────────────────────────────────────────────────
            log.info("Clustering %d nodes into k=%d ...", n_current, k)
            t_cl = time.time()
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = km.fit_predict(current_vecs)
            t_clustering = time.time() - t_cl

            # ── Compute cluster stats ────────────────────────────────────────
            cluster_sizes = []
            coherence_scores = []
            cluster_groups: Dict[int, List[int]] = {}
            for i, lbl in enumerate(labels):
                cluster_groups.setdefault(lbl, []).append(i)
            for lbl, idxs in sorted(cluster_groups.items()):
                cluster_sizes.append(len(idxs))
                coherence_scores.append(cluster_coherence(current_vecs[idxs]))

            noise_r = noise_ratio(labels, min_cluster_size=2)
            budget_remaining = budget_usd - ledger.total_cost_usd

            log.info("Cluster stats: k=%d  coh_mean=%.3f  noise=%.3f  budget_left=$%.2f",
                     k, np.mean(coherence_scores), noise_r, budget_remaining)

            # ── LLM depth decision ───────────────────────────────────────────
            t_dec = time.time()
            depth_prompt = build_depth_prompt(
                current_depth=depth,
                n_current_nodes=n_current,
                n_candidate_clusters=k,
                cluster_sizes=cluster_sizes,
                coherence_scores=coherence_scores,
                noise_r=noise_r,
                budget_remaining_usd=budget_remaining,
                recluster_attempts=recluster_attempts,
            )
            decision = call_gpt54_depth(depth_prompt, ledger, dry_run=dry_run)
            t_llm_depth_total += time.time() - t_dec
            n_llm_depth_calls += 1

            action = decision.get("decision", "STOP").upper()
            reason = decision.get("reason", "")
            suggested_k = int(decision.get("suggested_k", 0) or 0)

            log.info("Depth decision: %s  reason=%s  suggested_k=%s", action, reason, suggested_k)
            decision_log.append({
                "depth": depth, "k_tried": k, "action": action,
                "reason": reason, "suggested_k": suggested_k,
                "coh_mean": float(np.mean(coherence_scores)),
                "noise": noise_r,
                "cost_so_far": ledger.total_cost_usd,
            })

            if action == "RECLUSTER" and recluster_attempts < MAX_RECLUSTER_ATTEMPTS:
                recluster_attempts += 1
                k = max(4, suggested_k) if suggested_k > 0 else max(4, k - max(2, k//4))
                log.info("RECLUSTER attempt %d, new k=%d", recluster_attempts, k)
                continue

            if action == "STOP":
                log.info("LLM chose STOP at depth %d", depth)
                break

            # CREATE_NEXT_LEVEL: build L(depth+1) nodes
            log.info("Creating L%d nodes (k=%d clusters) ...", depth+1, k)
            t_sum = time.time()
            cluster_nodes = []
            for lbl, idxs in sorted(cluster_groups.items()):
                children = [current_nodes[i] for i in idxs]
                child_texts = [c.distilled_text for c in children]

                # LLM summary (gpt-5.4-mini)
                summary_prompt = build_summary_prompt(child_texts, depth+1)
                summary = call_mini(summary_prompt, ledger, dry_run=dry_run, max_tokens=300)

                # Detailed = joined children distilled texts
                detailed = "\n\n".join(f"[{c.node_id}] {c.distilled_text}" for c in children)

                # All L0 evidence IDs from children
                all_evidence = []
                for c in children:
                    all_evidence.extend(c.source_evidence_ids)

                # Use most common tenant_id
                from collections import Counter
                tenant_id = Counter(c.tenant_id for c in children).most_common(1)[0][0]

                cnode = DynamicNode(
                    node_id=f"L{depth+1}_{lbl}_{uuid.uuid4().hex[:6]}",
                    level=depth+1,
                    tenant_id=tenant_id,
                    distilled_text=summary,
                    detailed_text=detailed,
                    distilled_tokens=len(summary.split()),
                    detailed_tokens=len(detailed.split()),
                    children_ids=[c.node_id for c in children],
                    source_evidence_ids=all_evidence,
                    cluster_coherence=coherence_scores[lbl],
                )
                # Set parent_id on children
                for c in children:
                    c.parent_id = cnode.node_id
                cluster_nodes.append(cnode)
                log.debug("  L%d node %s: %d children, coh=%.3f",
                          depth+1, cnode.node_id, len(children), coherence_scores[lbl])

            t_llm_summary_total += time.time() - t_sum
            log.info("Created %d L%d nodes in %.1fs", len(cluster_nodes), depth+1, time.time()-t_sum)
            break  # exit recluster loop

        if action != "CREATE_NEXT_LEVEL" or not cluster_nodes:
            break

        # Advance
        all_nodes.extend(cluster_nodes)
        nodes_per_level.append(len(cluster_nodes))

        # Embed new level
        new_texts = [n.distilled_text for n in cluster_nodes]
        new_vecs = embed_texts(new_texts)

        current_nodes = cluster_nodes
        current_vecs = new_vecs
        depth += 1

    # ── Final stats ─────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    final_depth = max(n.level for n in all_nodes)

    # Build npl dict {L0: n, L1: n, ...} per v5_checklist §6.1
    nodes_per_level_dict = {f"L{i}": n for i, n in enumerate(nodes_per_level)}

    build_report = {
        "tier": "unknown",
        "n_l0_nodes": len(l0_nodes),
        "total_nodes": len(all_nodes),
        # v5_checklist §6.1 canonical field names
        "final_hierarchy_depth": final_depth,
        "nodes_per_level": nodes_per_level_dict,
        "t_l0_load_seconds": 0.0,          # filled at save time (loader is in main())
        "t_embedding_build_seconds": round(t_embed_seconds, 2),
        "t_hierarchy_build_seconds": round(t_total - t_embed_seconds, 2),
        "t_vector_index_build_seconds": 0.0,  # in-memory cosine, no FAISS add cost
        "t_total_construction_seconds": round(t_total, 2),
        "t_clustering_seconds": round(t_total - t_embed_seconds - t_llm_depth_total - t_llm_summary_total, 2),
        "t_llm_depth_decision_seconds": round(t_llm_depth_total, 2),
        "t_llm_distillation_seconds": round(t_llm_summary_total, 2),
        "n_llm_calls_low": ledger.n_calls - n_llm_depth_calls,    # gpt_5_4_mini (distillation)
        "n_llm_calls_high": n_llm_depth_calls,                    # gpt_5_4 (depth decisions)
        "cost_construction_usd": round(ledger.total_cost_usd, 4),
        # Legacy / back-compat fields (kept for older consumers)
        "final_depth": final_depth,
        "nodes_per_level_list": nodes_per_level,
        "t_embedding_seconds": round(t_embed_seconds, 2),
        "t_llm_summary_seconds": round(t_llm_summary_total, 2),
        "n_llm_depth_calls": n_llm_depth_calls,
        "n_llm_summary_calls": ledger.n_calls - n_llm_depth_calls,
        "total_llm_calls": ledger.n_calls,
        "total_tokens_in": ledger.total_in,
        "total_tokens_out": ledger.total_out,
        "cost_usd": round(ledger.total_cost_usd, 4),
        "dry_run": dry_run,
        "acceptance": {},
    }

    # ── Acceptance checks ────────────────────────────────────────────────────
    n_with_both = sum(1 for n in all_nodes if n.distilled_text and n.detailed_text)
    n_distilled_lt_detailed = sum(
        1 for n in all_nodes
        if len(n.distilled_text) < len(n.detailed_text)
    )
    l0_ids = {n.node_id for n in all_nodes if n.level == 0}
    all_evidence_ids = set()
    for n in all_nodes:
        all_evidence_ids.update(n.source_evidence_ids)
    l0_traceable = l0_ids.issubset(all_evidence_ids)

    build_report["acceptance"] = {
        "dual_rep_100pct": n_with_both == len(all_nodes),
        "dual_rep_pct": round(100 * n_with_both / len(all_nodes), 1),
        "distilled_lt_detailed_pct": round(100 * n_distilled_lt_detailed / len(all_nodes), 1),
        "l0_traceable": l0_traceable,
        "final_depth_ge_2": final_depth >= 2,
        "at_least_one_create": final_depth >= 1,
        "cost_le_budget": ledger.total_cost_usd <= budget_usd,
    }

    passed = all(v for v in build_report["acceptance"].values() if isinstance(v, bool))
    build_report["acceptance"]["ALL_PASS"] = passed

    return all_nodes, build_report, decision_log


# ─── Save ────────────────────────────────────────────────────────────────────

def save_outputs(
    all_nodes: List[DynamicNode],
    build_report: Dict[str, Any],
    decision_log: List[Dict],
    out_dir: Path,
    tier: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    build_report["tier"] = tier

    # hierarchy.json (ndjson)
    hier_path = out_dir / "hierarchy.json"
    with open(hier_path, "w") as f:
        for node in all_nodes:
            f.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")
    log.info("Wrote %d nodes to %s", len(all_nodes), hier_path)

    # build_report.json
    rpt_path = out_dir / "build_report.json"
    with open(rpt_path, "w") as f:
        json.dump(build_report, f, indent=2, ensure_ascii=False)
    log.info("Wrote build_report to %s", rpt_path)

    # decision_log.ndjson
    dec_path = out_dir / "decision_log.ndjson"
    with open(dec_path, "w") as f:
        for entry in decision_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Wrote %d decision log entries to %s", len(decision_log), dec_path)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V5 Dynamic Hierarchy Builder")
    parser.add_argument("--tier", required=True, help="Tier name, e.g. 10M")
    parser.add_argument("--l0_parquet", required=True, help="Path to L0 parquet file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--budget_usd", type=float, default=5.0)
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip LLM calls, use mock responses (fast smoke test)")
    args = parser.parse_args()

    log.info("=== V5 Dynamic Hierarchy Builder ===")
    log.info("tier=%s  l0=%s  out=%s  budget=$%.1f  dry_run=%s",
             args.tier, args.l0_parquet, args.out_dir, args.budget_usd, args.dry_run)

    # Load L0
    t_load_start = time.time()
    df = pd.read_parquet(args.l0_parquet)
    t_l0_load_seconds = time.time() - t_load_start
    log.info("Loaded L0: %d rows in %.2fs, cols=%s", len(df), t_l0_load_seconds, list(df.columns))

    out_dir = Path(args.out_dir)

    # Build
    all_nodes, build_report, decision_log = build_dynamic_hierarchy(
        l0_df=df,
        budget_usd=args.budget_usd,
        dry_run=args.dry_run,
        out_dir=out_dir,
    )
    build_report["t_l0_load_seconds"] = round(t_l0_load_seconds, 2)
    build_report["t_total_construction_seconds"] = round(
        build_report["t_total_construction_seconds"] + t_l0_load_seconds, 2
    )

    # Save
    save_outputs(all_nodes, build_report, decision_log, out_dir, tier=args.tier)

    # Summary
    log.info("=== BUILD COMPLETE ===")
    log.info("Tier: %s | Nodes: %d | Depth: %d | Cost: $%.4f | Time: %.0fs",
             args.tier,
             build_report["total_nodes"],
             build_report["final_depth"],
             build_report["cost_usd"],
             build_report["t_total_construction_seconds"])
    log.info("Nodes per level: %s", build_report["nodes_per_level"])
    log.info("Acceptance: %s", build_report["acceptance"])

    if not build_report["acceptance"].get("ALL_PASS"):
        log.error("❌ ACCEPTANCE CHECKS FAILED")
        raise SystemExit(1)
    log.info("✅ ALL ACCEPTANCE CHECKS PASS")


if __name__ == "__main__":
    main()
