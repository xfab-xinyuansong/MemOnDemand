"""Build MemOnDemand's dynamic hierarchy from source-resolved L0 records.

Pipeline:
    L0 source records
        ├─ detailed_text  = canonical_label + raw_text + tags + entities + as_of + tenant
        ├─ distilled_text = canonical_label-derived short summary (deterministic; no LLM)
        │   With explicit fallback when the label is degenerate.
        └─ source_evidence_ids = [evidence_span_id, ...] (real L0 evidence refs)

    L1 (per-tenant KMeans cluster of L0 distilled embeddings)
        ├─ clustering decision validated by gpt_5_4_mini (low-level)
        ├─ detailed_text = JSON-ish bundle of child distilled_text + per-cluster stats
        ├─ distilled_text = gpt_5_4_mini summary
        └─ source_evidence_ids = union of all descendants' L0 evidence

    L2 (per-tenant root summarising L1 nodes)
        ├─ detailed_text = bundle of L1 distilled_text
        ├─ distilled_text = gpt_5_4 (high-level abstraction)
        └─ source_evidence_ids = union of descendants

Outputs (in --out):
    hierarchy.json     -- list of node records (all levels)
    build_report.json  -- aggregated build stats + LLM call counts
    token_ledger.json  -- per-phase / per-model token + cost accounting

The output is a source-resolved, multi-level structure suitable for dual-view
retrieval and on-demand promotion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tiktoken

REPO_ROOT = Path(os.environ.get("MEMONDEMAND_REPO_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(REPO_ROOT))

from memondemand.pipelines.enterprise_rag_1_14b.runtime import (  # noqa: E402
    APIError,
    ConfiguredEmbedder,
    call as api_call,
)
from memondemand.methods.dual_node import (  # noqa: E402
    DualNode,
    NODE_STATE_LIGHT,
)
from memondemand.methods.token_ledger import (  # noqa: E402
    PHASE_HIERARCHY_BUILD,
    PHASE_DISTILLED_GEN,
    TokenLedger,
)

logger = logging.getLogger("memondemand.enterprise_rag_1_14b.build_hierarchy")

# ---------------------------------------------------------------------------
# Hyper-params
# ---------------------------------------------------------------------------

L1_TARGET_NODES_PER_CLUSTER = 8     # ~8 L0 children per L1 node
L1_MAX_CLUSTERS_PER_TENANT = 32     # safety cap; for small tenants we fall back
L1_MIN_CLUSTER_SIZE = 2             # singletons stay at L0
L2_MAX_CHILDREN = 12                # if tenant has >12 L1 nodes we still create one L2 root (summary handles all)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

GPT54_MINI = "gpt_5_4_mini"
GPT54 = "gpt_5_4"

# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

_ENC = None
def enc():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def tok_count(text: str) -> int:
    if not text:
        return 0
    return len(enc().encode(text))


# ---------------------------------------------------------------------------
# L0 loader (from raw medium_night1 directory)
# ---------------------------------------------------------------------------


def load_l0_raw(medium_night1_dir: Path) -> List[Dict[str, Any]]:
    """Load every L0 node from every release tenant (35 dirs, 33 release tenants)."""
    out: List[Dict[str, Any]] = []
    tenant_dirs = sorted([d for d in medium_night1_dir.iterdir()
                          if d.is_dir() and d.name.startswith("m1_")])
    for td in tenant_dirs:
        p = td / "jsonl" / "08a_memory_l0.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                for n in rec.get("nodes", []):
                    out.append(n)
    logger.info(f"loaded {len(out)} L0 nodes from {len(tenant_dirs)} tenant dirs")
    return out


def filter_to_manifest(l0_nodes: List[Dict[str, Any]],
                       manifest_index: pd.DataFrame) -> List[Dict[str, Any]]:
    """Keep only the 9203 nodes listed in own_full_l0_nodes.parquet."""
    keep = set(manifest_index["node_id"].astype(str))
    filtered = [n for n in l0_nodes if n.get("node_id") in keep]
    logger.info(f"filtered to manifest: {len(filtered)} / {len(l0_nodes)} raw, "
                f"{len(keep)} expected")
    return filtered


# ---------------------------------------------------------------------------
# L0 dual-rep construction
# ---------------------------------------------------------------------------


def _l0_distilled(label: str, tags: List[str]) -> str:
    """Short deterministic distilled rep — first ~10 words of canonical_label
    + up to 3 topic tags. Always non-empty unless input is fully empty."""
    label = (label or "").strip()
    if label:
        words = label.split()
        head = " ".join(words[:10])
    else:
        head = "[empty-label]"
    if tags:
        head = f"{head} [{', '.join(tags[:3])}]"
    return head.strip()


def _l0_detailed(node: Dict[str, Any]) -> str:
    """Rich detailed rep — canonical_label + raw_text + structured metadata block."""
    cl = (node.get("canonical_label") or "").strip()
    ls = node.get("level_specific") or {}
    rt = (ls.get("raw_text") or "").strip() if isinstance(ls, dict) else ""
    tags = node.get("topic_tags") or []
    ents_raw = node.get("entity_refs") or []
    ents: List[str] = []
    for e in ents_raw:
        if isinstance(e, dict):
            v = e.get("value") or e.get("name") or e.get("id") or ""
            if v:
                ents.append(str(v))
        elif isinstance(e, str) and e:
            ents.append(e)
    aof = node.get("as_of") or ""
    valid_from = node.get("valid_from") or ""
    tenant = node.get("tenant_id") or ""
    lifecycle = node.get("lifecycle_status") or ""

    parts: List[str] = []
    if cl:
        parts.append(cl)
    if rt and rt != cl:
        parts.append(f"Raw: {rt}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    if ents:
        parts.append(f"Entities: {', '.join(ents[:8])}")
    if aof:
        parts.append(f"AsOf: {aof}")
    if valid_from and valid_from != aof:
        parts.append(f"ValidFrom: {valid_from}")
    if lifecycle:
        parts.append(f"Lifecycle: {lifecycle}")
    if tenant:
        parts.append(f"Tenant: {tenant}")
    if not parts:
        parts.append("[empty-l0-record]")
    return "\n".join(parts)


def build_l0_dualnodes(l0_raw: List[Dict[str, Any]]) -> List[DualNode]:
    nodes: List[DualNode] = []
    for n in l0_raw:
        cl = (n.get("canonical_label") or "").strip()
        tags = n.get("topic_tags") or []
        ls = n.get("level_specific") or {}
        ev_ids: List[str] = []
        if isinstance(ls, dict):
            esid = ls.get("evidence_span_id")
            if esid:
                ev_ids.append(str(esid))
        for sid in n.get("source_evidence_span_ids") or []:
            ev_ids.append(str(sid))
        # dedupe preserving order
        ev_ids = list(dict.fromkeys(ev_ids))
        if not ev_ids:
            ev_ids = [n.get("node_id", "")]

        distilled = _l0_distilled(cl, tags)
        detailed = _l0_detailed(n)

        dt_tok = tok_count(distilled)
        det_tok = tok_count(detailed)
        # If accidentally distilled >= detailed (rare empty-record case), force
        # distilled to be a strict prefix so the invariant holds.
        if det_tok > 0 and dt_tok >= det_tok:
            # Take first ~half of detailed tokens as distilled fallback
            ids_full = enc().encode(detailed)
            half_ids = ids_full[: max(1, len(ids_full) // 2)]
            distilled = enc().decode(half_ids)
            dt_tok = len(half_ids)
        nodes.append(DualNode(
            node_id=n.get("node_id", ""),
            level="L0",
            tenant_id=n.get("tenant_id", ""),
            distilled_text=distilled,
            detailed_text=detailed,
            distilled_tokens=dt_tok,
            detailed_tokens=det_tok,
            source_evidence_ids=ev_ids,
            state=NODE_STATE_LIGHT,
            distilled_text_model_alias="deterministic_label_truncation",
            distilled_text_model_status="ACTIVE",
            extra={"l0_raw_origin": True},
        ))
    return nodes


# ---------------------------------------------------------------------------
# Embedding + clustering
# ---------------------------------------------------------------------------


_EMBED_MODEL = None


def embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"loading embedding model {EMBED_MODEL_NAME}")
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL


_AZURE_EMB = None
def embed_texts(texts: List[str]) -> np.ndarray:
    import os as _os
    if not texts:
        _d = 1536 if _os.environ.get("MEMONDEMAND_BUILD_EMBED")=="azure_small" else EMBED_DIM
        return np.zeros((0, _d), dtype=np.float32)
    if _os.environ.get("MEMONDEMAND_BUILD_EMBED")=="azure_small":
        global _AZURE_EMB
        if _AZURE_EMB is None:
            _AZURE_EMB = ConfiguredEmbedder(expected_dim=1536)
            logger.info("build using configured embedding backend")
        return np.asarray(_AZURE_EMB.encode(texts), dtype=np.float32)
    m = embedder()
    return np.asarray(m.encode(
        texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)


def kmeans_cluster(vecs: np.ndarray, n_clusters: int, seed: int = 20260608) -> np.ndarray:
    from sklearn.cluster import KMeans
    n_clusters = max(1, min(n_clusters, len(vecs)))
    if n_clusters == 1:
        return np.zeros(len(vecs), dtype=int)
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed)
    return km.fit_predict(vecs)


# ---------------------------------------------------------------------------
# LLM call wrappers (with simple retry, recording into ledger)
# ---------------------------------------------------------------------------


def llm_call_with_ledger(
    alias: str, system_prompt: str, user_prompt: str, ledger: TokenLedger,
    *, phase: str, max_tokens: int = 220, temperature: float = 0.0,
    max_retries: int = 4, node_id: str = "",
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            r = api_call(
                alias,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=120.0,
                max_retries=1,
            )
            wall = time.time() - t0
            text = (r.get("text") or "").strip()
            in_t = int(r.get("usage", {}).get("input_tokens", 0))
            out_t = int(r.get("usage", {}).get("output_tokens", 0))
            ledger.record(
                phase=phase, model_alias=alias,
                input_tokens=in_t, output_tokens=out_t,
                wall_seconds=wall, node_id=node_id,
            )
            return {"text": text, "input_tokens": in_t, "output_tokens": out_t,
                    "wall_seconds": wall, "success": True, "attempts": attempt + 1}
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                import random
                backoff = min(20.0, (2 ** attempt) + random.random())
                logger.warning(
                    f"llm_call retry {attempt+1}/{max_retries} after "
                    f"{type(exc).__name__}: {str(exc)[:120]} — sleeping {backoff:.1f}s"
                )
                time.sleep(backoff)
                continue
            break
    return {"text": "", "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0.0,
            "success": False, "error": f"{type(last_err).__name__}: {str(last_err)[:200]}",
            "attempts": max_retries + 1}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

L1_DISTILL_SYSTEM = """You are an enterprise memory aggregator. Given a small cluster of related L0 memory snippets from one tenant, produce a SHORT distilled summary that:

- captures the central topic, entities, and time references shared by the cluster
- preserves enough signal that a retrieval system can route relevant queries here
- is strictly SHORTER than the input bundle (target: 20-50 words)
- contains NO speculation or invented content
- contains NO gold answer, ground truth, evidence_link kinds, or expected_doc_ids tokens

Return ONLY the summary sentence(s); no prefix, no JSON, no quotation marks."""

L1_DISTILL_USER = """Cluster snippets (tenant={tenant}, count={n}):
---
{body}
---

Distilled summary:"""


L1_CLUSTER_DECISION_SYSTEM = """You are a memory-hierarchy validator. Given a candidate L1 cluster of L0 snippets, decide whether they cohere (true) or whether the cluster is too heterogeneous to be a meaningful node (false).

Reply with exactly one token: COHERENT or HETEROGENEOUS. No other text."""

L1_CLUSTER_DECISION_USER = """Candidate cluster (tenant={tenant}, count={n}):
{body}

Verdict:"""


L2_DISTILL_SYSTEM = """You are an enterprise memory abstractor. Given the L1-level distilled summaries that belong to one tenant, produce a SHORT high-level abstract of the tenant's overall memory state.

- 30-60 words
- captures dominant themes, key entities, time spans
- omit any single specific incident unless it is central
- NO speculation, NO gold-answer tokens, NO JSON

Return ONLY the summary; no prefix, no quotation marks."""

L2_DISTILL_USER = """Tenant: {tenant}
L1 distilled summaries (count={n}):
---
{body}
---

Tenant-level abstract:"""


def _bundle_children_distilled(children: List[DualNode], max_chars: int = 4000) -> str:
    """Render a children's distilled_text list as a numbered bundle."""
    out = []
    used = 0
    for i, c in enumerate(children, 1):
        snip = f"{i}. {c.distilled_text}"
        if used + len(snip) > max_chars:
            out.append(f"... ({len(children) - i + 1} more snippets truncated)")
            break
        out.append(snip)
        used += len(snip)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Build L1 from L0 (per-tenant clustering)
# ---------------------------------------------------------------------------


def _decide_n_clusters(n_l0: int) -> int:
    if n_l0 <= L1_MIN_CLUSTER_SIZE:
        return 1
    target = max(2, math.ceil(n_l0 / L1_TARGET_NODES_PER_CLUSTER))
    return min(target, L1_MAX_CLUSTERS_PER_TENANT)


def build_l1(l0_nodes: List[DualNode], ledger: TokenLedger,
             max_workers: int = 8, alias_status: str = "ACTIVE",
             model_alias: str = GPT54_MINI) -> Tuple[List[DualNode], Dict[str, Any]]:
    """Cluster L0 within each tenant and produce low-level summaries."""
    stats = {
        "n_tenants": 0,
        "n_l1_clusters": 0,
        "n_l1_skipped_singleton": 0,
        "n_l1_validation_pass": 0,
        "n_l1_validation_fail": 0,
        "n_l1_distill_fail": 0,
        "per_tenant": {},
    }

    by_tenant: Dict[str, List[DualNode]] = {}
    for n in l0_nodes:
        by_tenant.setdefault(n.tenant_id, []).append(n)

    cluster_tasks: List[Dict[str, Any]] = []
    for tenant, children in by_tenant.items():
        n_l0 = len(children)
        if n_l0 == 0:
            continue
        stats["n_tenants"] += 1
        n_clusters = _decide_n_clusters(n_l0)
        # Embed once per tenant
        texts = [c.distilled_text or c.detailed_text[:200] for c in children]
        vecs = embed_texts(texts)
        if n_clusters == 1:
            labels = np.zeros(n_l0, dtype=int)
        else:
            labels = kmeans_cluster(vecs, n_clusters)
        # Group L0 by cluster label
        clusters: Dict[int, List[DualNode]] = {}
        for c, lbl in zip(children, labels):
            clusters.setdefault(int(lbl), []).append(c)
        # Drop singleton clusters (they'd violate L1 min size)
        keep_clusters = []
        for lbl, members in clusters.items():
            if len(members) < L1_MIN_CLUSTER_SIZE:
                stats["n_l1_skipped_singleton"] += 1
                # Re-bind singletons into the largest neighbouring cluster
                continue
            keep_clusters.append(members)

        # Reabsorb singletons into the largest cluster
        if stats["n_l1_skipped_singleton"] and keep_clusters:
            singletons = [m for lbl, members in clusters.items() if len(members) < L1_MIN_CLUSTER_SIZE for m in members]
            biggest = max(keep_clusters, key=lambda x: len(x))
            biggest.extend(singletons)
        elif not keep_clusters and clusters:
            # All clusters are singletons (tiny tenant) — merge into one L1
            merged = [m for members in clusters.values() for m in members]
            keep_clusters = [merged]

        stats["per_tenant"][tenant] = {
            "n_l0": n_l0,
            "n_l1_target": n_clusters,
            "n_l1_kept": len(keep_clusters),
        }

        for cidx, members in enumerate(keep_clusters):
            cluster_tasks.append({
                "tenant": tenant,
                "cluster_idx": cidx,
                "members": members,
            })
            stats["n_l1_clusters"] += 1

    logger.info(f"L1: {stats['n_l1_clusters']} clusters across {stats['n_tenants']} tenants; "
                f"calling {model_alias} for cluster decisions + summaries")

    l1_nodes: List[DualNode] = []

    def _process_cluster(task: Dict[str, Any]) -> DualNode:
        tenant = task["tenant"]
        cidx = task["cluster_idx"]
        members: List[DualNode] = task["members"]
        n = len(members)
        body = _bundle_children_distilled(members)

        # 1. Cluster coherence decision (gpt_5_4_mini, very cheap)
        decision = llm_call_with_ledger(
            model_alias,
            L1_CLUSTER_DECISION_SYSTEM,
            L1_CLUSTER_DECISION_USER.format(tenant=tenant, n=n, body=body),
            ledger,
            phase=PHASE_HIERARCHY_BUILD,
            max_tokens=8,
            node_id=f"l1_{tenant}_c{cidx}",
        )
        decision_text = decision["text"].strip().upper()
        if "COHERENT" in decision_text:
            verdict = "COHERENT"
        else:
            verdict = "HETEROGENEOUS"

        # 2. Distill (gpt_5_4_mini)
        distill = llm_call_with_ledger(
            model_alias,
            L1_DISTILL_SYSTEM,
            L1_DISTILL_USER.format(tenant=tenant, n=n, body=body),
            ledger,
            phase=PHASE_DISTILLED_GEN,
            max_tokens=180,
            node_id=f"l1_{tenant}_c{cidx}",
        )
        distilled_text = distill["text"]
        # Fallback if distill failed: take first child's distilled
        if not distilled_text:
            distilled_text = (members[0].distilled_text or "[distill-fail]")[:200]

        # detailed_text = full bundle (rich)
        detailed_text = (
            f"L1 cluster (tenant={tenant}, cluster_idx={cidx}, n_children={n}, "
            f"coherence={verdict})\n"
            f"Children distilled snippets:\n{body}\n"
        )

        # Collect provenance: union of children's evidence ids
        ev: List[str] = []
        for c in members:
            ev.extend(c.source_evidence_ids)
        # Dedupe preserving order
        ev = list(dict.fromkeys(ev))

        node_id = f"L1_{tenant}_c{cidx:03d}"
        dt_tok = tok_count(distilled_text)
        det_tok = tok_count(detailed_text)
        # Guard against the pathological case where distilled accidentally
        # grew larger than detailed (very rare for L1 since detailed
        # repeats children). If so, hard-truncate distilled.
        if det_tok > 0 and dt_tok >= det_tok:
            ids_full = enc().encode(distilled_text)
            keep_ids = ids_full[: max(1, det_tok - 1)]
            distilled_text = enc().decode(keep_ids)
            dt_tok = len(keep_ids)

        return DualNode(
            node_id=node_id,
            level="L1",
            tenant_id=tenant,
            distilled_text=distilled_text,
            detailed_text=detailed_text,
            distilled_tokens=dt_tok,
            detailed_tokens=det_tok,
            source_evidence_ids=ev,
            state=NODE_STATE_LIGHT,
            distilled_text_model_alias=model_alias,
            distilled_text_model_status=alias_status,
            extra={
                "child_node_ids": [c.node_id for c in members],
                "cluster_idx": cidx,
                "coherence_verdict": verdict,
                "distill_success": distill["success"],
                "decision_success": decision["success"],
            },
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_cluster, t): t for t in cluster_tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                n = fut.result()
                l1_nodes.append(n)
                v = n.extra.get("coherence_verdict")
                if v == "COHERENT":
                    stats["n_l1_validation_pass"] += 1
                else:
                    stats["n_l1_validation_fail"] += 1
                if not n.extra.get("distill_success", False):
                    stats["n_l1_distill_fail"] += 1
            except Exception:
                logger.error(traceback.format_exc())
                stats["n_l1_distill_fail"] += 1
            if i % 25 == 0 or i == len(futures):
                logger.info(f"  L1 progress: {i}/{len(futures)}")
    return l1_nodes, stats


# ---------------------------------------------------------------------------
# L2 from L1 (one node per tenant, gpt_5_4 high-level abstraction)
# ---------------------------------------------------------------------------


def build_l2(l1_nodes: List[DualNode], ledger: TokenLedger,
             max_workers: int = 4, alias_status: str = "ACTIVE",
             model_alias: str = GPT54) -> Tuple[List[DualNode], Dict[str, Any]]:
    stats = {"n_l2_nodes": 0, "n_l2_distill_fail": 0}
    by_tenant: Dict[str, List[DualNode]] = {}
    for n in l1_nodes:
        by_tenant.setdefault(n.tenant_id, []).append(n)
    logger.info(f"L2: building {len(by_tenant)} tenant roots via {model_alias} (high-level)")

    def _process_tenant(tenant: str, l1s: List[DualNode]) -> DualNode:
        body = _bundle_children_distilled(l1s, max_chars=6000)
        n = len(l1s)
        distill = llm_call_with_ledger(
            model_alias,
            L2_DISTILL_SYSTEM,
            L2_DISTILL_USER.format(tenant=tenant, n=n, body=body),
            ledger,
            phase=PHASE_DISTILLED_GEN,
            max_tokens=200,
            node_id=f"L2_{tenant}",
        )
        distilled_text = distill["text"]
        if not distilled_text:
            distilled_text = f"[distill-fail] tenant={tenant} n_l1={n}"
        detailed_text = (
            f"L2 root (tenant={tenant}, n_l1_children={n})\n"
            f"L1 distilled summaries:\n{body}\n"
        )
        ev: List[str] = []
        for c in l1s:
            ev.extend(c.source_evidence_ids)
        ev = list(dict.fromkeys(ev))
        node_id = f"L2_{tenant}_root"
        dt_tok = tok_count(distilled_text)
        det_tok = tok_count(detailed_text)
        if det_tok > 0 and dt_tok >= det_tok:
            ids_full = enc().encode(distilled_text)
            keep_ids = ids_full[: max(1, det_tok - 1)]
            distilled_text = enc().decode(keep_ids)
            dt_tok = len(keep_ids)
        return DualNode(
            node_id=node_id,
            level="L2",
            tenant_id=tenant,
            distilled_text=distilled_text,
            detailed_text=detailed_text,
            distilled_tokens=dt_tok,
            detailed_tokens=det_tok,
            source_evidence_ids=ev,
            state=NODE_STATE_LIGHT,
            distilled_text_model_alias=model_alias,
            distilled_text_model_status=alias_status,
            extra={
                "child_node_ids": [c.node_id for c in l1s],
                "distill_success": distill["success"],
            },
        )

    l2_nodes: List[DualNode] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_tenant, t, l1s): t for t, l1s in by_tenant.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                node = fut.result()
                l2_nodes.append(node)
                stats["n_l2_nodes"] += 1
                if not node.extra.get("distill_success", False):
                    stats["n_l2_distill_fail"] += 1
            except Exception:
                logger.error(traceback.format_exc())
                stats["n_l2_distill_fail"] += 1
            if i % 5 == 0 or i == len(futures):
                logger.info(f"  L2 progress: {i}/{len(futures)}")
    return l2_nodes, stats


# ---------------------------------------------------------------------------
# Provenance graph (parent->children) for traceability check
# ---------------------------------------------------------------------------


def build_parent_child_index(all_nodes: List[DualNode]) -> Dict[str, List[str]]:
    """node_id -> list of child node_ids (empty for L0)."""
    out: Dict[str, List[str]] = {n.node_id: list(n.extra.get("child_node_ids", []) or [])
                                 for n in all_nodes}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", required=True,
                    help="Path to own_full_l0_nodes.parquet (the manifest index)")
    ap.add_argument("--raw_dir", required=True,
                    help="Directory containing source-resolved L0 JSONL records")
    ap.add_argument("--out", required=True)
    ap.add_argument("--low_level_alias", default=GPT54_MINI)
    ap.add_argument("--high_level_alias", default=GPT54)
    ap.add_argument("--tokenizer", default="cl100k_base")
    ap.add_argument("--max_workers_l1", type=int, default=8)
    ap.add_argument("--max_workers_l2", type=int, default=4)
    ap.add_argument("--alias_status", default="ACTIVE",
                    help="ACTIVE | PROVISIONAL — written into ledger for replay tracking")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.low_level_alias != GPT54_MINI or args.high_level_alias != GPT54:
        logger.warning(f"non-canonical aliases requested: low={args.low_level_alias} high={args.high_level_alias}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tokenizer != "cl100k_base":
        raise SystemExit("Only cl100k_base is supported (matches manifest tokenizer)")

    # 1. Load L0 manifest index, then resolve to raw L0 records
    logger.info(f"loading manifest index {args.l0}")
    idx = pd.read_parquet(args.l0)
    raw_all = load_l0_raw(Path(args.raw_dir))
    l0_raw = filter_to_manifest(raw_all, idx)
    if len(l0_raw) != len(idx):
        raise SystemExit(
            f"L0 raw load mismatch: got {len(l0_raw)} but manifest expects {len(idx)}"
        )

    ledger = TokenLedger(
        run_id=f"step3_build_own_full_{int(time.time())}",
        method="V4_step3_hierarchy",
        alias_status=args.alias_status,
        alias_chosen_at=dt.datetime.utcnow().isoformat() + "Z",
        alias_chosen_by="ml_engineer_v4_step3",
    )

    # 2. Build L0 dual nodes (deterministic, no LLM)
    t_l0 = time.time()
    l0_nodes = build_l0_dualnodes(l0_raw)
    t_l0 = time.time() - t_l0
    logger.info(f"L0: {len(l0_nodes)} dual nodes built in {t_l0:.1f}s")

    # 3. Build L1 (per-tenant clusters + gpt_5_4_mini decision/distill)
    t_l1 = time.time()
    l1_nodes, l1_stats = build_l1(
        l0_nodes,
        ledger,
        max_workers=args.max_workers_l1,
        alias_status=args.alias_status,
        model_alias=args.low_level_alias,
    )
    t_l1 = time.time() - t_l1
    logger.info(f"L1: {len(l1_nodes)} nodes built in {t_l1:.1f}s")

    # 4. Build L2 (per-tenant root + gpt_5_4 abstraction)
    t_l2 = time.time()
    l2_nodes, l2_stats = build_l2(
        l1_nodes,
        ledger,
        max_workers=args.max_workers_l2,
        alias_status=args.alias_status,
        model_alias=args.high_level_alias,
    )
    t_l2 = time.time() - t_l2
    logger.info(f"L2: {len(l2_nodes)} nodes built in {t_l2:.1f}s")

    all_nodes = l0_nodes + l1_nodes + l2_nodes
    logger.info(f"hierarchy total: {len(all_nodes)} nodes "
                f"(L0={len(l0_nodes)} L1={len(l1_nodes)} L2={len(l2_nodes)})")

    # 5. Write outputs
    hierarchy_path = out_dir / "hierarchy.json"
    with open(hierarchy_path, "w") as f:
        for n in all_nodes:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"wrote {hierarchy_path}")

    parent_child_path = out_dir / "parent_child_index.json"
    with open(parent_child_path, "w") as f:
        json.dump(build_parent_child_index(all_nodes), f, indent=2)

    ledger.export(str(out_dir / "token_ledger.json"), include_raw=False)
    # Also keep a raw-records copy for deeper analysis if needed
    ledger.export(str(out_dir / "token_ledger_with_records.json"), include_raw=True)

    report = {
        "build_step": "v4_step3_build_hierarchy",
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "tokenizer": "cl100k_base",
        "model_aliases": {
            "low_level": args.low_level_alias,
            "high_level": args.high_level_alias,
            "alias_status": args.alias_status,
        },
        "n_nodes": {
            "L0": len(l0_nodes),
            "L1": len(l1_nodes),
            "L2": len(l2_nodes),
            "total": len(all_nodes),
        },
        "wall_seconds": {
            "L0": round(t_l0, 1),
            "L1": round(t_l1, 1),
            "L2": round(t_l2, 1),
        },
        "l1_stats": l1_stats,
        "l2_stats": l2_stats,
        "ledger_totals": ledger.grand_total(),
    }
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"wrote {out_dir/'build_report.json'}")

    logger.info("=== BUILD DONE ===")
    logger.info(f"  L0={len(l0_nodes)} L1={len(l1_nodes)} L2={len(l2_nodes)}")
    logger.info(f"  ledger totals={ledger.grand_total()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
