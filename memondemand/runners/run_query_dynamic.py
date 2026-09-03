"""
V5 Dynamic Query Runner
==============================================
Query-time LLM Navigation (DESCEND/LATERAL/ANSWER/STOP_INSUFFICIENT)
+ On-demand Promotion + Answer generation.

Usage:
  python run_query_dynamic.py \
    --tier 10M \
    --hierarchy results/v5/erag_10M/hierarchy/hierarchy.json \
    --queries manifests/erag_queries.parquet \
    --out_dir results/v5/erag_10M/query_run \
    --method V5 \
    --max_queries 500 \
    --max_nav_steps 8 \
    --top_k_candidates 6

Outputs:
  out_dir/answers.jsonl       # one JSON per query
  out_dir/run_summary.json    # aggregate stats + timing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
AZURE_DEPLOYMENT = "gpt-5.4-mini"
AZURE_API_VERSION = "2024-12-01-preview"

BEDROCK_ENDPOINT = "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
BEDROCK_MODEL    = "openai.gpt-5.4"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NAV_ACTION_DESCEND = "DESCEND"
NAV_ACTION_LATERAL = "LATERAL"
NAV_ACTION_ANSWER  = "ANSWER"
NAV_ACTION_STOP    = "STOP_INSUFFICIENT"

PROMO_ACTION_PROMOTE    = "PROMOTE"
PROMO_ACTION_KEEP_LIGHT = "KEEP_LIGHT"

# ── Node structure ─────────────────────────────────────────────────────────────
@dataclass
class Node:
    node_id: str
    level: int
    tenant_id: str
    distilled_text: str
    detailed_text: str
    parent_id: Optional[str]
    children_ids: List[str]
    source_evidence_ids: List[str]
    state: str = "LIGHT"   # LIGHT | PROMOTED
    vec: Optional[np.ndarray] = None  # populated at index time

    @classmethod
    def from_dict(cls, d: Dict) -> "Node":
        return cls(
            node_id=d["node_id"],
            level=d["level"],
            tenant_id=d.get("tenant_id", ""),
            distilled_text=d.get("distilled_text", ""),
            detailed_text=d.get("detailed_text", ""),
            parent_id=d.get("parent_id"),
            children_ids=d.get("children_ids", []),
            source_evidence_ids=d.get("source_evidence_ids", []),
            state=d.get("state", "LIGHT"),
        )


# ── Hierarchy ──────────────────────────────────────────────────────────────────
class Hierarchy:
    def __init__(self, nodes: List[Node]):
        self.by_id: Dict[str, Node] = {n.node_id: n for n in nodes}
        self.max_depth = max(n.level for n in nodes)
        # Group by level
        self.by_level: Dict[int, List[Node]] = defaultdict(list)
        for n in nodes:
            self.by_level[n.level].append(n)
        # child → parent map
        self.parent_map: Dict[str, str] = {}
        for n in nodes:
            for cid in n.children_ids:
                self.parent_map[cid] = n.node_id

    def top_nodes(self) -> List[Node]:
        return self.by_level[self.max_depth]

    def children_of(self, node_id: str) -> List[Node]:
        n = self.by_id.get(node_id)
        if not n:
            return []
        return [self.by_id[c] for c in n.children_ids if c in self.by_id]

    def siblings_of(self, node_id: str) -> List[Node]:
        """Same-level nodes with the same parent."""
        parent_id = self.parent_map.get(node_id)
        if not parent_id:
            return []
        parent = self.by_id.get(parent_id)
        if not parent:
            return []
        return [self.by_id[c] for c in parent.children_ids
                if c in self.by_id and c != node_id]


def load_hierarchy(path: str) -> Hierarchy:
    nodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(Node.from_dict(json.loads(line)))
    log.info("Loaded %d nodes from %s (max_depth=%d)",
             len(nodes), path, max(n.level for n in nodes))
    return Hierarchy(nodes)


# ── Embedder ───────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedder %s ...", model_name)
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        ).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def index_hierarchy(hier: Hierarchy, embedder: Embedder) -> None:
    """Populate vec field for all nodes (distilled embedding)."""
    log.info("Embedding %d nodes for cosine index ...", len(hier.by_id))
    t0 = time.time()
    nodes = list(hier.by_id.values())
    texts = [n.distilled_text for n in nodes]
    vecs = embedder.encode(texts)
    for n, v in zip(nodes, vecs):
        n.vec = v
    log.info("Indexed %d nodes in %.1fs", len(nodes), time.time() - t0)


def top_k_by_cosine(query_vec: np.ndarray, candidates: List[Node], k: int) -> List[Tuple[Node, float]]:
    if not candidates:
        return []
    vecs = np.stack([n.vec for n in candidates if n.vec is not None])
    sims = (vecs @ query_vec).tolist()
    ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
    return ranked[:k]


# ── LLM Clients ───────────────────────────────────────────────────────────────
def _azure_client():
    from openai import AzureOpenAI
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_LLM_API_KEY", "")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT not set")
    if not key:
        raise RuntimeError("AZURE_LLM_API_KEY not set")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key,
                       api_version=AZURE_API_VERSION)


def call_mini(prompt: str, max_tokens: int = 512) -> Tuple[str, int, int]:
    """Returns (text, n_in, n_out)."""
    client = _azure_client()
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


# ── Navigator ─────────────────────────────────────────────────────────────────
def build_nav_prompt(query: str, current_nodes: List[Tuple[Node, float]],
                     evidence_so_far: List[Node], steps_left: int) -> str:
    node_list = "\n".join(
        f"  [{i+1}] node_id={n.node_id} level=L{n.level} sim={sim:.3f}\n"
        f"      distilled: {n.distilled_text[:200]}"
        for i, (n, sim) in enumerate(current_nodes[:8])
    )
    evidence_summary = (
        f"{len(evidence_so_far)} nodes collected so far."
        if evidence_so_far else "No evidence collected yet."
    )
    return f"""You are navigating a memory hierarchy to answer a query.

Query: {query}

Current candidate nodes (top by relevance):
{node_list}

Evidence status: {evidence_summary}
Navigation budget remaining: {steps_left} steps

Choose ONE action:
- DESCEND: Go deeper into the most relevant node (expand children)
- LATERAL: Search sibling nodes at same level for more coverage
- ANSWER: Current evidence is sufficient to answer the query
- STOP_INSUFFICIENT: Cannot find relevant evidence, stop

Respond with ONLY valid JSON (no markdown, no explanation):
{{"action": "DESCEND|LATERAL|ANSWER|STOP_INSUFFICIENT", "target_node_id": "<node_id or empty>", "reason": "<brief>"}}"""


def build_promo_prompt(query: str, candidates: List[Tuple[Node, float]]) -> str:
    node_list = "\n".join(
        f"  [{i+1}] node_id={n.node_id} (L{n.level}) sim={sim:.3f}\n"
        f"      distilled: {n.distilled_text[:150]}"
        for i, (n, sim) in enumerate(candidates)
    )
    return f"""You are deciding which memory nodes to promote (load full detail) for answering a query.

Query: {query}

Candidate nodes:
{node_list}

For each node, decide: PROMOTE (load full detailed text) or KEEP_LIGHT (distilled only).
Promote a node if its full text is likely needed to answer the query precisely.

Respond with ONLY valid JSON array (no markdown):
[{{"node_id": "...", "decision": "PROMOTE|KEEP_LIGHT"}}]"""


def build_answer_prompt(query: str, context_items: List[Tuple[Node, str]]) -> str:
    context = "\n\n".join(
        f"[Source {i+1} | {n.node_id} | L{n.level}]\n{text[:1500]}"
        for i, (n, text) in enumerate(context_items[:8])
    )
    return f"""Answer the following query using only the provided context. Be specific and cite evidence.
If the answer cannot be found in the context, say "Insufficient information."

Query: {query}

Context:
{context}

Answer:"""


def parse_json_safe(text: str, default: Any) -> Any:
    """Extract first JSON object/array from text."""
    text = text.strip()
    for pat in [r'\[.*?\]', r'\{.*?\}']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    try:
        return json.loads(text)
    except Exception:
        return default


# ── Per-query pipeline ─────────────────────────────────────────────────────────
def run_single_query(
    query_id: str,
    query_text: str,
    tenant_id: str,
    hier: Hierarchy,
    embedder: Embedder,
    max_nav_steps: int = 8,
    top_k: int = 6,
) -> Dict[str, Any]:
    t_wall_start = time.time()

    # ── Embed query ────────────────────────────────────────────────────────
    t0 = time.time()
    q_vec = embedder.encode_one(query_text)
    t_embedding_ms = (time.time() - t0) * 1000

    # ── Filter by tenant ───────────────────────────────────────────────────
    tenant_nodes_by_level: Dict[int, List[Node]] = defaultdict(list)
    for n in hier.by_id.values():
        if n.tenant_id == tenant_id or n.level > 0:
            tenant_nodes_by_level[n.level].append(n)

    # Start at top level
    current_level = hier.max_depth
    current_pool = hier.top_nodes()

    evidence_nodes: List[Node] = []
    visited_ids: set = set()

    t_retrieval_ms = 0.0
    t_navigation_llm_ms = 0.0
    n_llm_calls_navigation = 0
    nav_steps_taken = 0
    nav_actions: List[str] = []

    # ── Navigation loop ────────────────────────────────────────────────────
    for step in range(max_nav_steps):
        steps_left = max_nav_steps - step

        # Rank current pool
        t0 = time.time()
        ranked = top_k_by_cosine(q_vec, current_pool, k=top_k)
        t_retrieval_ms += (time.time() - t0) * 1000

        if not ranked:
            nav_actions.append("STOP_EMPTY_POOL")
            break

        # LLM navigation decision
        nav_prompt = build_nav_prompt(query_text, ranked, evidence_nodes, steps_left)
        t0 = time.time()
        try:
            nav_text, nav_in, nav_out = call_mini(nav_prompt, max_tokens=128)
            nav_decision = parse_json_safe(nav_text, {"action": "ANSWER", "target_node_id": "", "reason": "parse_fail"})
        except Exception as e:
            log.warning("Nav LLM error: %s", e)
            nav_decision = {"action": "ANSWER", "target_node_id": "", "reason": "llm_error"}
            nav_in, nav_out = 0, 0
        t_navigation_llm_ms += (time.time() - t0) * 1000
        n_llm_calls_navigation += 1
        nav_steps_taken += 1

        action = nav_decision.get("action", "ANSWER").upper().replace("-", "_")
        target_id = nav_decision.get("target_node_id", "")
        nav_actions.append(action)

        if action == NAV_ACTION_ANSWER or action == NAV_ACTION_STOP:
            # Collect top-k from current pool as evidence
            for n, _ in ranked[:top_k]:
                if n.node_id not in visited_ids:
                    evidence_nodes.append(n)
                    visited_ids.add(n.node_id)
            break

        elif action == NAV_ACTION_DESCEND:
            # Find target node (use top-1 if target_id missing)
            target = None
            if target_id and target_id in hier.by_id:
                target = hier.by_id[target_id]
            else:
                target = ranked[0][0]
            # Collect target as evidence, then move to children
            if target.node_id not in visited_ids:
                evidence_nodes.append(target)
                visited_ids.add(target.node_id)
            children = hier.children_of(target.node_id)
            if children:
                current_pool = children
                current_level = max(n.level for n in children)
            else:
                # At leaf — done
                for n, _ in ranked[:top_k]:
                    if n.node_id not in visited_ids:
                        evidence_nodes.append(n)
                        visited_ids.add(n.node_id)
                break

        elif action == NAV_ACTION_LATERAL:
            # Add current evidence, then expand to siblings
            for n, _ in ranked[:min(2, top_k)]:
                if n.node_id not in visited_ids:
                    evidence_nodes.append(n)
                    visited_ids.add(n.node_id)
            target = ranked[0][0] if not target_id else hier.by_id.get(target_id, ranked[0][0])
            siblings = hier.siblings_of(target.node_id)
            if siblings:
                current_pool = siblings
            else:
                # No siblings: try parent's siblings
                parent_id = hier.parent_map.get(target.node_id)
                if parent_id:
                    current_pool = hier.siblings_of(parent_id) or current_pool
                break
        else:
            # Unknown action → collect and stop
            for n, _ in ranked[:top_k]:
                if n.node_id not in visited_ids:
                    evidence_nodes.append(n)
                    visited_ids.add(n.node_id)
            break

    # Always add direct L0 fallback to ensure answerability
    # Navigation gives hierarchy context; L0 gives grounded detail
    t0 = time.time()
    l0_nodes = hier.by_level.get(0, [])
    if l0_nodes:
        l0_ranked = top_k_by_cosine(q_vec, l0_nodes, k=top_k)
        for n, _ in l0_ranked:
            if n.node_id not in visited_ids:
                evidence_nodes.append(n)
                visited_ids.add(n.node_id)
    t_retrieval_ms += (time.time() - t0) * 1000

    # Final fallback: still empty → use top-level nodes
    if not evidence_nodes:
        t0 = time.time()
        fallback = top_k_by_cosine(q_vec, hier.top_nodes(), k=top_k)
        t_retrieval_ms += (time.time() - t0) * 1000
        evidence_nodes = [n for n, _ in fallback]

    # ── Promotion ──────────────────────────────────────────────────────────
    # Score candidates
    t0 = time.time()
    evidence_nodes_unique = list({n.node_id: n for n in evidence_nodes}.values())
    candidate_scores = top_k_by_cosine(q_vec, evidence_nodes_unique, k=len(evidence_nodes_unique))

    promo_prompt = build_promo_prompt(query_text, candidate_scores[:6])
    try:
        promo_text, promo_in, promo_out = call_mini(promo_prompt, max_tokens=256)
        promo_decisions = parse_json_safe(promo_text, [])
        if not isinstance(promo_decisions, list):
            promo_decisions = []
    except Exception as e:
        log.warning("Promo LLM error: %s", e)
        promo_decisions = []
        promo_in, promo_out = 0, 0

    t_promotion_llm_ms = (time.time() - t0) * 1000
    n_llm_calls_promotion = 1

    # Build promoted set
    promo_map: Dict[str, str] = {
        d.get("node_id", ""): d.get("decision", PROMO_ACTION_KEEP_LIGHT)
        for d in promo_decisions if isinstance(d, dict)
    }
    n_promoted = 0
    context_items: List[Tuple[Node, str]] = []
    for n, _ in candidate_scores[:8]:
        decision = promo_map.get(n.node_id, PROMO_ACTION_KEEP_LIGHT)
        # L0 nodes always use detailed_text (full content); upper nodes use distilled unless promoted
        if decision == PROMO_ACTION_PROMOTE or n.level == 0:
            context_items.append((n, n.detailed_text))
            n_promoted += 1
        else:
            context_items.append((n, n.distilled_text))

    # ── Answer generation ──────────────────────────────────────────────────
    t0 = time.time()
    answer_prompt = build_answer_prompt(query_text, context_items)
    try:
        answer_text, ans_in, ans_out = call_mini(answer_prompt, max_tokens=512)
    except Exception as e:
        log.warning("Answer LLM error: %s", e)
        answer_text, ans_in, ans_out = "", 0, 0
    t_answer_llm_ms = (time.time() - t0) * 1000

    # ── Citation check ─────────────────────────────────────────────────────
    # L0 citation = any L0 node's doc_id appears in evidence
    l0_evidence_ids = set()
    for n in evidence_nodes_unique:
        l0_evidence_ids.update(n.source_evidence_ids)
    has_l0_cite = len(l0_evidence_ids) > 0

    t_wall_total_ms = (time.time() - t_wall_start) * 1000

    return {
        "query_id": query_id,
        "tenant_id": tenant_id,
        "answer": answer_text,
        "answer_nonempty": bool(answer_text and answer_text != "Insufficient information."),
        "has_l0_cite": has_l0_cite,
        "n_evidence_nodes": len(evidence_nodes_unique),
        "n_promoted": n_promoted,
        "nav_actions": nav_actions,
        "nav_steps_taken": nav_steps_taken,
        # Timing
        "t_embedding_ms": round(t_embedding_ms, 1),
        "t_retrieval_ms": round(t_retrieval_ms, 1),
        "t_navigation_llm_ms": round(t_navigation_llm_ms, 1),
        "t_promotion_llm_ms": round(t_promotion_llm_ms, 1),
        "t_answer_llm_ms": round(t_answer_llm_ms, 1),
        "t_wall_total_ms": round(t_wall_total_ms, 1),
        # LLM call counts
        "n_llm_calls_navigation": n_llm_calls_navigation,
        "n_llm_calls_promotion": n_llm_calls_promotion,
        "n_llm_calls_answer": 1,
        # Tokens
        "tokens_answer_in": ans_in,
        "tokens_answer_out": ans_out,
        "tokens_promo_in": promo_in,
        "tokens_promo_out": promo_out,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",       required=True)
    parser.add_argument("--hierarchy",  required=True)
    parser.add_argument("--queries",    required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--method",     default="V5")
    parser.add_argument("--max_queries", type=int, default=500)
    parser.add_argument("--max_nav_steps", type=int, default=8)
    parser.add_argument("--top_k_candidates", type=int, default=6)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"

    log.info("=== V5 Dynamic Query Runner ===")
    log.info("tier=%s  method=%s  max_q=%d  nav_steps=%d",
             args.tier, args.method, args.max_queries, args.max_nav_steps)

    # Load hierarchy + index
    hier = load_hierarchy(args.hierarchy)
    embedder = Embedder()
    index_hierarchy(hier, embedder)

    # Load queries
    df = pd.read_parquet(args.queries)
    df = df.head(args.max_queries)
    log.info("Loaded %d queries", len(df))

    # Resume support
    done_ids: set = set()
    if args.resume and answers_path.exists():
        with open(answers_path) as f:
            for line in f:
                d = json.loads(line)
                done_ids.add(d.get("query_id", ""))
        log.info("Resuming: %d already done", len(done_ids))

    # Run queries
    t_run_start = time.time()
    n_done = len(done_ids)
    n_errors = 0
    with open(answers_path, "a") as fout:
        for _, row in df.iterrows():
            qid = str(row["query_id"])
            if qid in done_ids:
                continue
            try:
                result = run_single_query(
                    query_id=qid,
                    query_text=str(row["query_text"]),
                    tenant_id=str(row.get("tenant_id", "")),
                    hier=hier,
                    embedder=embedder,
                    max_nav_steps=args.max_nav_steps,
                    top_k=args.top_k_candidates,
                )
                result["method"] = args.method
                result["tier"] = args.tier
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 20 == 0:
                    elapsed = time.time() - t_run_start
                    log.info("Progress: %d/%d  elapsed=%.0fs  avg=%.1fs/q",
                             n_done, len(df), elapsed, elapsed / max(1, n_done - len(done_ids)))
            except Exception as e:
                log.error("Query %s error: %s", qid, e)
                n_errors += 1

    # Summary
    answers = []
    with open(answers_path) as f:
        for line in f:
            answers.append(json.loads(line))

    n_total = len(answers)
    n_nonempty = sum(1 for a in answers if a.get("answer_nonempty"))
    n_l0_cite  = sum(1 for a in answers if a.get("has_l0_cite"))
    mean_wall  = np.mean([a["t_wall_total_ms"] for a in answers]) if answers else 0
    mean_nav   = np.mean([a["t_navigation_llm_ms"] for a in answers]) if answers else 0
    mean_promo = np.mean([a["t_promotion_llm_ms"] for a in answers]) if answers else 0
    mean_ans   = np.mean([a["t_answer_llm_ms"] for a in answers]) if answers else 0
    mean_steps = np.mean([a["nav_steps_taken"] for a in answers]) if answers else 0
    mean_promo_n = np.mean([a["n_promoted"] for a in answers]) if answers else 0

    # Cost estimate
    total_in  = sum(a.get("tokens_answer_in",0) + a.get("tokens_promo_in",0) for a in answers)
    total_out = sum(a.get("tokens_answer_out",0) + a.get("tokens_promo_out",0) for a in answers)
    cost_usd = (total_in * 0.40 + total_out * 1.60) / 1_000_000

    summary = {
        "tier": args.tier, "method": args.method,
        "n_queries": n_total, "n_errors": n_errors,
        "n_answers_nonempty": n_nonempty,
        "n_with_l0_cite": n_l0_cite,
        "citation_rate": round(n_l0_cite / max(1, n_total), 4),
        "mean_wall_ms": round(mean_wall, 1),
        "mean_navigation_llm_ms": round(mean_nav, 1),
        "mean_promotion_llm_ms": round(mean_promo, 1),
        "mean_answer_llm_ms": round(mean_ans, 1),
        "mean_nav_steps": round(mean_steps, 2),
        "mean_n_promoted": round(mean_promo_n, 2),
        "total_tokens_in": total_in, "total_tokens_out": total_out,
        "cost_usd": round(cost_usd, 4),
    }

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=== DONE ===")
    log.info("n=%d  citation_rate=%.3f  mean_wall=%.0fms  cost=$%.4f",
             n_total, summary["citation_rate"], mean_wall, cost_usd)
    log.info("Nav breakdown: nav=%.0fms  promo=%.0fms  ans=%.0fms  steps=%.1f",
             mean_nav, mean_promo, mean_ans, mean_steps)


if __name__ == "__main__":
    main()
