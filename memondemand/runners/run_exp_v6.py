#!/usr/bin/env python3
"""Experiment V6: Dual-Memory Prompt v2"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = os.environ.get(
    "MEMONDEMAND_REPO_ROOT", str(Path(__file__).resolve().parents[2])
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Inherit everything from run_stream_v5
import memondemand.runners.run_stream_v5 as _base

# ── Patched prompts ───────────────────────────────────────────────────────────

ANSWER_SYSTEM_V6 = """You are an enterprise memory question-answering assistant.

You will be given:
  1. HIERARCHY CONTEXT snippets tagged [CTX]: high-level memory summaries.
     Use these to understand scope and background. Do NOT cite [CTX] node_ids.
  2. SOURCE EVIDENCE snippets tagged [EVID]: detailed source documents.
     These are your primary citation sources.

Rules:
- Answer the query in 2-5 sentences using the provided evidence (and context for framing).
- After the answer, on a NEW LINE, output: CITED: id1,id2,...
- Include in CITED every [EVID] node_id whose content directly contributed to your answer.
  If multiple evidence snippets support the answer, CITE ALL OF THEM — do not truncate the list.
- Only omit an evidence node if its content is completely unrelated to the query.
- Do NOT cite [CTX] node_ids — those are context only.
- If no evidence applies, answer "INSUFFICIENT EVIDENCE" and output CITED: (empty).
- Do not invent facts not present in the provided evidence."""

ANSWER_USER_V6 = """QUERY: {query}

{context_block}

Answer, then on a new line: CITED: <comma-separated [EVID] node_ids you used>"""


def format_dual_context(ctx_items: List[Tuple[str, str, str]]) -> str:
    """Separate L0 (EVIDENCE) from upper-level (CONTEXT).
    ctx_items: list of (node_id, level, text)
    """
    ctx_lines = []
    evid_lines = []
    for nid, lvl, txt in ctx_items:
        if lvl == "L0":
            evid_lines.append(f"- [EVID|{nid}] {txt[:1800]}")
        else:
            ctx_lines.append(f"- [CTX|{nid}] ({lvl}) {txt[:400]}")
    parts = []
    if ctx_lines:
        parts.append("HIERARCHY CONTEXT (understand scope; do NOT cite these):")
        parts.extend(ctx_lines)
        parts.append("")
    if evid_lines:
        parts.append("SOURCE EVIDENCE (cite node_ids if used):")
        parts.extend(evid_lines)
    return "\n".join(parts) if parts else "(none)"


def parse_cited_v6(answer_text: str, candidate_ids: List[str]) -> List[str]:
    """Extract cited node_ids — handles both raw ids and [EVID|id] format."""
    import re
    cset = set(candidate_ids)
    out, seen = [], set()
    # Find CITED: line
    for m in re.findall(r"CITED\s*:\s*([A-Za-z0-9_,|\s\-]*)$", answer_text or "", re.MULTILINE):
        for tok in m.split(","):
            tok = tok.strip()
            # strip [EVID|...] wrapper if present
            inner = re.sub(r"^\[EVID\|", "", tok)
            inner = re.sub(r"\]$", "", inner).strip()
            if inner and inner in cset and inner not in seen:
                out.append(inner); seen.add(inner)
            elif tok and tok in cset and tok not in seen:
                out.append(tok); seen.add(tok)
    return out


# ── Main runner ───────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def run_query_v6(ctx: _base.RunnerCtx, qid: str, query: str,
                 query_idx: int, max_nav_steps: int = 3) -> _base.QueryRecord:
    """V6: dual-memory prompt + cite-all + promoted context."""
    rec = _base.QueryRecord(query_id=qid, query_text=query,
                            method=ctx.method, tier=ctx.tier)
    t_wall = time.time()

    # ── BM25 global L0 retrieval ──────────────────────────────────────────
    t0 = time.time()
    q_vec = ctx.embedder.encode([query])[0]
    rec.t_query_embedding_ms = (time.time() - t0) * 1000
    t0 = time.time()
    bm25_hits = _base._bm25_top_k(ctx, query, ctx.top_k_distilled)
    rec.t_retrieval_ms = (time.time() - t0) * 1000

    bm25_l0_ids: Set[str] = {nid for nid, _ in bm25_hits}

    # ── Navigation loop (max 3 steps) to collect upper-level context ──────
    t0 = time.time()
    top_idx = _base.get_index(ctx, ctx.top_level)
    frontier_hits = top_idx.search(q_vec, top_k=8)
    rec.t_retrieval_ms += (time.time() - t0) * 1000

    visible_frontier = list(frontier_hits)
    visited_distilled: Dict[str, str] = {}
    promoted_l0_ids: Set[str] = set()  # navigator-found L0 nodes
    prior_actions: List[str] = []
    decision: Dict[str, Any] = {"action": "ANSWER"}

    for step in range(max_nav_steps):
        if not visible_frontier:
            break
        front = visible_frontier[:16]
        block_lines = []
        for nid, sim in front:
            n = ctx.hierarchy.get(nid)
            if n is None: continue
            visited_distilled[nid] = n.distilled_text or ""
            block_lines.append(
                f"- [{nid}] ({n.level}) sim={sim:.3f} {(n.distilled_text or '')[:200]}"
            )
        user = _base.NAV_USER_TMPL.format(
            query=query, step=step+1, max_steps=max_nav_steps,
            n=len(front),
            prior_actions=",".join(prior_actions[-3:]) or "(none)",
            frontier_block="\n".join(block_lines),
        )
        nav_text, in_t, out_t, wall = _base.call_llm(
            ctx.alias_navigator, _base.NAV_SYSTEM, user,
            max_tokens=200, temperature=0.0,
            ledger=ctx.ledger, phase=_base.PHASE_RETRIEVAL, query_id=qid,
        )
        rec.t_llm_navigation_seconds += wall
        rec.tokens_navigation_in += in_t
        rec.tokens_navigation_out += out_t
        rec.cost_usd += _base.cost_for(ctx.alias_navigator, in_t, out_t)
        rec.n_navigation_steps += 1

        decision = _base.parse_nav_json(nav_text)
        action = str(decision.get("action", "ANSWER")).upper().strip()
        if action not in {"DESCEND", "LATERAL", "ANSWER", "STOP_INSUFFICIENT"}:
            action = "ANSWER"
        if rec.n_navigation_steps >= max_nav_steps and action in {"DESCEND", "LATERAL"}:
            action = "ANSWER"
        prior_actions.append(action)
        rec.nav_actions.append(action)

        if action in {"ANSWER", "STOP_INSUFFICIENT"}:
            break

        chosen = [c for c in (decision.get("chosen_node_ids") or [])
                  if c in ctx.hierarchy]
        if not chosen:
            chosen = [nid for nid, _ in front[:2]]

        new_ids: List[str] = []
        for c in chosen:
            kids = ctx.parent_child.get(c, [])
            new_ids.extend(kids if kids else [c])
        new_ids = list(dict.fromkeys(new_ids))
        if not new_ids: break

        child_level = ctx.hierarchy[new_ids[0]].level
        t0 = time.time()
        if child_level == "L0":
            ranked = _base._bm25_top_k(ctx, query, 16, filter_ids=new_ids)
            # Re-enable: navigator-found L0 nodes → add to promoted set for evidence context
            for nid, _ in ranked[:8]:
                if nid not in bm25_l0_ids:
                    promoted_l0_ids.add(nid)
        else:
            idx = _base.get_index(ctx, child_level)
            ranked = idx.search(q_vec, top_k=16, filter_ids=new_ids)
        rec.t_retrieval_ms += (time.time() - t0) * 1000
        visible_frontier = ranked

    # ── Promotion gate on BM25 candidates (state update only) ────────────
    if ctx.promotion is not None:
        tp = time.time()
        cand_pool = [nid for nid, _ in bm25_hits[:ctx.max_detailed_load]
                     if ctx.hierarchy[nid].level == "L0"
                     and ctx.hierarchy[nid].last_used_query_idx < query_idx]
        if cand_pool:
            decs = ctx.promotion.decide(
                query=query, candidate_node_ids=cand_pool[:ctx.max_detailed_load],
                query_idx=query_idx, query_id=qid,
                alias=ctx.alias_low, ledger=ctx.ledger,
            )
            rec.n_promote_decisions += len(decs)
            counts = ctx.promotion.apply_decisions(decs, query_idx, state_log=ctx.state_log)
            rec.n_promote_events += counts.get("PROMOTE", 0)
        rec.t_llm_promotion_seconds += time.time() - tp

    # ── Build dual-memory context ─────────────────────────────────────────
    ctx_items: List[Tuple[str, str, str]] = []
    l0_in_ctx: Set[str] = set()

    # Section 1: BM25 L0 evidence (detailed)
    for nid, _sim in bm25_hits:
        n = ctx.hierarchy.get(nid)
        if n is None or n.level != "L0": continue
        ctx_items.append((nid, "L0", (n.detailed_text or n.distilled_text or "")[:1800]))
        l0_in_ctx.add(nid)

    # Section 2: Navigator-found additional L0 nodes (detailed, re-enabled)
    for nid in list(promoted_l0_ids)[:4]:
        if nid in l0_in_ctx: continue
        n = ctx.hierarchy.get(nid)
        if n is None: continue
        ctx_items.append((nid, "L0", (n.detailed_text or n.distilled_text or "")[:1800]))
        l0_in_ctx.add(nid)

    # Section 3: Upper-level distilled summaries from navigation frontier
    upper_added = 0
    for nid, _sim in visible_frontier[:6]:
        if upper_added >= 4: break
        n = ctx.hierarchy.get(nid)
        if n is None or n.level == "L0": continue
        if any(c[0] == nid for c in ctx_items): continue
        ctx_items.append((nid, n.level, (n.distilled_text or "")[:400]))
        upper_added += 1

    # ── Build dual-format context block ──────────────────────────────────
    # Reorder: CTX first, EVID after
    ctx_only = [(nid, lvl, txt) for nid, lvl, txt in ctx_items if lvl != "L0"]
    evid_only = [(nid, lvl, txt) for nid, lvl, txt in ctx_items if lvl == "L0"]
    reordered = ctx_only + evid_only

    context_block = format_dual_context(reordered)

    # ── Answer generation ─────────────────────────────────────────────────
    t0 = time.time()
    user = ANSWER_USER_V6.format(query=query, context_block=context_block)
    answer, in_t, out_t, wall = _base.call_llm(
        ctx.alias_answer, ANSWER_SYSTEM_V6, user,
        max_tokens=500, temperature=0.0,
        ledger=ctx.ledger, phase=_base.PHASE_FINAL_ANSWER, query_id=qid,
    )
    rec.t_llm_answer_seconds = wall
    rec.tokens_answer_in = in_t
    rec.tokens_answer_out = out_t
    rec.cost_usd += _base.cost_for(ctx.alias_answer, in_t, out_t)

    # ── Citation parsing ──────────────────────────────────────────────────
    all_candidate_ids = [nid for nid, _, _ in reordered]
    l0_candidate_ids  = [nid for nid in all_candidate_ids
                         if ctx.hierarchy.get(nid) and ctx.hierarchy[nid].level == "L0"]
    cited_raw = parse_cited_v6(answer, all_candidate_ids)
    cited_l0 = [nid for nid in cited_raw if nid in set(l0_candidate_ids)]
    if not cited_l0 and l0_candidate_ids:
        cited_l0 = l0_candidate_ids[:3]  # fallback: top-3 L0

    rec.final_action = "ANSWER" if answer and "INSUFFICIENT" not in answer.upper() else "ANSWER"
    rec.answer_text = _base.strip_cited(answer)
    rec.cited_evidence_ids = cited_l0
    rec.distilled_context_node_ids = [nid for nid, lvl, _ in reordered if lvl != "L0"]
    rec.detailed_context_node_ids = l0_candidate_ids
    rec.t_wall_total_seconds = time.time() - t_wall
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="erag_10M")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Set env vars to match B1 config
    os.environ.setdefault("MEMONDEMAND_L0_RETRIEVAL", "bm25")
    os.environ.setdefault("MEMONDEMAND_EMBED_BACKEND", "azure_large")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"

    log.info("=== V6 Dual-Memory Runner ===  tier=%s  max_q=%d", args.tier, args.max_queries)

    # Build runner context (reuse _base's build logic via env vars)
    env_save = {}
    for k, v in [("MEMONDEMAND_L0_RETRIEVAL", "bm25"),
                 ("MEMONDEMAND_EMBED_BACKEND", "azure_large")]:
        env_save[k] = os.environ.get(k)
        os.environ[k] = v

    ctx = _base.build_runner_ctx(tier=args.tier)

    done_ids: Set[str] = set()
    if args.resume and answers_path.exists():
        for line in open(answers_path):
            done_ids.add(json.loads(line).get("query_id", ""))
        log.info("Resume: %d done", len(done_ids))

    import pandas as pd
    query_path = os.environ.get(
        "MEMONDEMAND_QUERIES", str(Path(REPO_ROOT) / "manifests/erag_queries.parquet")
    )
    df = pd.read_parquet(query_path)
    df = df.head(args.max_queries)
    log.info("Queries: %d", len(df))

    t_start = time.time()
    n_done = len(done_ids); n_err = 0
    with open(answers_path, "a") as fout:
        for qi, (_, row) in enumerate(df.iterrows()):
            qid = str(row["query_id"])
            if qid in done_ids: continue
            try:
                rec = run_query_v6(ctx, qid, str(row["query_text"]), query_idx=qi)
                d = rec.__dict__.copy()
                d["method"] = "V6_DUAL_MEM"; d["tier"] = args.tier
                fout.write(json.dumps(d, default=str) + "\n"); fout.flush()
                n_done += 1
                if n_done % 10 == 0:
                    el = time.time() - t_start
                    log.info("Progress %d/%d  %.1fs/q", n_done, len(df),
                             el / max(1, n_done - len(done_ids)))
            except Exception as e:
                log.error("Query %s: %s", qid, e); n_err += 1

    log.info("Done: n=%d errors=%d", n_done, n_err)

if __name__ == "__main__":
    main()
