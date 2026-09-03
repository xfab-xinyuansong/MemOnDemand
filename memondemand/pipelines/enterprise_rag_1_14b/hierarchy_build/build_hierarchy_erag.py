"""Build a dynamic hierarchy over EnterpriseRAG-Bench L0 records.

ERAG L0 parquet columns: doc_id, source_type, title, content, level, text
Supports the benchmark's corpus-scale tiers, including the expanded 1.14B setting.

Differences from build_hierarchy.py (own_full):
- No --raw_dir / --l0 split; single --erag_parquet input
- Column mapping: doc_id→node_id, source_type→tenant_id
- distilled_text (L0): deterministic  "title: content[:150]"  (no LLM)
- detailed_text  (L0): "title\\ntext"  (full text)
- source_evidence_ids: [doc_id]
- Source types are preserved as tenant identifiers for provenance isolation.

L1/L2 clustering + LLM summarisation logic is identical to build_hierarchy.py
(imported directly).

Outputs in --out:
    hierarchy.json          ndjson, one record per line
    parent_child_index.json
    build_report.json
    token_ledger.json
    token_ledger_with_records.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(os.environ.get("MEMONDEMAND_REPO_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Reuse shared utilities from build_hierarchy.py
# ---------------------------------------------------------------------------
from memondemand.pipelines.enterprise_rag_1_14b.hierarchy_build.build_hierarchy import (  # noqa: E402
    # data-classes / ledger
    DualNode,
    TokenLedger,
    # token utilities
    tok_count,
    enc,
    # LLM call wrapper
    llm_call_with_ledger,
    # embedding + clustering
    embed_texts,
    kmeans_cluster,
    _decide_n_clusters,
    # hierarchy builders
    build_l1,
    build_l2,
    # provenance
    build_parent_child_index,
    # alias constants
    GPT54_MINI,
    GPT54,
)
from memondemand.methods.dual_node import NODE_STATE_LIGHT  # noqa: E402
from memondemand.methods.token_ledger import (  # noqa: E402
    PHASE_HIERARCHY_BUILD,
    PHASE_DISTILLED_GEN,
)
from memondemand.pipelines.enterprise_rag_1_14b.runtime import get_alias_config  # noqa: E402

logger = logging.getLogger("memondemand.enterprise_rag_1_14b.build_hierarchy_erag")

# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

def _check_api_keys() -> None:
    """Resolve semantic aliases without reading or printing secret values."""
    get_alias_config(GPT54_MINI)
    get_alias_config(GPT54)
    logger.info("low- and high-level model aliases resolved")


# ---------------------------------------------------------------------------
# ERAG L0 loader
# ---------------------------------------------------------------------------

ERAG_TENANTS = {
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
}


# tiktoken `encode()` rejects literal special-token strings (`<|endoftext|>` etc.)
# that appear verbatim in some ERAG documents. Replace them with safe ASCII
# equivalents before any tokenization. Same character count, no semantic loss
# for downstream summarization or embedding.
_TIKTOKEN_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endofprompt|>",
)


def _sanitize_tiktoken_special(text: str) -> str:
    if not text:
        return text
    for tok in _TIKTOKEN_SPECIAL_TOKENS:
        if tok in text:
            text = text.replace(tok, tok.replace("|", "_"))
    return text


def load_erag_l0(parquet_path: Path, head: int = 0) -> List[DualNode]:
    """Load ERAG L0 parquet and produce DualNode list.

    Column mapping:
        doc_id       → node_id
        source_type  → tenant_id
        title + content[:150] → distilled_text  (deterministic, no LLM)
        title + "\\n" + text  → detailed_text

    Args:
        parquet_path: path to erag_{tier}_l0_nodes.parquet
        head: if > 0, only load first N rows (dry-run mode)
    """
    df = pd.read_parquet(parquet_path)
    if head > 0:
        df = df.head(head)
    logger.info(f"loaded {len(df)} rows from {parquet_path.name}")

    nodes: List[DualNode] = []
    for _, row in df.iterrows():
        doc_id = str(row["doc_id"])
        tenant_id = str(row.get("source_type", "unknown"))
        # Sanitize tiktoken special tokens that appear literally in some ERAG docs
        title = _sanitize_tiktoken_special(str(row.get("title") or "").strip())
        content = _sanitize_tiktoken_special(str(row.get("content") or "").strip())
        text = _sanitize_tiktoken_special(str(row.get("text") or "").strip())

        # Deterministic dual representations — no LLM involved for L0
        raw_distilled = (title + ": " + content[:150]).strip()
        raw_detailed = (title + "\n" + text).strip()

        # Guard: distilled must be strictly shorter than detailed in tokens
        dt_tok = tok_count(raw_distilled)
        det_tok = tok_count(raw_detailed)

        if det_tok == 0:
            # Edge case: empty detailed — pad with a sentinel so invariant holds
            raw_detailed = raw_distilled + "\n[empty-text]"
            det_tok = tok_count(raw_detailed)

        if dt_tok >= det_tok:
            # Truncate distilled to strictly fewer tokens
            ids_full = enc().encode(raw_distilled)
            keep_n = max(1, det_tok - 1)
            raw_distilled = enc().decode(ids_full[:keep_n])
            dt_tok = keep_n

        nodes.append(DualNode(
            node_id=doc_id,
            level="L0",
            tenant_id=tenant_id,
            distilled_text=raw_distilled,
            distilled_tokens=dt_tok,
            detailed_text=raw_detailed,
            detailed_tokens=det_tok,
            source_evidence_ids=[doc_id],
            state=NODE_STATE_LIGHT,
            distilled_text_model_alias="deterministic_title_content_truncation",
            distilled_text_model_status="ACTIVE",
            extra={"source_type": tenant_id, "l0_erag_origin": True},
        ))

    logger.info(f"built {len(nodes)} ERAG L0 DualNodes "
                f"(tenants: {sorted({n.tenant_id for n in nodes})})")
    return nodes


# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------

def _acceptance_checks(all_nodes: List[DualNode], l0_nodes: List[DualNode]) -> Dict[str, Any]:
    """Run the 3 required acceptance checks and return a dict of results."""
    total = len(all_nodes)
    if total == 0:
        return {"error": "no nodes"}

    # 1. 100% dual representation (both texts non-empty)
    dual_ok = sum(1 for n in all_nodes if n.distilled_text and n.detailed_text)
    dual_pct = 100.0 * dual_ok / total

    # 2. ≥95% distilled_tokens < detailed_tokens
    dt_lt_det = sum(
        1 for n in all_nodes
        if n.distilled_tokens > 0 and n.detailed_tokens > 0
        and n.distilled_tokens < n.detailed_tokens
    )
    dt_lt_det_pct = 100.0 * dt_lt_det / total

    # 3. 100% L0 source_evidence_ids traceable back to original doc_ids
    l0_ids = {n.node_id for n in l0_nodes}
    all_ev: List[str] = []
    for n in all_nodes:
        all_ev.extend(n.source_evidence_ids)
    traceable = sum(1 for ev in all_ev if ev in l0_ids)
    trace_pct = 100.0 * traceable / len(all_ev) if all_ev else 0.0

    results = {
        "dual_representation_pct": round(dual_pct, 2),
        "dual_representation_pass": dual_pct == 100.0,
        "distilled_lt_detailed_pct": round(dt_lt_det_pct, 2),
        "distilled_lt_detailed_pass": dt_lt_det_pct >= 95.0,
        "l0_traceable_pct": round(trace_pct, 2),
        "l0_traceable_pass": trace_pct == 100.0,
        "total_nodes": total,
        "total_evidence_refs": len(all_ev),
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# v5_llm_distill 2026-06-28: LLM-based L0 distillation (replaces deterministic title+content[:150])

L0_DISTILL_SYSTEM = """You are a precise document summarizer. Given a document, write a single 1-2 sentence summary that captures the main entity, topic, and most important factual claim. No preamble. No quoting. Output the summary text only."""

L0_DISTILL_USER_TMPL = """Title: {title}

Document:
{content}

Summary (1-2 sentences, focus on main entity, topic, and key facts):"""


def _llm_distill_one(node, ledger, alias=GPT54_MINI, max_tokens=200):
    """Call gpt_5_4_mini to produce L0 distilled_text. Returns updated distilled string."""
    # Extract title + content from existing distilled ("title: content[:150]") and detailed
    detailed = node.detailed_text or ''
    # Take first 1000 chars of detailed text as content sample for summarizer
    title_part, _, _ = detailed.partition(chr(10))
    body = detailed[len(title_part)+1:][:1000] if len(detailed) > len(title_part)+1 else ''
    if not title_part and not body:
        return node.distilled_text  # fallback: keep original
    user = L0_DISTILL_USER_TMPL.format(title=title_part[:200], content=body)
    try:
        result = llm_call_with_ledger(
            alias, L0_DISTILL_SYSTEM, user, ledger,
            phase=PHASE_DISTILLED_GEN, max_tokens=max_tokens,
            node_id=f'l0_distill_{node.node_id}'
        )
        text = (result.get('text') or '').strip()
        if not text:
            return node.distilled_text  # fallback
        # Sanitize and ensure shorter than detailed
        text = _sanitize_tiktoken_special(text)
        dt_tok = tok_count(text)
        det_tok = node.detailed_tokens or tok_count(detailed)
        if det_tok > 0 and dt_tok >= det_tok:
            # truncate
            ids_full = enc().encode(text)
            text = enc().decode(ids_full[:max(1, det_tok-1)])
            dt_tok = max(1, det_tok-1)
        return text, dt_tok
    except Exception as e:
        logger.warning(f'L0 distill failed for {node.node_id}: {type(e).__name__}: {str(e)[:120]}')
        return node.distilled_text, node.distilled_tokens


def llm_distill_l0_nodes(nodes, ledger, alias=GPT54_MINI, max_workers=12,
                          checkpoint_path=None, checkpoint_every=200):
    """Replace deterministic L0 distilled_text with LLM-generated summaries.
    Parallel calls, checkpoint to disk to resume."""
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Load checkpoint if exists
    done = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        for l in open(checkpoint_path):
            d = _json.loads(l)
            done[d['node_id']] = (d['distilled_text'], d['distilled_tokens'])
        logger.info(f'L0 LLM distill checkpoint: {len(done)} pre-loaded from {checkpoint_path}')

    todo = [n for n in nodes if n.node_id not in done]
    logger.info(f'L0 LLM distill: {len(todo)} nodes to process (skipping {len(done)} cached)')

    t_start = time.time()
    n_completed = 0
    ckpt_fp = None
    if checkpoint_path:
        ckpt_fp = open(checkpoint_path, 'a')

    def _process(node):
        result = _llm_distill_one(node, ledger, alias=alias)
        if isinstance(result, tuple):
            return node.node_id, result[0], result[1]
        return node.node_id, result, tok_count(result)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process, n): n.node_id for n in todo}
        for i, fut in enumerate(as_completed(futures)):
            try:
                nid, distilled, dist_tok = fut.result()
                done[nid] = (distilled, dist_tok)
                if ckpt_fp:
                    ckpt_fp.write(_json.dumps({'node_id': nid, 'distilled_text': distilled, 'distilled_tokens': dist_tok}) + chr(10))
                    if (i+1) % 50 == 0:
                        ckpt_fp.flush()
            except Exception as e:
                logger.warning(f'L0 distill task error: {e}')
            n_completed += 1
            if n_completed % checkpoint_every == 0:
                elapsed = time.time() - t_start
                logger.info(f'L0 LLM distill: {n_completed}/{len(todo)} elapsed={elapsed:.0f}s')

    if ckpt_fp:
        ckpt_fp.close()

    # Apply distilled back to nodes
    for n in nodes:
        if n.node_id in done:
            new_dist, new_tok = done[n.node_id]
            n.distilled_text = new_dist
            n.distilled_tokens = new_tok
            n.distilled_text_model_alias = alias
            n.distilled_text_model_status = 'ACTIVE'

    logger.info(f'L0 LLM distill done: {len(done)}/{len(nodes)} nodes (cost so far: ledger)')
    return nodes


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a MemOnDemand hierarchy from one EnterpriseRAG-Bench L0 parquet."
    )
    ap.add_argument("--erag_parquet", required=True,
                    help="Path to erag_{tier}_l0_nodes.parquet")
    ap.add_argument("--out", required=True,
                    help="Output directory for hierarchy.json + reports")
    ap.add_argument("--tier_label", required=True,
                    help="Tier label for logging, e.g. '250M' or 'FULL'")
    ap.add_argument("--low_level_alias", default=GPT54_MINI,
                    help=f"Alias for L1 distillation (default: {GPT54_MINI})")
    ap.add_argument("--high_level_alias", default=GPT54,
                    help=f"Alias for L2 abstraction (default: {GPT54})")
    ap.add_argument("--max_workers_l1", type=int, default=4,
                    help="Thread workers for L1 (default: 4)")
    ap.add_argument("--max_workers_l2", type=int, default=2,
                    help="Thread workers for L2 (default: 2)")
    ap.add_argument("--alias_status", default="ACTIVE",
                    help="ACTIVE | PROVISIONAL — written into ledger")
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--dry_run_head", type=int, default=0,
                    help="If > 0, load only first N rows and skip LLM calls (dry-run)")
    ap.add_argument("--llm_l0_distill", action="store_true",
                    help="Use the low-level model for L0 distilled text (default off)")
    ap.add_argument("--l0_distill_workers", type=int, default=12,
                    help="ThreadPool workers for L0 LLM distill (default 12)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format=f"%(asctime)s %(levelname)s %(name)s [{args.tier_label}]: %(message)s",
    )

    logger.info(f"=== ERAG Hierarchy Build — tier={args.tier_label} ===")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.erag_parquet)
    if not parquet_path.exists():
        raise SystemExit(f"ERROR: parquet not found: {parquet_path}")

    # -----------------------------------------------------------------------
    # 1. Load ERAG L0
    # -----------------------------------------------------------------------
    t_l0 = time.time()
    l0_nodes = load_erag_l0(parquet_path, head=args.dry_run_head)
    t_l0 = time.time() - t_l0
    logger.info(f"L0: {len(l0_nodes)} nodes loaded in {t_l0:.1f}s")

    if args.dry_run_head > 0:
        logger.info("[DRY-RUN] head=%d, skipping LLM calls. Exiting.", args.dry_run_head)
        print(f"\n[DRY-RUN] ERAG L0 sample loaded OK:")
        print(f"  parquet : {parquet_path}")
        print(f"  rows    : {len(l0_nodes)}")
        print(f"  tenants : {sorted({n.tenant_id for n in l0_nodes})}")
        print(f"  first node: node_id={l0_nodes[0].node_id} "
              f"tenant={l0_nodes[0].tenant_id} "
              f"dist_tok={l0_nodes[0].distilled_tokens} "
              f"det_tok={l0_nodes[0].detailed_tokens}")
        return 0

    # --- API key check (only needed for LLM calls; skipped in dry-run) ---
    _check_api_keys()

    # -----------------------------------------------------------------------
    # 2. Initialise ledger
    # -----------------------------------------------------------------------
    ledger = TokenLedger(
        run_id=f"step7_erag_{args.tier_label}_{int(time.time())}",
        method="V4_step7_erag_hierarchy",
        alias_status=args.alias_status,
        alias_chosen_at=dt.datetime.utcnow().isoformat() + "Z",
        alias_chosen_by="memondemand_pipeline",
    )

    # -----------------------------------------------------------------------
    # 2b. (Optional) Replace deterministic L0 distilled_text with LLM
    # -----------------------------------------------------------------------
    if args.llm_l0_distill:
        t_l0_llm = time.time()
        logger.info(f"L0 LLM distill enabled (alias={args.low_level_alias}, workers={args.l0_distill_workers})")
        ckpt_path = out_dir / "l0_distill_checkpoint.jsonl"
        l0_nodes = llm_distill_l0_nodes(
            l0_nodes, ledger,
            alias=args.low_level_alias,
            max_workers=args.l0_distill_workers,
            checkpoint_path=str(ckpt_path),
        )
        t_l0_llm = time.time() - t_l0_llm
        logger.info(f"L0 LLM distill total wall: {t_l0_llm:.1f}s")

    # -----------------------------------------------------------------------
    # 3. Build L1 (per-tenant KMeans + gpt_5_4_mini decision/distill)
    # -----------------------------------------------------------------------
    t_l1 = time.time()
    logger.info(f"Building L1 (max_workers={args.max_workers_l1}) ...")
    l1_nodes, l1_stats = build_l1(
        l0_nodes, ledger,
        max_workers=args.max_workers_l1,
        alias_status=args.alias_status,
        model_alias=args.low_level_alias,
    )
    t_l1 = time.time() - t_l1
    logger.info(f"L1: {len(l1_nodes)} nodes built in {t_l1:.1f}s")

    # -----------------------------------------------------------------------
    # 4. Build L2 (per-tenant root + gpt_5_4 abstraction)
    # -----------------------------------------------------------------------
    t_l2 = time.time()
    logger.info(f"Building L2 (max_workers={args.max_workers_l2}) ...")
    l2_nodes, l2_stats = build_l2(
        l1_nodes, ledger,
        max_workers=args.max_workers_l2,
        alias_status=args.alias_status,
        model_alias=args.high_level_alias,
    )
    t_l2 = time.time() - t_l2
    logger.info(f"L2: {len(l2_nodes)} nodes built in {t_l2:.1f}s")

    all_nodes = l0_nodes + l1_nodes + l2_nodes
    logger.info(
        f"hierarchy total: {len(all_nodes)} nodes "
        f"(L0={len(l0_nodes)} L1={len(l1_nodes)} L2={len(l2_nodes)})"
    )

    # -----------------------------------------------------------------------
    # 5. Write outputs
    # -----------------------------------------------------------------------
    hierarchy_path = out_dir / "hierarchy.json"
    with open(hierarchy_path, "w") as f:
        for n in all_nodes:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"wrote {hierarchy_path}  ({len(all_nodes)} lines)")

    parent_child_path = out_dir / "parent_child_index.json"
    with open(parent_child_path, "w") as f:
        json.dump(build_parent_child_index(all_nodes), f, indent=2)
    logger.info(f"wrote {parent_child_path}")

    ledger.export(str(out_dir / "token_ledger.json"), include_raw=False)
    ledger.export(str(out_dir / "token_ledger_with_records.json"), include_raw=True)

    # -----------------------------------------------------------------------
    # 6. Acceptance checks
    # -----------------------------------------------------------------------
    ac = _acceptance_checks(all_nodes, l0_nodes)
    logger.info("=== ACCEPTANCE CHECKS ===")
    logger.info(f"  100% dual repr:            {'PASS' if ac['dual_representation_pass'] else 'FAIL'} "
                f"({ac['dual_representation_pct']}%)")
    logger.info(f"  ≥95% distilled<detailed:   {'PASS' if ac['distilled_lt_detailed_pass'] else 'FAIL'} "
                f"({ac['distilled_lt_detailed_pct']}%)")
    logger.info(f"  100% L0 traceable:         {'PASS' if ac['l0_traceable_pass'] else 'FAIL'} "
                f"({ac['l0_traceable_pct']}%)")

    # Console summary
    print(f"\n{'='*60}")
    print(f"ERAG Hierarchy Build — tier={args.tier_label}  DONE")
    print(f"{'='*60}")
    print(f"  L0={len(l0_nodes)}  L1={len(l1_nodes)}  L2={len(l2_nodes)}  total={len(all_nodes)}")
    print(f"  100%  dual representation:  {'✓ PASS' if ac['dual_representation_pass'] else '✗ FAIL'}"
          f"  ({ac['dual_representation_pct']}%)")
    print(f"  ≥95% distilled<detailed:    {'✓ PASS' if ac['distilled_lt_detailed_pass'] else '✗ FAIL'}"
          f"  ({ac['distilled_lt_detailed_pct']}%)")
    print(f"  100% L0 traceable:          {'✓ PASS' if ac['l0_traceable_pass'] else '✗ FAIL'}"
          f"  ({ac['l0_traceable_pct']}%)")
    print(f"  ledger: {ledger.grand_total()}")
    print(f"  output: {out_dir}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 7. Write build report
    # -----------------------------------------------------------------------
    report = {
        "build_step": "enterprise_rag_1_14b_hierarchy",
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "tier_label": args.tier_label,
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
        "acceptance_checks": ac,
        "ledger_totals": ledger.grand_total(),
    }
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"wrote {out_dir / 'build_report.json'}")
    logger.info("=== BUILD COMPLETE ===")

    all_pass = (
        ac["dual_representation_pass"]
        and ac["distilled_lt_detailed_pass"]
        and ac["l0_traceable_pass"]
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
