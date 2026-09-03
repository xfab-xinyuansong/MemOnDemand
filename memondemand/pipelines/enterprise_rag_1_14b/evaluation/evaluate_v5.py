"""Evaluator for the MemOnDemand 1.14B EnterpriseRAG-Bench pipeline.

Loads answers.jsonl (no gold info inside) + erag_query_gold_evaluator_only.jsonl
(gold info, evaluator-only — never exposed to the runner).

Per-query metrics:
  precision = |cited & expected| / max(1, |cited|)
  recall    = |cited & expected| / max(1, |expected|)
  f1        = 2 P R / (P + R)

LLM score (0..1): the configured judge judges the answer 0..5 vs the reference;
                  if reference is empty, score based on factuality vs query alone.

Aggregate per (method, tier):
  mean_f1, mean_precision, mean_recall, mean_llm_score
  n_answered, n_stop, n_total, mean_wall_seconds, mean_cost_usd, total_cost_usd
  V5: n_promote_events_total, n_demote_events_total, mean_n_navigation_steps
"""
from __future__ import annotations
try:
    from memondemand.core import dns_patch  # noqa: F401
except ImportError:
    dns_patch = None


import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = os.environ.get("MEMONDEMAND_REPO_ROOT", str(Path.cwd()))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memondemand.pipelines.enterprise_rag_1_14b.runtime import call as api_call  # noqa: E402

JUDGE_PRICE_PER_M = {"input": 0.00, "output": 0.00}
JUDGE_ALIAS = os.environ.get("MEMONDEMAND_JUDGE_ALIAS", "gpt_5_4")


JUDGE_SYSTEM = """You are a lenient evaluator scoring whether a model's answer is broadly aligned with the reference.
A submitted answer that captures the gist or major substantive facts (even if it lacks every detail or paraphrases differently) should score high.
Only score low if the answer is clearly wrong, off-topic, or contradicts the reference.
Output STRICT JSON only — no commentary, no markdown.
Schema: {"score": <int 0..5>, "rationale": "<one short sentence>"}"""

JUDGE_USER_TMPL = """Question: {question}
Reference answer: {reference}
Submitted answer: {submitted}

Score the submitted answer 0-5 on factual correctness vs the reference.
- 5 = fully correct and complete
- 3 = partially correct
- 0 = wrong, empty, or refuses to answer when reference is non-trivial
If the reference is empty, score on plausibility & specificity vs the question only.

Return JSON only."""


def parse_score(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'"score"\s*:\s*(\d+)', text)
    if m:
        try:
            v = int(m.group(1))
            return max(0, min(5, v))
        except Exception:
            return None
    # fallback: first integer in [0..5]
    m = re.search(r"\b([0-5])\b", text)
    if m:
        return int(m.group(1))
    return None


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("429" in s) or ("usage limit" in s) or ("rate_limit" in s) or ("rate limit" in s)


def llm_judge(question: str, reference: str, submitted: str) -> Dict[str, Any]:
    if not submitted or not submitted.strip():
        return {"score": 0, "rationale": "empty_submission"}
    user = JUDGE_USER_TMPL.format(
        question=question[:1000],
        reference=(reference or "")[:1500],
        submitted=submitted[:1500],
    )
    # Rate-limit responses are retried so infrastructure throttling is not
    # silently converted into a zero judge score.
    MAX_ATTEMPTS = int(os.environ.get("MEMONDEMAND_JUDGE_MAX_ATTEMPTS", "200"))
    resp = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = api_call(
                JUDGE_ALIAS,
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=400, temperature=0.0, timeout=60.0, max_retries=1, backoff_base=2.0,
            )
            break  # success
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc):
                sleep_s = 30.0 + random.uniform(0, 15)
                print(f"    [judge 429] attempt {attempt+1}/{MAX_ATTEMPTS}, sleeping {sleep_s:.0f}s ...", flush=True)
                time.sleep(sleep_s)
                continue  # retry, do NOT give up
            # genuine non-429 error -> judge_error allowed
            return {"score": 0, "rationale": f"judge_error:{type(exc).__name__}"}
    if resp is None:
        return {"score": 0, "rationale": "judge_error:max_attempts_exceeded"}
    text = resp.get("text", "") or ""
    in_t = int(resp.get("usage", {}).get("input_tokens", 0))
    out_t = int(resp.get("usage", {}).get("output_tokens", 0))
    cost = (in_t * JUDGE_PRICE_PER_M["input"] + out_t * JUDGE_PRICE_PER_M["output"]) / 1_000_000
    score = parse_score(text)
    if score is None:
        score = 0
    return {"score": score, "rationale": text[:200], "in_t": in_t, "out_t": out_t,
            "cost": cost}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_llm_score", action="store_true",
                    help="Skip the LLM-judge step entirely.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume per_query_eval.jsonl if it exists.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = out_dir / "per_query_eval.jsonl"
    eval_path = out_dir / "eval.json"

    # Load gold
    gold: Dict[str, Dict[str, Any]] = {}
    with open(args.gold) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = d.get("question_id") or d.get("query_id")
            if not qid:
                continue
            gold[qid] = {
                "expected_doc_ids": list(d.get("expected_doc_ids") or []),
                "gold_answer": d.get("gold_answer") or "",
            }
    print(f"loaded {len(gold)} gold records")

    # Load answers
    answers: List[Dict[str, Any]] = []
    with open(args.answers) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                answers.append(json.loads(line))
            except Exception:
                pass
    print(f"loaded {len(answers)} answer records")

    # Resume support
    done_ids = set()
    if args.resume and per_q_path.exists():
        with open(per_q_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Only skip rows with valid LLM scores; retry judge_errors
                    has_valid = (d.get("llm_judge_raw") or 0) > 0
                    is_non_answer = d.get("final_action", "") != "ANSWER"
                    if has_valid or is_non_answer:
                        done_ids.add(d.get("query_id"))
                except Exception:
                    pass
        print(f"resume: {len(done_ids)} per-query evals done (judge_errors will be retried)")

    per_q_recs: List[Dict[str, Any]] = []
    judge_cost_total = 0.0
    t_start = time.time()
    fout = open(per_q_path, "a", encoding="utf-8") if args.resume else open(per_q_path, "w", encoding="utf-8")
    try:
        for i, ar in enumerate(answers):
            qid = ar.get("query_id", "")
            if qid in done_ids:
                continue
            g = gold.get(qid, {})
            expected_ids = set(g.get("expected_doc_ids", []))
            gold_answer = g.get("gold_answer", "")
            cited = set(ar.get("cited_evidence_ids", []) or [])
            inter = cited & expected_ids
            precision = len(inter) / max(1, len(cited))
            recall = len(inter) / max(1, len(expected_ids))
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            final_action = ar.get("final_action", "STOP_INSUFFICIENT")
            answer_text = ar.get("answer_text", "")
            llm_score_norm = 0.0
            judge_info: Dict[str, Any] = {}
            if final_action == "ANSWER" and not args.skip_llm_score:
                time.sleep(0.7)
                judge_info = llm_judge(ar.get("query_text", ""), gold_answer, answer_text)
                llm_score_norm = judge_info.get("score", 0) / 5.0
                judge_cost_total += judge_info.get("cost", 0.0)
            elif final_action != "ANSWER":
                # STOP_INSUFFICIENT → all 0
                precision = recall = f1 = llm_score_norm = 0.0

            rec = {
                "query_id": qid,
                "method": ar.get("method", ""),
                "tier": ar.get("tier", ""),
                "final_action": final_action,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "retrieval_recall": round(recall, 4),
                "f1": round(f1, 4),
                "llm_score": round(llm_score_norm, 4),
                "llm_judge_raw": judge_info.get("score") if judge_info else None,
                "llm_judge_rationale": judge_info.get("rationale", "")[:200] if judge_info else "",
                "n_cited": len(cited),
                "n_expected": len(expected_ids),
                "n_intersect": len(inter),
                "judge_cost_usd": round(judge_info.get("cost", 0.0), 6),
                "answer_cost_usd": ar.get("cost_usd", 0.0),
                "t_wall_total_seconds": ar.get("t_wall_total_seconds", 0.0),
                "n_navigation_steps": ar.get("n_navigation_steps", 0),
                "n_promote_events": ar.get("n_promote_events", 0),
                "n_demote_events": ar.get("n_demote_events", 0),
            }
            per_q_recs.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i+1) % 20 == 0:
                print(f"  progress {i+1}/{len(answers)}  elapsed={time.time()-t_start:.0f}s  "
                      f"judge_cost=${judge_cost_total:.3f}")
    finally:
        fout.close()

    # Re-read all per-query for aggregate (including any already done)
    all_per = []
    with open(per_q_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: all_per.append(json.loads(line))
            except Exception: pass

    n_total = len(all_per)
    if n_total == 0:
        print("no records to aggregate")
        return 1
    n_answered = sum(1 for r in all_per if r["final_action"] == "ANSWER")
    n_stop = sum(1 for r in all_per if r["final_action"] != "ANSWER")
    mean_f1 = float(np.mean([r["f1"] for r in all_per]))
    mean_prec = float(np.mean([r["precision"] for r in all_per]))
    mean_rec = float(np.mean([r["recall"] for r in all_per]))
    mean_llm = float(np.mean([r["llm_score"] for r in all_per]))
    mean_wall = float(np.mean([r["t_wall_total_seconds"] for r in all_per]))
    mean_cost = float(np.mean([r["answer_cost_usd"] for r in all_per]))
    total_cost_answer = float(sum(r["answer_cost_usd"] for r in all_per))
    total_cost_judge = float(sum(r["judge_cost_usd"] for r in all_per))
    n_prom_events = int(sum(r["n_promote_events"] for r in all_per))
    n_dem_events = int(sum(r["n_demote_events"] for r in all_per))
    mean_nav_steps = float(np.mean([r["n_navigation_steps"] for r in all_per]))

    agg = {
        "method": all_per[0].get("method", ""),
        "tier": all_per[0].get("tier", ""),
        "n_total": n_total,
        "n_answered": n_answered,
        "n_stop": n_stop,
        "mean_f1": round(mean_f1, 4),
        "mean_precision": round(mean_prec, 4),
        "mean_recall": round(mean_rec, 4),
        "mean_llm_score": round(mean_llm, 4),
        "mean_t_wall_total_seconds": round(mean_wall, 3),
        "mean_cost_usd_answer": round(mean_cost, 6),
        "total_cost_usd_answer": round(total_cost_answer, 4),
        "total_cost_usd_judge": round(total_cost_judge, 4),
        "total_cost_usd": round(total_cost_answer + total_cost_judge, 4),
        "n_promote_events_total": n_prom_events,
        "n_demote_events_total": n_dem_events,
        "mean_n_navigation_steps": round(mean_nav_steps, 3),
    }

    with open(eval_path, "w") as f:
        json.dump(agg, f, indent=2)
    print("=== EVAL DONE ===")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
