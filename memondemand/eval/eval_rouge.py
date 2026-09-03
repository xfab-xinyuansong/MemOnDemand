import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""
ROUGE-based evaluation for MemOnDemand runs.
Zero API calls — uses local rouge_score library.
Outputs per_query_eval.jsonl (same format as evaluate_v5.py).

Metric mapping:
  rouge1_f1 >= 0.40  →  llm_judge_raw = 3 (broadly correct)
  Correct_i = llm_judge_raw > 2  →  rouge1_f1 >= 0.40
  llm_score = rouge1_f1 (completeness proxy)
  Combined  = mean(Correct_i × llm_score_i)
"""
import json, argparse, pathlib, ast, re
from rouge_score import rouge_scorer

SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
CORRECT_THRESHOLD = 0.40   # rouge1 f1 >= this = broadly correct

def rouge1_to_raw(f1):
    if f1 >= 0.80: return 5
    if f1 >= 0.60: return 4
    if f1 >= 0.40: return 3
    if f1 >= 0.20: return 2
    return 1

def strip_cited(text):
    if not text:
        return ""
    text = re.sub(r"\nCITED\s*:[^\n]*", "", text)
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers",  required=True)
    ap.add_argument("--gold",     required=True)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--resume",   action="store_true")
    args = ap.parse_args()

    answers  = [json.loads(l) for l in open(args.answers)]
    gold_map = {}
    for l in open(args.gold):
        g = json.loads(l)
        qid = g.get("question_id") or g.get("query_id") or ""
        raw = g.get("expected_doc_ids", []) or []
        if isinstance(raw, str):
            try: raw = ast.literal_eval(raw)
            except: raw = []
        g["_doc_ids"] = list(raw)
        gold_map[qid] = g

    out_dir  = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_query_eval.jsonl"

    done_ids = set()
    if args.resume and out_path.exists():
        for l in open(out_path):
            try:
                d = json.loads(l)
                if (d.get("llm_judge_raw") or 0) > 0 or d.get("final_action") != "ANSWER":
                    done_ids.add(d.get("query_id"))
            except Exception:
                pass
        print(f"resume: {len(done_ids)} done")

    n_written = 0
    fout = open(out_path, "a" if args.resume else "w", encoding="utf-8")

    for i, a in enumerate(answers):
        qid = a.get("query_id", "")
        if qid in done_ids:
            continue

        g = gold_map.get(qid, {})
        gold_answer = g.get("gold_answer", "") or ""
        gold_ids    = g.get("_doc_ids", [])
        cited       = a.get("cited_evidence_ids") or []
        intersect   = set(cited) & set(gold_ids)
        dr          = len(intersect) / max(1, len(gold_ids))

        final_action = a.get("final_action", "STOP_INSUFFICIENT")
        if final_action == "ANSWER" and gold_answer:
            pred_text = strip_cited(a.get("answer_text", "") or "")
            if pred_text:
                scores = SCORER.score(gold_answer, pred_text)
                f1 = scores["rouge1"].fmeasure
            else:
                f1 = 0.0
            raw    = rouge1_to_raw(f1)
            score  = round(f1, 4)
            rationale = f"rouge1_f1:{f1:.4f}"
        else:
            raw, score, rationale = 0, 0.0, "non_answer"

        row = {
            "query_id":             qid,
            "final_action":         final_action,
            "doc_recall_this_query": round(dr, 4),
            "llm_judge_raw":        raw,
            "llm_score":            score,
            "llm_judge_rationale":  rationale,
            "cited_evidence_ids":   cited,
            "gold_ids":             gold_ids,
        }
        fout.write(json.dumps(row) + "\n")
        n_written += 1

    fout.close()
    print(f"wrote {n_written} new rows → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    all_rows  = [json.loads(l) for l in open(out_path)]
    n_total   = 500
    n_gold    = 470
    doc_rcl   = sum(r["doc_recall_this_query"] for r in all_rows) / n_gold * 100
    correct   = sum(1 for r in all_rows if (r.get("llm_judge_raw") or 0) > 2) / n_total * 100
    completeness = sum(r.get("llm_score", 0) for r in all_rows) / n_total * 100
    combined  = sum(
        (1 if (r.get("llm_judge_raw") or 0) > 2 else 0) * r.get("llm_score", 0.0)
        for r in all_rows
    ) / n_total * 100
    inv_doc   = sum(
        max(0, len(r.get("cited_evidence_ids") or []) - len(r.get("gold_ids") or []))
        for r in all_rows
    ) / max(1, len(all_rows))

    print(f"\n{'='*52}")
    print(f"{'DocRcl%':<12} {'Correct%':<12} {'Complete%':<12} {'Combined%':<12} {'InvDoc'}")
    print(f"{doc_rcl:<12.2f} {correct:<12.1f} {completeness:<12.1f} {combined:<12.1f} {inv_doc:.2f}")
    print(f"  (correct = ROUGE-1 F1 >= {CORRECT_THRESHOLD:.2f})")
    print(f"{'='*52}")

if __name__ == "__main__":
    main()
