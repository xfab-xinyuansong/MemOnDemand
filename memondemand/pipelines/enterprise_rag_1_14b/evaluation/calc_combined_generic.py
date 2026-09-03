#!/usr/bin/env python3
"""Compute Combined/DocRcl/F1/Correct/Compl/InvDoc_ratio/Promo for a tier
using the SAME formula as all other v3-canonical rows in README.
STOP_INSUFFICIENT excluded from denominator.
"""
import json, sys

def num(x):
    return x if isinstance(x, (int, float)) else 0

def main(per_query_path, promo_from_eval_json=None):
    rows = []
    bad = 0
    for l in open(per_query_path):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except Exception:
            bad += 1
    seen = {}
    for r in rows:
        qid = r.get("query_id")
        if qid:
            seen[qid] = r
    rows = list(seen.values())

    # exclude STOP_INSUFFICIENT
    judged = [r for r in rows if not r.get("is_stop_insufficient") and r.get("llm_judge_raw") is not None]
    n = len(judged)

    correct = [1 if num(r.get("llm_judge_raw")) > 2 else 0 for r in judged]
    comb = [(1 if num(r.get("llm_judge_raw")) > 2 else 0) * num(r.get("llm_judge_raw")) / 5 for r in judged]
    compl = [num(r.get("llm_judge_raw")) / 5 for r in judged]

    dr_rows = [r for r in judged if num(r.get("n_expected")) > 0]
    dr = [num(r.get("retrieval_recall")) for r in dr_rows]
    f1s = [
        num(r.get("f1", r.get("retrieval_f1")))
        for r in dr_rows
        if r.get("f1", r.get("retrieval_f1")) is not None
    ]

    inv = []
    for r in judged:
        n_cited = num(r.get("n_cited"))
        n_int = num(r.get("n_intersect"))
        if n_cited > 0:
            inv.append(max(0, n_cited - n_int) / n_cited)

    print("bad_lines", bad)
    print("N (judged)", n)
    print("Combined", round(100 * sum(comb) / n, 2) if n else "NA")
    print("DocRcl%", round(100 * sum(dr) / len(dr), 2) if dr else "NA", f"(n={len(dr)})")
    print("F1", round(sum(f1s) / len(f1s), 2) if f1s else "NA")
    print("Correct%", round(100 * sum(correct) / n, 1) if n else "NA")
    print("Compl%", round(100 * sum(compl) / n, 2) if n else "NA")
    print("InvDoc_ratio", round(sum(inv) / len(inv), 4) if inv else "NA", f"(n={len(inv)})")

if __name__ == "__main__":
    main(sys.argv[1])
