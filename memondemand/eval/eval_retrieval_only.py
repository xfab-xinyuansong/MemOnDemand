import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
#!/usr/bin/env python3
"""Evaluate DocRcl/F1/InvDoc from answers.jsonl without LLM judge.
Usage: python eval_retrieval_only.py <out_dir> [--gold gold.jsonl] [--tag exp_name]
"""
import json, sys, argparse, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    default_gold = str(Path(REPO_ROOT) / 'manifests/erag_query_gold_evaluator_only.jsonl')
    default_report = str(Path(REPO_ROOT) / 'reports/retrieval_only.json')
    ap.add_argument('--gold', default=default_gold)
    ap.add_argument('--tag', default='')
    ap.add_argument('--report', default=default_report)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ans_path = out_dir / 'answers.jsonl'
    if not ans_path.exists():
        print(f'ERROR: {ans_path} not found'); sys.exit(1)

    # Load gold
    gold = {}
    for l in open(args.gold):
        r = json.loads(l)
        qid = r.get('query_id') or r.get('question_id', '')
        gold[qid] = set(r.get('expected_doc_ids', []))

    rows = [json.loads(l) for l in open(ans_path)]
    print(f'Evaluating {len(rows)} queries from {out_dir}...')

    total_recall = total_prec = total_f1 = 0.0
    total_inv = n_cited_total = n_promo = 0
    n_valid = 0

    for r in rows:
        qid = r.get('query_id', '')
        cited = set(r.get('cited_evidence_ids', []))
        gold_ids = gold.get(qid, set())
        n_expected = len(gold_ids)
        n_cited = len(cited)
        n_intersect = len(cited & gold_ids)
        n_promo += int(r.get('n_promote_events', 0))
        n_cited_total += n_cited

        recall = n_intersect / n_expected if n_expected > 0 else 0.0
        prec   = n_intersect / n_cited   if n_cited   > 0 else 0.0
        f1     = 2*recall*prec/(recall+prec) if (recall+prec) > 0 else 0.0
        inv    = (n_cited - n_intersect) / n_cited if n_cited > 0 else 0.0

        total_recall += recall
        total_prec   += prec
        total_f1     += f1
        total_inv    += inv
        if n_expected > 0: n_valid += 1

    n = len(rows)
    n_with_gold = sum(1 for r in rows if gold.get(r.get('query_id',''), set()))
    dr   = total_recall / n_with_gold * 100 if n_with_gold else 0
    f1v  = total_f1     / n_with_gold       if n_with_gold else 0
    invd = total_inv    / n * 100            if n else 0
    avg_promo = n_promo / n if n else 0

    tag = args.tag or out_dir.name
    print(f'\n=== {tag} (n={n}) ===')
    print(f'  DocRcl:  {dr:.2f}%')
    print(f'  F1:      {f1v:.4f}')
    print(f'  InvDoc:  {invd:.2f}%')
    print(f'  Promo:   {n_promo} ({avg_promo:.2f}/q)')

    # Load / update report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if report_path.exists():
        try: results = json.load(open(report_path))
        except: results = {}

    results[tag] = {
        'n': n, 'n_with_gold': n_with_gold,
        'DocRcl': round(dr, 2), 'F1': round(f1v, 4),
        'InvDoc': round(invd, 2), 'n_promote': n_promo,
        'avg_promo_per_q': round(avg_promo, 2)
    }
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved → {report_path}')
    return results[tag]

if __name__ == '__main__':
    main()
