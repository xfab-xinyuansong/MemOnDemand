import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""BM25 solo retrieval on ERAG 10M, paired with B_flat dense via RRF hybrid.

Outputs:
  experiments/v5/reports/bm25_solo_eval.jsonl   — per-query retrieved doc ids + recall
  experiments/v5/reports/hybrid_bflat_eval.jsonl — RRF (BM25 + dense) per-query
  experiments/v5/reports/bm25_summary.json     — aggregated metrics
"""
import json, time, re, argparse, math
from collections import defaultdict
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

# Hyperparameters
TOP_K = 12  # match B_flat retrieval depth
RRF_K = 60
ALSO_K = [5, 10, 20]  # report @5/@10/@20 too


def simple_tokenize(text):
    if not text: return []
    text = text.lower()
    # alphanumeric tokens len>=2
    return re.findall(r"[a-z0-9]{2,}", text)


def load_l0(parquet_path):
    df = pd.read_parquet(parquet_path)
    docs = []
    for _, r in df.iterrows():
        title = r.get('title','') or ''
        content = r.get('content','') or ''
        full = (title + ' ' + content)
        docs.append({'doc_id': r['doc_id'], 'text': full})
    return docs


def load_gold(jsonl_path):
    out = {}
    for l in open(jsonl_path):
        d = json.loads(l)
        out[d['question_id']] = {'expected': set(d.get('expected_doc_ids',[])),
                                 'question': d.get('question_text', d.get('query_text',''))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--l0_parquet', default='manifests/erag_10M_l0_nodes.parquet')
    ap.add_argument('--gold', default='manifests/erag_query_gold_with_answers.jsonl')
    ap.add_argument('--queries', default='manifests/erag_queries.parquet')
    ap.add_argument('--bflat_answers', default='results/v5_emb_large/erag_10M/B_flat/seed_20260608/answers.jsonl')
    ap.add_argument('--out_dir', default='experiments/v5/reports')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f'Loading L0 from {args.l0_parquet} ...')
    docs = load_l0(args.l0_parquet)
    print(f'  {len(docs)} docs in {time.time()-t0:.1f}s')

    print('Tokenizing ...')
    t1 = time.time()
    tokenized = [simple_tokenize(d['text']) for d in docs]
    doc_ids = [d['doc_id'] for d in docs]
    print(f'  done in {time.time()-t1:.1f}s')

    print('Building BM25 index ...')
    t2 = time.time()
    bm25 = BM25Okapi(tokenized)
    print(f'  done in {time.time()-t2:.1f}s')

    # Queries: prefer gold file, fallback to query parquet
    gold = load_gold(args.gold)
    # Replace empty question_text from queries parquet
    qdf = pd.read_parquet(args.queries)
    qmap = dict(zip(qdf['query_id'], qdf['query_text']))
    for qid in gold:
        if not gold[qid]['question']:
            gold[qid]['question'] = qmap.get(qid, '')
    print(f'Loaded {len(gold)} queries')

    # Load B_flat dense retrieval (top-12 from detailed_context_node_ids, order preserved)
    bflat = {}
    if Path(args.bflat_answers).exists():
        for l in open(args.bflat_answers):
            d = json.loads(l)
            bflat[d['query_id']] = d.get('detailed_context_node_ids', []) or []
        print(f'Loaded B_flat dense retrieval for {len(bflat)} queries (top-{TOP_K} expected)')

    bm25_out = []
    hybrid_out = []
    bm25_recalls = defaultdict(list)
    hybrid_recalls = defaultdict(list)

    print('Running BM25 + Hybrid retrieval on 500 queries ...')
    t3 = time.time()
    sorted_qids = sorted(gold.keys())
    for i, qid in enumerate(sorted_qids):
        q_text = gold[qid]['question']
        exp = gold[qid]['expected']
        q_tokens = simple_tokenize(q_text)
        # BM25 scoring
        scores = bm25.get_scores(q_tokens)
        # Top-K by index
        # Get top-50 then slice (cheaper than full argsort)
        top_idx = scores.argsort()[::-1][:max(ALSO_K + [TOP_K])]
        bm25_ranking = [(doc_ids[idx], float(scores[idx])) for idx in top_idx]
        bm25_topk = [d for d, _ in bm25_ranking[:TOP_K]]

        # Hybrid via RRF
        dense_ranking = bflat.get(qid, [])
        rrf = {}
        for rank, did in enumerate(bm25_topk, 1):
            rrf[did] = rrf.get(did, 0.0) + 1.0 / (RRF_K + rank)
        for rank, did in enumerate(dense_ranking[:TOP_K], 1):
            rrf[did] = rrf.get(did, 0.0) + 1.0 / (RRF_K + rank)
        hybrid_topk = [did for did, _ in sorted(rrf.items(), key=lambda x: -x[1])[:TOP_K]]

        rec_bm25 = {}
        rec_hybrid = {}
        if exp:
            for k in ALSO_K + [TOP_K]:
                bm25_k = bm25_topk[:k]
                rec_bm25[k] = len(set(bm25_k) & exp) / len(exp)
                hybrid_k = hybrid_topk[:k]
                rec_hybrid[k] = len(set(hybrid_k) & exp) / len(exp)
                bm25_recalls[k].append(rec_bm25[k])
                hybrid_recalls[k].append(rec_hybrid[k])

        bm25_out.append({
            'query_id': qid,
            'question_text': q_text,
            'expected_doc_ids': list(exp),
            'n_expected': len(exp),
            'bm25_top12': bm25_topk,
            'recall_at_5': rec_bm25.get(5, 0.0),
            'recall_at_10': rec_bm25.get(10, 0.0),
            'recall_at_12': rec_bm25.get(TOP_K, 0.0),
            'recall_at_20': rec_bm25.get(20, 0.0),
        })
        hybrid_out.append({
            'query_id': qid,
            'question_text': q_text,
            'expected_doc_ids': list(exp),
            'n_expected': len(exp),
            'hybrid_top12': hybrid_topk,
            'recall_at_5': rec_hybrid.get(5, 0.0),
            'recall_at_10': rec_hybrid.get(10, 0.0),
            'recall_at_12': rec_hybrid.get(TOP_K, 0.0),
            'recall_at_20': rec_hybrid.get(20, 0.0),
        })

        if (i+1) % 100 == 0:
            print(f'  {i+1}/{len(sorted_qids)}  elapsed={time.time()-t3:.0f}s')

    print(f'Retrieval done in {time.time()-t3:.1f}s')

    # Write per-query JSONLs
    bm25_path = Path(args.out_dir) / 'bm25_solo_eval.jsonl'
    with open(bm25_path, 'w') as f:
        for r in bm25_out:
            f.write(json.dumps(r) + '\n')
    print(f'Wrote {bm25_path}')

    hyb_path = Path(args.out_dir) / 'hybrid_bflat_eval.jsonl'
    with open(hyb_path, 'w') as f:
        for r in hybrid_out:
            f.write(json.dumps(r) + '\n')
    print(f'Wrote {hyb_path}')

    # Summary
    summary = {
        'n_total_queries': len(sorted_qids),
        'n_queries_with_expected': len(bm25_recalls[TOP_K]),
        'top_k': TOP_K,
        'bm25_solo': {f'recall_at_{k}': sum(v)/len(v)*100 if v else 0.0 for k, v in bm25_recalls.items()},
        'hybrid_rrf': {f'recall_at_{k}': sum(v)/len(v)*100 if v else 0.0 for k, v in hybrid_recalls.items()},
        'rrf_k': RRF_K,
    }
    sum_path = Path(args.out_dir) / 'bm25_summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {sum_path}')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
