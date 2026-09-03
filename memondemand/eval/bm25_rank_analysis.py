import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""Q3: BM25 direct rank analysis on same queries (10M vs 60M).

For each query with non-empty expected_doc_ids:
  - Run BM25 on 10M corpus → expected docs' ranks
  - Run BM25 on 60M corpus → expected docs' ranks
  - Recall@K for K in [5, 10, 12, 20, 50]
"""
import sys
import json, re, time
import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text):
    if not text: return []
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def build_bm25(parquet_path):
    df = pd.read_parquet(parquet_path)
    doc_ids = df["doc_id"].astype(str).tolist()
    # combine title + content + text
    texts = []
    for _, r in df.iterrows():
        title = str(r.get("title") or "")
        content = str(r.get("content") or "")
        text_field = str(r.get("text") or "")
        full = title + " " + content + " " + text_field
        texts.append(full)
    print(f"  tokenizing {len(texts)} docs ...")
    tokenized = [tokenize(t) for t in texts]
    print(f"  building BM25 ...")
    bm = BM25Okapi(tokenized)
    return bm, doc_ids


def compute_recall_at_k(retrieved_ids, expected_set):
    """Return dict of recall at various K."""
    out = {}
    for K in [5, 10, 12, 20, 50]:
        topk = set(retrieved_ids[:K])
        out[K] = len(topk & expected_set) / len(expected_set) if expected_set else 0.0
    return out


def find_ranks(retrieved_ids, expected_set):
    """For each expected id, find its rank (1-indexed); -1 if not in top-1000."""
    ranks = []
    pos = {did: i+1 for i, did in enumerate(retrieved_ids[:1000])}
    for eid in expected_set:
        ranks.append(pos.get(eid, -1))
    return ranks


def main():
    ROOT = REPO_ROOT
    gold = {}
    with open(f"{ROOT}/manifests/erag_query_gold_with_answers.jsonl") as f:
        for l in f:
            d = json.loads(l)
            if d.get("expected_doc_ids"):
                gold[d["question_id"]] = d

    qids = sorted(gold.keys())
    print(f"Gold queries with expected_doc_ids: {len(qids)}")

    results = {}
    for tier in ["10M", "60M"]:
        print(f"\n--- Building BM25 on {tier} ---")
        t0 = time.time()
        bm, doc_ids = build_bm25(f"{ROOT}/manifests/erag_{tier}_l0_nodes.parquet")
        print(f"  {tier} ready in {time.time()-t0:.1f}s, |corpus|={len(doc_ids)}")

        recalls_at_k = {5: [], 10: [], 12: [], 20: [], 50: []}
        # Track expected-doc ranks: only count those present in this tier's manifest
        all_ranks = []
        n_skip_no_coverage = 0
        n_skip_missing_doc = 0
        for qid in qids:
            g = gold[qid]
            qtext = g["question_text"]
            exp = set(g["expected_doc_ids"])
            # Check if all expected docs are in this tier's manifest
            tier_set = set(doc_ids)
            exp_in_tier = exp & tier_set
            if not exp_in_tier:
                n_skip_no_coverage += 1
                continue
            if exp_in_tier != exp:
                n_skip_missing_doc += 1
            q_toks = tokenize(qtext)
            scores = bm.get_scores(q_toks)
            top_idx = scores.argsort()[::-1][:100]
            retrieved = [doc_ids[i] for i in top_idx]
            rcl = compute_recall_at_k(retrieved, exp)
            for K in recalls_at_k:
                recalls_at_k[K].append(rcl[K])
            ranks = find_ranks(retrieved, exp_in_tier)
            all_ranks.extend(ranks)

        print(f"  {tier} skipped (no coverage): {n_skip_no_coverage}")
        print(f"  {tier} partial coverage: {n_skip_missing_doc}")
        avg_rank_in_top1000 = [r for r in all_ranks if r > 0]
        avg_rank = sum(avg_rank_in_top1000) / len(avg_rank_in_top1000) if avg_rank_in_top1000 else 0
        n_in_top12 = sum(1 for r in all_ranks if 0 < r <= 12)
        n_in_top20 = sum(1 for r in all_ranks if 0 < r <= 20)
        n_in_top50 = sum(1 for r in all_ranks if 0 < r <= 50)
        n_not_found = sum(1 for r in all_ranks if r == -1)
        print(f"  expected docs total: {len(all_ranks)}")
        print(f"    in top-12: {n_in_top12} ({n_in_top12/len(all_ranks)*100:.1f}%)")
        print(f"    in top-20: {n_in_top20} ({n_in_top20/len(all_ranks)*100:.1f}%)")
        print(f"    in top-50: {n_in_top50} ({n_in_top50/len(all_ranks)*100:.1f}%)")
        print(f"    not in top-1000: {n_not_found} ({n_not_found/len(all_ranks)*100:.1f}%)")
        print(f"    avg rank (if in top-1000): {avg_rank:.1f}")
        print(f"  per-query mean recall:")
        for K in [5, 10, 12, 20, 50]:
            vals = recalls_at_k[K]
            avg = sum(vals)/len(vals)*100 if vals else 0
            print(f"    @{K}: {avg:.2f}%")
        results[tier] = {
            "n_queries": len(qids) - n_skip_no_coverage,
            "n_partial": n_skip_missing_doc,
            "recalls_at_k": {K: sum(v)/len(v)*100 for K, v in recalls_at_k.items()},
            "n_in_top12": n_in_top12,
            "n_in_top20": n_in_top20,
            "n_in_top50": n_in_top50,
            "n_not_found": n_not_found,
            "avg_rank_in_top1000": avg_rank,
            "total_expected": len(all_ranks),
        }

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
