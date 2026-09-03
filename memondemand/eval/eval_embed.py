import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""
Embedding-based evaluation for MemOnDemand runs.
Replaces LLM judge with cosine similarity between predicted and gold answers.
Uses Azure text-embedding-3-large (3072d) - no rate limit issues.

Output: per_query_eval.jsonl (same format as evaluate_v5.py)
  llm_judge_raw : int 1-5  (from sim × 5, rounded)
  llm_score     : float 0-1 (cosine similarity)
  llm_judge_rationale : "embed_sim:<score>"
  Correct_i = llm_judge_raw > 2  →  sim >= 0.50 (lenient threshold)
"""
import json, argparse, os, math, pathlib, sys
import numpy as np

from memondemand.core.api_adapter import load_env
load_env()

EMBED_BATCH = 1           # Azure embed hangs on batch>5; single safer
EMBED_MODEL = os.environ.get("AZURE_EMBED_LARGE_DEPLOYMENT", "text-embedding-3-large")
EMBED_API_VERSION = "2024-02-01"
CORRECT_THRESHOLD = 0.50  # sim >= this → Correct (lenient, matches LLM "broadly aligned")

def get_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=os.environ["AZURE_LLM_API_KEY"],
        api_version=EMBED_API_VERSION,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )

def embed_batch(client, texts):
    """Embed a list of texts, return list of np.ndarray."""
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=60.0)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)

def sim_to_raw(sim):
    """Map 0-1 similarity to 1-5 int scale."""
    return max(1, min(5, round(sim * 5)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--gold",    required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--resume",  action="store_true")
    args = ap.parse_args()

    # Load inputs
    answers = [json.loads(l) for l in open(args.answers)]
    gold_map = {}
    for l in open(args.gold):
        g = json.loads(l)
        gold_map[g.get("question_id", g.get("query_id",""))] = g

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_query_eval.jsonl"

    # Resume: skip already-done (valid score) rows
    done_ids = set()
    if args.resume and out_path.exists():
        for l in open(out_path):
            try:
                d = json.loads(l)
                if (d.get("llm_judge_raw") or 0) > 0:
                    done_ids.add(d["query_id"])
            except Exception:
                pass
        print(f"resume: {len(done_ids)} already done")

    # Filter to-do
    todo = [a for a in answers if a.get("query_id") not in done_ids]
    # Only ANSWER queries need embedding
    answer_todo = [a for a in todo if a.get("final_action") == "ANSWER"]
    nonanswer_todo = [a for a in todo if a.get("final_action") != "ANSWER"]
    print(f"to eval: {len(answer_todo)} ANSWER + {len(nonanswer_todo)} non-ANSWER")

    client = get_client()

    # Gather predicted and gold texts
    pred_texts, gold_texts, valid_mask = [], [], []
    for a in answer_todo:
        qid = a.get("query_id", "")
        g = gold_map.get(qid, {})
        gold_ans = g.get("gold_answer", "")
        pred_ans = a.get("answer_text", "") or a.get("answer", "") or ""
        # strip "CITED: ..." from pred
        if "CITED:" in pred_ans:
            pred_ans = pred_ans[:pred_ans.index("CITED:")].strip()
        pred_texts.append(pred_ans or "N/A")
        gold_texts.append(gold_ans or "N/A")
        valid_mask.append(bool(gold_ans) and bool(pred_ans))

    # Batch embed
    print(f"embedding {len(pred_texts) * 2} texts in batches of {EMBED_BATCH}...")
    all_texts = pred_texts + gold_texts
    all_vecs = []
    for i in range(0, len(all_texts), EMBED_BATCH):
        batch = all_texts[i: i + EMBED_BATCH]
        vecs = embed_batch(client, batch)
        all_vecs.extend(vecs)
        if i % 50 == 0:
            print(f"  embedded {min(i+EMBED_BATCH, len(all_texts))}/{len(all_texts)}")

    pred_vecs = all_vecs[:len(pred_texts)]
    gold_vecs = all_vecs[len(pred_texts):]

    # Write results
    fout = open(out_path, "a", encoding="utf-8") if args.resume else open(out_path, "w", encoding="utf-8")
    n_written = 0

    # Non-ANSWER rows: score=0
    for a in nonanswer_todo:
        qid = a.get("query_id", "")
        g = gold_map.get(qid, {})
        cited = a.get("cited_evidence_ids") or []
        raw_ids = g.get("expected_doc_ids", []) or []
        if isinstance(raw_ids, str):
            import ast
            try: raw_ids = ast.literal_eval(raw_ids)
            except: raw_ids = []
        gold_ids = list(raw_ids)
        intersect = set(cited) & set(gold_ids)
        dr = len(intersect) / max(1, len(gold_ids))
        row = {
            "query_id": qid,
            "final_action": a.get("final_action", "STOP_INSUFFICIENT"),
            "doc_recall_this_query": dr,
            "llm_judge_raw": 0,
            "llm_score": 0.0,
            "llm_judge_rationale": "non_answer",
            "cited_evidence_ids": cited,
            "gold_ids": gold_ids,
        }
        fout.write(json.dumps(row) + "\n")
        n_written += 1

    # ANSWER rows: embedding similarity
    for i, a in enumerate(answer_todo):
        qid = a.get("query_id", "")
        g = gold_map.get(qid, {})
        cited = a.get("cited_evidence_ids") or []
        raw_ids = g.get("expected_doc_ids", []) or []
        if isinstance(raw_ids, str):
            import ast
            try: raw_ids = ast.literal_eval(raw_ids)
            except: raw_ids = []
        gold_ids = list(raw_ids)
        intersect = set(cited) & set(gold_ids)
        dr = len(intersect) / max(1, len(gold_ids))

        if valid_mask[i]:
            sim = cosine_sim(pred_vecs[i], gold_vecs[i])
        else:
            sim = 0.0

        raw = sim_to_raw(sim)
        row = {
            "query_id": qid,
            "final_action": "ANSWER",
            "doc_recall_this_query": dr,
            "llm_judge_raw": raw,
            "llm_score": round(sim, 4),
            "llm_judge_rationale": f"embed_sim:{sim:.4f}",
            "cited_evidence_ids": cited,
            "gold_ids": gold_ids,
        }
        fout.write(json.dumps(row) + "\n")
        n_written += 1

    fout.close()
    print(f"\nwrote {n_written} rows → {out_path}")

    # Summary
    all_rows = [json.loads(l) for l in open(out_path)]
    n_total = 500
    n_gold  = 470
    rcl     = sum(r["doc_recall_this_query"] for r in all_rows)
    correct = sum(1 for r in all_rows if r.get("llm_judge_raw", 0) > 2)
    combined = sum(
        (1 if r.get("llm_judge_raw", 0) > 2 else 0) * r.get("llm_score", 0.0)
        for r in all_rows
    )
    inv_doc = sum(
        max(0, len(r.get("cited_evidence_ids") or []) - len(r.get("gold_ids") or []))
        for r in all_rows
    ) / len(all_rows)
    valid_scores = [r["llm_score"] for r in all_rows if r.get("llm_judge_raw", 0) > 0]
    mean_sim = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    print(f"\n=== SUMMARY ({len(all_rows)}/500 rows) ===")
    print(f"DocRcl%   : {rcl/n_gold*100:.2f}")
    print(f"Correct%  : {correct/n_total*100:.1f}  (sim >= {CORRECT_THRESHOLD})")
    print(f"Combined% : {combined/n_total*100:.1f}")
    print(f"InvDoc    : {inv_doc:.2f}")
    print(f"Mean sim  : {mean_sim:.3f}")

if __name__ == "__main__":
    main()
