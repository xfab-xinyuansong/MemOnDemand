"""
V5 Baseline: B_flat — Flat Dense RAG (no hierarchy, no navigation, no promotion)
Top-k embedding search over ALL L0 nodes → direct answer.
"""
from __future__ import annotations
import argparse, json, logging, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

AZURE_DEPLOYMENT  = "gpt-5.4-mini"
AZURE_API_VERSION = "2024-12-01-preview"

def _client():
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT not set")
    return AzureOpenAI(azure_endpoint=endpoint,
                       api_key=os.environ["AZURE_LLM_API_KEY"],
                       api_version=AZURE_API_VERSION)

def call_mini(prompt, max_tokens=512):
    r = _client().chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=max_tokens)
    t = (r.choices[0].message.content or "").strip()
    return t, r.usage.prompt_tokens, r.usage.completion_tokens

def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return m.encode(texts, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--hierarchy", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_queries", type=int, default=500)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"

    # Load L0 nodes only
    nodes, texts = [], []
    with open(args.hierarchy) as f:
        for line in f:
            d = json.loads(line)
            if d["level"] == 0:
                nodes.append(d)
                texts.append(d.get("distilled_text",""))
    log.info("Loaded %d L0 nodes", len(nodes))

    log.info("Embedding L0 nodes...")
    vecs = embed_texts(texts)

    # Load queries
    df = pd.read_parquet(args.queries).head(args.max_queries)

    done_ids = set()
    if args.resume and answers_path.exists():
        done_ids = {json.loads(l)["query_id"] for l in open(answers_path)}
        log.info("Resuming: %d done", len(done_ids))

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    n_done = 0
    with open(answers_path, "a") as fout:
        for _, row in df.iterrows():
            qid = str(row["query_id"])
            if qid in done_ids: continue
            t_start = time.time()

            t0 = time.time()
            q_vec = embedder.encode([str(row["query_text"])], normalize_embeddings=True, convert_to_numpy=True)[0]
            t_embed = (time.time()-t0)*1000

            t0 = time.time()
            sims = vecs @ q_vec
            top_idx = np.argsort(-sims)[:args.top_k]
            t_retrieval = (time.time()-t0)*1000

            ctx = "\n\n".join(f"[{i+1}] {nodes[i]['distilled_text'][:600]}" for i in top_idx)
            prompt = f"Answer using only the context. Query: {row['query_text']}\n\nContext:\n{ctx}\n\nAnswer:"
            t0 = time.time()
            ans, n_in, n_out = call_mini(prompt)
            t_ans = (time.time()-t0)*1000

            l0_ids = set(nodes[i]["node_id"] for i in top_idx)
            result = {
                "query_id": qid, "tier": args.tier, "method": "B_flat",
                "answer": ans, "answer_nonempty": bool(ans),
                "has_l0_cite": bool(l0_ids),
                "t_embedding_ms": round(t_embed,1),
                "t_retrieval_ms": round(t_retrieval,1),
                "t_navigation_llm_ms": 0.0,
                "t_promotion_llm_ms": 0.0,
                "t_answer_llm_ms": round(t_ans,1),
                "t_wall_total_ms": round((time.time()-t_start)*1000,1),
                "n_llm_calls_navigation": 0,
                "n_llm_calls_promotion": 0,
                "n_llm_calls_answer": 1,
                "tokens_answer_in": n_in, "tokens_answer_out": n_out,
            }
            fout.write(json.dumps(result)+"\n"); fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                log.info("Progress %d/%d", n_done, len(df))

    answers = [json.loads(l) for l in open(answers_path)]
    n = len(answers)
    summary = {
        "tier": args.tier, "method": "B_flat", "n_queries": n,
        "citation_rate": round(sum(1 for a in answers if a["has_l0_cite"])/max(1,n),4),
        "mean_wall_ms": round(np.mean([a["t_wall_total_ms"] for a in answers]),1),
        "mean_answer_llm_ms": round(np.mean([a["t_answer_llm_ms"] for a in answers]),1),
        "cost_usd": round(sum(a["tokens_answer_in"]*0.40+a["tokens_answer_out"]*1.60 for a in answers)/1e6,4),
    }
    import json as _j
    open(out_dir/"run_summary.json","w").write(_j.dumps(summary,indent=2))
    log.info("DONE: n=%d cite=%.3f wall=%.0fms cost=$%.4f", n, summary["citation_rate"], summary["mean_wall_ms"], summary["cost_usd"])

if __name__=="__main__": main()
