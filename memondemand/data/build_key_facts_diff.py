import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
#!/usr/bin/env python3
"""
build_key_facts_diff.py — 差集 L0 key_facts 构建
从上一规模的 hierarchy (已有 key_facts) 中复用已有 doc_id 的 key_facts，
只对新增 doc_id 调用 gpt_5_4_mini，大幅减少 API 用量。

用法:
  python build_key_facts_diff.py \
    --tier 60M \
    --prev_kf_json results/.../erag_20M/hierarchy/hierarchy_v4l1kf.json \
    --workers 12

输出: results/v5_emb_large_llm_distill/erag_{tier}/hierarchy/hierarchy_v3keyfacts.json
"""
import sys, json, pathlib, time, threading, queue, argparse
try: from memondemand.core import dns_patch  # noqa: F401
except: pass

from memondemand.core.api_adapter import call as api_call

BASE = str(
    pathlib.Path(_os.environ.get("MEMONDEMAND_REPO_ROOT", REPO_ROOT))
    / "results/v5_emb_large_llm_distill"
)

KEY_FACTS_SYSTEM = (
    "You are a fact extractor for an enterprise knowledge base.\n"
    "Given a document, extract 4-8 key facts as bullet points.\n"
    "Each bullet must:\n"
    "  - Be a single atomic fact (one sentence)\n"
    "  - Preserve EXACT names: config keys, field names, system names, people names, identifiers\n"
    "  - Preserve EXACT values: thresholds, dates, percentages, version numbers\n"
    "  - Be self-contained (readable without the document)\n"
    "Format: one bullet per line starting with '• '\n"
    "Output ONLY the bullets. No headers, no preamble, no explanation."
)

def extract_key_facts(node, alias="gpt_5_4_mini"):
    content = (node.get("detailed_text") or "").strip()
    if not content:
        return node.get("distilled_text", "")
    user = ("Document (tenant: " + node.get("tenant_id","unknown") + "):\n"
            + content[:5000] + "\n\nKey facts (4-8 bullets, preserve exact names/values):")
    try:
        resp = api_call(alias,
            [{"role":"system","content":KEY_FACTS_SYSTEM},
             {"role":"user","content":user}],
            max_tokens=400, temperature=0.0,
            max_retries=5, backoff_base=4.0)
        return (resp.get("text") or "").strip()
    except Exception as e:
        print(f"  [WARN] {node.get('node_id')}: {e}", flush=True)
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, help="e.g. 60M")
    ap.add_argument("--prev_kf_json", required=True,
                    help="上一规模已有 key_facts 的 hierarchy JSON (如 20M 的 hierarchy_v4l1kf.json)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    hier_path = pathlib.Path(f"{BASE}/erag_{args.tier}/hierarchy/hierarchy.json")
    out_path  = hier_path.parent / "hierarchy_v3keyfacts.json"

    if not hier_path.exists():
        print(f"ERROR: {hier_path} not found"); return

    print(f"[{args.tier}] Loading prev key_facts from {args.prev_kf_json} ...", flush=True)
    prev_kf = {}
    for line in open(args.prev_kf_json):
        n = json.loads(line)
        kf = n.get("key_facts", "")
        if kf and (n.get("level") == 0 or n.get("level") == "L0"):
            prev_kf[n["node_id"]] = kf
    print(f"  prev_kf loaded: {len(prev_kf)} doc_ids", flush=True)

    print(f"[{args.tier}] Loading current hierarchy ...", flush=True)
    nodes_all = [json.loads(l) for l in open(hier_path)]
    l0_nodes = [n for n in nodes_all if n.get("level") == 0 or n.get("level") == "L0"]

    reused   = {n["node_id"]: prev_kf[n["node_id"]] for n in l0_nodes if n["node_id"] in prev_kf}
    new_l0   = [n for n in l0_nodes if n["node_id"] not in prev_kf]
    print(f"  L0 total={len(l0_nodes)}  reused={len(reused)}  new_to_extract={len(new_l0)}", flush=True)

    if args.dry_run:
        print(f"[dry_run] would extract {len(new_l0)} new L0 key_facts"); return

    # Extract key_facts for NEW L0 nodes only
    extracted = dict(reused)
    lock = threading.Lock()
    q = queue.Queue()
    for n in new_l0: q.put(n)
    total_new = len(new_l0); done=[0]; errs=[0]

    def worker():
        while True:
            try: n = q.get_nowait()
            except queue.Empty: break
            kf = extract_key_facts(n)
            with lock:
                if kf: extracted[n["node_id"]] = kf
                else: errs[0] += 1
                done[0] += 1
                if done[0] % 200 == 0:
                    print(f"  [{done[0]}/{total_new} new | {len(extracted)} total]", flush=True)
            q.task_done()

    t0 = time.time()
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in ts: t.start()
    for t in ts: t.join()

    print(f"\n  Done: {len(extracted)}/{len(l0_nodes)} key_facts  errs={errs[0]}  "
          f"wall={int(time.time()-t0)}s", flush=True)

    # Write output
    print(f"[{args.tier}] Writing {out_path} ...", flush=True)
    with open(out_path, "w") as f:
        for n in nodes_all:
            nid = n["node_id"]
            if nid in extracted:
                n = dict(n)
                n["key_facts"] = extracted[nid]
                n["key_facts_model"] = "gpt_5_4_mini_keyfacts"
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"  Written -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
