import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
#!/usr/bin/env python3
"""rebuild_distilled_v2.py — 改进 L0 节点 distilled_text（精确事实提取 prompt）"""
try:
    from memondemand.core import dns_patch  # noqa: F401
except Exception:
    pass
import argparse, json, pathlib, time, threading, queue
from memondemand.core.api_adapter import call as api_call

L0_DISTILL_V2_SYSTEM = (
    "You are a precision memory extractor for an enterprise knowledge base.\n"
    "Given a document, produce a compact fact-summary (3-5 sentences) that preserves:\n"
    "1. KEY ENTITIES: system names, people, organizations, product names\n"
    "2. EXACT VALUES: config keys, field names, thresholds, dates, numbers, identifiers\n"
    "3. MAIN DECISION or FACTUAL CLAIM: what was decided, required, or established\n"
    "4. UNIQUE TECHNICAL NAMES: endpoint names, workflow names, process identifiers\n"
    "Preserve specifics. Do NOT generalize away named entities or exact values.\n"
    "Output the fact-summary text only. No headers, no bullets, no preamble."
)

def distill_one(node, alias="gpt_5_4_mini"):
    content = (node.get("detailed_text") or "").strip()
    if not content:
        return node.get("distilled_text", "")
    user = ("Document (tenant: " + node.get("tenant_id","unknown") + "):\n"
            + content[:6000] + "\n\nFact-summary (3-5 sentences, preserve all specific names, values, identifiers):")
    try:
        resp = api_call(alias,
            [{"role":"system","content":L0_DISTILL_V2_SYSTEM},
             {"role":"user","content":user}],
            max_tokens=300, temperature=0.0)
        return (resp.get("text") or "").strip()
    except Exception as e:
        print("  ERROR " + str(node.get("node_id")) + ": " + str(e))
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="10M")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = pathlib.Path(_os.environ.get("MEMONDEMAND_REPO_ROOT", REPO_ROOT))
    candidates = [
        root / ("results/v5_emb_large_llm_distill/erag_" + args.tier + "/hierarchy/hierarchy.json"),
        root / ("results/v5_emb_large_llm_distill/erag_" + args.tier + "_gold/hierarchy/hierarchy.json"),
    ]
    h_path = next((p for p in candidates if p.exists()), None)
    if not h_path:
        print("ERROR: hierarchy not found for " + args.tier); return
    out_path = h_path.parent / "hierarchy_v2distilled.json"
    print("Hierarchy: " + str(h_path))
    print("Output:    " + str(out_path))

    nodes_all = [json.loads(l) for l in open(h_path)]
    l0_nodes  = [n for n in nodes_all if n["level"] == "L0"]
    if args.limit:
        l0_nodes = l0_nodes[:args.limit]
    print("L0 nodes to re-distill: " + str(len(l0_nodes)))

    if args.dry_run:
        for n in l0_nodes[:3]:
            old = n.get("distilled_text","")
            new = distill_one(n) or ""
            print("\n[" + n["node_id"] + "] tenant=" + str(n.get("tenant_id")))
            print("  OLD (" + str(len(old)) + "c): " + old[:200])
            print("  NEW (" + str(len(new)) + "c): " + new[:300])
        return

    updated = {}
    lock = threading.Lock()
    q = queue.Queue()
    for n in l0_nodes:
        q.put(n)
    total = len(l0_nodes)
    done = [0]; errs = [0]

    def worker():
        while True:
            try:
                n = q.get_nowait()
            except queue.Empty:
                break
            nd = distill_one(n)
            with lock:
                if nd:
                    updated[n["node_id"]] = nd
                else:
                    errs[0] += 1
                done[0] += 1
                if done[0] % 200 == 0:
                    print("  [" + str(done[0]) + "/" + str(total) + "]", flush=True)
            q.task_done()

    t0 = time.time()
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in ts: t.start()
    for t in ts: t.join()
    print("\nUpdated " + str(len(updated)) + "/" + str(total)
          + ", errs=" + str(errs[0]) + ", wall=" + str(int(time.time()-t0)) + "s")

    with open(out_path, "w") as f:
        for n in nodes_all:
            if n["node_id"] in updated:
                n = dict(n)
                n["distilled_text"] = updated[n["node_id"]]
                n["distilled_text_model_alias"] = "gpt_5_4_mini_v2fact"
                n["distilled_tokens"] = len(updated[n["node_id"]].split())
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print("Done -> " + str(out_path))

if __name__ == "__main__":
    main()
