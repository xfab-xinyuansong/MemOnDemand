import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
#!/usr/bin/env python3
"""
rebuild_key_facts.py — 为每个 L0 节点提取结构化 key_facts 字段
格式: bullet-point 关键事实，每条保留精确实体/值/名称
写入 hierarchy_v3keyfacts.json
"""
try: from memondemand.core import dns_patch  # noqa: F401
except: pass
import argparse, json, pathlib, time, threading, queue
from memondemand.core.api_adapter import call as api_call

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
            max_tokens=400, temperature=0.0)
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
    ap.add_argument("--use_v2distilled", action="store_true",
                    help="Read from hierarchy_v2distilled.json as input")
    args = ap.parse_args()

    root = pathlib.Path(_os.environ.get("MEMONDEMAND_REPO_ROOT", REPO_ROOT))
    in_suffix = "hierarchy_v2distilled.json" if args.use_v2distilled else "hierarchy.json"
    h_candidates = [
        root / ("results/v5_emb_large_llm_distill/erag_" + args.tier + "/hierarchy/" + in_suffix),
        root / ("results/v5_emb_large_llm_distill/erag_" + args.tier + "_gold/hierarchy/" + in_suffix),
    ]
    h_path = next((p for p in h_candidates if p.exists()), None)
    if not h_path:
        print("ERROR: hierarchy not found for " + args.tier); return
    out_path = h_path.parent / "hierarchy_v3keyfacts.json"
    print("Input:  " + str(h_path))
    print("Output: " + str(out_path))

    nodes_all = [json.loads(l) for l in open(h_path)]
    l0_nodes  = [n for n in nodes_all if n["level"] == "L0"]
    if args.limit: l0_nodes = l0_nodes[:args.limit]
    print("L0 nodes: " + str(len(l0_nodes)))

    if args.dry_run:
        for n in l0_nodes[:3]:
            kf = extract_key_facts(n) or ""
            old_dis = (n.get("distilled_text") or "")[:120]
            print("\n[" + n["node_id"] + "]")
            print("  distilled: " + old_dis)
            print("  key_facts:")
            for line in kf.split("\n"):
                print("    " + line)
        return

    updated = {}
    lock = threading.Lock(); q = queue.Queue()
    for n in l0_nodes: q.put(n)
    total = len(l0_nodes); done=[0]; errs=[0]

    def worker():
        while True:
            try: n = q.get_nowait()
            except queue.Empty: break
            kf = extract_key_facts(n)
            with lock:
                if kf: updated[n["node_id"]] = kf
                else: errs[0] += 1
                done[0] += 1
                if done[0] % 500 == 0:
                    print("  [" + str(done[0]) + "/" + str(total) + "]", flush=True)
            q.task_done()

    t0 = time.time()
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in ts: t.start()
    for t in ts: t.join()
    print("\nDone: " + str(len(updated)) + "/" + str(total) + " errs=" + str(errs[0])
          + " wall=" + str(int(time.time()-t0)) + "s")

    with open(out_path, "w") as f:
        for n in nodes_all:
            if n["node_id"] in updated:
                n = dict(n)
                n["key_facts"] = updated[n["node_id"]]
                n["key_facts_model"] = "gpt_5_4_mini_keyfacts"
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print("Written -> " + str(out_path))

if __name__ == "__main__":
    main()
