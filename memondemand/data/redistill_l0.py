import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
#!/usr/bin/env python3
"""Rebuild L0 distilled_text with a fact-preserving prompt (threaded).
Usage: python redistill_l0.py --hierarchy <path> --out <path> [--workers 16]
"""
import sys, argparse, json, time, pathlib, logging, threading, queue
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
try:
    from memondemand.core import dns_patch  # noqa: F401
except Exception:
    pass
from memondemand.core.api_adapter import call as api_call

DISTILL_SYSTEM = """You are a fact-preserving memory distiller for enterprise documents.

Write a compact factual summary (80-130 words) that a retrieval query can use to decide whether this document contains the answer.

MANDATORY rules — preserve ALL of the following:
1. Proper nouns: company names, person names, product names, system names, team names
2. Numerical specifics: thresholds, dates, counts, percentages, durations, sizes, limits
3. Technical identifiers: config keys, API names, CLI flags, file paths, schema fields, error codes, version numbers
4. Key decisions, conclusions, policy statements (quote exactly if short)
5. Actions taken, who was involved, what was changed or resolved

Format: Write ONLY the summary. No preamble, no markdown. Fact-dense is better than fluent-vague."""

DISTILL_USER = """Document:
{text}

Fact-preserving summary (80-130 words):"""

def redistill_one(det: str, alias: str = "gpt_5_4_mini") -> str:
    user = DISTILL_USER.format(text=det[:6000])
    resp = api_call(alias, [
        {"role": "system", "content": DISTILL_SYSTEM},
        {"role": "user",   "content": user},
    ], max_tokens=200, temperature=0.0)
    return (resp.get("text") or "").strip()

def worker(task_q, result_q, alias):
    while True:
        item = task_q.get()
        if item is None:
            break
        obj, idx = item
        det = obj.get("detailed_text", "") or ""
        try:
            new_dis = redistill_one(det, alias) if det.strip() else (obj.get("distilled_text","") or "")
        except Exception as e:
            log.warning("Error node %s: %s", obj["node_id"], e)
            new_dis = obj.get("distilled_text","") or det[:200]
        obj["distilled_text"] = new_dis
        obj["distilled_tokens"] = len(new_dis.split())
        obj["distilled_text_model_alias"] = alias
        obj["distilled_text_model_status"] = "REDIST_V2"
        result_q.put((idx, obj))
        task_q.task_done()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alias", default="gpt_5_4_mini")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.hierarchy)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all nodes
    all_nodes = []
    with open(in_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            all_nodes.append(obj)
    log.info("Loaded %d nodes total", len(all_nodes))

    # Resume: load already-done
    done = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                obj = json.loads(line)
                done[obj["node_id"]] = obj
        log.info("Resume: %d already done", len(done))

    # Partition
    l0_todo = [(i, obj) for i, obj in enumerate(all_nodes)
                if obj["level"] == "L0" and obj["node_id"] not in done]
    l0_skip = [(i, obj) for i, obj in enumerate(all_nodes)
                if obj["level"] == "L0" and obj["node_id"] in done]
    non_l0  = [(i, obj) for i, obj in enumerate(all_nodes) if obj["level"] != "L0"]
    log.info("L0 todo=%d  skip=%d  non-L0=%d", len(l0_todo), len(l0_skip), len(non_l0))

    # Thread pool
    task_q = queue.Queue(maxsize=args.workers * 4)
    result_q = queue.Queue()
    threads = [threading.Thread(target=worker, args=(task_q, result_q, args.alias), daemon=True)
               for _ in range(args.workers)]
    for t in threads: t.start()

    # Feed tasks
    def feed():
        for idx, obj in l0_todo:
            task_q.put((obj, idx))
        for _ in threads:
            task_q.put(None)
    feed_t = threading.Thread(target=feed, daemon=True)
    feed_t.start()

    # Collect results
    results = {}
    t0 = time.time()
    for _ in range(len(l0_todo)):
        idx, obj = result_q.get()
        results[obj["node_id"]] = (idx, obj)
        n = len(results)
        if n % 200 == 0:
            rate = n / max(1, time.time()-t0)
            log.info("Done %d/%d L0 nodes (%.1f/s, eta %.0fs)",
                     n, len(l0_todo), rate, (len(l0_todo)-n)/max(0.01,rate))

    for t in threads: t.join()

    # Write output (preserve original order)
    # Merge: non-L0 + skipped + new results
    all_out = {}
    for i, obj in non_l0:
        all_out[i] = obj
    for i, obj in l0_skip:
        all_out[i] = done[obj["node_id"]]
    for nid, (i, obj) in results.items():
        all_out[i] = obj

    with open(out_path, "w") as f:
        for i in sorted(all_out):
            f.write(json.dumps(all_out[i], ensure_ascii=False) + "\n")
    log.info("Written %d nodes to %s", len(all_out), out_path)

if __name__ == "__main__":
    main()
