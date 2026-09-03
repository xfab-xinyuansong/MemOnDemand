#!/usr/bin/env python3
"""Backfill missing L0 key facts for the full EnterpriseRAG-Bench tier.

The incremental output is resumable and uses a quality gate before accepting
each generated record.
"""
import os, json, time, pathlib, argparse
import concurrent.futures

try:
    from memondemand.core import dns_patch  # noqa: F401
except ImportError:
    dns_patch = None

from memondemand.pipelines.enterprise_rag_1_14b.runtime import call as api_call

KEY_FACTS_SYSTEM = (
    "You are a fact extractor for an enterprise knowledge base.\n"
    "Given a document, extract 4-8 key facts as bullet points.\n"
    "Each bullet must:\n"
    "  - Be a single atomic fact (one sentence)\n"
    "  - Preserve EXACT names: config keys, field names, system names, people names, identifiers\n"
    "  - Preserve EXACT values: thresholds, dates, percentages, version numbers\n"
    "  - Be self-contained (readable without the document)\n"
    "Format: one bullet per line starting with '\u2022 '\n"
    "Output ONLY the bullets. No headers, no preamble, no explanation."
)

def is_valid_key_facts(text):
    if not text or not isinstance(text, str):
        return False, "empty_or_none"
    t = text.strip()
    if len(t) < 20:
        return False, f"too_short({len(t)}chars)"
    if "\u2022" not in t:
        return False, "no_bullet_marker"
    if len(t) < 150:
        lower = t.lower()
        for bad in ["i cannot", "i can't", "as an ai", "error:", "http 429", "http 500"]:
            if bad in lower:
                return False, f"error_like_content:{bad}"
    return True, "ok"

def extract_key_facts_with_retry(node, alias="gpt_5_4", max_attempts=200):
    content = (node.get("detailed_text") or "").strip()
    if not content:
        fallback = (node.get("distilled_text") or "").strip()
        return fallback, "fallback_distilled" if fallback else "empty_content"
    user = ("Document (tenant: " + node.get("tenant_id", "unknown") + "):\n"
            + content[:5000] + "\n\nKey facts (4-8 bullets, preserve exact names/values):")
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            resp = api_call(alias,
                [{"role": "system", "content": KEY_FACTS_SYSTEM},
                 {"role": "user", "content": user}],
                max_tokens=400, temperature=0.0)
            text = (resp.get("text") or "").strip()
            valid, reason = is_valid_key_facts(text)
            if valid:
                return text, "ok"
            if attempt % 10 == 1:
                print(f"  [invalid_output] {node.get('node_id')} attempt={attempt} reason={reason}", flush=True)
            time.sleep(min(20, 3 + attempt))
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "rate_limit" in msg.lower() or "usage limit" in msg.lower()
            sleep_s = min(60, 5 + attempt * 1.5) if is_429 else min(15, 3 + attempt)
            if attempt % 10 == 1:
                print(f"  [retry] {node.get('node_id')} attempt={attempt} err={msg[:100]} sleeping={sleep_s:.0f}s", flush=True)
            time.sleep(sleep_s)
    return None, f"giving_up_after_{max_attempts}_attempts"


def load_already_done(out_jsonl_path):
    done = set()
    if not os.path.exists(out_jsonl_path):
        return done
    with open(out_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("key_facts"):
                done.add(r["node_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier_path", required=True)
    ap.add_argument("--missing_ids_path", required=True)
    ap.add_argument("--out_incremental", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print(f"Loading missing ids from {args.missing_ids_path} ...", flush=True)
    missing_ids = set(json.load(open(args.missing_ids_path)))
    print(f"target nodes: {len(missing_ids)}", flush=True)

    already_done = load_already_done(args.out_incremental)
    print(f"already done (resume): {len(already_done)}", flush=True)

    todo_ids = missing_ids - already_done
    print(f"todo: {len(todo_ids)}", flush=True)

    print(f"Loading L0 nodes from {args.hier_path} (filtering to todo set) ...", flush=True)
    todo_nodes = []
    with open(args.hier_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n = json.loads(line)
            if n.get("level") == "L0" and n["node_id"] in todo_ids:
                todo_nodes.append(n)
    print(f"loaded {len(todo_nodes)} target nodes", flush=True)

    out_f = open(args.out_incremental, "a")
    done_count = [0]
    failed_count = [0]
    lock = __import__("threading").Lock()

    def process_one(node):
        text, reason = extract_key_facts_with_retry(node)
        with lock:
            if text and reason in ("ok", "fallback_distilled"):
                rec = {"node_id": node["node_id"], "key_facts": text, "key_facts_model": "gpt-5.4", "reason": reason}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                done_count[0] += 1
            else:
                failed_count[0] += 1
            total = done_count[0] + failed_count[0]
            if total % 100 == 0:
                print(f"progress {total}/{len(todo_nodes)} (done={done_count[0]} failed={failed_count[0]})", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(process_one, todo_nodes))

    print(f"DONE: {done_count[0]} done, {failed_count[0]} failed", flush=True)


if __name__ == "__main__":
    main()
