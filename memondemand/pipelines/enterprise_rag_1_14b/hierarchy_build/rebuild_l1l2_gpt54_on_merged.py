#!/usr/bin/env python3
"""Rebuild L1/L2 key facts after merging the L0 backfill.

Usage:
    python3 rebuild_l1l2_gpt54_on_merged.py --tier 60M --hier_path <path to merged hierarchy>
    python3 rebuild_l1l2_gpt54_on_merged.py --tier 100M --hier_path <path to merged hierarchy>
"""
import argparse, json, time, pathlib

try:
    from memondemand.core import dns_patch  # noqa: F401
except ImportError:
    dns_patch = None

from memondemand.pipelines.enterprise_rag_1_14b.runtime import call as call_llm

L1_PROMPT = """\
Below are key-fact bullet points from {n_children} child documents grouped into this cluster.
Produce exactly 6 concise bullet points that aggregate the MOST IMPORTANT entities, exact values,
technical names, people, thresholds, and identifiers that span this cluster.
Each bullet: <=25 words. Preserve exact strings (config keys, names, numbers).
Respond ONLY with the 6 bullets, one per line, starting with "-".

CHILD KEY FACTS (truncated):
{child_kf}
"""

L2_PROMPT = """\
Below are key-fact summaries from {n_children} child clusters.
Produce exactly 4 concise bullet points capturing the TOP entities, themes, exact values,
and identifiers that span all child clusters.
Each bullet: <=30 words. Respond ONLY with the 4 bullets starting with "-".

CHILD SUMMARIES:
{child_kf}
"""

MAX_ATTEMPTS = 50

def is_rate_limit(exc):
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "quota" in s

def build_key_facts(node_id, children, nodes, level, alias="gpt_5_4"):
    if level == "L1":
        kf_parts = []
        for cid in children[:60]:
            c = nodes.get(cid, {})
            kf = (c.get("key_facts") or "").strip() or (c.get("distilled_text") or "").strip()
            if kf:
                kf_parts.append(kf[:400])
        if not kf_parts:
            return None
        child_kf = "\n\n".join(kf_parts[:40])[:6000]
        prompt = L1_PROMPT.format(n_children=len(children), child_kf=child_kf)
    else:
        kf_parts = []
        for cid in children:
            c = nodes.get(cid, {})
            kf = (c.get("key_facts") or "").strip() or (c.get("distilled_text") or "").strip()
            if kf:
                kf_parts.append(kf[:500])
        if not kf_parts:
            return None
        child_kf = "\n\n".join(kf_parts)[:4000]
        prompt = L2_PROMPT.format(n_children=len(children), child_kf=child_kf)

    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = call_llm(alias=alias, messages=[{"role": "user", "content": prompt}],
                            max_tokens=300, temperature=0.0)
            text = (resp.get("text") or resp.get("content") or "").strip()
            if text:
                return text
            return None
        except Exception as e:
            if is_rate_limit(e):
                sleep_s = 20.0 + attempt * 3
                print(f"    [429] {node_id} attempt {attempt+1}, sleeping {sleep_s:.0f}s", flush=True)
                time.sleep(sleep_s)
                continue
            print(f"  [WARN] {node_id}: {e}", flush=True)
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--hier_path", required=True, help="path to the ALREADY L0-key_facts-merged hierarchy file")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    hier_in = args.hier_path
    hier_out = str(pathlib.Path(hier_in).with_name(
        pathlib.Path(hier_in).stem + "_l1l2gpt54.json"))

    print(f"=== [{args.tier}] Loading merged hierarchy from {hier_in} ===", flush=True)
    nodes = {}
    with open(hier_in) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n = json.loads(line)
            nodes[n["node_id"]] = n

    # Build parent->children map from extra.child_node_ids (verified field
    # in this hierarchy format: L1 nodes have extra.child_node_ids listing
    # L0 dsid children; L2 nodes have extra.child_node_ids listing L1_xxx
    # cluster-id children).
    parent_to_children = {}
    for nid, n in nodes.items():
        extra = n.get("extra") or {}
        kids = n.get("children_ids") or extra.get("child_node_ids") or extra.get("children_ids")
        if kids:
            parent_to_children[nid] = list(kids)

    l1_nodes = [n for n in nodes.values() if n.get("level") == "L1"]
    l2_nodes = [n for n in nodes.values() if n.get("level") == "L2"]
    print(f"  L1={len(l1_nodes)} L2={len(l2_nodes)}", flush=True)

    print(f"  Rebuilding L1 key_facts with gpt-5.4 (SERIAL, sleep={args.sleep}s) ...", flush=True)
    done = 0
    for n in l1_nodes:
        kids = parent_to_children.get(n["node_id"], [])
        kf = build_key_facts(n["node_id"], kids, nodes, "L1", alias="gpt_5_4")
        if kf:
            nodes[n["node_id"]]["key_facts"] = kf
            nodes[n["node_id"]]["key_facts_model"] = "gpt-5.4"
        done += 1
        if done % 30 == 0:
            print(f"    [{done}/{len(l1_nodes)}]", flush=True)
        time.sleep(args.sleep)
    print(f"  L1 done: {sum(1 for n in l1_nodes if nodes[n['node_id']].get('key_facts'))}/{len(l1_nodes)} have key_facts", flush=True)

    print(f"  Rebuilding L2 key_facts with gpt-5.4 (SERIAL, sleep={args.sleep}s) ...", flush=True)
    for n in l2_nodes:
        kids = parent_to_children.get(n["node_id"], [])
        kf = build_key_facts(n["node_id"], kids, nodes, "L2", alias="gpt_5_4")
        if kf:
            nodes[n["node_id"]]["key_facts"] = kf
            nodes[n["node_id"]]["key_facts_model"] = "gpt-5.4"
        time.sleep(args.sleep)
    print(f"  L2 done: {sum(1 for n in l2_nodes if nodes[n['node_id']].get('key_facts'))}/{len(l2_nodes)} have key_facts", flush=True)

    print(f"  Writing {hier_out} ...", flush=True)
    with open(hier_out, "w") as out:
        for n in nodes.values():
            out.write(json.dumps(n, ensure_ascii=False) + "\n")

    # verify counts by level
    kf_counts, total = {}, {}
    for n in nodes.values():
        lv = n.get("level", "?")
        total[lv] = total.get(lv, 0) + 1
        if (n.get("key_facts") or "").strip():
            kf_counts[lv] = kf_counts.get(lv, 0) + 1
    for lv in sorted(total):
        print(f"  {lv}: {kf_counts.get(lv,0)}/{total[lv]} have key_facts")
    print(f"[{args.tier}] Done -> {hier_out}", flush=True)


if __name__ == "__main__":
    main()
