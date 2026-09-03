import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""
rebuild_l1l2_keyfacts.py
Rebuild L1/L2 key_facts by aggregating children's key_facts via gpt_5_4_mini.
Output: hierarchy_v4l1kf.json  (same JSONL format, L1/L2 now have key_facts)
Usage: python rebuild_l1l2_keyfacts.py --tier 10M [--tier 20M]
"""
import argparse, json, os, sys, time, concurrent.futures, pathlib

try: from memondemand.core import dns_patch  # noqa: F401
except: pass

from memondemand.core.api_adapter import call as call_llm

BASE = os.path.join(
    os.environ.get("MEMONDEMAND_REPO_ROOT", REPO_ROOT),
    "results/v5_emb_large_llm_distill",
)

L1_PROMPT = """\
Below are key-fact bullet points from {n_children} child documents grouped into this cluster.
Produce exactly 6 concise bullet points that aggregate the MOST IMPORTANT entities, exact values,
technical names, people, thresholds, and identifiers that span this cluster.
Each bullet: ≤25 words. Preserve exact strings (config keys, names, numbers).
Respond ONLY with the 6 bullets, one per line, starting with "•".

CHILD KEY FACTS (truncated):
{child_kf}
"""

L2_PROMPT = """\
Below are key-fact summaries from {n_children} child clusters.
Produce exactly 4 concise bullet points capturing the TOP entities, themes, exact values,
and identifiers that span all child clusters.
Each bullet: ≤30 words. Respond ONLY with the 4 bullets starting with "•".

CHILD SUMMARIES:
{child_kf}
"""

def build_key_facts(node_id, children, nodes, level, alias="gpt_5_4_mini"):
    if level == "L1":
        # collect children L0 key_facts
        kf_parts = []
        for cid in children[:60]:   # cap at 60 children
            c = nodes.get(cid, {})
            kf = c.get("key_facts", "").strip()
            if kf:
                kf_parts.append(kf[:400])  # cap each child
        if not kf_parts:
            return None
        child_kf = "\n\n".join(kf_parts[:40])[:6000]  # max 6K chars
        prompt = L1_PROMPT.format(n_children=len(children), child_kf=child_kf)
    else:  # L2
        kf_parts = []
        for cid in children:
            c = nodes.get(cid, {})
            kf = c.get("key_facts", "").strip() or c.get("distilled_text", "").strip()
            if kf:
                kf_parts.append(kf[:500])
        if not kf_parts:
            return None
        child_kf = "\n\n".join(kf_parts)[:4000]
        prompt = L2_PROMPT.format(n_children=len(children), child_kf=child_kf)

    try:
        resp = call_llm(alias=alias, messages=[{"role": "user", "content": prompt}],
                        max_tokens=300)
        return resp.get("text", resp.get("content", "")).strip()
    except Exception as e:
        print(f"  [WARN] {node_id}: {e}", flush=True)
        return None


def process_tier(tier: str):
    hier_in  = f"{BASE}/erag_{tier}/hierarchy/hierarchy.json"
    hier_out = f"{BASE}/erag_{tier}/hierarchy/hierarchy_v4l1kf.json"

    print(f"\n=== [{tier}] Loading hierarchy ===", flush=True)
    nodes = {}
    with open(hier_in) as f:
        for line in f:
            n = json.loads(line)
            nodes[n["node_id"]] = n

    # build parent→children map from L1/L2 extra.child_node_ids
    parent_child = {}
    for nid, n in nodes.items():
        extra = n.get("extra") or {}
        kids = extra.get("child_node_ids", []) or extra.get("children_ids", [])
        if not kids and n.get("level") == "L1":
            kids = list(n.get("source_evidence_ids", []) or [])
        if kids:
            parent_child[nid] = kids

    l1_nodes = [n for n in nodes.values() if n.get("level") == "L1"]
    l2_nodes = [n for n in nodes.values() if n.get("level") == "L2"]
    print(f"  L1={len(l1_nodes)} L2={len(l2_nodes)}", flush=True)

    # ---- L1 key_facts ----
    print(f"  Building L1 key_facts ({len(l1_nodes)} nodes, 8 workers) ...", flush=True)
    done = 0
    def do_l1(n):
        kids = parent_child.get(n["node_id"], [])
        kf = build_key_facts(n["node_id"], kids, nodes, "L1")
        return n["node_id"], kf

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(do_l1, n): n["node_id"] for n in l1_nodes}
        for fut in concurrent.futures.as_completed(futures):
            nid, kf = fut.result()
            if kf:
                nodes[nid]["key_facts"] = kf
            done += 1
            if done % 30 == 0:
                print(f"    [{done}/{len(l1_nodes)}]", flush=True)
    print(f"  L1 done: {sum(1 for n in l1_nodes if nodes[n['node_id']].get('key_facts'))} have key_facts", flush=True)

    # ---- L2 key_facts (using L1 key_facts just built) ----
    print(f"  Building L2 key_facts ({len(l2_nodes)} nodes) ...", flush=True)
    for n in l2_nodes:
        kids = parent_child.get(n["node_id"], [])
        kf = build_key_facts(n["node_id"], kids, nodes, "L2")
        if kf:
            nodes[n["node_id"]]["key_facts"] = kf
    print(f"  L2 done", flush=True)

    # ---- write output ----
    print(f"  Writing {hier_out} ...", flush=True)
    with open(hier_out, "w") as out:
        for n in nodes.values():
            out.write(json.dumps(n, ensure_ascii=False) + "\n")

    # verify
    kf_counts = {}
    for n in nodes.values():
        lv = n.get("level", "?")
        kf_counts[lv] = kf_counts.get(lv, 0) + (1 if n.get("key_facts", "").strip() else 0)
    total = {}
    for n in nodes.values():
        lv = n.get("level", "?")
        total[lv] = total.get(lv, 0) + 1
    for lv in sorted(total):
        print(f"  {lv}: {kf_counts.get(lv,0)}/{total[lv]} have key_facts")

    print(f"[{tier}] Done → {hier_out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", nargs="+", default=["10M", "20M"])
    args = ap.parse_args()
    for t in args.tier:
        process_tier(t)
