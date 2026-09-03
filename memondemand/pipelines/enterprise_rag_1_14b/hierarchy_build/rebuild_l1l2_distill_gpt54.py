#!/usr/bin/env python3
"""Rebuild L1/L2 navigation summaries with the high-level model.

Does NOT touch key_facts (separate field, separate pipeline).
Does NOT re-cluster or change node structure -- reads existing hierarchy,
regenerates distilled_text for L1/L2 only, writes to a NEW output file
(does not overwrite input).

Usage:
    python3 rebuild_l1l2_distill_gpt54.py --tier 60M --hier_path <path>
"""
import argparse, json, time, pathlib

try:
    from memondemand.core import dns_patch  # noqa: F401
except ImportError:
    dns_patch = None

from memondemand.pipelines.enterprise_rag_1_14b.runtime import call as call_llm

# Prompts used by the hierarchy builder.
L1_DISTILL_SYSTEM = """You are an enterprise memory aggregator. Given a small cluster of related L0 memory snippets from one tenant, produce a SHORT distilled summary that:

- captures the central topic, entities, and time references shared by the cluster
- preserves enough signal that a retrieval system can route relevant queries here
- is strictly SHORTER than the input bundle (target: 20-50 words)
- contains NO speculation or invented content
- contains NO gold answer, ground truth, evidence_link kinds, or expected_doc_ids tokens

Return ONLY the summary sentence(s); no prefix, no JSON, no quotation marks."""

L1_DISTILL_USER = """Cluster snippets (tenant={tenant}, count={n}):
---
{body}
---

Distilled summary:"""

L2_DISTILL_SYSTEM = """You are an enterprise memory abstractor. Given the L1-level distilled summaries that belong to one tenant, produce a SHORT high-level abstract of the tenant's overall memory state.

- 30-60 words
- captures dominant themes, key entities, time spans
- omit any single specific incident unless it is central
- NO speculation, NO gold-answer tokens, NO JSON

Return ONLY the summary; no prefix, no quotation marks."""

L2_DISTILL_USER = """Tenant: {tenant}
L1 distilled summaries (count={n}):
---
{body}
---

Tenant-level abstract:"""

MAX_ATTEMPTS = 50

def is_rate_limit(exc):
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "quota" in s

def bundle_children_distilled(children_texts, max_chars=4000):
    out, used = [], 0
    for i, txt in enumerate(children_texts, 1):
        snip = f"{i}. {txt}"
        if used + len(snip) > max_chars:
            out.append(f"... ({len(children_texts) - i + 1} more snippets truncated)")
            break
        out.append(snip)
        used += len(snip)
    return "\n".join(out)

def call_distill(system, user, alias="gpt_5_4", max_tokens=150):
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = call_llm(alias=alias, messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.0)
            text = (resp.get("text") or "").strip()
            return text or None
        except Exception as e:
            if is_rate_limit(e):
                sleep_s = 20.0 + attempt * 3
                print(f"    [429] attempt {attempt+1}, sleeping {sleep_s:.0f}s", flush=True)
                time.sleep(sleep_s)
                continue
            print(f"  [WARN] {e}", flush=True)
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--hier_path", required=True,
                     help="path to hierarchy file to read (JSONL, L0/L1/L2)")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    hier_in = args.hier_path
    hier_out = str(pathlib.Path(hier_in).with_name(
        pathlib.Path(hier_in).stem + "_l1l2distillgpt54.json"))

    print(f"=== [{args.tier}] Loading hierarchy from {hier_in} ===", flush=True)
    nodes = {}
    with open(hier_in) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n = json.loads(line)
            nodes[n["node_id"]] = n

    parent_to_children = {}
    for nid, n in nodes.items():
        extra = n.get("extra") or {}
        kids = n.get("children_ids") or extra.get("child_node_ids") or extra.get("children_ids")
        if kids:
            parent_to_children[nid] = list(kids)

    l1_nodes = [n for n in nodes.values() if n.get("level") == "L1"]
    l2_nodes = [n for n in nodes.values() if n.get("level") == "L2"]
    print(f"  L1={len(l1_nodes)} L2={len(l2_nodes)}", flush=True)

    print(f"  Rebuilding L1 distilled_text with gpt-5.4 (SERIAL, sleep={args.sleep}s) ...", flush=True)
    done = 0
    for n in l1_nodes:
        kids = parent_to_children.get(n["node_id"], [])
        tenant = n.get("tenant_id", "unknown")
        child_texts = []
        for cid in kids[:60]:
            c = nodes.get(cid, {})
            txt = (c.get("distilled_text") or "").strip()
            if txt:
                child_texts.append(txt[:400])
        if not child_texts:
            done += 1
            continue
        body = bundle_children_distilled(child_texts)
        user = L1_DISTILL_USER.format(tenant=tenant, n=len(kids), body=body)
        new_text = call_distill(L1_DISTILL_SYSTEM, user, alias="gpt_5_4", max_tokens=150)
        if new_text:
            nodes[n["node_id"]]["distilled_text"] = new_text
            nodes[n["node_id"]]["distilled_text_model_alias"] = "gpt_5_4"
        done += 1
        if done % 30 == 0:
            print(f"    [{done}/{len(l1_nodes)}]", flush=True)
        time.sleep(args.sleep)
    print(f"  L1 done: {done}/{len(l1_nodes)}", flush=True)

    print(f"  Rebuilding L2 distilled_text with gpt-5.4 (SERIAL, sleep={args.sleep}s) ...", flush=True)
    for n in l2_nodes:
        kids = parent_to_children.get(n["node_id"], [])
        tenant = n.get("tenant_id", "unknown")
        child_texts = []
        for cid in kids:
            c = nodes.get(cid, {})
            # L2's children may be L1_xxx string ids OR distilled just-updated L1 nodes
            txt = (c.get("distilled_text") or "").strip()
            if txt:
                child_texts.append(txt[:500])
        if not child_texts:
            continue
        body = bundle_children_distilled(child_texts)
        user = L2_DISTILL_USER.format(tenant=tenant, n=len(kids), body=body)
        new_text = call_distill(L2_DISTILL_SYSTEM, user, alias="gpt_5_4", max_tokens=200)
        if new_text:
            nodes[n["node_id"]]["distilled_text"] = new_text
            nodes[n["node_id"]]["distilled_text_model_alias"] = "gpt_5_4"
        time.sleep(args.sleep)
    print(f"  L2 done", flush=True)

    print(f"  Writing {hier_out} ...", flush=True)
    with open(hier_out, "w") as out:
        for n in nodes.values():
            out.write(json.dumps(n, ensure_ascii=False) + "\n")

    # verify
    from collections import Counter
    counters = {"L0": Counter(), "L1": Counter(), "L2": Counter()}
    for n in nodes.values():
        lv = n.get("level", "?")
        if lv in counters:
            counters[lv][n.get("distilled_text_model_alias", "NONE")] += 1
    for lv in ("L0", "L1", "L2"):
        print(f"  {lv}: {dict(counters[lv])}")
    print(f"[{args.tier}] Done -> {hier_out}", flush=True)


if __name__ == "__main__":
    main()
