#!/usr/bin/env python3
"""Merge reusable L0 key facts into a full-corpus hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_facts", required=True, help="Incremental L0 key-facts JSONL")
    parser.add_argument("--hierarchy", required=True, help="Full hierarchy JSONL")
    parser.add_argument("--out", required=True, help="Merged hierarchy JSONL")
    parser.add_argument(
        "--missing_ids",
        required=True,
        help="JSON output containing L0 node IDs that still need backfilling",
    )
    args = parser.parse_args()

    key_facts = {}
    with Path(args.key_facts).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("key_facts"):
                key_facts[record["node_id"]] = record["key_facts"]

    nodes = []
    l0_total = matched = 0
    with Path(args.hierarchy).open(encoding="utf-8") as handle:
        for line in handle:
            node = json.loads(line)
            if node.get("level") == "L0":
                l0_total += 1
                facts = key_facts.get(node["node_id"])
                if facts:
                    node["key_facts"] = facts
                    node["key_facts_model"] = "gpt-5.4"
                    matched += 1
            nodes.append(node)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for node in nodes:
            handle.write(json.dumps(node, ensure_ascii=False) + "\n")

    missing = [
        node["node_id"]
        for node in nodes
        if node.get("level") == "L0" and not node.get("key_facts")
    ]
    missing_path = Path(args.missing_ids)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.write_text(json.dumps(missing), encoding="utf-8")

    print(
        f"L0 total={l0_total} reused={matched} missing={len(missing)}; "
        f"merged hierarchy written to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
