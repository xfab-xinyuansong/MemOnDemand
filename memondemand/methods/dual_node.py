from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


NODE_STATE_LIGHT = "LIGHT"
NODE_STATE_PROMOTED = "PROMOTED"
VALID_NODE_STATES = {NODE_STATE_LIGHT, NODE_STATE_PROMOTED}


@dataclass
class DualNode:

    node_id: str
    level: str
    tenant_id: str = ""

    distilled_text: str = ""
    detailed_text: str = ""
    detail_ref: str = ""
    distilled_tokens: int = 0
    detailed_tokens: int = 0

    source_evidence_ids: List[str] = field(default_factory=list)

    state: str = NODE_STATE_LIGHT
    promotion_score: float = 0.0
    last_promoted_query_idx: int = -1
    last_used_query_idx: int = -1
    promotion_count: int = 0
    detail_use_count: int = 0

    distilled_text_model_alias: str = ""
    distilled_text_model_status: str = ""

    key_facts: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DualNode":
        return cls(
            node_id=d["node_id"],
            level=d.get("level", "L0"),
            tenant_id=d.get("tenant_id", ""),
            distilled_text=d.get("distilled_text", ""),
            detailed_text=d.get("detailed_text", ""),
            detail_ref=d.get("detail_ref", ""),
            distilled_tokens=int(d.get("distilled_tokens", 0)),
            detailed_tokens=int(d.get("detailed_tokens", 0)),
            source_evidence_ids=list(d.get("source_evidence_ids", []) or []),
            state=d.get("state", NODE_STATE_LIGHT),
            promotion_score=float(d.get("promotion_score", 0.0)),
            last_promoted_query_idx=int(d.get("last_promoted_query_idx", -1)),
            last_used_query_idx=int(d.get("last_used_query_idx", -1)),
            promotion_count=int(d.get("promotion_count", 0)),
            detail_use_count=int(d.get("detail_use_count", 0)),
            distilled_text_model_alias=d.get("distilled_text_model_alias", ""),
            distilled_text_model_status=d.get("distilled_text_model_status", ""),
            key_facts=d.get("key_facts", ""),
            extra=dict(d.get("extra", {}) or {}),
        )


class DualNodeError(ValueError):
    pass


def validate_one(node: DualNode) -> List[str]:
    errs: List[str] = []
    if not node.node_id:
        errs.append("node_id is empty")
    if node.state not in VALID_NODE_STATES:
        errs.append(f"state={node.state!r} not in {VALID_NODE_STATES}")

    if not node.distilled_text:
        errs.append("distilled_text is empty")
    if not node.detailed_text and not node.detail_ref:
        errs.append("both detailed_text and detail_ref are empty")
    if node.distilled_tokens <= 0:
        errs.append(f"distilled_tokens must be > 0, got {node.distilled_tokens}")
    if node.detailed_tokens <= 0 and not node.detail_ref:
        errs.append(f"detailed_tokens must be > 0 (or detail_ref must be set), got {node.detailed_tokens}")

    if not node.source_evidence_ids:
        errs.append("source_evidence_ids is empty (provenance violation)")

    return errs


def validate_batch(nodes: List[DualNode]) -> Dict[str, Any]:
    per_node_errors: Dict[str, List[str]] = {}
    total = len(nodes)
    n_have_both_repr = 0
    n_have_token_counts = 0
    n_distilled_lt_detailed = 0
    n_have_provenance = 0
    invalid_nodes: List[str] = []

    for n in nodes:
        errs = validate_one(n)
        if errs:
            per_node_errors[n.node_id] = errs
            invalid_nodes.append(n.node_id)
        has_distilled = bool(n.distilled_text) and n.distilled_tokens > 0
        has_detailed = (bool(n.detailed_text) or bool(n.detail_ref)) and (
            n.detailed_tokens > 0 or bool(n.detail_ref)
        )
        if has_distilled and has_detailed:
            n_have_both_repr += 1
        if n.distilled_tokens > 0 and n.detailed_tokens > 0:
            n_have_token_counts += 1
        if n.distilled_tokens > 0 and n.detailed_tokens > 0 and n.distilled_tokens < n.detailed_tokens:
            n_distilled_lt_detailed += 1
        if n.source_evidence_ids:
            n_have_provenance += 1

    pct_both_repr = (n_have_both_repr / total) if total else 0.0
    pct_token_counts = (n_have_token_counts / total) if total else 0.0
    pct_distilled_lt_detailed = (n_distilled_lt_detailed / total) if total else 0.0
    pct_provenance = (n_have_provenance / total) if total else 0.0

    acceptance = {
        "criterion_1_100pct_dual_repr_and_tokens": {
            "value": pct_both_repr,
            "threshold": 1.0,
            "pass": pct_both_repr >= 1.0 and pct_token_counts >= 1.0,
        },
        "criterion_2_95pct_distilled_lt_detailed": {
            "value": pct_distilled_lt_detailed,
            "threshold": 0.95,
            "pass": pct_distilled_lt_detailed >= 0.95,
        },
        "criterion_3_100pct_provenance": {
            "value": pct_provenance,
            "threshold": 1.0,
            "pass": pct_provenance >= 1.0,
        },
    }
    overall_pass = all(c["pass"] for c in acceptance.values())

    return {
        "total_nodes": total,
        "invalid_node_count": len(invalid_nodes),
        "invalid_node_sample": invalid_nodes[:20],
        "per_node_errors_sample": dict(list(per_node_errors.items())[:10]),
        "criteria": acceptance,
        "overall_pass": overall_pass,
        "counts": {
            "n_have_both_repr": n_have_both_repr,
            "n_have_token_counts": n_have_token_counts,
            "n_distilled_lt_detailed": n_distilled_lt_detailed,
            "n_have_provenance": n_have_provenance,
        },
    }


def write_nodes_jsonl(nodes: List[DualNode], path: str) -> None:
    with open(path, "w") as f:
        for n in nodes:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False) + "\n")


def read_nodes_jsonl(path: str) -> List[DualNode]:
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(DualNode.from_dict(json.loads(ln)))
    return out


def _self_test() -> int:
    cases = []
    n1 = DualNode(
        node_id="m0_t1_aaa",
        level="L0",
        tenant_id="t1",
        distilled_text="Short summary.",
        detailed_text="A longer, more detailed body of text describing the full evidence.",
        distilled_tokens=3,
        detailed_tokens=12,
        source_evidence_ids=["ev_01"],
        state=NODE_STATE_LIGHT,
        distilled_text_model_alias="gpt_5_4_mini",
        distilled_text_model_status="ACTIVE",
    )
    cases.append(("valid L0", n1, True))
    n2 = DualNode(node_id="m0_t1_bbb", level="L0", distilled_text="", detailed_text="x",
                  distilled_tokens=0, detailed_tokens=2, source_evidence_ids=["ev_02"])
    cases.append(("missing distilled", n2, False))
    n3 = DualNode(node_id="m0_t1_ccc", level="L0",
                  distilled_text="long " * 30, detailed_text="short",
                  distilled_tokens=30, detailed_tokens=2,
                  source_evidence_ids=["ev_03"])
    cases.append(("distilled longer (single-node valid but batch criterion 2 may fail)", n3, True))
    n4 = DualNode(node_id="m0_t1_ddd", level="L0", distilled_text="x", detailed_text="xxxx",
                  distilled_tokens=1, detailed_tokens=4)
    cases.append(("missing provenance", n4, False))

    failures = 0
    for label, n, want_valid in cases:
        errs = validate_one(n)
        ok_single = (not errs) == want_valid
        print(f"  [{'PASS' if ok_single else 'FAIL'}] single-node: {label} — errors={errs}")
        if not ok_single:
            failures += 1

    batch = [n1, n3]
    report = validate_batch(batch)
    print(f"\nbatch report on (valid, distilled>detailed): {json.dumps(report['criteria'], indent=2)}")
    if not (
        report["criteria"]["criterion_1_100pct_dual_repr_and_tokens"]["pass"]
        and not report["criteria"]["criterion_2_95pct_distilled_lt_detailed"]["pass"]
    ):
        print("  [FAIL] expected criterion 1 PASS + criterion 2 FAIL")
        failures += 1
    else:
        print("  [PASS] batch criteria flags as expected")

    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
