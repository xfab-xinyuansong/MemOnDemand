from __future__ import annotations

from memondemand.methods import DualNode, validate_one


def test_dual_node_round_trip() -> None:
    node = DualNode(
        node_id="doc-1",
        level="L0",
        tenant_id="demo",
        distilled_text="Short routing memory.",
        detailed_text="Detailed evidence used for grounded answering.",
        distilled_tokens=3,
        detailed_tokens=7,
        source_evidence_ids=["doc-1"],
    )

    restored = DualNode.from_dict(node.to_dict())

    assert restored.node_id == "doc-1"
    assert restored.level == "L0"
    assert restored.source_evidence_ids == ["doc-1"]
    assert validate_one(restored) == []
