"""Create a tiny local MemOnDemand dataset for quickstart commands.

The generated files are intentionally small and synthetic. They let users test
the CLI wiring before pointing MemOnDemand at private enterprise data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "demo"
MANIFESTS = ROOT / "manifests"


DOCS = [
    (
        "doc-001",
        "handbook",
        "Security onboarding",
        "Security onboarding requires hardware-key enrollment before production access.",
        "The security handbook says every new engineer must enroll a hardware key before production access is granted.",
    ),
    (
        "doc-002",
        "handbook",
        "Incident response",
        "Critical incidents require an incident commander and a customer-impact note.",
        "The incident response guide assigns an incident commander and requires a customer-impact note for critical events.",
    ),
    (
        "doc-003",
        "policy",
        "Data retention",
        "Billing exports are retained for seven years and access is logged.",
        "The data-retention policy states that billing exports are kept for seven years and every access is logged.",
    ),
    (
        "doc-004",
        "policy",
        "PII handling",
        "Customer identifiers must be redacted before sharing debugging traces.",
        "The PII handling policy requires customer identifiers to be redacted before debugging traces are shared outside the owning team.",
    ),
    (
        "doc-005",
        "runbook",
        "Search service rollback",
        "Search service rollback starts with traffic draining and index snapshot validation.",
        "The search rollback runbook starts by draining traffic, validating the last healthy index snapshot, and notifying support.",
    ),
    (
        "doc-006",
        "runbook",
        "Embedding cache rebuild",
        "Embedding cache rebuilds should run in batches and preserve the old cache until validation passes.",
        "The embedding cache runbook recommends batched rebuilds and keeping the old cache available until validation succeeds.",
    ),
    (
        "doc-007",
        "ticket",
        "Latency regression",
        "A retrieval latency regression was traced to dense reranking on too many candidates.",
        "The latency ticket found that dense reranking over a wide candidate set caused p95 retrieval latency to rise.",
    ),
    (
        "doc-008",
        "ticket",
        "Citation mismatch",
        "Citation mismatch errors came from stale document ids after a manifest refresh.",
        "The citation mismatch ticket links the error to stale document ids produced by a manifest refresh.",
    ),
    (
        "doc-009",
        "design",
        "Hierarchy levels",
        "L0 stores raw evidence, L1 groups by team, and L2 groups by organization.",
        "The hierarchy design uses L0 for raw evidence, L1 for team-level groups, and L2 for organization-level groups.",
    ),
    (
        "doc-010",
        "design",
        "Promotion budget",
        "Promotion should prefer high-overlap nodes and cap detailed context per query.",
        "The promotion design prefers nodes with high query overlap and caps detailed context for each query.",
    ),
    (
        "doc-011",
        "memo",
        "Quarterly audit",
        "The quarterly audit sampled access logs and found no unapproved export activity.",
        "The quarterly audit memo sampled access logs and reported no unapproved export activity.",
    ),
    (
        "doc-012",
        "memo",
        "Support readiness",
        "Support readiness requires an escalation owner and a verified rollback plan.",
        "The support readiness memo requires each launch to name an escalation owner and verify the rollback plan.",
    ),
]


QUERIES = [
    ("q-001", "What has to happen before a new engineer gets production access?"),
    ("q-002", "Why did retrieval latency increase?"),
    ("q-003", "How long are billing exports retained?"),
    ("q-004", "What should a launch have for support readiness?"),
    ("q-005", "How are stale document ids connected to citation errors?"),
]


GOLD = [
    {
        "query_id": "q-001",
        "expected_doc_ids": ["doc-001"],
        "gold_answer": "New engineers must enroll a hardware key before production access is granted.",
    },
    {
        "query_id": "q-002",
        "expected_doc_ids": ["doc-007"],
        "gold_answer": "Latency increased because dense reranking ran over too many candidates.",
    },
    {
        "query_id": "q-003",
        "expected_doc_ids": ["doc-003"],
        "gold_answer": "Billing exports are retained for seven years.",
    },
    {
        "query_id": "q-004",
        "expected_doc_ids": ["doc-012"],
        "gold_answer": "A launch needs an escalation owner and a verified rollback plan.",
    },
    {
        "query_id": "q-005",
        "expected_doc_ids": ["doc-008"],
        "gold_answer": "Citation errors came from stale document ids after a manifest refresh.",
    },
]


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "doc_id": doc_id,
                "source_type": source_type,
                "title": title,
                "content": content,
                "text": text,
            }
            for doc_id, source_type, title, content, text in DOCS
        ]
    ).to_parquet(MANIFESTS / "l0_nodes.parquet", index=False)
    pd.DataFrame(
        [{"query_id": query_id, "query_text": query_text} for query_id, query_text in QUERIES]
    ).to_parquet(MANIFESTS / "queries.parquet", index=False)
    with (MANIFESTS / "gold.jsonl").open("w", encoding="utf-8") as f:
        for row in GOLD:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote demo manifests to {MANIFESTS}")


if __name__ == "__main__":
    main()
