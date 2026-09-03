# Data Contract

MemOnDemand expects local parquet and JSONL files. The package does not ship private
enterprise data.

## L0 Evidence

Parquet file with one row per raw evidence item.

| Field | Type | Description |
| --- | --- | --- |
| `doc_id` | string | Stable source evidence id. |
| `source_type` | string | Tenant, corpus, or artifact type used for grouping. |
| `title` | string | Short display title. |
| `content` | string | Compact source summary or excerpt. |
| `text` | string | Detailed source text used for grounded answering. |

## Queries

Parquet file with one row per question.

| Field | Type | Description |
| --- | --- | --- |
| `query_id` | string | Stable query id. |
| `query_text` | string | User-facing question. |

## Gold Labels

JSONL file used by evaluation.

| Field | Type | Description |
| --- | --- | --- |
| `query_id` or `question_id` | string | Query id. |
| `expected_doc_ids` | list[string] | Source ids expected in the answer evidence. |
| `gold_answer` | string | Reference answer text. |

## Local Layout

```text
manifests/
  l0_nodes.parquet
  queries.parquet
  gold.jsonl
results/
  hierarchy/
  runs/
```
