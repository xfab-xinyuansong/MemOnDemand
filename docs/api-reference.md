# API Reference

This page documents the supported public Python surface. Lower-level modules are
available for advanced users, but the objects below are the stable integration
points for applications, tests, and orchestration code.

## Memory Node Schema

```python
from memondemand.methods import DualNode
```

`DualNode` represents one hierarchy node with both compact and detailed memory.

| Field | Purpose |
| --- | --- |
| `node_id` | Stable node identifier used by indexes and parent-child links. |
| `level` | Hierarchy level, for example L0 raw evidence or higher-level summaries. |
| `distilled_text` | Compact routing and answer-context text. |
| `detailed_text` | Source-level evidence used for grounded answering. |
| `children` | Child node ids in the hierarchy. |
| `metadata` | Optional operational metadata such as source type or title. |

## Node IO

```python
from memondemand.methods import read_nodes_jsonl, write_nodes_jsonl, validate_one
```

Use JSONL node files when integrating MemOnDemand with external storage or test
fixtures.

```python
nodes = list(read_nodes_jsonl("results/hierarchy/nodes.jsonl"))
validate_one(nodes[0])
write_nodes_jsonl("results/hierarchy/export.jsonl", nodes)
```

## Promotion Control

```python
from memondemand.methods import PromotionController
```

`PromotionController` manages which detailed evidence is allowed into the
answer context. Use it when a service needs a bounded promoted set rather than
dumping every retrieved source into the prompt.

```python
promotion = PromotionController(nodes_by_id, embedder=None, promotion_budget=20)
```

## Decay Control

```python
from memondemand.methods import DecayController
```

`DecayController` ages out stale promoted state so long-running sessions do not
carry irrelevant evidence forever.

```python
decay = DecayController(nodes_by_id, decay_window=15)
```

## Token Accounting

```python
from memondemand.methods import TokenLedger
```

`TokenLedger` records token use per run. Store it with answer artifacts so
latency, cost, and context budget can be audited later.

```python
ledger = TokenLedger(run_id="prod-2026-07-23", method="V5")
```

## Model Gateway

```python
from memondemand.core.api_adapter import call
```

The public gateway uses the general chat-completions endpoint configured by
`MEMONDEMAND_API_BASE_URL`, `MEMONDEMAND_API_KEY`, and `MEMONDEMAND_API_MODEL`:

```python
response = call(
    "general",
    [{"role": "user", "content": "Summarize the selected evidence."}],
    max_tokens=256,
    temperature=0.0,
)

text = response["text"]
usage = response["usage"]
```

The returned record includes text, token usage, latency, model, alias, and estimated
cost. Gateway errors raise `APIError`; exception text and logs are sanitized before
credentials can be emitted. The adapter retries transient failures with bounded
exponential backoff and does not retry non-transient authentication or request errors.

## CLI First

Use the CLI for production batch runs:

```bash
memondemand build-hierarchy --help
memondemand run --help
memondemand evaluate --help
```

Use the Python API for service integration, custom orchestration, tests, and
adapters around existing storage systems.
