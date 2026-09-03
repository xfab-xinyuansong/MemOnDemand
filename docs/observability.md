# Observability

MemOnDemand runs should be easy to audit after the fact. Every production or
benchmark run should keep enough metadata to explain which evidence was used,
how much context was sent to the model, and what changed between releases.

## Core Signals

| Signal | Why it matters |
| --- | --- |
| Answer-input tokens | Primary driver of context cost and latency. |
| Navigator tokens | Shows whether hierarchy routing is growing unexpectedly. |
| Promotion events | Indicates how often compact memory needs source-level detail. |
| Invalid citations | Tracks grounding quality and source precision. |
| Completeness | Detects over-compression or missing evidence. |
| Document recall | Measures whether expected source documents are reachable. |
| Resume state | Confirms long jobs can continue without duplicate work. |

## Run Directory

Use a stable run layout:

```text
results/
  runs/
    2026-07-23-prod-smoke/
      answers.jsonl
      eval/
      logs/
      token_ledger.jsonl
      command.txt
      metadata.json
```

`metadata.json` should include:

```json
{
  "git_commit": "REPLACE_WITH_COMMIT",
  "corpus_tier": "20M",
  "query_count": 500,
  "seed": 20260608,
  "method": "V5",
  "model_alias": "your-model-alias"
}
```

## Release Gates

For product releases, compare new runs against the previous accepted run:

- Combined score does not regress beyond the accepted threshold.
- Invalid citations do not increase unexpectedly.
- Answer-input tokens stay within the context budget.
- Promotion events remain explainable for the corpus size.
- Generated logs do not contain secrets or private text that should not be
  shared.

## Incident Triage

| Symptom | First checks |
| --- | --- |
| Token spike | Inspect answer-input tokens, promoted node count, and top-k settings. |
| Citation drift | Check gold labels, retrieved candidates, and promoted evidence. |
| Low completeness | Inspect whether distilled context lost required details. |
| Slow cold start | Verify hierarchy size, cache directory, and eager preprocessing. |
| Failed resume | Confirm output directory, answer ids, and partial JSONL files. |
