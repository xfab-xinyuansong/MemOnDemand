# Integration Guide

MemOnDemand is designed to sit inside an existing enterprise RAG stack. It does not
own your document store, model gateway, job scheduler, or evaluation dashboard.
Instead, it provides a package boundary around hierarchy construction, query
navigation, promotion, answer-context assembly, and evaluation.

## Deployment Modes

| Mode | Best for | Integration pattern |
| --- | --- | --- |
| Local CLI | Development, smoke tests, offline evaluation | Run `memondemand` commands against local manifests. |
| Batch job | Nightly hierarchy builds and benchmark runs | Call the CLI from a scheduler and persist `results/`. |
| Service worker | Product RAG requests with controlled context | Wrap the Python API and run retrieval inside a worker. |
| Evaluation pipeline | Regression testing and release gates | Run `memondemand evaluate` and publish metrics to your dashboard. |

## Model Gateway

MemOnDemand uses provider-neutral environment variables for OpenAI-compatible chat
and embedding endpoints:

```bash
export MEMONDEMAND_API_BASE_URL=https://your-gateway.example/v1
export MEMONDEMAND_API_KEY=...
export MEMONDEMAND_API_MODEL=...
export MEMONDEMAND_EMBED_API_BASE_URL=https://your-gateway.example/v1
export MEMONDEMAND_EMBED_API_KEY=...
export MEMONDEMAND_EMBED_API_MODEL=...
```

Keep gateway-specific retry, tenancy, and audit logic at your gateway layer
when possible. Keep MemOnDemand configuration limited to endpoint, key, model, and
runtime behavior.

## Storage Boundary

MemOnDemand expects local parquet and JSONL inputs at runtime. Production systems
usually add a thin adapter before invoking the package:

1. Export evidence and queries from the source system.
2. Write manifests to a run-scoped local directory.
3. Run `memondemand build-hierarchy` or `memondemand run`.
4. Upload answer files, ledgers, and reports to your artifact store.

## Orchestration Boundary

Use `--resume` for long jobs and keep each run in a unique output directory.
Store these values with the run:

- Git commit
- Corpus tier
- Query count
- Seed
- Model aliases
- Environment template version
- Command line

## Evaluation Boundary

MemOnDemand writes local evaluation outputs. Most teams connect those outputs to a
dashboard or release gate that tracks:

- Combined score
- Correctness
- Completeness
- Document recall
- Invalid citations
- Answer-input tokens per query
- Promotion events per query
