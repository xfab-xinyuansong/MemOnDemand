# Production Checklist

Use this checklist before running MemOnDemand in a shared or enterprise
environment.

## Packaging

- Install from a pinned git commit or release tag.
- Use Python 3.10 or newer.
- Install only the extras your deployment needs: core, local embeddings, model
  APIs, evaluation, or index backends.
- Run `memondemand doctor` in the deployment environment.

## Data

- Keep parquet manifests, generated answers, JSONL logs, embedding caches, and
  retrieval indexes outside the source tree.
- Store run outputs under a versioned `results/` directory.
- Record corpus tier, query count, seed, model aliases, and git commit for each
  run.
- Review generated logs before sharing artifacts.

## Secrets

- Load credentials from `.env`, environment variables, or a secrets manager.
- Never commit provider keys, bearer tokens, generated answers, or customer
  corpora.
- Use separate model credentials for development, evaluation, and production
  environments when possible.

## Operations

- Use `--resume` for long-running jobs.
- Keep token ledgers and promotion logs with each run directory.
- Run answer evaluation and retrieval-only evaluation before publishing result
  tables.
- Monitor invalid citations, answer completeness, promoted node counts, and
  answer-input tokens per query.

## Release Readiness

- `python -m compileall -q memondemand`
- `pytest`
- `memondemand doctor`
- `python -m build`
- Secret scan with the README commands.
- Confirm that no private manifests, results, caches, or logs are staged.
