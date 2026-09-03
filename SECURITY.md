# Security

MemOnDemand is designed to run against private local data. Treat configuration,
manifests, answers, logs, and caches as sensitive unless your deployment policy
says otherwise.

## Reporting

Open a private security advisory or contact the repository owner if you find a
vulnerability that could expose credentials, private documents, or generated
answer logs.

## Data Handling

- Store provider keys in `.env` or a secrets manager.
- Keep private corpora and generated outputs outside Git.
- Review logs before sharing run artifacts.
- Avoid committing cached embeddings or retrieval indexes.

## Operational Checks

- Run the README secret scan before pushing public commits.
- Keep `results/`, `logs/`, caches, generated JSONL, and parquet manifests out
  of commits.
- See `docs/production.md` for release safety checks and
  `docs/observability.md` for run-audit guidance.
