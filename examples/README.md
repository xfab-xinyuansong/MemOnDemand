# Examples

The examples directory contains a tiny local dataset generator used for smoke
tests and onboarding.

## Demo Dataset

```bash
python examples/create_demo_dataset.py
```

Generated files:

```text
examples/demo/manifests/l0_nodes.parquet
examples/demo/manifests/queries.parquet
examples/demo/manifests/gold.jsonl
```

The generated demo data is intentionally small and local. It is useful for CLI
checks, docs validation, and integration tests. Production deployments should
create manifests from their own document store and keep those manifests outside
Git.

## Smoke Commands

```bash
memondemand build-hierarchy \
  --tier demo \
  --l0_parquet examples/demo/manifests/l0_nodes.parquet \
  --out_dir examples/demo/results/hierarchy \
  --dry_run

memondemand doctor
memondemand run --help
memondemand evaluate --help
```
