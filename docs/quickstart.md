# Quickstart

This guide creates a tiny synthetic dataset, builds a dry-run hierarchy, and
checks the main CLI entry points.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## 2. Create Demo Data

```bash
python examples/create_demo_dataset.py
```

This writes:

```text
examples/demo/manifests/l0_nodes.parquet
examples/demo/manifests/queries.parquet
examples/demo/manifests/gold.jsonl
```

## 3. Build A Hierarchy

```bash
memondemand build-hierarchy \
  --tier demo \
  --l0_parquet examples/demo/manifests/l0_nodes.parquet \
  --out_dir examples/demo/results/hierarchy \
  --dry_run
```

## 4. Check The CLI

```bash
memondemand doctor
memondemand run --help
memondemand evaluate --help
```

For model-backed runs, fill `.env` from `.env.example` and point the runner at
your own manifests.
