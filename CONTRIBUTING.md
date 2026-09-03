# Contributing

Thanks for improving MemOnDemand. Keep changes product-focused, documented, and
easy to test locally.

## Local Checks

```bash
pip install -e ".[dev]"
python -m compileall -q memondemand
pytest
memondemand doctor
python -m build
```

You can also use the convenience targets:

```bash
make install-dev
make test
make build
```

## Pull Request Checklist

- Update README or docs when behavior changes.
- Add focused tests for new public behavior.
- Do not commit `.env`, generated outputs, parquet manifests, caches, or logs.
- Keep provider credentials in environment variables only.
- Run package build checks before cutting a release.

## Documentation Map

- API settings: the short `API` section in `README.md`
- Deployment patterns: `docs/integration.md`
- Operational checks: `docs/observability.md` and `docs/production.md`
- Release process: `docs/release.md`
