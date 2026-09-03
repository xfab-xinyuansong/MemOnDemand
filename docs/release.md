# Release Process

Use this process when publishing a package, tagging a benchmark-ready commit, or
handing a MemOnDemand build to another team.

## Versioning

MemOnDemand follows semantic versioning:

- Patch: documentation, packaging, tests, or compatible fixes.
- Minor: new CLI flags, optional dependencies, or compatible APIs.
- Major: breaking changes to manifests, CLI behavior, or public Python APIs.

Update the version in `pyproject.toml` and `memondemand/__init__.py` together.

## Pre-release Checks

```bash
python -m compileall -q memondemand
pytest
memondemand doctor
python -m build
python -m twine check dist/*
```

Run the secret scan from the README before pushing public release commits.

## Release Notes

Each release note should include:

- User-visible changes
- Migration steps
- New or changed environment variables
- Evaluation or benchmark changes
- Known limitations

## Artifact Policy

Do not attach private corpora, generated answers, parquet manifests, API logs,
or caches to public releases. Publish only package artifacts, documentation,
small fixtures, and reproducible command lines.
