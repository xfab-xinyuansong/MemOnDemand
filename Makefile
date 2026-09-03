.PHONY: install install-dev doctor test build clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

doctor:
	memondemand doctor

test:
	python -m compileall -q memondemand
	pytest

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
