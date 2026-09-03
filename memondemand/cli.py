"""Product command line interface for MemOnDemand."""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path
from typing import Iterable

from memondemand import __version__


_COMMAND_MODULES = {
    "build-hierarchy": "memondemand.data.build_hierarchy_dynamic",
    "run": "memondemand.runners.run_stream_v5",
    "run-v6": "memondemand.runners.run_exp_v6",
    "evaluate": "memondemand.eval.evaluate_v5",
    "eval-retrieval": "memondemand.eval.eval_retrieval_only",
    "eval-rouge": "memondemand.eval.eval_rouge",
    "api-smoke": "memondemand.core.api_adapter",
}


def _run_module(module: str, argv: list[str], prog: str) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = [prog, *argv]
        try:
            result = runpy.run_module(module, run_name="__main__")
        except ModuleNotFoundError as exc:
            missing = exc.name or "a runtime dependency"
            print(
                f"memondemand: missing dependency '{missing}'. "
                'Install the needed extras, for example: pip install -e ".[all]"',
                file=sys.stderr,
            )
            return 2
        returned = result.get("__return_code__")
        return int(returned) if returned is not None else 0
    finally:
        sys.argv = old_argv


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _doctor(_: argparse.Namespace) -> int:
    root = Path(os.environ.get("MEMONDEMAND_REPO_ROOT", Path.cwd())).resolve()
    env_path = Path(os.environ.get("MEMONDEMAND_ENV_FILE", root / ".env"))
    checks = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "memondemand": __version__,
        "project_root": str(root),
        "env_file": str(env_path) if env_path.exists() else "not found",
        "numpy": "ok" if _has_module("numpy") else "missing",
        "pandas": "ok" if _has_module("pandas") else "missing",
        "pyarrow": "ok" if _has_module("pyarrow") else "missing",
        "rank_bm25": "ok" if _has_module("rank_bm25") else "missing",
        "sentence_transformers": "ok" if _has_module("sentence_transformers") else "optional",
        "rouge_score": "ok" if _has_module("rouge_score") else "optional",
    }
    width = max(len(k) for k in checks)
    for key, value in checks.items():
        print(f"{key.ljust(width)}  {value}")
    return 0


def _add_passthrough(subparsers: argparse._SubParsersAction, name: str, help_text: str) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("module_args", nargs=argparse.REMAINDER)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memondemand",
        description="MemOnDemand command line tools for hierarchical memory retrieval.",
    )
    parser.add_argument("--version", action="version", version=f"memondemand {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="check the local MemOnDemand environment")
    doctor.set_defaults(func=_doctor)

    _add_passthrough(subparsers, "build-hierarchy", "build a hierarchy from L0 parquet data")
    _add_passthrough(subparsers, "run", "run the V5 retrieval pipeline")
    _add_passthrough(subparsers, "run-v6", "run the V6 dual-memory experiment wrapper")
    _add_passthrough(subparsers, "evaluate", "evaluate generated answers")
    _add_passthrough(subparsers, "eval-retrieval", "run retrieval-only evaluation")
    _add_passthrough(subparsers, "eval-rouge", "run ROUGE evaluation")
    _add_passthrough(subparsers, "api-smoke", "exercise the configured API adapter")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv and raw_argv[0] in _COMMAND_MODULES:
        command = raw_argv[0]
        return _run_module(
            _COMMAND_MODULES[command],
            raw_argv[1:],
            prog=f"memondemand {command}",
        )
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if not args.command:
        parser.print_help()
        return 0
    if hasattr(args, "func"):
        return int(args.func(args))
    module = _COMMAND_MODULES[args.command]
    return _run_module(module, getattr(args, "module_args", []), prog=f"memondemand {args.command}")
