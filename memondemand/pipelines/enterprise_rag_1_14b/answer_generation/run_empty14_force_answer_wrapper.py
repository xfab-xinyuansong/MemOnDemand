#!/usr/bin/env python3
"""Force-answer wrapper for a manifest of previously unanswered queries.

Pass the rerun manifest through the normal ``--queries`` argument. The wrapper
enables force-answer mode, applies the same detailed-memory truncation used by
the full run, and optionally loads the checked-in answer prompt.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MEMONDEMAND_FORCE_ANSWER", "1")

from memondemand.pipelines.enterprise_rag_1_14b.answer_generation import (  # noqa: E402
    run_stream_v5_force_answer_patched as R,
)

_TRUNCATE_CHARS = int(os.environ.get("MEMONDEMAND_TRUNCATE_CHARS", "2000"))
_original_node_text = R._node_text_for_answer


def _truncated_node_text(n, is_top_detail: bool = False) -> str:
    mode = os.environ.get("MEMONDEMAND_ANSWER_MODE", "detailed_truncated")
    if mode == "detailed_truncated":
        detailed = n.detailed_text or ""
        distilled = n.distilled_text or ""
        return (detailed or distilled)[:_TRUNCATE_CHARS]
    return _original_node_text(n, is_top_detail)


R._node_text_for_answer = _truncated_node_text

_prompt_path = os.environ.get("MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE", "")
if _prompt_path:
    path = Path(_prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Answer prompt not found: {path}")
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("memondemand_answer_prompt", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load answer prompt module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        R.ANSWER_SYSTEM = module.ANSWER_SYSTEM
    else:
        R.ANSWER_SYSTEM = path.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(R.main())
