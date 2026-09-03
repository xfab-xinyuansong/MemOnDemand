#!/usr/bin/env python3
"""Run the 1.14B pipeline with configurable detailed-memory truncation.

``MEMONDEMAND_TRUNCATE_CHARS`` controls the per-node detailed-text budget.
``MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE`` optionally loads a replacement
answer prompt without changing the underlying runner.
"""
import os
import sys
import importlib.util
from pathlib import Path

from memondemand.pipelines.enterprise_rag_1_14b.answer_generation import run_stream_v5 as R

_TRUNCATE_CHARS = int(os.environ.get("MEMONDEMAND_TRUNCATE_CHARS", "2000"))

_orig_node_text = R._node_text_for_answer


def _patched_node_text_for_answer(n, is_top_detail: bool = False) -> str:
    mode = os.environ.get("MEMONDEMAND_ANSWER_MODE", "detailed_truncated")
    if mode == "detailed_truncated":
        det = n.detailed_text or ""
        dis = n.distilled_text or ""
        return (det or dis)[:_TRUNCATE_CHARS]
    return _orig_node_text(n, is_top_detail)


R._node_text_for_answer = _patched_node_text_for_answer
print(f"[patch] MEMONDEMAND_TRUNCATE_CHARS={_TRUNCATE_CHARS} applied to _node_text_for_answer", flush=True)

# Optional: improved answer system prompt override
_override_prompt_path = os.environ.get("MEMONDEMAND_ANSWER_SYSTEM_OVERRIDE_FILE", "")
if _override_prompt_path and os.path.exists(_override_prompt_path):
    prompt_path = Path(_override_prompt_path)
    if prompt_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("memondemand_answer_prompt", prompt_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load prompt module: {prompt_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _new_prompt = module.ANSWER_SYSTEM
    else:
        _new_prompt = prompt_path.read_text(encoding="utf-8")
    R.ANSWER_SYSTEM = _new_prompt
    print(f"[patch] ANSWER_SYSTEM overridden from {_override_prompt_path} ({len(_new_prompt)} chars)", flush=True)

if __name__ == "__main__":
    sys.exit(R.main())
