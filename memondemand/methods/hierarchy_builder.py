"""Hierarchy builder."""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import tiktoken

from memondemand.core.api_adapter import get_alias_config  # noqa: E402
from memondemand.methods.dual_node import DualNode, NODE_STATE_LIGHT  # noqa: E402
from memondemand.methods.token_ledger import (  # noqa: E402
    PHASE_DISTILLED_GEN,
    PHASE_HIERARCHY_BUILD,
    TokenLedger,
)

logger = logging.getLogger(__name__)


DISTILL_SYSTEM_PROMPT = """You are an enterprise memory summarizer. Given a long-form L0 memory record from a corporate dataset, produce a SHORT distilled summary that:

- captures the central topic, entities, time references, and any explicit decisions or values stated
- preserves enough downstream signal that a retrieval system can route relevant queries to the correct node
- is strictly SHORTER than the input (target: 15-40 words)
- contains NO speculation or invented content
- contains NO gold answer fields, ground truth, evidence_link kinds, or expected_doc_ids tokens

Return ONLY the summary sentence(s); no prefix, no JSON, no quotation marks."""

DISTILL_USER_TEMPLATE = """L0 record body:
---
{body}
---

Distilled summary:"""


# Module-level Azure client cache
_AZURE_CLIENT = None
_AZURE_LOCK_KEY = ""
_ENC = None


def _get_enc():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _ensure_azure_client():
    """Return a shared Azure client with a bounded HTTP connection pool."""
    global _AZURE_CLIENT
    if _AZURE_CLIENT is not None:
        return _AZURE_CLIENT
    import httpx
    import os
    from openai import AzureOpenAI
    spec = get_alias_config("gpt_5_4_mini")
    api_key = _read_azure_key()
    if not api_key:
        raise RuntimeError("AZURE_LLM_API_KEY or AZURE_OPENAI_KEY is required")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required")
    http_client = httpx.Client(
        timeout=httpx.Timeout(120.0, connect=20.0, read=120.0, write=30.0),
        limits=httpx.Limits(
            max_connections=256, max_keepalive_connections=128, keepalive_expiry=300.0,
        ),
    )
    _AZURE_CLIENT = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=spec.api_version,
        http_client=http_client,
        max_retries=3,
    )
    return _AZURE_CLIENT


def _read_azure_key() -> str:
    import os
    return os.environ.get("AZURE_LLM_API_KEY") or os.environ.get("AZURE_OPENAI_KEY") or ""


def llm_distill_one(body: str, max_retries: int = 4) -> Dict[str, Any]:
    """Call gpt_5_4_mini for one distilled summary.

    Returns dict with:
        text, input_tokens, output_tokens, wall_seconds, success, error
    """
    spec = get_alias_config("gpt_5_4_mini")
    client = _ensure_azure_client()
    enc = _get_enc()
    user_prompt = DISTILL_USER_TEMPLATE.format(body=body[:6000])
    input_tokens_est = len(enc.encode(DISTILL_SYSTEM_PROMPT)) + len(enc.encode(user_prompt))

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=spec.model,
                messages=[
                    {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=120,
            )
            wall = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            in_t = getattr(usage, "prompt_tokens", input_tokens_est) if usage else input_tokens_est
            out_t = getattr(usage, "completion_tokens", len(enc.encode(text))) if usage else len(enc.encode(text))
            return {
                "text": text,
                "input_tokens": int(in_t),
                "output_tokens": int(out_t),
                "wall_seconds": wall,
                "success": True,
                "error": None,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            is_429 = "429" in msg or "ratelimit" in msg or "too many" in msg
            if attempt < max_retries and (is_429 or "timeout" in msg):
                backoff = (2 ** attempt) + random.random()
                logger.warning(
                    f"llm_distill_one retry {attempt+1}/{max_retries} after {type(exc).__name__}, sleeping {backoff:.1f}s"
                )
                time.sleep(backoff)
                continue
            break
    return {
        "text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_seconds": 0.0,
        "success": False,
        "error": f"{type(last_err).__name__}: {str(last_err)[:200]}",
        "attempts": max_retries + 1,
    }


def build_l0_dualnodes(
    l0_records: List[Dict[str, Any]],
    *,
    ledger: TokenLedger,
    max_workers: int = 8,
    progress_every: int = 100,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    alias_status_tag: str = "",
) -> List[DualNode]:
    """Build a DualNode per L0 record. L0 records are dicts like
        {node_id, tenant_id, canonical_label, level_specific.raw_text,
         level_specific.evidence_span_id (or source_evidence_span_ids)}

    Returns a list of DualNodes (one per input record). Failed-to-distill
    records still produce a DualNode but with `distilled_text == ""` and the
    error captured in `extra["distill_error"]`. The Step 3 acceptance check
    will catch these and fail the run.
    """
    enc = _get_enc()

    def _node_body_and_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
        label = rec.get("canonical_label", "") or ""
        ls = rec.get("level_specific", {}) or {}
        raw = ls.get("raw_text", "") if isinstance(ls, dict) else ""
        body = label + ("\n" + raw if raw else "")
        # Collect provenance: evidence_span_id from level_specific, else node_id self-ref
        ev_ids: List[str] = []
        if isinstance(ls, dict):
            esid = ls.get("evidence_span_id")
            if esid:
                ev_ids.append(str(esid))
        for sid in rec.get("source_evidence_span_ids", []) or []:
            ev_ids.append(str(sid))
        if not ev_ids:
            ev_ids = [rec.get("node_id", "")]
        return {"body": body, "evidence_ids": ev_ids}

    def _process_one(rec: Dict[str, Any]) -> DualNode:
        info = _node_body_and_meta(rec)
        body = info["body"]
        detailed_tokens = len(enc.encode(body)) if body else 0
        distilled = llm_distill_one(body) if body else {
            "text": "", "input_tokens": 0, "output_tokens": 0,
            "wall_seconds": 0.0, "success": False, "error": "empty body",
        }
        distilled_text = distilled["text"]
        distilled_tokens = len(enc.encode(distilled_text)) if distilled_text else 0
        ledger.record(
            phase=PHASE_DISTILLED_GEN, model_alias="gpt_5_4_mini",
            input_tokens=distilled["input_tokens"], output_tokens=distilled["output_tokens"],
            wall_seconds=distilled["wall_seconds"],
            node_id=rec.get("node_id", ""),
        )
        node = DualNode(
            node_id=rec.get("node_id", ""),
            level=rec.get("level", "L0"),
            tenant_id=rec.get("tenant_id", ""),
            distilled_text=distilled_text,
            detailed_text=body,
            distilled_tokens=distilled_tokens,
            detailed_tokens=detailed_tokens,
            source_evidence_ids=info["evidence_ids"],
            state=NODE_STATE_LIGHT,
            distilled_text_model_alias="gpt_5_4_mini",
            distilled_text_model_status=alias_status_tag,
        )
        if not distilled["success"]:
            node.extra["distill_error"] = distilled["error"]
            node.extra["distill_attempts"] = distilled["attempts"]
        return node

    nodes: List[DualNode] = []
    n_total = len(l0_records)
    done = 0
    fails = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one, rec) for rec in l0_records]
        for fut in as_completed(futures):
            n = fut.result()
            nodes.append(n)
            done += 1
            if "distill_error" in n.extra:
                fails += 1
            if done % progress_every == 0 or done == n_total:
                logger.info(
                    f"  hierarchy_build: {done}/{n_total} done ({fails} distill failures so far)"
                )
                if progress_cb:
                    progress_cb(done, n_total)
    logger.info(f"hierarchy_build complete: {done}/{n_total} nodes, {fails} distill failures")
    return nodes
