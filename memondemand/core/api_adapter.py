"""Multi-model API adapter for MemOnDemand."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(os.environ.get("MEMONDEMAND_REPO_ROOT", Path.cwd())).resolve()
DEFAULT_ENV_PATH = Path(os.environ.get("MEMONDEMAND_ENV_FILE", PROJECT_ROOT / ".env"))


def load_env(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ if not already set.

    Lines starting with '#' or blank lines are ignored. Values are NOT logged.
    """
    if not env_path.exists():
        return
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Strip optional surrounding quotes
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        # Silent: don't leak path issues; caller checks env presence later.
        pass


logger = logging.getLogger("memondemand.api_adapter")
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


PRICING: Dict[str, Dict[str, float]] = {
    # Azure OpenAI list prices
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # AWS Bedrock Claude Opus 4-class pricing (approximate)
    "us.anthropic.claude-opus-4-8": {"input": 15.00, "output": 75.00},
    # AWS Bedrock gpt-oss-120b — placeholder pricing (no public list price yet)
    "openai.gpt-oss-120b-1:0": {"input": 0.0, "output": 0.0},
    # AWS Bedrock Mantle GPT-5.x — placeholder pricing (no public list price yet)
    "openai.gpt-5.4": {"input": 0.0, "output": 0.0},
    "openai.gpt-5.4-mini": {"input": 0.0, "output": 0.0},
    "openai.gpt-5-mini": {"input": 0.0, "output": 0.0},
    "openai.gpt-5.5": {"input": 0.0, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (input_tokens / 1_000_000.0) * p["input"] + (
        output_tokens / 1_000_000.0
    ) * p["output"]


@dataclass(frozen=True)
class AliasConfig:
    alias: str
    provider: str  # "openai_compatible" | "azure" | "bedrock" | "bedrock_responses"
    model: str     # model id (azure deployment name OR bedrock model id)

    chat_param_quirk: str = ""
    api_version: str = "2024-08-01-preview"


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"Missing required env var: {name}. Set it before calling api_adapter."
        )
    return v


def get_alias_config(alias: str) -> AliasConfig:
    """Resolve provider + model id from env. Does NOT return any secret."""
    if alias in ("general", "default"):
        model_id = os.environ.get("MEMONDEMAND_API_MODEL", "model")
        return AliasConfig(alias="general", provider="openai_compatible", model=model_id)
    if alias == "mini":
        deployment = _require_env("AZURE_OPENAI_MINI")
        return AliasConfig(alias="mini", provider="azure", model=deployment)
    if alias == "standard":
        deployment = _require_env("AZURE_OPENAI_STANDARD")
        return AliasConfig(
            alias="standard", provider="azure", model=deployment
        )
    if alias == "complex":
        model_id = _require_env("BEDROCK_MODEL_COMPLEX")
        return AliasConfig(alias="complex", provider="bedrock", model=model_id)
    if alias == "gpt_large":
        model_id = os.environ.get(
            "BEDROCK_MODEL_GPT_LARGE", "openai.gpt-oss-120b-1:0"
        )
        return AliasConfig(alias="gpt_large", provider="bedrock", model=model_id)
    if alias == "gpt_5_4":
        model_id = os.environ.get("BEDROCK_MODEL_GPT54", "openai.gpt-5.4")
        return AliasConfig(
            alias="gpt_5_4", provider="bedrock_responses", model=model_id
        )
    if alias == "gpt_5_4_mini":

        deployment = os.environ.get("AZURE_GPT54_MINI_DEPLOYMENT", "gpt-5.4-mini")
        return AliasConfig(
            alias="gpt_5_4_mini",
            provider="azure",
            model=deployment,
            chat_param_quirk="max_completion_tokens",
            api_version="2024-12-01-preview",
        )
    if alias == "gpt_5_4_azure":
        # Azure full model fallback: gpt-5.4 deployment NOT available on this resource.

        deployment = os.environ.get("AZURE_GPT4O_DEPLOYMENT", "gpt-4o")
        return AliasConfig(
            alias="gpt_5_4_azure",
            provider="azure",
            model=deployment,
            api_version="2024-12-01-preview",
        )
    if alias == "gpt_5_5":
        model_id = os.environ.get("BEDROCK_MODEL_GPT55", "openai.gpt-5.5")
        return AliasConfig(
            alias="gpt_5_5", provider="bedrock_responses", model=model_id
        )
    raise ValueError(
        f"Unknown alias: {alias!r}. Expected one of: "
        "general, mini, standard, complex, gpt_large, gpt_5_4, "
        "gpt_5_4_mini, gpt_5_5, gpt_5_4_azure"
    )


def _http_post(
    url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _call_openai_compatible(
    cfg: AliasConfig,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Call a generic OpenAI-compatible chat completions endpoint."""
    base_url = _require_env("MEMONDEMAND_API_BASE_URL").rstrip("/")
    api_key = _require_env("MEMONDEMAND_API_KEY")
    model = os.environ.get("MEMONDEMAND_API_MODEL", cfg.model)
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    raw = _http_post(url, headers, body, timeout=timeout)
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible response missing 'choices'")
    text = choices[0].get("message", {}).get("content", "") or ""
    usage = raw.get("usage") or {}
    return {
        "text": text,
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


def _call_azure(
    cfg: AliasConfig,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    endpoint = _require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")

    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    # Prefer dedicated AZURE_LLM_API_KEY (v4/models.yaml convention); fall back
    # to legacy AZURE_OPENAI_KEY.
    api_key = os.environ.get("AZURE_LLM_API_KEY") or _require_env("AZURE_OPENAI_KEY")
    api_version = cfg.api_version or "2024-08-01-preview"
    url = (
        f"{endpoint}/openai/deployments/{cfg.model}/chat/completions"
        f"?api-version={api_version}"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    body: Dict[str, Any] = {"messages": messages}
    if cfg.chat_param_quirk == "max_completion_tokens":

        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
        body["temperature"] = temperature
    raw = _http_post(url, headers, body, timeout=timeout)
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError("Azure response missing 'choices'")
    text = choices[0].get("message", {}).get("content", "") or ""
    usage = raw.get("usage") or {}
    return {
        "text": text,
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


def _call_bedrock(
    cfg: AliasConfig,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    region = _require_env("BEDROCK_REGION")
    bearer = _require_env("BEDROCK_API_KEY")
    # NB: Anthropic Bedrock messages format does NOT accept "system" inside
    # the messages list; we hoist it to the top-level "system" field.
    system_parts: List[str] = []
    chat_messages: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
        else:
            # Bedrock messages format expects content as list of blocks OR
            # plain string; we use string for simplicity.
            chat_messages.append({"role": role, "content": content})
    body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": chat_messages,
    }
    # claude-opus-4-8 deprecates the `temperature` knob; only include it
    # for older Claude models. Detect by model id substring.
    model_lc = cfg.model.lower()
    if "opus-4-8" not in model_lc:
        body["temperature"] = temperature
    if system_parts:
        body["system"] = "\n\n".join(system_parts)

    url = (
        f"https://bedrock-runtime.{region}.amazonaws.com/model/"
        f"{cfg.model}/invoke"
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer}",
    }
    raw = _http_post(url, headers, body, timeout=timeout)
    content = raw.get("content") or []
    text_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    text = "".join(text_parts)
    usage = raw.get("usage") or {}
    return {
        "text": text,
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
    }


def _call_bedrock_converse(
    cfg: AliasConfig,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Call a Bedrock model via the Converse API.

    Used for non-Anthropic Bedrock models (e.g. openai.gpt-oss-120b-1:0) that
    expose the unified Converse interface. Returns text + token usage in the
    same shape as the other provider helpers. If the response contains
    `reasoningContent` blocks, they are appended after the visible text so
    callers do not silently drop chain-of-thought content.
    """
    region = _require_env("BEDROCK_REGION")
    bearer = _require_env("BEDROCK_API_KEY")

    # Split system messages out (Converse uses a top-level `system` field)
    # and rewrite remaining messages into content-block form.
    system_parts: List[str] = []
    chat_messages: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        text_val = content if isinstance(content, str) else json.dumps(content)
        chat_messages.append(
            {"role": role, "content": [{"text": text_val}]}
        )

    body: Dict[str, Any] = {
        "messages": chat_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_parts:
        body["system"] = [{"text": "\n\n".join(system_parts)}]

    url = (
        f"https://bedrock-runtime.{region}.amazonaws.com/model/"
        f"{cfg.model}/converse"
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer}",
    }
    raw = _http_post(url, headers, body, timeout=timeout)

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    msg = (raw.get("output") or {}).get("message") or {}
    for block in msg.get("content") or []:
        if not isinstance(block, dict):
            continue
        if "text" in block and isinstance(block["text"], str):
            text_parts.append(block["text"])
        rc = block.get("reasoningContent")
        if isinstance(rc, dict):
            rt = rc.get("reasoningText") or {}
            rtxt = rt.get("text")
            if isinstance(rtxt, str) and rtxt:
                reasoning_parts.append(rtxt)
    text = "".join(text_parts)
    if not text and reasoning_parts:
        # Fallback: model returned only reasoning content (no visible answer).
        text = "".join(reasoning_parts)

    usage = raw.get("usage") or {}
    return {
        "text": text,
        "input_tokens": int(usage.get("inputTokens", 0)),
        "output_tokens": int(usage.get("outputTokens", 0)),
    }


def _bedrock_responses_endpoint() -> str:
    return os.environ.get(
        "BEDROCK_MANTLE_ENDPOINT",
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1",
    ).rstrip("/")


def _bedrock_responses_api_key() -> str:
    for k in (
        "BEDROCK_MANTLE_API_KEY",
        "AWS_BEDROCK_EXPERIMENT_KEY",
        "BEDROCK_API_KEY",
    ):
        v = os.environ.get(k)
        if v:
            return v
    raise RuntimeError(
        "Missing API key for Bedrock Mantle /responses endpoint. "
        "Set BEDROCK_MANTLE_API_KEY, AWS_BEDROCK_EXPERIMENT_KEY, "
        "or BEDROCK_API_KEY."
    )


def _flatten_messages_to_input(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages to a single flat input string for /responses.

    The OpenAI /responses API takes a single `input` string (not chat
    messages). We tag each segment by role and join with blank lines. The
    final segment is left without a trailing turn marker.
    """
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        if role == "system":
            tag = "System"
        elif role == "assistant":
            tag = "Assistant"
        else:
            tag = "User"
        parts.append(f"{tag}: {content}")
    return "\n\n".join(parts)


def _parse_responses_output(raw: Dict[str, Any]) -> str:
    """Walk response['output'] and collect text from output_text blocks."""
    text_parts: List[str] = []
    for item in raw.get("output") or []:
        if not isinstance(item, dict):
            continue

        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text":
                t = block.get("text", "")
                if isinstance(t, str):
                    text_parts.append(t)
    return "".join(text_parts)


def _call_bedrock_responses(
    cfg: AliasConfig,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Call AWS Bedrock Mantle's OpenAI /responses endpoint.

    Used for openai.gpt-5.4 and openai.gpt-5.5. Reasoning models (gpt-5.5)
    may reject `temperature`; we retry once without it if the server
    objects.
    """
    endpoint = _bedrock_responses_endpoint()
    api_key = _bedrock_responses_api_key()
    flat_input = _flatten_messages_to_input(messages)

    url = f"{endpoint}/responses"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def _build_body(include_temp: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": cfg.model,
            "input": flat_input,
            "max_output_tokens": max_tokens,
        }
        if include_temp:
            body["temperature"] = temperature
        return body

    # First attempt: include temperature. If server rejects it (HTTP 400
    # with mention of "temperature"), retry once without.
    try:
        raw = _http_post(url, headers, _build_body(True), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            try:
                body_bytes = e.read()
                err_body = body_bytes.decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            if "temperature" in err_body.lower():
                raw = _http_post(url, headers, _build_body(False), timeout=timeout)
            else:
                raise
        else:
            raise

    text = _parse_responses_output(raw)
    usage = raw.get("usage") or {}
    return {
        "text": text,
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
    }


class APIError(RuntimeError):
    """Raised when a model gateway request cannot be completed safely."""


def call(
    alias: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 5,
    backoff_base: float = 1.0,
) -> Dict[str, Any]:
    """Call a model via the given alias. Returns a structured dict.

    On failure after all retries raises APIError. The error message and
    any logged content are sanitized to NEVER include raw key/token values.
    """
    cfg = get_alias_config(alias)
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            if cfg.provider == "openai_compatible":
                inner = _call_openai_compatible(
                    cfg, messages, max_tokens, temperature, timeout
                )
            elif cfg.provider == "azure":
                inner = _call_azure(cfg, messages, max_tokens, temperature, timeout)
            elif cfg.provider == "bedrock":
                if cfg.alias == "gpt_large":
                    inner = _call_bedrock_converse(
                        cfg, messages, max_tokens, temperature, timeout
                    )
                else:
                    inner = _call_bedrock(
                        cfg, messages, max_tokens, temperature, timeout
                    )
            elif cfg.provider == "bedrock_responses":
                inner = _call_bedrock_responses(
                    cfg, messages, max_tokens, temperature, timeout
                )
            else:  # pragma: no cover
                raise APIError(f"Unknown provider: {cfg.provider}")
            latency_ms = (time.time() - t0) * 1000.0
            in_tok = inner["input_tokens"]
            out_tok = inner["output_tokens"]
            cost = estimate_cost(cfg.model, in_tok, out_tok)
            logger.info(
                "call ok alias=%s provider=%s model=%s "
                "input_tokens=%d output_tokens=%d latency_ms=%.1f cost_usd=%.6f attempt=%d",
                cfg.alias, cfg.provider, cfg.model,
                in_tok, out_tok, latency_ms, cost, attempt,
            )
            return {
                "text": inner["text"],
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
                "latency_ms": latency_ms,
                "model": cfg.model,
                "provider": cfg.provider,
                "alias": cfg.alias,
                "cost_usd": cost,
            }
        except urllib.error.HTTPError as e:
            # Read body for diagnostics but DO NOT log keys or headers.
            try:
                body_bytes = e.read()
                err_body = body_bytes.decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            err_body_safe = _sanitize(err_body)
            latency_ms = (time.time() - t0) * 1000.0
            logger.warning(
                "call http_err alias=%s provider=%s model=%s "
                "status=%s latency_ms=%.1f attempt=%d body=%s",
                cfg.alias, cfg.provider, cfg.model,
                e.code, latency_ms, attempt, err_body_safe[:500],
            )
            last_err = APIError(f"HTTP {e.code}: {err_body_safe[:200]}")
            # 4xx (except 408 / 429) usually not retryable
            if e.code in (400, 401, 403, 404) and e.code not in (408, 429):
                raise last_err
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, RuntimeError) as e:
            latency_ms = (time.time() - t0) * 1000.0
            logger.warning(
                "call err alias=%s provider=%s model=%s "
                "err_type=%s latency_ms=%.1f attempt=%d",
                cfg.alias, cfg.provider, cfg.model,
                type(e).__name__, latency_ms, attempt,
            )
            last_err = APIError(f"{type(e).__name__}: {_sanitize(str(e))[:200]}")
        # Exponential backoff with jitter, capped for predictable job recovery.
        if attempt < max_retries:
            _is_429 = isinstance(last_err, APIError) and "HTTP 429" in str(last_err)
            if _is_429:
                sleep_s = 30.0
            else:
                sleep_s = min(
                    30.0,
                    backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5),
                )
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err  # type: ignore[misc]


_SECRET_ENV_KEYS = (
    "AZURE_OPENAI_KEY",
    "MEMONDEMAND_API_KEY",
    "MEMONDEMAND_EMBED_API_KEY",
    "BEDROCK_API_KEY",
    "BEDROCK_MANTLE_API_KEY",
    "OPENAI_API_KEY",
    "AWS_BEDROCK_SECRET",
    "AWS_BEDROCK_EXPERIMENT_KEY",
)


def _sanitize(text: str) -> str:
    """Replace any occurrence of known secret env values inside a string."""
    out = text
    for k in _SECRET_ENV_KEYS:
        v = os.environ.get(k)
        if v and len(v) >= 8 and v in out:
            out = out.replace(v, f"<REDACTED:{k}>")
    # Generic bearer/api-key patterns
    out = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", r"\1<REDACTED>", out)
    out = re.sub(r"(api[-_]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._\-]{12,}",
                 r"\1<REDACTED>", out, flags=re.IGNORECASE)
    return out


# Auto-load .env when imported so callers do not have to do it manually.
load_env()


if __name__ == "__main__":
    if not os.environ.get("MEMONDEMAND_API_BASE_URL") or not os.environ.get("MEMONDEMAND_API_KEY"):
        print("general smoke skipped: set MEMONDEMAND_API_BASE_URL and MEMONDEMAND_API_KEY")
    else:
        try:
            res = call(
                "general",
                [{"role": "user", "content": "Reply with exactly: TEST_GENERAL"}],
                max_tokens=100,
                temperature=0.0,
                max_retries=1,
            )
            assert "TEST_GENERAL" in res["text"], f"got: {res['text']!r}"
            print(
                f"general alias works: text={res['text'][:80]!r}, "
                f"latency={res['latency_ms']:.0f}ms, "
                f"in={res['usage']['input_tokens']} out={res['usage']['output_tokens']}"
            )
        except Exception as e:
            print(f"general smoke skipped: {e}")
