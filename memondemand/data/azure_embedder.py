"""Azure OpenAI embedding helper with environment-only credentials."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Sequence

import numpy as np

from memondemand.core import dns_patch  # noqa: F401

log = logging.getLogger(__name__)

EMBED_MODEL_NAME = os.environ.get("EMB_AZURE_DEPLOYMENT", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMB_AZURE_DIM", "1536"))
AZURE_API_VERSION = os.environ.get("EMB_AZURE_API_VERSION", "2024-12-01-preview")


def _embedding_url() -> str:
    endpoint = (
        os.environ.get("EMB_AZURE_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or ""
    ).rstrip("/")
    if not endpoint:
        raise RuntimeError("Set EMB_AZURE_ENDPOINT or AZURE_OPENAI_ENDPOINT.")
    return (
        f"{endpoint}/openai/deployments/{EMBED_MODEL_NAME}"
        f"/embeddings?api-version={AZURE_API_VERSION}"
    )


def _api_key() -> str:
    key = (
        os.environ.get("EMB_AZURE_KEY")
        or os.environ.get("AZURE_LLM_API_KEY")
        or os.environ.get("AZURE_OPENAI_KEY")
        or ""
    )
    if not key:
        raise RuntimeError("Set EMB_AZURE_KEY, AZURE_LLM_API_KEY, or AZURE_OPENAI_KEY.")
    return key


def _post(payload: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _embedding_url(),
        data=body,
        headers={"Content-Type": "application/json", "api-key": _api_key()},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def embed_batch(texts: Sequence[str], max_retries: int = 12, batch_size: int = 128) -> np.ndarray:
    """Embed texts with Azure OpenAI and return L2-normalized float32 vectors."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = [(text or " ")[:32000] for text in texts[start:end]]
        attempt = 0
        while True:
            try:
                response = _post({"input": batch_texts, "model": EMBED_MODEL_NAME})
                data = response.get("data", [])
                if len(data) != len(batch_texts):
                    raise RuntimeError(
                        f"Got {len(data)} embeddings for {len(batch_texts)} inputs."
                    )
                for i, item in enumerate(data):
                    vec = np.asarray(item["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        vec = vec / norm
                    out[start + i] = vec
                break
            except (RuntimeError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                if attempt >= max_retries:
                    raise RuntimeError("Azure embedding failed after retries.") from exc
                wait = 65 if "429" in str(exc) else min(2**attempt, 30)
                attempt += 1
                log.warning(
                    "Azure embed batch %d-%d attempt %d failed; sleeping %ds",
                    start,
                    end,
                    attempt,
                    wait,
                )
                time.sleep(wait)

        if (start // batch_size) % 10 == 0:
            log.info("Azure embed progress: %d/%d", end, len(texts))
    return out


class AzureEmbedder:
    """Drop-in embedder class used by the V5 runner."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self.dim = EMBED_DIM
        log.info("AzureEmbedder ready: model=%s dim=%d", model_name, EMBED_DIM)

    def encode(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        batch_size = int(kwargs.get("batch_size", 128))
        return embed_batch(list(texts), batch_size=batch_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sample = ["hello world", "this is a test", "azure embedding works"]
    embeddings = embed_batch(sample)
    print(f"shape={embeddings.shape}, norms={np.linalg.norm(embeddings, axis=1)}")
