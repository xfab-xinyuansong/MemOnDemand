"""Portable runtime helpers for the public 1.14B pipeline.

The two public MemOnDemand repositories expose different model-gateway
configurations.  This module keeps the pipeline independent of either
deployment: semantic aliases are preserved when supported and otherwise map
to the repository's public ``general`` alias.  Embeddings similarly use the
repository's Azure client when present and fall back to its general embedding
endpoint.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np

from memondemand.core import api_adapter as _adapter


APIError = _adapter.APIError


def resolve_alias(alias: str) -> str:
    """Resolve a semantic model alias without encoding deployment details."""
    env_name = "MEMONDEMAND_MODEL_ALIAS_" + alias.upper()
    configured = os.environ.get(env_name, "").strip()
    candidate = configured or alias
    try:
        _adapter.get_alias_config(candidate)
        return candidate
    except ValueError:
        _adapter.get_alias_config("general")
        return "general"


def get_alias_config(alias: str):
    """Return the underlying repository configuration for ``alias``."""
    return _adapter.get_alias_config(resolve_alias(alias))


def call(
    alias: str,
    messages: List[Dict[str, str]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call the configured chat backend through the repository adapter."""
    return _adapter.call(resolve_alias(alias), messages, **kwargs)


class ConfiguredEmbedder:
    """Small adapter over either public embedding implementation."""

    def __init__(self, expected_dim: int = 1536):
        self.dim = expected_dim
        try:
            from memondemand.data.azure_embedder import AzureEmbedder

            self._client = AzureEmbedder()
            self._general_embed = None
            self.dim = int(getattr(self._client, "dim", expected_dim))
        except ImportError:
            embed_fn = getattr(_adapter, "embed", None)
            if embed_fn is None:
                raise RuntimeError(
                    "No embedding backend is available. Configure the public "
                    "embedding endpoint or install the repository embedding client."
                )
            self._client = None
            self._general_embed = embed_fn

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._client is not None:
            return np.asarray(self._client.encode(values, **kwargs), dtype=np.float32)
        response = self._general_embed(values)
        vectors = np.asarray(response["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(values):
            raise RuntimeError("Embedding backend returned an invalid matrix")
        self.dim = int(vectors.shape[1])
        return vectors
