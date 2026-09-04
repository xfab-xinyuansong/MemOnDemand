"""Dual memory index with distilled and detailed ChromaDB collections."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import EmbeddingFunction

logger = logging.getLogger(__name__)


# Embedding model cache (module-global so multiple DualIndex instances share)
_CACHED_ST_MODEL: Dict[str, Any] = {}


def _get_local_embedder(model_name: str = "all-MiniLM-L6-v2"):
    if model_name not in _CACHED_ST_MODEL:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedder: {model_name}")
        _CACHED_ST_MODEL[model_name] = SentenceTransformer(model_name)
    return _CACHED_ST_MODEL[model_name]


class LocalEmbeddingFunction(EmbeddingFunction):
    """ChromaDB EmbeddingFunction backed by local sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def __call__(self, input):
        if self._model is None:
            self._model = _get_local_embedder(self.model_name)
        vecs = self._model.encode(
            list(input),
            batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return [v.tolist() for v in vecs]

    @staticmethod
    def name() -> str:
        return "v4_local_st"


class DualIndex:
    """A pair of (distilled, detailed) ChromaDB collections."""

    def __init__(
        self,
        chroma_path: str,
        dataset: str = "own_full",
        run_id: str = "step3",
        embedding_function: Optional[EmbeddingFunction] = None,
    ):
        os.makedirs(chroma_path, exist_ok=True)
        self.chroma_path = chroma_path
        self.dataset = dataset
        self.run_id = run_id
        self.embedding_function = embedding_function or LocalEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.distilled_collection_name = f"distilled_{dataset}_{run_id}"
        self.detailed_collection_name = f"detailed_{dataset}_{run_id}"
        self._distilled_col = None
        self._detailed_col = None

    def reset(self) -> None:
        for name in (self.distilled_collection_name, self.detailed_collection_name):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        self._distilled_col = None
        self._detailed_col = None
        _ = self.distilled
        _ = self.detailed

    @property
    def distilled(self):
        if self._distilled_col is None:
            self._distilled_col = self.client.get_or_create_collection(
                name=self.distilled_collection_name,
                metadata={"hnsw:space": "cosine", "memondemand_kind": "distilled"},
                embedding_function=self.embedding_function,
            )
        return self._distilled_col

    @property
    def detailed(self):
        if self._detailed_col is None:
            self._detailed_col = self.client.get_or_create_collection(
                name=self.detailed_collection_name,
                metadata={"hnsw:space": "cosine", "memondemand_kind": "detailed"},
                embedding_function=self.embedding_function,
            )
        return self._detailed_col

    def upsert_distilled(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> int:
        if not ids:
            return 0
        self.distilled.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(ids)

    def upsert_detailed(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> int:
        if not ids:
            return 0
        self.detailed.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(ids)

    def query_distilled(self, query_text: str, n_results: int = 10,
                        where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        kwargs = dict(query_texts=[query_text], n_results=n_results,
                      include=["documents", "metadatas", "distances"])
        if where:
            kwargs["where"] = where
        return self.distilled.query(**kwargs)

    def query_detailed_ids_only(self, query_text: str, n_results: int = 10,
                                where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detailed-side search: returns ONLY ids + distances + minimal metadata."""
        kwargs = dict(query_texts=[query_text], n_results=n_results,
                      include=["metadatas", "distances"])
        if where:
            kwargs["where"] = where
        return self.detailed.query(**kwargs)

    def load_detailed_payload(self, ids: List[str]) -> Dict[str, Any]:
        """Explicit detailed-body loader. Caller (promotion controller) MUST pay
        the token cost recorded via TokenLedger."""
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.detailed.get(ids=ids, include=["documents", "metadatas"])

    def count_distilled(self) -> int:
        return self.distilled.count()

    def count_detailed(self) -> int:
        return self.detailed.count()


def _self_test() -> int:
    import shutil
    p = "/tmp/dual_index_test"
    shutil.rmtree(p, ignore_errors=True)
    idx = DualIndex(p, dataset="t", run_id="smoke")
    idx.reset()
    ids = ["n1", "n2"]
    distilled_texts = ["short hello world summary", "brief weather note"]
    detailed_texts = [
        "A long elaboration of hello world with many extra details.",
        "Detailed multi-sentence weather report covering temperature and humidity.",
    ]
    meta = [{"tenant": "t1"}, {"tenant": "t1"}]
    idx.upsert_distilled(ids, distilled_texts, meta)
    idx.upsert_detailed(ids, detailed_texts, meta)
    assert idx.count_distilled() == 2
    assert idx.count_detailed() == 2

    res_d = idx.query_distilled("hello", n_results=2)
    assert res_d.get("documents") and res_d["documents"][0]

    res_det = idx.query_detailed_ids_only("hello", n_results=2)
    # detailed query MUST NOT include populated documents field
    docs = res_det.get("documents")
    assert docs is None or all(x is None for sub in docs for x in sub), \
        f"detailed query leaked documents: {docs}"
    assert "ids" in res_det

    payload = idx.load_detailed_payload(["n1"])
    assert payload["documents"][0].startswith("A long elaboration")
    print("[PASS] DualIndex self-test")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
