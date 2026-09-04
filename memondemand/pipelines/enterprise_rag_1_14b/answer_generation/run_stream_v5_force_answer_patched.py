"""Force-answer query runner for the MemOnDemand full-corpus pipeline."""
from __future__ import annotations

try:
    from memondemand.core import dns_patch  # noqa: F401
except ImportError:
    dns_patch = None

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Running this dedicated module enables force-answer mode unless a caller
# explicitly disables it with MEMONDEMAND_FORCE_ANSWER=0.
os.environ.setdefault("MEMONDEMAND_FORCE_ANSWER", "1")

REPO_ROOT = os.environ.get("MEMONDEMAND_REPO_ROOT", str(Path.cwd()))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from memondemand.pipelines.enterprise_rag_1_14b.runtime import (  # noqa: E402
    APIError,
    ConfiguredEmbedder,
    call as api_call,
)
from memondemand.methods.dual_node import (  # noqa: E402
    DualNode,
    NODE_STATE_LIGHT,
    NODE_STATE_PROMOTED,
)
from memondemand.methods.token_ledger import (  # noqa: E402
    PHASE_FINAL_ANSWER,
    PHASE_PROMOTION_DECISION,
    PHASE_RETRIEVAL,
    TokenLedger,
)
from memondemand.methods.promotion_controller import PromotionController  # noqa: E402
from memondemand.methods.decay_controller import DecayController  # noqa: E402
from memondemand.methods.state_log import (  # noqa: E402
    StateLog,
    StateLogEntry,
    EVENT_PROMOTE,
    EVENT_KEEP_LIGHT,
    EVENT_DEMOTE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("memondemand.enterprise_rag_1_14b.runner")


VALID_METHODS = {"B_flat", "B_fixed_hier", "B_dynamic_hier", "B_llm_nav", "V5"}


def load_hierarchy_jsonl(path: str) -> Dict[str, DualNode]:
    """V5 hierarchy is newline-delimited JSON. level is an int (0..4).
    We translate to DualNode with level="L<int>" for V4 compat."""
    nodes: Dict[str, DualNode] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lvl = d["level"]
            if isinstance(lvl, int):
                d["level"] = f"L{lvl}"
            n = DualNode.from_dict(d)
            nodes[n.node_id] = n
    return nodes


def build_parent_child(hierarchy: Dict[str, DualNode]) -> Dict[str, List[str]]:
    """Index child identifiers for fast parent-to-child lookup."""
    out = {}
    for nid, n in hierarchy.items():
        kids = n.extra.get("children_ids", []) or n.extra.get("child_node_ids", []) or []
        if not kids:
            # Fallback: for L1 nodes, use source_evidence_ids (L0 leaves under this cluster)
            if n.level == "L1":
                kids = list(n.source_evidence_ids or [])
        out[nid] = list(kids)
    return out


def build_parent_child_from_file(path: str) -> Dict[str, List[str]]:
    """Build a parent-to-child index from hierarchy records."""
    parent_child: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            nid = d["node_id"]
            extra = d.get("extra", {}) or {}
            children = d.get("children_ids", []) or extra.get("children_ids", []) or extra.get("child_node_ids", []) or []
            # Fallback for L1 nodes in erag schema: use source_evidence_ids (L0 leaves)
            if not children:
                lvl = d.get("level")
                lvl_str = f"L{lvl}" if isinstance(lvl, int) else lvl
                if lvl_str == "L1":
                    children = list(d.get("source_evidence_ids", []) or [])
            parent_child[nid] = list(children)
    return parent_child


def build_parent_map(parent_child: Dict[str, List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for parent, children in parent_child.items():
        for c in children:
            out[c] = parent
    return out


def bucket_by_level(hierarchy: Dict[str, DualNode]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for nid, n in hierarchy.items():
        out[n.level].append(nid)
    # Sort levels: L0, L1, ...
    return dict(out)


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        import os as _os, json as _json
        backend = _os.environ.get("MEMONDEMAND_EMBED_BACKEND", "azure_small")
        self.backend = backend
        # Optional query embedding cache (for azure_large to avoid per-query API)
        self._query_cache = {}
        npy_path = _os.environ.get("MEMONDEMAND_QUERY_EMB_NPY", "")
        ids_path = _os.environ.get("MEMONDEMAND_QUERY_EMB_IDS", "")
        if npy_path and ids_path and _os.path.exists(npy_path) and _os.path.exists(ids_path):
            try:
                _vecs = np.load(npy_path)
                with open(ids_path) as _f: _meta = _json.load(_f)
                for _i, _t in enumerate(_meta.get("query_texts", [])):
                    self._query_cache[_t] = _vecs[_i].astype(np.float32)
                log.info("loaded query embedding cache: %d texts from %s", len(self._query_cache), npy_path)
            except Exception as _e:
                log.warning("query cache load failed: %s", _e)
        if backend in ("azure_large", "azure_small", "configured"):
            _dim = 1536 if backend != "azure_large" else 3072
            log.info("loading configured embedder (%s, expected_dim=%d) ...", backend, _dim)
            self._azure = ConfiguredEmbedder(expected_dim=_dim)
            self.dim = self._azure.dim
            self.model = None
        else:
            from sentence_transformers import SentenceTransformer
            log.info("loading embedder %s ...", model_name)
            self.model = SentenceTransformer(model_name)
            self.dim = 384
            self._azure = None

    def encode(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        texts_list = list(texts)
        # Try cache first for all texts; only call backend for misses
        if self._query_cache:
            out = np.zeros((len(texts_list), self.dim), dtype=np.float32)
            misses = []
            miss_idx = []
            for i, t in enumerate(texts_list):
                if t in self._query_cache:
                    out[i] = self._query_cache[t]
                else:
                    miss_idx.append(i)
                    misses.append(t)
            if misses:
                if self.backend in ("azure_large", "azure_small"):
                    miss_vecs = self._azure.encode(misses, **kwargs).astype(np.float32)
                else:
                    params = {"batch_size": 64, "normalize_embeddings": True,
                              "show_progress_bar": False, "convert_to_numpy": True}
                    params.update(kwargs)
                    miss_vecs = self.model.encode(misses, **params).astype(np.float32)
                for j, idx in enumerate(miss_idx):
                    out[idx] = miss_vecs[j]
                    # populate cache so subsequent queries reuse
                    self._query_cache[misses[j]] = miss_vecs[j]
            return out
        # No cache configured
        if self.backend in ("azure_large", "azure_small"):
            return self._azure.encode(texts_list, **kwargs).astype(np.float32)
        params = {
            "batch_size": 64,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        params.update(kwargs)
        vecs = self.model.encode(texts_list, **params).astype(np.float32)
        return vecs


_BM25_CACHE = {'index': None, 'doc_ids': None}
def _bm25_tokenize(text):
    if not text: return []
    return re.findall(r"[a-z0-9]{2,}", text.lower())

def _build_bm25_index(hierarchy):
    """Build BM25Okapi index over L0 nodes' detailed_text (full content)."""
    from rank_bm25 import BM25Okapi
    l0_ids = []
    l0_texts = []
    for nid, n in hierarchy.items():
        if n.level == "L0":
            l0_ids.append(nid)
            # BM25 augment: concat detailed_text + key_facts for better entity matching
            _kf = (getattr(n, "key_facts", None) or "").strip()
            _base = (n.detailed_text or n.distilled_text or "")
            l0_texts.append((_base + " " + _kf).strip() if _kf else _base)
    log.info("BM25: tokenizing %d L0 docs ...", len(l0_ids))
    t0 = time.time()
    tokenized = [_bm25_tokenize(t) for t in l0_texts]
    log.info("BM25: building Okapi index ...")
    bm = BM25Okapi(tokenized)
    log.info("BM25: ready (%.1fs)", time.time() - t0)
    return bm, l0_ids

def _get_bm25_index(ctx):
    if _BM25_CACHE['index'] is None:
        bm, l0_ids = _build_bm25_index(ctx.hierarchy)
        _BM25_CACHE['index'] = bm
        _BM25_CACHE['doc_ids'] = l0_ids
    return _BM25_CACHE['index'], _BM25_CACHE['doc_ids']


# ===== C4: te3-large RRF — module-level globals =====
import numpy as _np_te3
import json as _json_te3
from pathlib import Path as _Path_te3

_TE3_ARR: "_np_te3.ndarray | None" = None
_TE3_IDS: "list | None" = None

def _load_te3_index_once():
    global _TE3_ARR, _TE3_IDS
    if _TE3_ARR is not None:
        return
    cache = _Path_te3(os.environ.get(
        "MEMONDEMAND_TE3_INDEX_CACHE",
        str(_Path_te3(REPO_ROOT) / "results" / "enterprise_rag_1_14b" / "_index_cache"),
    ))
    npy_files = sorted(cache.glob("index_L0_*.npy"))
    json_files = sorted(cache.glob("index_L0_*.json"))
    if not npy_files or not json_files:
        log.warning("C4: te3-large L0 index not found, RRF disabled")
        return
    import logging
    logging.getLogger(__name__).info("C4: loading te3-large L0 index (%s)...", npy_files[0].name)
    arr = _np_te3.load(str(npy_files[0]))          # (N, 3072) float32
    meta = _json_te3.loads(json_files[0].read_text())
    norms = _np_te3.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _TE3_ARR = arr / norms                         # pre-normalized
    _TE3_IDS = meta["ids"]
    logging.getLogger(__name__).info("C4: te3-large L0 index loaded: %d vectors dim=%d", len(_TE3_IDS), arr.shape[1])

# ===== END C4 module-level =====

def _bm25_top_k(ctx, query_text, k, filter_ids=None):
    """Return list of (doc_id, score) for top-k BM25 hits on L0."""
    bm, all_ids = _get_bm25_index(ctx)
    q_toks = _bm25_tokenize(query_text)
    scores = bm.get_scores(q_toks)
    if filter_ids is not None:
        filter_set = set(filter_ids)
        # Mask scores to filter set
        masked = []
        for i, did in enumerate(all_ids):
            if did in filter_set:
                masked.append((did, float(scores[i])))
        masked.sort(key=lambda x: -x[1])
        return masked[:k]
    # Global top-k
    import numpy as np
    top_idx = scores.argsort()[::-1][:k]
    return [(all_ids[i], float(scores[i])) for i in top_idx]


def _bm25_then_dense_rerank(ctx, query_text, k=12, wide_k=50):
    """BM25 retrieves wide_k candidates, then rerank by dense cosine sim using cached L0 index."""
    import os as _os, json as _json
    # 1. BM25 wide retrieval
    bm_hits = _bm25_top_k(ctx, query_text, wide_k, filter_ids=None)
    if not bm_hits:
        return []
    cand_ids = [did for did, _ in bm_hits]
    # 2. Get L0 dense index (from ctx.indexes or build via get_index)
    l0_idx = get_index(ctx, "L0")
    if l0_idx is None or len(l0_idx.ids) == 0:
        # No dense index → fall back to BM25
        return bm_hits[:k]
    # 3. Embed query (uses ctx.embedder, which has cache for azure_large via MEMONDEMAND_QUERY_EMB_NPY)
    q_vec = ctx.embedder.encode([query_text])[0]
    # 4. Cosine rerank filtered to candidate ids
    reranked = l0_idx.search(q_vec, top_k=k, filter_ids=cand_ids)
    if not reranked:
        # Fallback: BM25 top-k
        return bm_hits[:k]
    return reranked


def _bm25_guided_l1_frontier(ctx, query_text, bm25_top_k=50, max_l1=8):
    """BM25 top-N L0 hits -> L1 parents (sorted by coverage count) -> top-max_l1 L1 frontier."""
    bm_hits = _bm25_top_k(ctx, query_text, bm25_top_k, filter_ids=None)
    if not bm_hits:
        return []
    l0_set = set(did for did, _ in bm_hits)
    # Count L1 parents by how many of their L0 children are in the BM25 hit set
    l1_coverage = {}
    for nid, n in ctx.hierarchy.items():
        if n.level != "L1":
            continue
        kids = ctx.parent_child.get(nid, [])
        if not kids:
            continue
        hits = sum(1 for k in kids if k in l0_set)
        if hits > 0:
            l1_coverage[nid] = hits
    if not l1_coverage:
        return []
    sorted_l1 = sorted(l1_coverage.items(), key=lambda x: -x[1])[:max_l1]
    # Return list of (l1_id, score) where score = coverage / max_coverage
    max_cov = sorted_l1[0][1]
    return [(nid, cov / max_cov) for nid, cov in sorted_l1]

class CosineIndex:
    def __init__(self, ids: List[str], vecs: np.ndarray):
        self.ids = list(ids)
        self.vecs = vecs
        self.id_to_pos = {nid: i for i, nid in enumerate(self.ids)}

    def search(self, q: np.ndarray, top_k: int = 10,
               filter_ids: Optional[Sequence[str]] = None
               ) -> List[Tuple[str, float]]:
        if self.vecs.shape[0] == 0:
            return []
        if filter_ids is not None:
            idxs = [self.id_to_pos[i] for i in filter_ids if i in self.id_to_pos]
            if not idxs:
                return []
            sub = self.vecs[idxs]
            sims = sub @ q
            order = np.argsort(-sims)[:top_k]
            return [(self.ids[idxs[i]], float(sims[i])) for i in order]
        sims = self.vecs @ q
        order = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in order]


ANSWER_SYSTEM = """You are an enterprise memory question-answering assistant.

You will be given:
  1. A user QUERY.
  2. [CTX] hierarchy context blocks for background scope (do NOT cite these).
  3. [EVID] evidence blocks, each with a node_id. Some have [BRIEF] (distilled summary) only;
     top-ranked evidence also has [FULL] (complete source text).

Rules:
- Write a precise, factual answer (1-5 sentences) using ONLY information from [EVID] blocks.
- Your answer MUST be grounded PRIMARILY in [FULL] source text. Treat [BRIEF] as a topic hint only.
- Write 1-3 precise sentences. Every factual claim must be traceable to a specific [FULL] passage.
- After the answer, on a NEW LINE, output: CITED: id1,id2,...
- CITE EVERY [EVID] node whose [FULL] or [BRIEF] content you used. Do not omit citations.
  If 3+ nodes contributed, cite all of them.
- Do NOT cite [CTX] node_ids.
- If no [EVID] context applies, output "INSUFFICIENT EVIDENCE" and CITED: (empty).
- Never invent facts not present in the provided context."""

# V6: Dual-Memory prompt — controlled by MEMONDEMAND_V6_DUAL_MEM=1
ANSWER_SYSTEM_V6 = """You are an enterprise knowledge retrieval system. Answer questions using ONLY the provided source documents.

Context structure:
  [CTX] SCOPE SUMMARIES — cluster-level overviews for orientation. NEVER cite these.
  [EVID] SOURCE DOCUMENTS — enterprise records. Each has:
    • [BRIEF] key-fact bullets (topic index — for scanning only)
    • [FULL] complete source text (your primary evidence — between "--- FULL SOURCE TEXT ---" markers)

REQUIRED STEPS:

① IDENTIFY FOCUS ENTITY
  From the query, name the exact entity being asked about:
  person name / product / company / system / ticket ID / config key / metric name.

② TRIAGE ALL [EVID] NODES
  For each [EVID] block:
  — Does its [BRIEF] or [FULL] content mention the FOCUS entity specifically?
  — Does its [FULL] source text contain facts that DIRECTLY answer the query?
  Mark nodes that pass BOTH checks. Ignore the rest.
  (If entity names are similar but different — e.g. "Redwood Staging" vs "Redwood Prod" — do NOT conflate them.)

③ WRITE ANSWER (1-3 sentences)
  Using ONLY the marked nodes from step ②:
  • Base every factual claim on a [FULL] source passage. [BRIEF] bullets may orient you but never serve as sole evidence.
  • If two nodes provide complementary facts (e.g., one has the value, another has the date), combine them.
  • Be concise and direct. Do not hedge with phrases like "based on the provided context".
  • If no [FULL] text directly supports the query → output exactly: INSUFFICIENT EVIDENCE

④ CITE ALL SOURCES
  On a NEW LINE: CITED: node_id1,node_id2,...
  List EVERY [EVID] node_id whose [BRIEF] or [FULL] content contributed to your answer.
  Never cite [CTX] ids. Never fabricate ids. Never omit a contributing node."""

ANSWER_USER_V6 = """QUERY: {query}

{context_block}

Write your answer, then on a new line: CITED: <comma-separated node_ids from [EVID] that you used>"""
# V8-TOP5: prompt for top5_full_only mode (no [BRIEF], only [FULL] texts)
ANSWER_SYSTEM_V8_TOP5 = """You are an enterprise knowledge retrieval system.

Context:
  [CTX] SCOPE SUMMARIES — cluster-level overviews. NEVER cite these.
  [EVID] SOURCE DOCUMENTS — complete source texts. These are your ONLY evidence.

STEPS:

1. FOCUS: What is the exact entity / concept the query is about?
2. READ EVIDENCE: Read each [EVID] document. Does it specifically address the FOCUS entity with factual content?
3. ANSWER (1-3 sentences): Write a precise, direct answer using ONLY facts explicitly stated in matching [EVID] documents. No inference beyond what is written.
4. If no [EVID] document addresses the query → INSUFFICIENT EVIDENCE
5. CITED: id1,id2,... — every [EVID] node_id you drew facts from. No [CTX] ids."""

ANSWER_USER_V8_TOP5 = """QUERY: {query}

{context_block}

Answer (1-3 sentences), then: CITED: <node_ids you used>"""


# V6.1: Scan-first + citation completeness check — controlled by MEMONDEMAND_V61=1
ANSWER_SYSTEM_V61 = """You are an enterprise knowledge retrieval system.

Context:
  [CTX] SCOPE SUMMARIES — orientation only. NEVER cite.
  [EVID] SOURCE DOCUMENTS — your evidence. [BRIEF]=key bullets; [FULL]=full source text.

STEPS:

A. LOCK ENTITY: Name the exact thing the query is about.
B. SCAN: For each [EVID] block, check if [FULL] text contains a direct answer about that entity.
C. ANSWER (1-3 sentences): Use ONLY [FULL] passages from relevant blocks. Be precise.
D. CHECK: Re-scan — any relevant node not yet cited? Add it.
E. CITE: CITED: id1,id2,... — every contributing [EVID] node_id. No [CTX] ids.

No match → INSUFFICIENT EVIDENCE. Never invent facts."""

ANSWER_USER_V61 = """QUERY: {query}

{context_block}

Follow the procedure (A→B→C→D). Write your answer, then: CITED: <node_ids>"""

# V7: Entity-matching + exhaustive scan — controlled by MEMONDEMAND_V7_ENTITY=1
ANSWER_SYSTEM_V7 = """You are an enterprise knowledge retrieval system.

Context:
  [CTX] SCOPE SUMMARIES — orientation only. NEVER cite.
  [EVID] SOURCE DOCUMENTS — your evidence. Each has [BRIEF] bullets and optionally [FULL] source text.

STEP 1 — ENTITY LOCK: Identify the exact entity the query is about (person / product / company / ticket / config key).
STEP 2 — EVIDENCE SCAN: For each [EVID] block, ask: (a) does it name the same entity? (b) does its [FULL] text directly answer the query? Mark blocks where BOTH are true.
STEP 3 — ANSWER (1-3 sentences): Use ONLY marked blocks. Base claims on [FULL] text. Combine facts across nodes when complementary. If nothing matches → INSUFFICIENT EVIDENCE.
STEP 4 — CITE: CITED: id1,id2,... — all [EVID] ids from your marked set. No [CTX] ids.

Never invent facts. Never conflate similar-but-different entities."""

ANSWER_USER_V7 = """QUERY: {query}

{context_block}

Follow the 4-step procedure. Write your answer, then: CITED: <node_ids>"""

ANSWER_USER_TMPL = """QUERY: {query}

CONTEXT:
{context_block}

Answer with the format:
<answer text>
CITED: <comma-separated node_ids you actually used>"""


CITED_RE = re.compile(r"CITED\s*:\s*([A-Za-z0-9_,\s\-]*)$", re.MULTILINE)


def parse_cited(answer_text: str, candidate_ids: Sequence[str]) -> List[str]:
    cset = set(candidate_ids)
    out, seen = [], set()
    for m in CITED_RE.findall(answer_text or ""):
        for tok in m.split(","):
            tok = tok.strip()
            # V6: strip [EVID|...] wrapper if present
            if tok.startswith("[EVID|") and tok.endswith("]"):
                tok = tok[6:-1].strip()
            if tok and tok in cset and tok not in seen:
                out.append(tok); seen.add(tok)
    return out


def strip_cited(answer_text: str) -> str:
    if not answer_text:
        return ""
    return re.sub(r"\nCITED\s*:\s*[^\n]*", "", answer_text).strip()


import os as _os_ansmode

def _node_text_for_answer(n, is_top_detail: bool = False) -> str:
    """Return node text for answer context based on MEMONDEMAND_ANSWER_MODE.

    Modes:
      detailed_truncated (default): detailed_text[:2000chars]  (legacy)
      detailed_full               : detailed_text, no truncation
      distilled                   : distilled_text only, no truncation
      both                        : distilled + detailed, no truncation
      mixed (RECOMMENDED)         : all nodes get distilled; top-5 also get detailed
                                    is_top_detail=True  → brief + full text
                                    is_top_detail=False → brief only
    """
    mode = _os_ansmode.environ.get("MEMONDEMAND_ANSWER_MODE", "detailed_truncated")
    det  = n.detailed_text or ""
    dis  = n.distilled_text or ""
    if mode == "distilled":
        return dis
    if mode == "detailed_full":
        return det
    if mode == "both":
        parts = []
        if dis:
            parts.append(f"[BRIEF] {dis}")
        if det:
            parts.append(f"[FULL] {det}")
        return "\n".join(parts) if parts else (det or dis)
    if mode == "top5_full_only":
        # Only show full text for top-K nodes; hide everything else
        if is_top_detail:
            return f"[FULL]\n{det or dis}"
        return ""
    if mode == "mixed":
        # BRIEF content: prefer key_facts (structured bullets) if available, else distilled
        _brief_mode = _os_ansmode.environ.get("MEMONDEMAND_BRIEF_MODE", "distilled")
        if _brief_mode == "key_facts":
            brief = (getattr(n, "key_facts", None) or "").strip() or dis
        else:
            brief = dis
        # All nodes: BRIEF (no trunc). Top-5 by BM25 score: also detailed.
        if is_top_detail and det:
            return f"[BRIEF] {brief}\n--- FULL SOURCE TEXT ---\n{det}\n--- END SOURCE ---"
        return f"[BRIEF] {brief}" if brief else (det or dis)[:2000]
    # default: detailed_truncated (legacy)
    return (det or dis)[:2000]

def format_context_block(items: List[Tuple[str, str, str]]) -> str:
    """items: list of (node_id, level, text)."""
    if not items:
        return "(none)"
    import os as _os_fmt
    if _os_fmt.environ.get("MEMONDEMAND_V6_DUAL_MEM", "0") == "1":
        return _format_dual_context(items)
    return "\n".join(f"- [{nid}] ({lvl}) {txt}" for nid, lvl, txt in items)

def _format_dual_context(items: List[Tuple[str, str, str]]) -> str:
    """V6: separate CTX (upper-level) vs EVID (L0) with explicit labels."""
    ctx_lines, evid_lines = [], []
    for idx, (nid, lvl, txt) in enumerate(items):
        if lvl == "L0":
            evid_lines.append(f"- [EVID|{nid}] {txt}")
        else:
            ctx_lines.append(f"- [CTX|{nid}] ({lvl}) {txt}")
    parts = []
    if ctx_lines:
        parts += ["[CTX] HIERARCHY CONTEXT — background scope, do NOT cite:"] + ctx_lines + [""]
    if evid_lines:
        parts += [
            "[EVID] SOURCE EVIDENCE — cite node_ids you used:",
            "  (nodes with [BRIEF] = distilled summary; nodes with [FULL] = full source text)",
            "  Cite ALL that contributed — omitting relevant citations hurts recall.",
        ] + evid_lines
    return "\n".join(parts) if parts else "(none)"

def _format_dual_smart_context(
    l0_hits_with_scores,  # List[(nid, bm25_score)]
    upper_items,           # List[(nid, lvl, txt)] for L1/L2
    hierarchy,
    top_detailed_k: int = 5,
) -> str:
    """Dual-smart: ALL L0 nodes get distilled_text overview; top-K also get detailed_text.

    Format:
      [OVERVIEW] distilled summaries for ALL 25 BM25 hits (quick scan, no truncation)
      [EVIDENCE] distilled + detailed for top-K most relevant (cite these)
    """
    import os as _os_ds
    top_k = int(_os_ds.environ.get("MEMONDEMAND_DUAL_SMART_K", str(top_detailed_k)))

    overview_lines = []
    evidence_lines = []

    for rank, (nid, score) in enumerate(l0_hits_with_scores):
        n = hierarchy.get(nid)
        if n is None: continue
        dis = (n.distilled_text or "").strip()
        det = (n.detailed_text or "").strip()

        if rank < top_k:
            # Full dual-memory: distilled overview + detailed content
            parts = []
            if dis:
                parts.append(f"  [SUMMARY] {dis}")
            if det:
                parts.append(f"  [FULL] {det}")
            evidence_lines.append(f"- [EVID|{nid}]\n" + "\n".join(parts))
        else:
            # Distilled overview only
            overview_lines.append(f"- [EVID|{nid}] {dis}")

    # Upper-level context
    ctx_lines = [f"- [CTX|{nid}] ({lvl}) {txt}" for nid, lvl, txt in upper_items]

    parts = []
    if ctx_lines:
        parts += ["[CTX] HIERARCHY CONTEXT (background — do NOT cite):", *ctx_lines, ""]
    if evidence_lines:
        parts += [f"[EVIDENCE] TOP-{top_k} SOURCES — cite these node_ids (both summary and full text available):",
                  *evidence_lines, ""]
    if overview_lines:
        parts += [f"[OVERVIEW] ADDITIONAL SOURCES — distilled summaries only; cite if relevant:",
                  *overview_lines]
    return "\n".join(parts) if parts else "(none)"


NAV_SYSTEM = """You navigate a hierarchical enterprise memory to locate evidence for a query.

Hierarchy: L2 = broad topic clusters  →  L1 = fine-grained clusters  →  L0 = source documents

ACTIONS:
- DESCEND: enter the most relevant node(s) to find more specific evidence
- LATERAL: try sibling nodes if the current path is clearly off-topic
- ANSWER: commit — evidence retrieval will run from the current cluster
- STOP_INSUFFICIENT: only if the corpus has NO cluster related to the query

NAVIGATION STRATEGY:

Step 1 — IDENTIFY the FOCUS ENTITY from the query:
  the exact product name / company / person / system / ticket / configuration key being asked about.

Step 2 — MATCH against node summaries:
  Read each node's summary (key bullet points). Does it explicitly name the FOCUS entity?
  • Strong match (entity name present) → DESCEND into that node immediately.
  • Weak match (related topic but entity not named) → note it as fallback.
  • No match → skip.

Step 3 — NAVIGATE toward specificity:
  • Prefer narrow L1 clusters over broad L2 clusters when the entity name appears.
  • If you see 2+ clusters with the entity, DESCEND into ALL of them (list multiple ids).
  • Use LATERAL only if you've DESCENDed and the cluster turned out off-topic.
  • Do NOT ANSWER from L2 unless no L1 cluster is relevant.

SOURCE-TYPE HINTS (soft — override only if entity match is clear):
  - Policy / process / config / docs query → prefer confluence / google_drive / github
  - Message / ticket / incident / person query → prefer slack / linear / gmail

STEP CAP:
  You have a MAXIMUM of {max_steps} navigation actions (DESCEND or LATERAL combined).
  After reaching the cap, you MUST output ANSWER (or STOP_INSUFFICIENT if nothing relevant was ever seen).

OUTPUT: strict JSON only — no markdown, no extra text:
{"action": "DESCEND|LATERAL|ANSWER|STOP_INSUFFICIENT",
 "chosen_node_ids": ["<id>", ...],
 "rationale": "<one concise sentence explaining your choice>"}

ANSWER and STOP_INSUFFICIENT: chosen_node_ids may be empty."""

NAV_USER_TMPL = """QUERY: {query}

Navigation step {step}/{max_steps}  |  History: {prior_actions}

FRONTIER — candidate nodes ({n} shown):
{frontier_block}

Instructions:
1. Identify the FOCUS ENTITY from the query.
2. Find the frontier node(s) whose summary explicitly mentions it.
3. Output JSON: action + chosen_node_ids + rationale."""


def call_llm(alias: str, system: str, user: str, *,
             max_tokens: int = 400, temperature: float = 0.0,
             timeout: float = 90.0,
             ledger: Optional[TokenLedger] = None,
             phase: str = PHASE_FINAL_ANSWER,
             query_id: str = "",
             ) -> Tuple[str, int, int, float]:
    """Generic wrapper around api_adapter.call. Returns (text, in_tok, out_tok, wall_s)."""
    t0 = time.time()
    try:
        resp = api_call(
            alias,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_retries=120,
        )
    except (APIError, Exception) as exc:  # noqa: BLE001
        wall = time.time() - t0
        log.warning("LLM call failed alias=%s phase=%s: %s", alias, phase, str(exc)[:200])
        return "", 0, 0, wall
    wall = time.time() - t0
    in_t = int(resp.get("usage", {}).get("input_tokens", 0))
    out_t = int(resp.get("usage", {}).get("output_tokens", 0))
    if ledger is not None:
        ledger.record(
            phase=phase, model_alias=alias,
            input_tokens=in_t, output_tokens=out_t,
            wall_seconds=wall, query_id=query_id,
        )
    return resp.get("text", "") or "", in_t, out_t, wall


def parse_nav_json(text: str) -> Dict[str, Any]:
    """Extract first JSON object from LLM text.

    v5 Step C v3 fix 3: on parse failure / empty response, force ANSWER rather
    than STOP_INSUFFICIENT. Navigator must never refuse.
    """
    if not text:
        return {"action": "ANSWER", "chosen_node_ids": [], "rationale": "empty_response_forced_answer"}
    # Try direct
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try first balanced { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"action": "ANSWER", "chosen_node_ids": [], "rationale": "parse_fail_forced_answer"}


@dataclass
class QueryRecord:
    query_id: str
    query_text: str
    method: str
    tier: str
    final_action: str = "STOP_INSUFFICIENT"
    answer_text: str = ""
    cited_evidence_ids: List[str] = field(default_factory=list)
    # Timings
    t_query_embedding_ms: float = 0.0
    t_retrieval_ms: float = 0.0
    t_llm_navigation_seconds: float = 0.0
    t_llm_promotion_seconds: float = 0.0
    t_llm_answer_seconds: float = 0.0
    t_decay_ms: float = 0.0
    t_wall_total_seconds: float = 0.0
    # Counts
    n_navigation_steps: int = 0
    n_promote_decisions: int = 0
    n_promote_events: int = 0
    n_demote_events: int = 0
    # Tokens
    tokens_navigation_in: int = 0
    tokens_navigation_out: int = 0
    tokens_promotion_in: int = 0
    tokens_promotion_out: int = 0
    tokens_distilled_context_in: int = 0
    tokens_detailed_context_in: int = 0
    tokens_answer_in: int = 0
    tokens_answer_out: int = 0
    cost_usd: float = 0.0
    final_gov_decision: str = "ALLOW"
    # Bookkeeping
    nav_actions: List[str] = field(default_factory=list)
    distilled_context_node_ids: List[str] = field(default_factory=list)
    detailed_context_node_ids: List[str] = field(default_factory=list)
    promoted_node_ids: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        for k in ("t_query_embedding_ms", "t_retrieval_ms",
                  "t_llm_navigation_seconds", "t_llm_promotion_seconds",
                  "t_llm_answer_seconds", "t_decay_ms",
                  "t_wall_total_seconds", "cost_usd"):
            d[k] = round(d[k], 4)
        return d


PRICES = {
    "gpt_5_4_mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt_5_4":      {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
}


def cost_for(alias: str, in_t: int, out_t: int) -> float:
    p = PRICES.get(alias, PRICES["gpt_5_4_mini"])
    return in_t * p["input"] + out_t * p["output"]


@dataclass
class RunnerCtx:
    method: str
    tier: str
    hierarchy: Dict[str, DualNode]
    parent_child: Dict[str, List[str]]
    parent_map: Dict[str, str]
    by_level: Dict[str, List[str]]
    level_names_sorted: List[str]  # L0..LN
    top_level: str
    embedder: Embedder
    indexes: Dict[str, CosineIndex]  # level -> CosineIndex
    alias_answer: str
    alias_navigator: str
    alias_low: str
    top_k_distilled: int
    max_detailed_load: int
    max_steps: int
    ledger: TokenLedger
    state_log: Optional[StateLog]
    promotion: Optional[PromotionController]
    decay: Optional[DecayController]


def get_index(ctx: RunnerCtx, level: str) -> CosineIndex:
    if level in ctx.indexes:
        return ctx.indexes[level]
    ids = ctx.by_level.get(level, [])
    if not ids:
        ctx.indexes[level] = CosineIndex([], np.zeros((0, ctx.embedder.dim), dtype=np.float32))
        return ctx.indexes[level]
    # MEMONDEMAND_SKIP_L0_EMBED=1: skip ONLY L0 embedding (avoids 207K-node 429); L1/L2 nav still works
    import os as _os_skip
    if level == "L0" and _os_skip.environ.get("MEMONDEMAND_SKIP_L0_EMBED", "0") == "1":
        log.warning("SKIP_L0_EMBED: returning empty L0 cosine index; BM25-only for L0 retrieval")
        ctx.indexes[level] = CosineIndex([], np.zeros((0, ctx.embedder.dim), dtype=np.float32))
        return ctx.indexes[level]
    # Disk cache support
    import os as _os, json as _json, hashlib as _hash
    cache_dir = _os.environ.get("MEMONDEMAND_INDEX_CACHE_DIR", "")
    cache_npy = ""
    cache_meta = ""
    if cache_dir:
        _os.makedirs(cache_dir, exist_ok=True)
        key = _hash.sha1(f"{level}|{ctx.embedder.backend}|{ctx.embedder.dim}|{len(ids)}|{ids[0] if ids else None}|{ids[-1] if ids else None}".encode()).hexdigest()[:16]
        cache_npy = _os.path.join(cache_dir, f"index_{level}_{key}.npy")
        cache_meta = _os.path.join(cache_dir, f"index_{level}_{key}.json")
        if _os.path.exists(cache_npy) and _os.path.exists(cache_meta):
            try:
                with open(cache_meta) as _f: meta = _json.load(_f)
                if meta.get("ids") == ids and meta.get("dim") == ctx.embedder.dim:
                    vecs = np.load(cache_npy).astype(np.float32)
                    log.info("loaded cached index for level %s: %d nodes from %s", level, len(ids), cache_npy)
                    ctx.indexes[level] = CosineIndex(ids, vecs)
                    return ctx.indexes[level]
            except Exception as _e:
                log.warning("index cache load failed for %s: %s", level, _e)
    # Nav context: prefer key_facts (if present) for more precise entity navigation
    texts = [(getattr(ctx.hierarchy[i], "key_facts", None) or "").strip() or (ctx.hierarchy[i].distilled_text or i) for i in ids]
    log.info("indexing %d nodes at level %s ...", len(ids), level)
    vecs = ctx.embedder.encode(texts)
    ctx.indexes[level] = CosineIndex(ids, vecs)
    if cache_dir and cache_npy:
        try:
            np.save(cache_npy, vecs)
            with open(cache_meta, "w") as _f:
                _json.dump({"ids": ids, "dim": int(ctx.embedder.dim), "backend": ctx.embedder.backend}, _f)
            log.info("saved index cache for level %s -> %s", level, cache_npy)
        except Exception as _e:
            log.warning("index cache save failed for %s: %s", level, _e)
    return ctx.indexes[level]


def get_top_level(hierarchy: Dict[str, DualNode]) -> str:
    levels = sorted({n.level for n in hierarchy.values()},
                    key=lambda s: int(s.lstrip("L")))
    return levels[-1] if levels else "L0"


def get_sorted_levels(hierarchy: Dict[str, DualNode]) -> List[str]:
    return sorted({n.level for n in hierarchy.values()},
                  key=lambda s: int(s.lstrip("L")))


# --- Per-method implementations --------------------------------------------


MAX_CITED_PER_QUERY = 5

def _cap_cited(ids, cand_ids):
    """Truncate cited_l0 to MAX_CITED_PER_QUERY items. Preserve order: prefer
    items in cand_ids order (which is retrieval-rank order from final_l0_retrieval)."""
    if not ids:
        return ids
    seen = set()
    out = []
    # First pass: include cited items in cand_ids order (retrieval rank)
    for c in cand_ids:
        if c in ids and c not in seen:
            out.append(c)
            seen.add(c)
            if len(out) >= MAX_CITED_PER_QUERY:
                return out
    # Tail: any cited items not in cand_ids (shouldn't happen but defensive)
    for c in ids:
        if c not in seen:
            out.append(c)
            seen.add(c)
            if len(out) >= MAX_CITED_PER_QUERY:
                return out
    return out

def run_b_flat(ctx: RunnerCtx, qid: str, query: str) -> QueryRecord:
    rec = QueryRecord(query_id=qid, query_text=query,
                      method=ctx.method, tier=ctx.tier)
    t_wall = time.time()
    # Embed query
    t0 = time.time()
    q_vec = ctx.embedder.encode([query])[0]
    rec.t_query_embedding_ms = (time.time() - t0) * 1000

    # Top-k over L0 distilled
    t0 = time.time()
    l0 = get_index(ctx, "L0")
    hits = l0.search(q_vec, top_k=ctx.top_k_distilled)
    rec.t_retrieval_ms = (time.time() - t0) * 1000

    # Build context (L0 → use detailed_text)
    ctx_items: List[Tuple[str, str, str]] = []
    distilled_tok = 0
    detailed_tok = 0
    for nid, _sim in hits:
        n = ctx.hierarchy[nid]
        # B_flat over L0 — use detailed_text (raw content) — matches
        # B1FlatRAG-style retrieval, but we still cap text length.
        text = _node_text_for_answer(n)
        ctx_items.append((nid, n.level, text))
        detailed_tok += n.detailed_tokens or 0

    user = ANSWER_USER_TMPL.format(query=query, context_block=format_context_block(ctx_items))
    answer, in_t, out_t, wall = call_llm(
        ctx.alias_answer, ANSWER_SYSTEM, user,
        max_tokens=400, temperature=0.0,
        ledger=ctx.ledger, phase=PHASE_FINAL_ANSWER, query_id=qid,
    )
    rec.t_llm_answer_seconds = wall
    rec.tokens_answer_in = in_t
    rec.tokens_answer_out = out_t
    rec.tokens_detailed_context_in = detailed_tok

    cand_ids = [nid for nid, _, _ in ctx_items]
    cited = parse_cited(answer, cand_ids)
    # If model didn't cite explicitly, treat all shown candidates as cited
    # (still bounded by what we showed — no hallucinated ids).
    if not cited:
        cited = cand_ids

    rec.final_action = "ANSWER" if answer and "INSUFFICIENT" not in answer.upper() else "ANSWER"
    rec.answer_text = strip_cited(answer)
    rec.cited_evidence_ids = _cap_cited(cited, cand_ids)
    rec.distilled_context_node_ids = cand_ids
    rec.detailed_context_node_ids = cand_ids
    rec.t_wall_total_seconds = time.time() - t_wall
    rec.cost_usd = cost_for(ctx.alias_answer, in_t, out_t)
    return rec


def run_b_hier_bfs(ctx: RunnerCtx, qid: str, query: str) -> QueryRecord:
    """Top-down BFS over the depth-K hierarchy (same logic for fixed/dynamic)."""
    rec = QueryRecord(query_id=qid, query_text=query,
                      method=ctx.method, tier=ctx.tier)
    t_wall = time.time()
    t0 = time.time()
    q_vec = ctx.embedder.encode([query])[0]
    rec.t_query_embedding_ms = (time.time() - t0) * 1000

    # Descend each level, picking top-k per level (k_per_level decreasing).
    t0 = time.time()
    levels_desc = list(reversed(ctx.level_names_sorted))  # [L4, L3, L2, L1, L0]
    # k per level: 4 at top, then 8 at L1, then top_k_distilled at L0
    chosen_ids: List[str] = []
    parent_pool: List[str] = list(ctx.by_level.get(levels_desc[0], []))
    accumulated_hits: List[Tuple[str, str, float]] = []  # (id, level, sim)

    for li, lvl in enumerate(levels_desc):
        idx = get_index(ctx, lvl)
        if not idx.ids:
            continue
        k = max(2, ctx.top_k_distilled // 2) if lvl != "L0" else ctx.top_k_distilled
        # filter by children of last-chosen
        pool = parent_pool if parent_pool else idx.ids
        hits = idx.search(q_vec, top_k=k, filter_ids=pool)
        for nid, sim in hits:
            accumulated_hits.append((nid, lvl, sim))
        # next pool = children of these hits
        parent_pool = []
        for nid, _ in hits:
            parent_pool.extend(ctx.parent_child.get(nid, []))
    rec.t_retrieval_ms = (time.time() - t0) * 1000

    # Build context: prefer L0 detailed; upper levels distilled
    # Sort by (level ascending L0 first, then by sim desc) and cap to ~top_k_distilled L0 + upper summaries
    ctx_items: List[Tuple[str, str, str]] = []
    detailed_tok = 0
    distilled_tok = 0
    seen: Set[str] = set()
    # L0 first
    l0_hits = [(nid, sim) for nid, lvl, sim in accumulated_hits if lvl == "L0"]
    l0_hits.sort(key=lambda x: -x[1])
    for nid, _ in l0_hits[: ctx.top_k_distilled]:
        if nid in seen:
            continue
        seen.add(nid)
        n = ctx.hierarchy[nid]
        _ans_mode_cur = _os_ansmode.environ.get("MEMONDEMAND_ANSWER_MODE", "")
        _n_l0_so_far = sum(1 for _x in ctx_items if _x[1] == "L0")
        _top_k_det = int(_os_ansmode.environ.get("MEMONDEMAND_MIXED_TOP_K", "5"))
        _is_top_det = (_ans_mode_cur == "mixed" and _n_l0_so_far < _top_k_det)
        ctx_items.append((nid, n.level, _node_text_for_answer(n, is_top_detail=_is_top_det)))
        detailed_tok += n.detailed_tokens or 0
    # Upper-level summaries (cap 4)
    upper = [(nid, lvl, sim) for nid, lvl, sim in accumulated_hits if lvl != "L0"]
    upper.sort(key=lambda x: (int(x[1].lstrip("L")), -x[2]))
    for nid, lvl, _ in upper[:4]:
        if nid in seen:
            continue
        seen.add(nid)
        n = ctx.hierarchy[nid]
        ctx_items.append((nid, n.level, (n.distilled_text or "")[:600]))
        distilled_tok += n.distilled_tokens or 0

    user = ANSWER_USER_TMPL.format(query=query, context_block=format_context_block(ctx_items))
    answer, in_t, out_t, wall = call_llm(
        ctx.alias_answer, ANSWER_SYSTEM, user,
        max_tokens=400, temperature=0.0,
        ledger=ctx.ledger, phase=PHASE_FINAL_ANSWER, query_id=qid,
    )
    rec.t_llm_answer_seconds = wall
    rec.tokens_answer_in = in_t
    rec.tokens_answer_out = out_t
    rec.tokens_detailed_context_in = detailed_tok
    rec.tokens_distilled_context_in = distilled_tok

    cand_ids = [nid for nid, _, _ in ctx_items]
    cited = parse_cited(answer, cand_ids)

    if not cited:
        cited_l0 = [nid for nid in cand_ids if ctx.hierarchy[nid].level == "L0"]
    else:
        cited_l0 = [nid for nid in cited if ctx.hierarchy.get(nid) is not None
                    and ctx.hierarchy[nid].level == "L0"]
        if not cited_l0:
            cited_l0 = [nid for nid in cand_ids if ctx.hierarchy[nid].level == "L0"]

    rec.final_action = "ANSWER"
    rec.answer_text = strip_cited(answer)
    rec.cited_evidence_ids = _cap_cited(cited_l0, cand_ids)
    rec.distilled_context_node_ids = [c for c in cand_ids if ctx.hierarchy[c].level != "L0"]
    rec.detailed_context_node_ids = [c for c in cand_ids if ctx.hierarchy[c].level == "L0"]
    rec.t_wall_total_seconds = time.time() - t_wall
    rec.cost_usd = cost_for(ctx.alias_answer, in_t, out_t)
    return rec


def expand_to_l0(ctx: RunnerCtx, ids: Sequence[str]) -> List[str]:
    """Map cited node_ids to underlying L0 dsids. L0 nodes map to themselves."""
    out: List[str] = []
    seen: Set[str] = set()
    for nid in ids:
        n = ctx.hierarchy.get(nid)
        if n is None:
            continue
        if n.level == "L0":
            if nid not in seen:
                out.append(nid); seen.add(nid)
        else:
            for sid in n.source_evidence_ids:
                if sid not in seen:
                    out.append(sid); seen.add(sid)
    return out


def collect_descendant_l0(ctx: RunnerCtx, ids: Sequence[str]) -> List[str]:
    """BFS over parent_child to gather all L0 leaves under the given node ids.
    L0 nodes map to themselves. Order is preserved by first-visit."""
    out: List[str] = []
    seen: Set[str] = set()
    stack: List[str] = list(ids)
    while stack:
        nid = stack.pop(0)
        n = ctx.hierarchy.get(nid)
        if n is None:
            continue
        if n.level == "L0":
            if nid not in seen:
                out.append(nid); seen.add(nid)
            continue
        kids = ctx.parent_child.get(nid, [])
        if kids:
            stack.extend(kids)
        else:
            # No children but not L0 - fall back to source_evidence_ids
            for sid in n.source_evidence_ids:
                sn = ctx.hierarchy.get(sid)
                if sn is not None and sn.level == "L0" and sid not in seen:
                    out.append(sid); seen.add(sid)
    return out


def final_l0_retrieval(ctx: RunnerCtx, q_vec, seed_ids: Sequence[str],
                       k: int, query_text: str = "") -> List[Tuple[str, float]]:
    """Gather descendant leaf nodes and return the highest-scoring candidates."""
    import os as _os
    if _os.environ.get("MEMONDEMAND_L0_RETRIEVAL", "") == "bm25":

        if _os.environ.get("MEMONDEMAND_DENSE_RERANK", "0") == "1":
            try:
                wide_k = int(_os.environ.get("MEMONDEMAND_BM25_WIDE_K", "50"))
            except Exception:
                wide_k = 50
            return _bm25_then_dense_rerank(ctx, query_text, k=k, wide_k=wide_k)
        # Exp-A3: cluster-aware BM25 — filter by navigation frontier cluster
        if _os.environ.get("MEMONDEMAND_CLUSTER_FILTER", "0") == "1" and seed_ids:
            # Collect all L0 descendants of the frontier seed nodes (navigation cluster)
            cluster_l0: list = []
            _seen_cl: set = set()
            _stack = list(seed_ids)
            while _stack:
                _nid = _stack.pop()
                if _nid in _seen_cl: continue
                _seen_cl.add(_nid)
                _n = ctx.hierarchy.get(_nid)
                if _n is None: continue
                if _n.level == "L0":
                    cluster_l0.append(_nid)
                else:
                    _stack.extend(ctx.parent_child.get(_nid, []))
            # BM25 global wide retrieval, then filter to cluster + fallback
            _wide_k = int(_os.environ.get("MEMONDEMAND_BM25_WIDE_K", "50"))
            bm_wide = _bm25_top_k(ctx, query_text, _wide_k, filter_ids=None)
            cluster_set = set(cluster_l0)
            in_cluster  = [(did, sc) for did, sc in bm_wide if did in cluster_set]
            out_cluster = [(did, sc) for did, sc in bm_wide if did not in cluster_set]
            merged = in_cluster + out_cluster
            return merged[:k]
        # Exp-C1: Query Expansion — append L1/L2 distilled summary keywords to query
        if _os.environ.get("MEMONDEMAND_QUERY_EXPAND", "0") == "1" and seed_ids:
            _STOP = {'the','a','an','and','or','of','in','to','for','with','on','is','are',
                     'that','this','at','by','from','as','be','was','were','it','its',
                     'have','has','had','not','but','they','we','our','their','all','can',
                     'will','which','who','how','when','where','what','if','so','do','does'}
            _kw_pool = []
            for _sid in seed_ids[:8]:
                _nd = ctx.hierarchy.get(_sid)
                if _nd is None: continue
                # Walk up to L1/L2 for summary text
                _cur = _sid
                for _ in range(3):
                    _par = ctx.parent_map.get(_cur)
                    if not _par: break
                    _par_nd = ctx.hierarchy.get(_par)
                    if _par_nd and _par_nd.level in ('L1','L2') and _par_nd.distilled_text:
                        _words = [w.lower().strip(',.:;()[]"')
                                  for w in _par_nd.distilled_text.split()
                                  if len(w) > 4 and w.lower().strip(',.:;()[]"') not in _STOP]
                        _kw_pool.extend(_words[:10])
                    _cur = _par or _cur
                # Also get keywords from the node itself if L1
                if _nd.level in ('L1','L2') and _nd.distilled_text:
                    _words2 = [w.lower().strip(',.:;()[]"')
                               for w in _nd.distilled_text.split()
                               if len(w) > 4 and w.lower().strip(',.:;()[]"') not in _STOP]
                    _kw_pool.extend(_words2[:10])
            # Frequency-based top-8 (prefer longer, more specific terms)
            from collections import Counter as _Ctr
            _freq = _Ctr(_kw_pool)
            _top_kw = [w for w, _ in sorted(_freq.items(), key=lambda x: (-len(x[0]), -x[1]))[:8]]
            expanded_query = query_text + ' ' + ' '.join(_top_kw)
            return _bm25_top_k(ctx, expanded_query, k, filter_ids=None)

        # Exp-C2: RRF fusion — BM25 top-50 UNION Dense top-50, merged by RRF
        if _os.environ.get("MEMONDEMAND_RRF", "0") == "1":
            _rrf_wide = int(_os.environ.get("MEMONDEMAND_RRF_WIDE_K", "50"))
            _rrf_k = 60  # RRF constant
            # BM25 ranked list
            bm_hits = _bm25_top_k(ctx, query_text, _rrf_wide, filter_ids=None)
            bm_rank = {did: i for i, (did, _) in enumerate(bm_hits)}
            # Dense ranked list (minilm global L0)
            l0_idx = get_index(ctx, "L0")
            dense_hits = l0_idx.search(q_vec, top_k=_rrf_wide)
            dense_rank = {did: i for i, (did, _) in enumerate(dense_hits)}
            # Union of all candidate IDs
            all_ids = list(dict.fromkeys([did for did,_ in bm_hits] + [did for did,_ in dense_hits]))
            # RRF score
            rrf_scores = {}
            for did in all_ids:
                br = bm_rank.get(did, _rrf_wide + _rrf_k)
                dr = dense_rank.get(did, _rrf_wide + _rrf_k)
                rrf_scores[did] = 1.0/(br + _rrf_k) + 1.0/(dr + _rrf_k)
            sorted_ids = sorted(all_ids, key=lambda d: -rrf_scores[d])
            # Return as (id, score) tuples
            return [(did, rrf_scores[did]) for did in sorted_ids[:k]]

        # Exp-C4: te3-large RRF — BM25 top-50 UNION te3-large cosine top-50
        if _os.environ.get("MEMONDEMAND_TE3_RRF", "0") == "1":
            _load_te3_index_once()
            if _TE3_ARR is not None and len(_TE3_ARR) > 0:
                _rrf_wide_c4 = int(_os.environ.get("MEMONDEMAND_RRF_WIDE_K", "50"))
                _rrf_c4 = 60  # RRF constant k
                # BM25 ranked list (id only, for RRF)
                _bm25_hits_c4 = _bm25_top_k(ctx, query_text, _rrf_wide_c4, filter_ids=None)
                _bm25_ids_c4 = [_id for _id, _ in _bm25_hits_c4]
                # te3-large cosine ranked list
                _q_c4 = _np_te3.array(q_vec, dtype=_np_te3.float32)
                _q_c4 = _q_c4 / (_np_te3.linalg.norm(_q_c4) + 1e-9)
                _scores_c4 = _TE3_ARR @ _q_c4          # (N,) cosine scores
                _idxs_c4 = _np_te3.argpartition(_scores_c4, -_rrf_wide_c4)[-_rrf_wide_c4:]
                _idxs_c4 = _idxs_c4[_np_te3.argsort(_scores_c4[_idxs_c4])[::-1]]
                _dense_ids_c4 = [_TE3_IDS[_i] for _i in _idxs_c4]
                # RRF merge
                _rrf_scores_c4: dict = {}
                for _rank, _did in enumerate(_bm25_ids_c4, 1):
                    _rrf_scores_c4[_did] = _rrf_scores_c4.get(_did, 0.0) + 1.0 / (_rrf_c4 + _rank)
                for _rank, _did in enumerate(_dense_ids_c4, 1):
                    _rrf_scores_c4[_did] = _rrf_scores_c4.get(_did, 0.0) + 1.0 / (_rrf_c4 + _rank)
                _rrf_ranked_c4 = sorted(_rrf_scores_c4, key=lambda _d: -_rrf_scores_c4[_d])
                return [(_did, _rrf_scores_c4[_did]) for _did in _rrf_ranked_c4[:k]]

        # BM25 global top-k, ignore hierarchy seed pool entirely
        return _bm25_top_k(ctx, query_text, k, filter_ids=None)
    l0_pool = collect_descendant_l0(ctx, seed_ids)
    if not l0_pool:
        # Fallback: global L0 top-k (rare; only when frontier has no L0 lineage)
        idx = get_index(ctx, "L0")
        return idx.search(q_vec, top_k=k)
    idx = get_index(ctx, "L0")
    return idx.search(q_vec, top_k=k, filter_ids=l0_pool)


def run_llm_nav(ctx: RunnerCtx, qid: str, query: str, query_idx: int,
                use_promotion: bool) -> QueryRecord:
    """LLM navigator (B_llm_nav and V5)."""
    rec = QueryRecord(query_id=qid, query_text=query,
                      method=ctx.method, tier=ctx.tier)
    t_wall = time.time()
    t0 = time.time()
    q_vec = ctx.embedder.encode([query])[0]
    rec.t_query_embedding_ms = (time.time() - t0) * 1000

    # Start at top level
    t0 = time.time()
    import os as _os_dir
    _retr_mode = _os_dir.environ.get("MEMONDEMAND_L0_RETRIEVAL", "")
    if _retr_mode == "bm25_guided":
        # Direction A: use BM25-guided L1 frontier as starting point
        frontier_hits = _bm25_guided_l1_frontier(ctx, query, bm25_top_k=50, max_l1=8)
        if not frontier_hits:
            # Fallback to top-level cosine
            top_idx = get_index(ctx, ctx.top_level)
            frontier_hits = top_idx.search(q_vec, top_k=8)
    else:
        top_idx = get_index(ctx, ctx.top_level)
        frontier_hits = top_idx.search(q_vec, top_k=8)
    rec.t_retrieval_ms += (time.time() - t0) * 1000

    visible_frontier: List[Tuple[str, float]] = list(frontier_hits)
    visited_distilled: Dict[str, str] = {}  # node_id -> distilled_text
    promoted_detailed: Dict[str, str] = {}  # node_id -> detailed_text
    answered_evidence_ids: List[str] = []  # final cited candidate pool
    prior_actions: List[str] = []

    decision: Dict[str, Any] = {"action": "ANSWER"}
    step = 0
    for step in range(ctx.max_steps):
        if not visible_frontier:
            decision = {"action": "ANSWER", "chosen_node_ids": [],
                        "rationale": "empty_frontier_forced_answer"}
            break
        # Cap frontier to 16 items
        front = visible_frontier[:16]
        # Build prompt
        block_lines = []
        for nid, sim in front:
            n = ctx.hierarchy.get(nid)
            if n is None:
                continue
            visited_distilled[nid] = n.distilled_text or ""
            block_lines.append(
                f"- [{nid}] ({n.level}) sim={sim:.3f} {(n.distilled_text or '')[:200]}"
            )
        frontier_block = "\n".join(block_lines)
        user = NAV_USER_TMPL.format(
            query=query, step=step + 1, max_steps=ctx.max_steps,
            n=len(front),
            prior_actions=",".join(prior_actions[-3:]) or "(none)",
            frontier_block=frontier_block,
        )

        nav_text, in_t, out_t, wall = call_llm(
            ctx.alias_navigator, NAV_SYSTEM, user,
            max_tokens=200, temperature=0.0,
            ledger=ctx.ledger, phase=PHASE_RETRIEVAL, query_id=qid,
        )
        rec.t_llm_navigation_seconds += wall
        rec.tokens_navigation_in += in_t
        rec.tokens_navigation_out += out_t
        rec.cost_usd += cost_for(ctx.alias_navigator, in_t, out_t)

        decision = parse_nav_json(nav_text)
        action = str(decision.get("action", "ANSWER")).upper().strip()

        import os as _os_fd
        if rec.n_navigation_steps == 0 and _os_fd.environ.get("MEMONDEMAND_FORCE_FIRST_DESCEND", "0") == "1":
            _front_levels = set()
            for nid, _ in front:
                _n = ctx.hierarchy.get(nid)
                if _n is not None:
                    _front_levels.add(_n.level)
            _all_l0 = _front_levels and all(lvl == "L0" for lvl in _front_levels)
            if action == "ANSWER" and not _all_l0:
                # Coerce to DESCEND on top-2 candidates
                action = "DESCEND"
                if not decision.get("chosen_node_ids"):
                    decision["chosen_node_ids"] = [nid for nid, _ in front[:2]]

        if action not in {"DESCEND", "LATERAL", "ANSWER", "STOP_INSUFFICIENT"}:
            action = "ANSWER"
        # Hard cap: after 2 navigation actions (DESCEND/LATERAL) force ANSWER.
        import os as _os_max
        try:
            MAX_NAV_STEPS = int(_os_max.environ.get("MEMONDEMAND_MAX_NAV_STEPS", "2"))
        except Exception:
            MAX_NAV_STEPS = 2
        if rec.n_navigation_steps >= MAX_NAV_STEPS and action in {"DESCEND", "LATERAL"}:
            # Force ANSWER if any visible/visited evidence; else STOP.
            if visible_frontier or visited_distilled or promoted_detailed:
                action = "ANSWER"
            else:
                action = "STOP_INSUFFICIENT"
        # Disallow STOP_INSUFFICIENT if any evidence is already visible. Keep
        # the parsed decision in sync because final_action is read from it.
        if action == "STOP_INSUFFICIENT":
            if visible_frontier or visited_distilled or promoted_detailed:
                action = "ANSWER"
                decision["action"] = "ANSWER"
                decision["chosen_node_ids"] = []

        if (
            action == "STOP_INSUFFICIENT"
            and os.environ.get("MEMONDEMAND_FORCE_ANSWER", "0") == "1"
        ):
            action = "ANSWER"
            decision["action"] = "ANSWER"
            if not decision.get("chosen_node_ids") and front:
                decision["chosen_node_ids"] = [nid for nid, _ in front[:3]]
        chosen = [c for c in (decision.get("chosen_node_ids") or [])
                  if c in ctx.hierarchy]
        prior_actions.append(action)
        rec.nav_actions.append(action)
        rec.n_navigation_steps += 1

        if action == "ANSWER":
            break
        if action == "STOP_INSUFFICIENT":
            decision = {"action": "STOP_INSUFFICIENT", "chosen_node_ids": [],
                        "rationale": decision.get("rationale", "navigator_stop")}
            break

        if action == "DESCEND":
            if not chosen:
                # fallback: take top 2 from frontier
                chosen = [nid for nid, _ in front[:2]]

            # Exp-B1: multi-path — expand top-2 frontier nodes (not just the chosen one)
            import os as _os_mp
            if _os_mp.environ.get("MEMONDEMAND_MULTIPATH_TOP2", "0") == "1":
                # Always expand top-2 frontier nodes regardless of LLM choice
                top2_ids = [nid for nid, _ in front[:2]]
                chosen = list(dict.fromkeys(chosen + top2_ids))  # chosen first, then top2 extras

            # Expand to children of chosen (or keep chosen if leaf)
            new_frontier_ids: List[str] = []
            for c in chosen:
                kids = ctx.parent_child.get(c, [])
                if kids:
                    new_frontier_ids.extend(kids)
                else:
                    # leaf — keep the chosen node itself
                    new_frontier_ids.append(c)
            new_frontier_ids = list(dict.fromkeys(new_frontier_ids))
            if not new_frontier_ids:
                continue
            t0 = time.time()
            child_level = ctx.hierarchy[new_frontier_ids[0]].level
            import os as _os_dr
            _descend_ranking = _os_dr.environ.get("MEMONDEMAND_DESCEND_RANKING", "cosine")
            if _descend_ranking == "bm25":

                # For L0 children, use BM25; otherwise fall back to cosine
                if child_level == "L0":
                    bm_hits = _bm25_top_k(ctx, query, 16, filter_ids=new_frontier_ids)
                    ranked = bm_hits
                else:
                    idx = get_index(ctx, child_level)
                    ranked = idx.search(q_vec, top_k=16, filter_ids=new_frontier_ids)
            else:
                idx = get_index(ctx, child_level)
                ranked = idx.search(q_vec, top_k=16, filter_ids=new_frontier_ids)
            rec.t_retrieval_ms += (time.time() - t0) * 1000
            visible_frontier = ranked

            if use_promotion and ctx.promotion is not None:
                tp = time.time()
                # Candidates: top L0 nodes in the new frontier + any
                # chosen nodes that are already L0.
                cand_pool: List[str] = []
                for nid, _ in ranked[: ctx.max_detailed_load * 2]:
                    if ctx.hierarchy[nid].level == "L0":
                        cand_pool.append(nid)
                for c in chosen:
                    if ctx.hierarchy[c].level == "L0" and c not in cand_pool:
                        cand_pool.append(c)
                # Fairness: a node must not have been "marked used" this
                # query yet (mark_detail_used is only called AFTER answer).
                cand_pool = [c for c in cand_pool
                             if ctx.hierarchy[c].last_used_query_idx < query_idx]
                if cand_pool:
                    decisions = ctx.promotion.decide(
                        query=query,
                        candidate_node_ids=cand_pool[: ctx.max_detailed_load],
                        query_idx=query_idx,
                        query_id=qid,
                        alias=ctx.alias_low,
                        ledger=ctx.ledger,
                    )
                    rec.n_promote_decisions += len(decisions)
                    counts = ctx.promotion.apply_decisions(
                        decisions, query_idx, state_log=ctx.state_log,
                    )
                    rec.n_promote_events += counts.get("PROMOTE", 0)

                rec.t_llm_promotion_seconds += time.time() - tp
            continue

        if action == "LATERAL":
            if not chosen:
                chosen = [nid for nid, _ in front[:1]]
            # siblings = parent's other children
            sib_ids: List[str] = []
            for c in chosen:
                parent = ctx.parent_map.get(c)
                if parent:
                    for s in ctx.parent_child.get(parent, []):
                        if s != c:
                            sib_ids.append(s)
            sib_ids = list(dict.fromkeys(sib_ids))
            if not sib_ids:
                continue
            t0 = time.time()
            sib_level = ctx.hierarchy[sib_ids[0]].level
            idx = get_index(ctx, sib_level)
            ranked = idx.search(q_vec, top_k=16, filter_ids=sib_ids)
            rec.t_retrieval_ms += (time.time() - t0) * 1000
            visible_frontier = ranked
            continue

    # ---- build final context + answer (or honor STOP_INSUFFICIENT) ----

    final_action = str(decision.get("action", "ANSWER")).upper().strip()
    if final_action not in {"ANSWER", "STOP_INSUFFICIENT"}:
        # Out of steps with no terminal — default to ANSWER if any evidence
        final_action = "ANSWER" if (visited_distilled or promoted_detailed) else "STOP_INSUFFICIENT"
    rec.final_action = final_action

    if final_action == "ANSWER":

        chosen_ans = [c for c in (decision.get("chosen_node_ids") or [])
                      if c in ctx.hierarchy]
        seed_ids: List[str] = []
        seen_seed: Set[str] = set()
        for nid in chosen_ans + [n for n, _ in visible_frontier] + list(promoted_detailed.keys()):
            if nid not in seen_seed:
                seed_ids.append(nid); seen_seed.add(nid)

        final_l0_hits = final_l0_retrieval(ctx, q_vec, seed_ids, k=ctx.top_k_distilled, query_text=query)

        import os as _os_bm25p
        if (use_promotion and ctx.promotion is not None
                and _os_bm25p.environ.get("MEMONDEMAND_L0_RETRIEVAL", "") in ("bm25", "bm25_promo")):
            _tp_bm25promo = time.time()

            _bm25_cand_pool = [
                nid for nid, _ in final_l0_hits[: ctx.top_k_distilled]
                if (ctx.hierarchy.get(nid) is not None
                    and ctx.hierarchy[nid].level == "L0"
                    and ctx.hierarchy[nid].last_used_query_idx < query_idx)
            ]
            if _bm25_cand_pool:
                _bm25_decisions = ctx.promotion.decide(
                    query=query,
                    candidate_node_ids=_bm25_cand_pool[: ctx.max_detailed_load],
                    query_idx=query_idx,
                    query_id=qid,
                    alias=ctx.alias_low,
                    ledger=ctx.ledger,
                )
                rec.n_promote_decisions += len(_bm25_decisions)
                _bm25_counts = ctx.promotion.apply_decisions(
                    _bm25_decisions, query_idx, state_log=ctx.state_log,
                )
                # Exp-C3: promotion-driven target re-retrieval
                # For each accepted promotion, BM25 within cluster siblings → top-5 supplement
                import os as _os_c3
                if _os_c3.environ.get("MEMONDEMAND_PROMO_RETRIEVAL", "0") == "1":
                    _c3_extra_ids: list = []
                    _c3_seen: set = {did for did, _ in final_l0_hits}
                    for _dec in _bm25_decisions:
                        _pnid = getattr(_dec, "node_id", None) or ""
                        if not _pnid or getattr(_dec, "decision", "") != "PROMOTE": continue
                        # Get L0 siblings (parent's children) of promoted node
                        _ppid = ctx.parent_map.get(_pnid, "")
                        _siblings = ctx.parent_child.get(_ppid, []) if _ppid else []
                        _sib_l0 = [s for s in _siblings if s not in _c3_seen
                                   and ctx.hierarchy.get(s) is not None
                                   and ctx.hierarchy[s].level == "L0"]
                        if not _sib_l0: continue
                        # BM25 within siblings
                        _sib_hits = _bm25_top_k(ctx, query, 5, filter_ids=_sib_l0)
                        for _sh_id, _sh_sc in _sib_hits:
                            if _sh_id not in _c3_seen:
                                _c3_extra_ids.append((_sh_id, _sh_sc))
                                _c3_seen.add(_sh_id)
                    # Append supplemental hits to final_l0_hits (promotion cluster boost)
                    if _c3_extra_ids:
                        final_l0_hits = list(final_l0_hits) + _c3_extra_ids[:10]
                rec.n_promote_events += _bm25_counts.get("PROMOTE", 0)

                # Do NOT load detailed_text into promoted_detailed.
            rec.t_llm_promotion_seconds += time.time() - _tp_bm25promo

        ctx_items: List[Tuple[str, str, str]] = []
        detailed_tok = 0
        distilled_tok = 0
        # 1) Final L0 retrieval — these are the primary evidence
        l0_in_ctx: Set[str] = set()
        for nid, _sim in final_l0_hits:
            n = ctx.hierarchy.get(nid)
            if n is None or n.level != "L0":
                continue
            _ans_mode_cur = _os_ansmode.environ.get("MEMONDEMAND_ANSWER_MODE", "")
            _n_l0_so_far = sum(1 for _x in ctx_items if _x[1] == "L0")
            _top_k_det = int(_os_ansmode.environ.get("MEMONDEMAND_MIXED_TOP_K", "5"))
            _is_top_det = (_ans_mode_cur == "mixed" and _n_l0_so_far < _top_k_det)
            ctx_items.append((nid, n.level, _node_text_for_answer(n, is_top_detail=_is_top_det)))
            detailed_tok += n.detailed_tokens or 0
            l0_in_ctx.add(nid)

        import os as _os_sec2
        _skip_promoted_in_ctx = _os_sec2.environ.get("MEMONDEMAND_L0_RETRIEVAL", "") in ("bm25", "bm25_promo")
        if not _skip_promoted_in_ctx:
            for nid, txt in list(promoted_detailed.items())[: ctx.max_detailed_load]:
                if nid in l0_in_ctx:
                    continue
                n = ctx.hierarchy.get(nid)
                if n is None:
                    continue
                ctx_items.append((nid, n.level, (txt or "")[:2000]))
                if n.level == "L0":
                    detailed_tok += n.detailed_tokens or 0
                    l0_in_ctx.add(nid)
                else:
                    distilled_tok += n.distilled_tokens or 0
        # 3) Up to a few upper-level summaries for context (from visible frontier)
        upper_added = 0
        for nid, _sim in visible_frontier[:6]:
            if upper_added >= 4:
                break
            if nid in l0_in_ctx or any(c[0] == nid for c in ctx_items):
                continue
            n = ctx.hierarchy.get(nid)
            if n is None or n.level == "L0":
                continue
            ctx_items.append((nid, n.level, (n.distilled_text or "")[:500]))
            distilled_tok += n.distilled_tokens or 0
            upper_added += 1
        # 4) Top-up: backfill from visited distilled if we have <4 items
        if len(ctx_items) < 4:
            for nid, txt in list(visited_distilled.items())[:6]:
                if any(c[0] == nid for c in ctx_items):
                    continue
                n = ctx.hierarchy.get(nid)
                if n is None:
                    continue
                ctx_items.append((nid, n.level, (txt or "")[:500]))
                distilled_tok += n.distilled_tokens or 0

        # dual_smart mode: replace format_context_block with dual_smart formatter
        import os as _os_dsm
        _use_dual_smart = _os_dsm.environ.get("MEMONDEMAND_ANSWER_MODE", "") == "dual_smart"
        if _use_dual_smart:
            _upper_items = [(nid, lvl, txt) for nid, lvl, txt in ctx_items if lvl != "L0"]
            _context_block = _format_dual_smart_context(
                final_l0_hits, _upper_items, ctx.hierarchy
            )
        cand_ids = [nid for nid, _, _ in ctx_items]
        import os as _os_v6dm
        _use_v6 = _os_v6dm.environ.get("MEMONDEMAND_V6_DUAL_MEM", "0") == "1"
        _use_v7 = _os_v6dm.environ.get("MEMONDEMAND_V7_ENTITY", "0") == "1"
        _use_v61 = _os_v6dm.environ.get("MEMONDEMAND_V61", "0") == "1"
        _ans_mode_v8 = _os_v6dm.environ.get("MEMONDEMAND_ANSWER_MODE", "")
        if _ans_mode_v8 == "top5_full_only":
            _ans_sys = ANSWER_SYSTEM_V8_TOP5
            _ans_tmpl = ANSWER_USER_V8_TOP5
        elif _use_v7:
            _ans_sys = ANSWER_SYSTEM_V7
            _ans_tmpl = ANSWER_USER_V7
        elif _use_v61:
            _ans_sys = ANSWER_SYSTEM_V61
            _ans_tmpl = ANSWER_USER_V61
        elif _use_v6:
            _ans_sys = ANSWER_SYSTEM_V6
            _ans_tmpl = ANSWER_USER_V6
        else:
            _ans_sys = ANSWER_SYSTEM
            _ans_tmpl = ANSWER_USER_TMPL
        if _use_dual_smart:
            user = _ans_tmpl.format(query=query, context_block=_context_block)
        else:
            user = _ans_tmpl.format(query=query, context_block=format_context_block(ctx_items))
        _is_long_prompt = (_use_v7 or _use_v61 or _ans_mode_v8 == "top5_full_only")
        _max_out = 700 if _is_long_prompt else (600 if _use_v6 else 400)
        answer, in_t, out_t, wall = call_llm(
            ctx.alias_answer, _ans_sys, user,
            max_tokens=_max_out, temperature=0.0,
            ledger=ctx.ledger, phase=PHASE_FINAL_ANSWER, query_id=qid,
        )
        rec.t_llm_answer_seconds = wall
        rec.tokens_answer_in = in_t
        rec.tokens_answer_out = out_t
        rec.tokens_detailed_context_in = detailed_tok
        rec.tokens_distilled_context_in = distilled_tok
        rec.cost_usd += cost_for(ctx.alias_answer, in_t, out_t)

        cited = parse_cited(answer, cand_ids)
        if cited:
            cited_l0 = [nid for nid in cited if ctx.hierarchy.get(nid) is not None
                        and ctx.hierarchy[nid].level == "L0"]
        else:
            cited_l0 = []
        if not cited_l0:
            # Fallback: every L0 node actually shown in the answer context
            cited_l0 = [nid for nid in cand_ids
                        if ctx.hierarchy.get(nid) is not None
                        and ctx.hierarchy[nid].level == "L0"]
        rec.answer_text = strip_cited(answer)
        rec.cited_evidence_ids = _cap_cited(cited_l0, cand_ids)
        rec.distilled_context_node_ids = [c for c in cand_ids if ctx.hierarchy[c].level != "L0"]
        rec.detailed_context_node_ids = [c for c in cand_ids if ctx.hierarchy[c].level == "L0"]
        rec.promoted_node_ids = list(promoted_detailed.keys())

        # V5: mark used for promotion controller (after answer)
        if use_promotion and ctx.promotion is not None:
            for nid in promoted_detailed.keys():
                ctx.promotion.mark_detail_used(nid, query_idx,
                                               query_id=qid,
                                               state_log=ctx.state_log)
    else:
        # STOP_INSUFFICIENT: emit empty answer, no citations
        rec.answer_text = ""
        rec.cited_evidence_ids = []
        rec.distilled_context_node_ids = []
        rec.detailed_context_node_ids = []
        rec.promoted_node_ids = []

    # V5: decay cycle every query (only if promotion enabled)
    if use_promotion and ctx.decay is not None:
        tdc = time.time()
        demote_ids, reasons, keep_scores = ctx.decay.select_for_demotion(
            ctx.hierarchy, query_idx,
        )
        ctx.decay.apply_demotions(
            ctx.hierarchy, demote_ids, query_idx,
            query_id=qid, state_log=ctx.state_log,
            keep_scores=keep_scores, reasons=reasons,
        )
        rec.n_demote_events = len(demote_ids)
        rec.t_decay_ms = (time.time() - tdc) * 1000

    rec.t_wall_total_seconds = time.time() - t_wall
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(VALID_METHODS))
    ap.add_argument("--hierarchy", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--query_order_seed", type=int, default=20260608)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--alias_answer", default="gpt_5_4")
    ap.add_argument("--alias_navigator", default="gpt_5_4")
    ap.add_argument("--alias_low", default="gpt_5_4_mini")
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--top_k_distilled", type=int, default=8)
    ap.add_argument("--max_detailed_load", type=int, default=6)
    ap.add_argument("--promotion_budget", type=int, default=20)
    ap.add_argument("--decay_window", type=int, default=15)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--checkpoint_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_smoke", type=int, default=0,
                    help="If >0, run only first N queries (smoke test)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    state_log_path = out_dir / "state_log.ndjson"
    summary_path = out_dir / "run_summary.json"

    # Detect tier from output path or hierarchy
    tier = "unknown"
    joined_paths = f"{args.hierarchy} {args.out}".lower()
    tier_markers = (
        ("1.14B", ("erag_1.14b", "erag_1_14b", "erag_1140m")),
        ("FULL", ("erag_full", "erag_618m")),
        ("250M", ("erag_250m",)),
        ("150M", ("erag_150m",)),
        ("100M", ("erag_100m",)),
        ("60M", ("erag_60m",)),
        ("20M", ("erag_20m",)),
        ("10M", ("erag_10m",)),
    )
    for label, markers in tier_markers:
        if any(marker in joined_paths for marker in markers):
            tier = label
            break

    # ----- Load hierarchy -----
    log.info("loading hierarchy from %s ...", args.hierarchy)
    hierarchy = load_hierarchy_jsonl(args.hierarchy)
    parent_child = build_parent_child_from_file(args.hierarchy)
    parent_map = build_parent_map(parent_child)
    by_level = bucket_by_level(hierarchy)
    level_names = sorted(by_level.keys(), key=lambda s: int(s.lstrip("L")))
    top_level = level_names[-1]

    _nav_start = os.environ.get("MEMONDEMAND_NAV_START_LEVEL", "").strip()
    if _nav_start and _nav_start in level_names:
        log.info("hierarchy top is %s but MEMONDEMAND_NAV_START_LEVEL=%s — overriding nav start", top_level, _nav_start)
        top_level = _nav_start
    log.info("hierarchy: %d nodes, levels=%s, top=%s",
             len(hierarchy), level_names, top_level)

    # ----- Load queries -----
    df = pd.read_parquet(args.queries)
    # Deterministic order seed
    if args.query_order_seed:
        df = df.sample(frac=1.0, random_state=args.query_order_seed).reset_index(drop=True)
    if args.n_smoke > 0:
        df = df.head(args.n_smoke)
    log.info("loaded %d queries (seed=%d, n_smoke=%d)",
             len(df), args.query_order_seed, args.n_smoke)

    # ----- Resume support -----
    done_ids: Set[str] = set()
    if args.resume and answers_path.exists():
        with open(answers_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    qid = d.get("query_id", "")
                    if qid:
                        done_ids.add(qid)
                except Exception:
                    pass
        log.info("resume: %d already done", len(done_ids))

    # ----- Embedder + initial indexes -----
    embedder = Embedder()
    indexes: Dict[str, CosineIndex] = {}

    # ----- Controllers (V5 only) -----
    ledger = TokenLedger(run_id=f"v5_{args.method}_{tier}",
                         method=args.method, alias_status="CONFIRMED")
    state_log: Optional[StateLog] = None
    promotion: Optional[PromotionController] = None
    decay: Optional[DecayController] = None
    if args.method == "V5":
        state_log = StateLog(str(state_log_path))

        if args.promotion_budget > 0:
            promotion = PromotionController(
                hierarchy_dict=hierarchy,
                embedder=embedder,
                promotion_budget=args.promotion_budget,
                decay_window=args.decay_window,
                tau=args.tau,
                alias_high=args.alias_answer,
                alias_low=args.alias_low,
                max_candidates_per_decision=args.max_detailed_load,
                deterministic_only=False,
            )
            decay = DecayController(
                promotion_budget=args.promotion_budget,
                decay_window=args.decay_window,
                tau=args.tau,
            )
        else:
            log.info("promotion_budget=0 — disabling promotion and decay controllers")
            promotion = None
            decay = None

    ctx = RunnerCtx(
        method=args.method, tier=tier,
        hierarchy=hierarchy, parent_child=parent_child,
        parent_map=parent_map, by_level=by_level,
        level_names_sorted=level_names, top_level=top_level,
        embedder=embedder, indexes=indexes,
        alias_answer=args.alias_answer,
        alias_navigator=args.alias_navigator,
        alias_low=args.alias_low,
        top_k_distilled=args.top_k_distilled,
        max_detailed_load=args.max_detailed_load,
        max_steps=args.max_steps,
        ledger=ledger, state_log=state_log,
        promotion=promotion, decay=decay,
    )

    # Pre-build L0 index (always needed for B_flat)
    get_index(ctx, "L0")
    # Pre-build top-level for nav methods
    if args.method in {"B_llm_nav", "V5", "B_fixed_hier", "B_dynamic_hier"}:
        for lvl in level_names:
            get_index(ctx, lvl)

    # ----- Run loop -----
    n_done = len(done_ids)
    n_errors = 0
    t_run_start = time.time()
    f_out = open(answers_path, "a", encoding="utf-8")
    try:
        for query_idx, (_, row) in enumerate(df.iterrows()):
            qid = str(row["query_id"])
            if qid in done_ids:
                continue
            query = str(row["query_text"])
            try:
                if args.method == "B_flat":
                    rec = run_b_flat(ctx, qid, query)
                elif args.method in {"B_fixed_hier", "B_dynamic_hier"}:
                    rec = run_b_hier_bfs(ctx, qid, query)
                elif args.method == "B_llm_nav":
                    rec = run_llm_nav(ctx, qid, query, query_idx, use_promotion=False)
                elif args.method == "V5":

                    rec = run_llm_nav(ctx, qid, query, query_idx, use_promotion=(args.promotion_budget > 0))
                else:
                    raise ValueError(f"unknown method {args.method}")
            except Exception as exc:  # noqa: BLE001
                log.error("query %s error: %s", qid, str(exc)[:300])
                n_errors += 1
                rec = QueryRecord(query_id=qid, query_text=query,
                                  method=args.method, tier=tier,
                                  final_action="STOP_INSUFFICIENT",
                                  error=str(exc)[:300])

            f_out.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            f_out.flush()
            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t_run_start
                log.info("progress %d/%d  elapsed=%.0fs  avg=%.1fs/q",
                         n_done, len(df), elapsed,
                         elapsed / max(1, n_done - len(done_ids)))
            if n_done % args.checkpoint_every == 0:
                # Just write an intermediate summary file
                _write_summary(answers_path, summary_path, args, tier,
                               partial=True)
    finally:
        f_out.close()

    # Final summary
    _write_summary(answers_path, summary_path, args, tier, partial=False)
    # Export ledger
    try:
        ledger_path = out_dir / "token_ledger.json"
        out_data = {
            "run_id": ledger.run_id, "method": ledger.method,
            "alias_status": ledger.alias_status,
            "n_records": len(ledger._records),
            "total_in_tokens": sum(r.input_tokens for r in ledger._records),
            "total_out_tokens": sum(r.output_tokens for r in ledger._records),
            "total_cost_usd": round(sum(r.cost_usd for r in ledger._records), 4),
            "by_phase": {},
        }
        for r in ledger._records:
            ph = out_data["by_phase"].setdefault(r.phase, {"n":0, "in":0, "out":0, "cost":0.0})
            ph["n"] += 1; ph["in"] += r.input_tokens; ph["out"] += r.output_tokens; ph["cost"] += r.cost_usd
        for ph in out_data["by_phase"].values():
            ph["cost"] = round(ph["cost"], 4)
        with open(ledger_path, "w") as f:
            json.dump(out_data, f, indent=2)
    except Exception as e:
        log.warning("ledger export failed: %s", e)

    log.info("DONE method=%s tier=%s  n=%d errors=%d wall=%.0fs",
             args.method, tier, n_done, n_errors, time.time() - t_run_start)
    return 0


def _write_summary(answers_path: Path, summary_path: Path,
                   args: argparse.Namespace, tier: str, partial: bool = False) -> None:
    if not answers_path.exists():
        return
    recs = []
    with open(answers_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    n = len(recs)
    if n == 0:
        return
    n_ans = sum(1 for r in recs if r.get("final_action") == "ANSWER")
    n_stop = sum(1 for r in recs if r.get("final_action") == "STOP_INSUFFICIENT")
    n_prom = sum(r.get("n_promote_events", 0) for r in recs)
    n_dem = sum(r.get("n_demote_events", 0) for r in recs)
    total_cost = sum(r.get("cost_usd", 0.0) for r in recs)
    summary = {
        "method": args.method,
        "tier": tier,
        "n_queries": n,
        "n_answered": n_ans,
        "n_stop_insufficient": n_stop,
        "n_promote_events_total": n_prom,
        "n_demote_events_total": n_dem,
        "mean_n_navigation_steps": round(np.mean([r.get("n_navigation_steps", 0) for r in recs]), 2),
        "mean_t_wall_total_seconds": round(np.mean([r.get("t_wall_total_seconds", 0) for r in recs]), 3),
        "mean_t_llm_navigation_seconds": round(np.mean([r.get("t_llm_navigation_seconds", 0) for r in recs]), 3),
        "mean_t_llm_promotion_seconds": round(np.mean([r.get("t_llm_promotion_seconds", 0) for r in recs]), 3),
        "mean_t_llm_answer_seconds": round(np.mean([r.get("t_llm_answer_seconds", 0) for r in recs]), 3),
        "mean_cost_usd": round(total_cost / max(1, n), 6),
        "total_cost_usd": round(total_cost, 4),
        "distinct_nav_actions": sorted(set(a for r in recs for a in r.get("nav_actions", []))),
        "partial": partial,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
