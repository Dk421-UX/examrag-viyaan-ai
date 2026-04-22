"""
vector_store.py
───────────────
FAISS IndexFlatIP (inner product = cosine on normalised vectors).
Metadata stored in a parallel JSON list.

Imports: only embeddings (no other backend modules).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from backend.embeddings import encode, dim

logger = logging.getLogger(__name__)

_INDEX_PATH = "data/faiss_index/index.bin"
_META_PATH  = "data/faiss_index/metadata.json"


class VectorStore:
    """Thread-safe FAISS store with JSON metadata sidecar."""

    def __init__(
        self,
        index_path: str = _INDEX_PATH,
        meta_path: str  = _META_PATH,
    ) -> None:
        try:
            import faiss as _faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            ) from exc

        self._faiss      = _faiss
        self.index_path  = index_path
        self.meta_path   = meta_path
        self._dim        = dim()
        self._index      = _faiss.IndexFlatIP(self._dim)
        self._meta: List[Dict[str, Any]] = []

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self._faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        logger.info(f"VectorStore saved ({self._index.ntotal} vectors)")

    def load(self) -> bool:
        if not os.path.exists(self.index_path):
            logger.info("No index found — starting fresh.")
            return False
        self._index = self._faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        logger.info(f"VectorStore loaded ({self._index.ntotal} vectors)")
        return True

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        vecs  = encode(texts)
        self._index.add(vecs)
        self._meta.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks (total: {self._index.ntotal})")
        return len(chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        subject_filter: Optional[str] = None,
        min_score: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Return top_k most-similar chunks.
        - Filters by subject when subject_filter is set.
        - Drops results below min_score (cosine similarity).
        """
        if self._index.ntotal == 0:
            return []

        # Fetch extra when filtering so we still get enough after filter
        fetch = min(top_k * 6 if subject_filter else top_k * 2, self._index.ntotal)
        q_vec = encode(query)
        scores, indices = self._index.search(q_vec, fetch)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            if float(score) < min_score:
                continue
            meta = dict(self._meta[idx])
            meta["score"] = round(float(score), 4)
            if subject_filter and subject_filter.lower() not in meta.get("subject", "").lower():
                continue
            results.append(meta)
            if len(results) >= top_k:
                break

        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def get_subjects(self) -> List[str]:
        subjects = sorted({m.get("subject", "General") for m in self._meta})
        return ["All"] + subjects


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[VectorStore] = None


def get_store(
    index_path: str = _INDEX_PATH,
    meta_path: str  = _META_PATH,
) -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(index_path, meta_path)
        _store.load()
    return _store


def reset_store() -> None:
    global _store
    _store = None
