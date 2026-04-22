"""
embeddings.py
─────────────
Sentence-Transformers embedding wrapper.
Model: all-MiniLM-L6-v2  (80 MB, 384-dim, fast on CPU)

Design rules:
- Singleton model — loaded once per process, never reloaded
- Thread-safe lazy init via a module-level lock
- No imports from other backend modules (zero circular-import risk)
"""

from __future__ import annotations

import logging
import threading
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

_model = None
_lock  = threading.Lock()


def _get_model():
    """Return the singleton SentenceTransformer, loading it on first call."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:          # double-checked locking
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model '{_MODEL_NAME}'…")
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("Embedding model ready.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
    return _model


def encode(
    texts: Union[str, List[str]],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode one or more strings into L2-normalised float32 vectors.
    Returns shape (N, 384).
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    vecs  = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine via dot product
    )
    return vecs.astype(np.float32)


def dim() -> int:
    """Return the embedding dimensionality (384)."""
    return _EMBEDDING_DIM
