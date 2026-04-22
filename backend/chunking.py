"""
chunking.py
───────────
Token-aware sentence chunker.
Target: ~250 tokens per chunk, 40-token overlap.
Fallback token counter (no tiktoken required).

No imports from other backend modules.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Token counter (tiktoken optional) ────────────────────────────────────────
try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.get_encoding("cl100k_base")

    def _tok(text: str) -> int:
        return len(_enc.encode(text))

except Exception:
    def _tok(text: str) -> int:           # ~4 chars per token heuristic
        return max(1, len(text) // 4)


CHUNK_TOKENS   = 250
OVERLAP_TOKENS = 40
MIN_TOKENS     = 25

Chunk = Dict[str, Any]


# ── Sentence splitter ─────────────────────────────────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> List[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── Core chunker ──────────────────────────────────────────────────────────────
def _build_chunks(sentences: List[str], chunk_tokens: int, overlap_tokens: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_count = 0

    for sent in sentences:
        sc = _tok(sent)
        if current_count + sc > chunk_tokens and current:
            chunks.append(" ".join(current))
            # keep tail sentences as overlap
            tail: List[str] = []
            tail_count = 0
            for s in reversed(current):
                c = _tok(s)
                if tail_count + c > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_count += c
            current      = tail
            current_count = tail_count
        current.append(sent)
        current_count += sc

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    doc: Dict[str, Any],
    chunk_tokens: int  = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    """Split one document dict into overlapping Chunk dicts."""
    text = doc.get("text", "").strip()
    if not text:
        return []

    sentences  = _split_sentences(text)
    raw_chunks = _build_chunks(sentences, chunk_tokens, overlap_tokens)

    result: List[Chunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        if _tok(chunk_text) < MIN_TOKENS:
            continue
        result.append({
            "text":        chunk_text,
            "source":      doc.get("source", "unknown"),
            "page":        doc.get("page", 1),
            "subject":     doc.get("subject", "General"),
            "chunk_index": idx,
            "token_count": _tok(chunk_text),
        })
    return result


def chunk_documents(
    docs: List[Dict[str, Any]],
    chunk_tokens: int  = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    """Chunk a list of documents."""
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_tokens, overlap_tokens))
    logger.info(f"Chunked {len(docs)} docs → {len(all_chunks)} chunks")
    return all_chunks
