"""
retriever.py  (ExamRAG v5)
──────────────────────────
High-accuracy retrieval with:
  1. Hybrid scoring  — 70% semantic (embedding) + 30% keyword overlap
  2. Reranking       — final sort by hybrid score after dedup
  3. Confidence threshold filtering — drops low-quality chunks
  4. Context sanitization — no filenames, no scores, no metadata noise;
     only clean, readable text (≤ 500 tokens)
  5. Query rewriting — keyword expansion for better recall

v5 improvements vs v4:
  - Hybrid scoring replaces pure semantic search → higher recall + precision
  - Confidence threshold raised to 0.35 (was 0.25) → fewer irrelevant results
  - format_context() fully sanitized: no [N] headers, no filenames, no scores
  - Context capped at 500 tokens (was 900) to fit 768-token num_ctx window

Imports: only vector_store (no rag_pipeline, no llm).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from backend.vector_store import get_store

logger = logging.getLogger(__name__)

# ── Stopwords ─────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "what is are the a an of in on for how why explain define describe"
    " give write list state discuss i me my we our you your he she it"
    " its they them their this that these those be been am was were has"
    " have had do does did will would could should may might shall can"
    " to from with by at about into through during before after above"
    " below between among so and or but not no nor very just also then"
    " than too so up out if".split()
)

# ── Keyword scoring (pure Python, zero deps) ──────────────────────────────────

def _keyword_score(query: str, text: str) -> float:
    """
    Compute a keyword-overlap score [0.0, 1.0].
    Extracts non-stopword tokens from the query and counts their occurrences
    in the chunk text, normalised by chunk length.
    """
    q_tokens = re.sub(r"[^a-z0-9 ]", "", query.lower()).split()
    q_keywords = {t for t in q_tokens if t not in _STOPWORDS and len(t) > 2}
    if not q_keywords:
        return 0.0

    t_tokens = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()
    if not t_tokens:
        return 0.0

    hits = sum(1 for w in t_tokens if w in q_keywords)
    # Scale: hits-per-50-words, capped at 1.0
    return min(1.0, (hits / max(len(t_tokens), 1)) * 50)


# ── Query rewriting ───────────────────────────────────────────────────────────

def _rewrite_query(query: str) -> str:
    """
    Lightweight keyword-based query rewrite.
    Strips question words and stopwords, returns core keywords only.
    No LLM call — pure text processing.
    """
    tokens = re.sub(r"[?.!,]", "", query.lower()).split()
    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    if not keywords:
        return query
    return " ".join(keywords[:10])


# ── Confidence check ──────────────────────────────────────────────────────────

def _is_confident(chunks: List[Dict[str, Any]], min_score: float = 0.35) -> bool:
    """Return True if at least one chunk clears the confidence threshold."""
    return any(c.get("score", 0.0) >= min_score for c in chunks)


# ── De-duplication ────────────────────────────────────────────────────────────

def _dedup(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove near-duplicate chunks (same source + page + chunk_index)."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for c in chunks:
        key = (c.get("source", ""), c.get("page", 0), c.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# ── Hybrid reranking ──────────────────────────────────────────────────────────

def _hybrid_rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    semantic_weight: float = 0.70,
    keyword_weight:  float = 0.30,
) -> List[Dict[str, Any]]:
    """
    Compute hybrid score = semantic_weight * semantic + keyword_weight * keyword.
    Mutates chunk dicts in-place to store new 'score' + debug sub-scores.
    """
    for c in chunks:
        sem  = c.get("score", 0.0)           # already set by vector_store
        kw   = _keyword_score(query, c.get("text", ""))
        hybrid = round(semantic_weight * sem + keyword_weight * kw, 4)
        c["score"]          = hybrid
        c["_semantic_score"] = sem
        c["_keyword_score"]  = kw
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 3,
    subject_filter: Optional[str] = None,
    min_score: float = 0.35,          # v5: raised from 0.25 → higher confidence
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k high-confidence chunks for *query*.

    Pipeline:
      1. Semantic search (pass 1: original query)
      2. Semantic search (pass 2: rewritten query if needed)
      3. Deduplicate
      4. Hybrid rerank (semantic + keyword)
      5. Confidence-threshold filter
      6. Cap at top_k

    Returns empty list when index is empty or confidence is too low.
    A low-confidence result is better turned into a fallback than hallucinated.
    """
    store = get_store()
    if store.total_vectors == 0:
        return []

    sf = None if (not subject_filter or subject_filter.lower() == "all") else subject_filter

    # Pass 1: original query — fetch up to top_k * 4 to give reranking room
    fetch = min(top_k * 4, store.total_vectors)
    results = store.search(query, top_k=fetch, subject_filter=sf, min_score=0.15)

    # Pass 2: rewritten query if pass 1 gave fewer than top_k results
    if len(results) < top_k:
        rewritten = _rewrite_query(query)
        if rewritten.lower() != query.lower():
            extra = store.search(rewritten, top_k=fetch, subject_filter=sf, min_score=0.15)
            results = results + extra

    # Dedup → hybrid rerank → cap
    results = _dedup(results)
    results = _hybrid_rerank(query, results)
    results = results[:top_k]

    # Confidence gate — return empty rather than hallucinate
    if not results or not _is_confident(results, min_score):
        logger.info(
            f"Confidence too low for query: {query!r} "
            f"(best={results[0]['score']:.3f} < {min_score})" if results else ""
        )
        return []

    return results


# ── Context formatter (sanitized) ─────────────────────────────────────────────

_NOISE_RE = re.compile(
    r"(?i)"
    r"\(score\s*=\s*[\d.]+\)"   # (score=0.87) — full parens form
    r"|\bscore\s*=\s*[\d.]+"    # score=0.87   — bare form
    r"|\b[\w\-]+\.pdf\b"        # filenames like notes.pdf
    r"|\[[\d]+\]"               # index markers [1]
    r"|\bp\.\s*\d+"             # page refs like p.3
)


def _sanitize(text: str) -> str:
    """Strip metadata noise from a chunk's text."""
    text = _NOISE_RE.sub("", text)
    # Remove leftover empty parens like () or (  )
    text = re.sub(r"\(\s*\)", "", text)
    # Remove excess whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_context(chunks: List[Dict[str, Any]], max_tokens: int = 500) -> str:
    """
    Format chunks into a clean, sanitized context string.

    v5 changes:
      - No filenames, page numbers, or scores in output
      - Only clean, readable prose text
      - Capped at 500 tokens (≈ 2000 chars) to fit 768-token num_ctx
    """
    if not chunks:
        return ""

    max_chars = max_tokens * 4  # ~4 chars per token heuristic
    parts: List[str] = []
    used = 0

    for c in chunks:
        body = _sanitize(c.get("text", "").strip())
        if not body:
            continue

        if used + len(body) > max_chars:
            remaining = max_chars - used
            if remaining > 80:
                # Trim at sentence boundary where possible
                truncated = body[:remaining]
                last_stop = max(
                    truncated.rfind("."),
                    truncated.rfind("!"),
                    truncated.rfind("?"),
                )
                if last_stop > remaining // 2:
                    truncated = truncated[: last_stop + 1]
                parts.append(truncated)
            break

        parts.append(body)
        used += len(body)

    return "\n\n".join(parts)
