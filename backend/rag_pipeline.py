"""
rag_pipeline.py  (ExamRAG v5)
──────────────────────────────
Pure orchestrator. Strict one-directional dependency graph:

  rag_pipeline
    ├── retriever   → vector_store → embeddings
    ├── ingest
    ├── chunking
    └── llm

No module below this level may import from rag_pipeline.

v5 improvements:
  - Context reduced to 500 tokens (matches 768-token num_ctx window)
  - All prompts use STRICT anti-hallucination header
  - Fallback message when confidence is low (not hallucination)
  - Adaptive difficulty level hint injected into prompts
  - Strategy mode added + improved prompt copy for all modes

Public functions (DO NOT RENAME):
  query()   query_stream()   ingest_bytes()
  get_index_stats()   get_subjects()   list_models()   is_online()
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Generator, List, Optional

from backend.ingest      import load_pdf_bytes, load_text_bytes, load_directory
from backend.chunking    import chunk_documents
from backend.vector_store import get_store
from backend.retriever   import retrieve, format_context
from backend.llm         import generate, stream as llm_stream, is_online, list_models, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
#  MODE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════

MODES = ["Explanation", "5-Mark", "10-Mark", "Revision", "Quiz", "Strategy"]

_INTENT_MAP = {
    "5-Mark":      ["5 mark", "5-mark", "five mark", "short answer"],
    "10-Mark":     ["10 mark", "10-mark", "ten mark", "long answer", "essay", "detailed"],
    "Quiz":        ["quiz", "mcq", "multiple choice", "test me", "question", "challenge"],
    "Revision":    ["revise", "revision", "summarise", "summarize", "bullet", "notes", "summary"],
    "Strategy":    ["strategy", "how to study", "study plan", "approach", "technique", "tips"],
    "Explanation": ["explain", "what is", "what are", "define", "describe", "how does", "why"],
}


def detect_intent(query: str) -> str:
    """Detect the most likely exam mode from the query text."""
    q = query.lower()
    for mode, keywords in _INTENT_MAP.items():
        if any(kw in q for kw in keywords):
            return mode
    return "Explanation"


# ════════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE DIFFICULTY
# ════════════════════════════════════════════════════════════════════════════════

def get_difficulty_hint(query_count: int) -> str:
    """Return a difficulty instruction based on how many queries the student has made."""
    if query_count < 6:
        return "Use simple language suitable for a beginner student."
    elif query_count < 16:
        return "Use intermediate academic language with clear examples."
    else:
        return "Use advanced academic language with technical depth and nuance."


# ════════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES  (compact, strict, anti-hallucination)
# ════════════════════════════════════════════════════════════════════════════════

# v5: Shared preamble enforcing strict context-only answers
_STRICT_PREAMBLE = (
    "You are ExamRAG, a precise exam tutor.\n"
    "RULES: Answer ONLY from the CONTEXT below. "
    "Do NOT add facts not in the context. "
    "If the context does not contain enough information, say so.\n"
    "{difficulty}\n\n"
    "CONTEXT:\n{context}\n\n"
)

_TEMPLATES: Dict[str, str] = {
    "Explanation": (
        _STRICT_PREAMBLE +
        "QUESTION: {query}\n\n"
        "Write a clear explanation from the context only.\n"
        "Then on new lines:\n"
        "KEYPOINTS: bullet 1 | bullet 2 | bullet 3\n"
        "EXAMTIP: one practical exam tip\n\n"
        "EXPLANATION:\n"
    ),
    "5-Mark": (
        _STRICT_PREAMBLE +
        "QUESTION: {query}\n\n"
        "Write a 5-mark answer (≤130 words). "
        "Structure: one-line intro → 3 key points → one-line conclusion.\n"
        "Then:\nKEYPOINTS: point 1 | point 2 | point 3\n"
        "EXAMTIP: tip for full marks\n\n"
        "5-MARK ANSWER:\n"
    ),
    "10-Mark": (
        _STRICT_PREAMBLE +
        "QUESTION: {query}\n\n"
        "Write a 10-mark answer (≤280 words). "
        "Include: introduction, developed explanation with examples, conclusion.\n"
        "Then:\nKEYPOINTS: point 1 | point 2 | point 3 | point 4\n"
        "EXAMTIP: examiner focus tip\n\n"
        "10-MARK ANSWER:\n"
    ),
    "Revision": (
        _STRICT_PREAMBLE +
        "TOPIC: {query}\n\n"
        "Create a bullet-point revision summary with clear sub-headings.\n"
        "Each bullet must be a distinct, exam-worthy fact.\n"
        "Then: EXAMTIP: one revision tip\n\n"
        "REVISION NOTES:\n"
    ),
    "Quiz": (
        _STRICT_PREAMBLE +
        "TOPIC: {query}\n\n"
        "Generate 4 multiple-choice questions from the context ONLY.\n"
        "Format:\nQ1) ...\nA) ...\nB) ...\nC) ...\nD) ...\nAns: X\n\n"
        "QUIZ:\n"
    ),
    "Strategy": (
        _STRICT_PREAMBLE +
        "TOPIC: {query}\n\n"
        "Give an exam study strategy covering:\n"
        "1. How to approach this topic  "
        "2. Common exam mistakes  "
        "3. What examiners reward  "
        "4. One useful mnemonic\n\n"
        "STRATEGY:\n"
    ),
}

# Used when no documents are indexed yet
_NO_CONTEXT_TEMPLATE = (
    "You are a knowledgeable exam tutor. No study documents have been uploaded yet.\n"
    "{difficulty}\n"
    "Mode: {mode} | Question: {query}\n\n"
    "Give a helpful answer from general knowledge, then note that uploading "
    "relevant PDFs will improve accuracy significantly.\n\nAnswer:\n"
)

_OFFLINE_TEMPLATE = (
    "⚠️ **Ollama is not running.**\n\n"
    "Fix it in 3 steps:\n"
    "```bash\n"
    "ollama serve\n"
    "ollama pull phi\n"
    "```\n\n"
    "Then refresh this page and try again."
)

_LOW_CONFIDENCE_TEMPLATE = (
    "⚠️ **Not enough relevant content found** in your uploaded documents "
    "to answer this confidently.\n\n"
    "**Suggestions:**\n"
    "- Upload a PDF that covers this topic\n"
    "- Rephrase your question with more specific terms\n"
    "- Use the **Explanation** mode for broader answers\n\n"
    "*Falling back to general knowledge...*"
)


# ════════════════════════════════════════════════════════════════════════════════
#  OUTPUT PARSING
# ════════════════════════════════════════════════════════════════════════════════

def _parse(raw: str) -> Dict[str, Any]:
    """Extract answer / key_points / exam_tip from raw LLM output."""
    answer_block = re.split(r"\nKEYPOINTS:|\nEXAMTIP:", raw)[0].strip()

    # Key points — support both newline and pipe-separated formats
    kp_match = re.search(r"KEYPOINTS:\s*(.+?)(?:\nEXAMTIP:|$)", raw, re.DOTALL)
    key_points: List[str] = []
    if kp_match:
        raw_kp = kp_match.group(1).strip()
        # Handle pipe-separated inline format: "point 1 | point 2 | point 3"
        if "|" in raw_kp and "\n" not in raw_kp.strip():
            candidates = [p.strip() for p in raw_kp.split("|")]
        else:
            candidates = raw_kp.split("\n")
        for line in candidates:
            line = re.sub(r"^[-•*\d.)\s]+", "", line).strip()
            if len(line) > 8:
                key_points.append(line)
        key_points = key_points[:5]

    # Exam tip
    tip_match = re.search(r"EXAMTIP:\s*(.+?)$", raw, re.DOTALL)
    exam_tip  = tip_match.group(1).strip()[:280] if tip_match else ""

    # Fallback key points from answer sentences
    if not key_points:
        sentences = re.split(r"(?<=[.!?])\s+", answer_block)
        key_points = [s.strip() for s in sentences[:4] if len(s.strip()) > 20]

    return {
        "answer":     answer_block,
        "key_points": key_points,
        "exam_tip":   exam_tip,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  INGEST
# ════════════════════════════════════════════════════════════════════════════════

def ingest_bytes(
    data: bytes,
    filename: str     = "upload.pdf",
    chunk_tokens: int = 250,
    overlap: int      = 40,
) -> int:
    """Ingest raw bytes (PDF or text). Returns number of chunks added."""
    if filename.lower().endswith(".pdf"):
        docs = load_pdf_bytes(data, source=filename)
    else:
        docs = load_text_bytes(data, source=filename)
    chunks = chunk_documents(docs, chunk_tokens=chunk_tokens, overlap_tokens=overlap)
    store  = get_store()
    added  = store.add_chunks(chunks)
    store.save()
    return added


def ingest_directory(directory: str = "data") -> int:
    docs   = load_directory(directory)
    chunks = chunk_documents(docs)
    store  = get_store()
    added  = store.add_chunks(chunks)
    store.save()
    return added


# ════════════════════════════════════════════════════════════════════════════════
#  QUERY  (blocking)
# ════════════════════════════════════════════════════════════════════════════════

def query(
    user_query: str,
    mode: Optional[str]           = None,
    top_k: int                    = 3,
    subject_filter: Optional[str] = None,
    model: str                    = DEFAULT_MODEL,
    auto_detect_mode: bool        = True,
    query_count: int              = 0,   # v5: for adaptive difficulty
) -> Dict[str, Any]:
    """
    Full RAG pipeline — blocking.
    Returns a structured dict. Never raises.
    """
    if not mode or auto_detect_mode:
        detected       = detect_intent(user_query)
        effective_mode = mode if (mode and not auto_detect_mode) else detected
    else:
        effective_mode = mode

    if not is_online():
        return {
            "answer":     _OFFLINE_TEMPLATE,
            "key_points": [],
            "exam_tip":   "",
            "sources":    [],
            "context":    "",
            "mode":       effective_mode,
            "fallback":   True,
            "error":      "ollama_offline",
        }

    difficulty = get_difficulty_hint(query_count)

    # Retrieval
    try:
        chunks = retrieve(user_query, top_k=top_k, subject_filter=subject_filter)
    except Exception as exc:
        logger.warning(f"Retrieval error: {exc}")
        chunks = []

    fallback = len(chunks) == 0

    # Build prompt
    if fallback:
        store = get_store()
        if store.total_vectors == 0:
            # No docs indexed at all
            prompt  = _NO_CONTEXT_TEMPLATE.format(
                difficulty=difficulty, mode=effective_mode, query=user_query
            )
        else:
            # Docs indexed but low confidence — warn + use general knowledge
            prompt  = (
                _LOW_CONFIDENCE_TEMPLATE + "\n\n" +
                _NO_CONTEXT_TEMPLATE.format(
                    difficulty=difficulty, mode=effective_mode, query=user_query
                )
            )
        context = ""
    else:
        context  = format_context(chunks, max_tokens=500)
        template = _TEMPLATES.get(effective_mode, _TEMPLATES["Explanation"])
        prompt   = template.format(
            difficulty=difficulty, context=context, query=user_query
        )

    # Generate
    try:
        raw = generate(prompt, model=model)
        if not raw.strip():
            raw = "No response generated. Please try again."
    except RuntimeError as exc:
        return {
            "answer":     f"⚠️ {exc}",
            "key_points": [], "exam_tip": "",
            "sources": [], "context": context,
            "mode": effective_mode, "fallback": True, "error": str(exc),
        }

    parsed  = _parse(raw)
    sources = _build_sources(chunks)

    return {
        "answer":     parsed["answer"],
        "key_points": parsed["key_points"],
        "exam_tip":   parsed["exam_tip"],
        "sources":    sources,
        "context":    context,
        "mode":       effective_mode,
        "fallback":   fallback,
        "error":      None,
    }


# ════════════════════════════════════════════════════════════════════════════════
#  QUERY STREAM
# ════════════════════════════════════════════════════════════════════════════════

def query_stream(
    user_query: str,
    mode: Optional[str]           = None,
    top_k: int                    = 3,
    subject_filter: Optional[str] = None,
    model: str                    = DEFAULT_MODEL,
    query_count: int              = 0,   # v5: for adaptive difficulty
) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming RAG pipeline.
    Yields:
      {"type": "meta",  "data": {sources, context, mode, fallback}}
      {"type": "token", "data": "<token string>"}
      {"type": "done",  "data": None}

    Guaranteed to yield exactly ONE "meta" event, then tokens, then ONE "done".
    Never re-triggers. Never loops. Safe on repeated calls.
    """
    effective_mode = mode or detect_intent(user_query)
    difficulty     = get_difficulty_hint(query_count)

    if not is_online():
        yield {"type": "meta",  "data": {"sources": [], "context": "", "mode": effective_mode, "fallback": True}}
        yield {"type": "token", "data": _OFFLINE_TEMPLATE}
        yield {"type": "done",  "data": None}
        return

    try:
        chunks = retrieve(user_query, top_k=top_k, subject_filter=subject_filter)
    except Exception:
        chunks = []

    fallback = len(chunks) == 0
    context  = "" if fallback else format_context(chunks, max_tokens=500)
    sources  = _build_sources(chunks)

    yield {"type": "meta", "data": {
        "sources": sources, "context": context,
        "mode": effective_mode, "fallback": fallback,
    }}

    if fallback:
        store = get_store()
        if store.total_vectors == 0:
            prompt = _NO_CONTEXT_TEMPLATE.format(
                difficulty=difficulty, mode=effective_mode, query=user_query
            )
        else:
            # Emit low-confidence warning as a token before generation
            yield {"type": "token", "data": _LOW_CONFIDENCE_TEMPLATE + "\n\n"}
            prompt = _NO_CONTEXT_TEMPLATE.format(
                difficulty=difficulty, mode=effective_mode, query=user_query
            )
    else:
        template = _TEMPLATES.get(effective_mode, _TEMPLATES["Explanation"])
        prompt   = template.format(
            difficulty=difficulty, context=context, query=user_query
        )

    try:
        for token in llm_stream(prompt, model=model):
            yield {"type": "token", "data": token}
    except RuntimeError as exc:
        yield {"type": "token", "data": f"\n\n⚠️ {exc}"}

    yield {"type": "done", "data": None}


# ════════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build clean source list from retrieved chunks (no raw scores in UI)."""
    return [
        {
            "source":  c.get("source", "?"),
            "page":    c.get("page", "?"),
            "subject": c.get("subject", "General"),
            "score":   c.get("score", 0.0),
            "preview": c.get("text", "")[:100].strip() + "…",
        }
        for c in chunks
    ]


# ════════════════════════════════════════════════════════════════════════════════
#  INFO HELPERS  (public functions — do not rename)
# ════════════════════════════════════════════════════════════════════════════════

def get_index_stats() -> Dict[str, Any]:
    try:
        store = get_store()
        return {"total_vectors": store.total_vectors, "subjects": store.get_subjects()}
    except Exception:
        return {"total_vectors": 0, "subjects": ["All"]}


def get_subjects() -> List[str]:
    try:
        return get_store().get_subjects()
    except Exception:
        return ["All"]


def get_models() -> List[str]:
    return list_models()


def ollama_online() -> bool:
    return is_online()
