"""
ingest.py
─────────
Load PDFs and plain-text files into raw Document dicts.
Each Document: {text, source, page, subject}

No imports from other backend modules.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

Document = Dict[str, Any]

_SUBJECT_MAP: Dict[str, List[str]] = {
    "Mathematics":  ["math", "calculus", "algebra", "statistics", "geometry"],
    "Physics":      ["physics", "mechanics", "thermodynamics", "optics"],
    "Chemistry":    ["chem", "organic", "inorganic", "biochem"],
    "Biology":      ["bio", "ecology", "genetics", "anatomy"],
    "Computer Science": ["cs", "algorithm", "programming", "data structure", "software"],
    "History":      ["history", "medieval", "ancient", "civil"],
    "Economics":    ["econ", "micro", "macro", "finance"],
    "Geography":    ["geo", "climate", "topography"],
}


def _infer_subject(filename: str) -> str:
    name = Path(filename).stem.lower()
    for subject, keywords in _SUBJECT_MAP.items():
        if any(k in name for k in keywords):
            return subject
    return "General"


# ── PDF extraction ────────────────────────────────────────────────────────────
def _extract_pymupdf(data: bytes, source: str) -> List[Document]:
    docs: List[Document] = []
    try:
        import fitz  # PyMuPDF
        pdf = fitz.open(stream=data, filetype="pdf")
        for i in range(len(pdf)):
            text = pdf[i].get_text("text").strip()
            if text:
                docs.append({"text": text, "source": source,
                              "page": i + 1, "subject": _infer_subject(source)})
        pdf.close()
    except Exception as e:
        logger.debug(f"PyMuPDF failed for {source}: {e}")
    return docs


def _extract_pdfplumber(data: bytes, source: str) -> List[Document]:
    docs: List[Document] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append({"text": text, "source": source,
                                  "page": i + 1, "subject": _infer_subject(source)})
    except Exception as e:
        logger.debug(f"pdfplumber failed for {source}: {e}")
    return docs


def load_pdf_bytes(data: bytes, source: str = "upload.pdf") -> List[Document]:
    if not data:
        raise ValueError("Empty PDF bytes.")
    docs = _extract_pymupdf(data, source)
    if not docs:
        docs = _extract_pdfplumber(data, source)
    if not docs:
        raise ValueError(
            f"No text extracted from '{source}'. "
            "The PDF may be image-only or password-protected."
        )
    logger.info(f"Loaded {len(docs)} pages from '{source}'")
    return docs


def load_text_bytes(data: bytes, source: str = "note.txt") -> List[Document]:
    text = data.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Empty text file: '{source}'")
    return [{"text": text, "source": source, "page": 1, "subject": _infer_subject(source)}]


def load_directory(directory: str) -> List[Document]:
    """Recursively load all PDFs and text files from a directory."""
    root = Path(directory)
    all_docs: List[Document] = []
    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        try:
            if fp.suffix.lower() == ".pdf":
                all_docs.extend(load_pdf_bytes(fp.read_bytes(), source=fp.name))
            elif fp.suffix.lower() in (".txt", ".md"):
                all_docs.extend(load_text_bytes(fp.read_bytes(), source=fp.name))
        except Exception as e:
            logger.warning(f"Skipped {fp.name}: {e}")
    logger.info(f"Directory ingest: {len(all_docs)} documents from '{directory}'")
    return all_docs
