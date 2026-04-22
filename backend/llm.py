"""
llm.py  (ExamRAG v5)
────────────────────
Ollama HTTP client — optimised for 8 GB RAM / CPU-only systems.

v5 changes:
  - num_ctx    = 768  (was 1024)  — critical for 8 GB RAM
  - num_predict = 400  (was 600)  — keeps answers focused, not rambling
  - DEFAULT_MODEL = "phi"          — best quality/RAM trade-off (1.6 GB)
  - repeat_penalty = 1.1           — reduces repetition / hallucination
  - temperature = 0.2              — tighter, more accurate outputs

Public functions (DO NOT RENAME):
  is_online()   list_models()   generate()   stream()
"""

from __future__ import annotations

import json
import logging
from typing import Generator, List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL      = "http://localhost:11434"
DEFAULT_MODEL   = "phi"                       # v5: phi is best for 8 GB RAM
REQUEST_TIMEOUT = 75

RECOMMENDED_MODELS = ["phi", "gemma:2b", "tinyllama", "mistral", "llama2"]

_online_cache: Optional[bool] = None


def _invalidate() -> None:
    global _online_cache
    _online_cache = None


def is_online(force: bool = False) -> bool:
    """Return True if the Ollama server is reachable. Cached per process."""
    global _online_cache
    if not force and _online_cache is not None:
        return _online_cache
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        _online_cache = r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        _online_cache = False
    return _online_cache  # type: ignore[return-value]


def list_models() -> List[str]:
    """Return available model names, preferred lightweight ones first."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]

        def _rank(name: str) -> int:
            for i, rec in enumerate(RECOMMENDED_MODELS):
                if name.lower().startswith(rec):
                    return i
            return len(RECOMMENDED_MODELS)

        return sorted(names, key=_rank)
    except Exception:
        return []


def _base_payload(
    prompt: str,
    model: str,
    temperature: float,
    streaming: bool,
) -> dict:
    return {
        "model":  model,
        "prompt": prompt,
        "stream": streaming,
        "options": {
            "temperature":    temperature,
            "num_ctx":        768,    # v5: 768 (was 1024) — RAM-safe
            "num_predict":    400,    # v5: 400 (was 600)  — focused answers
            "num_thread":     4,
            "repeat_penalty": 1.1,   # v5: reduce repetition
        },
    }


def generate(
    prompt: str,
    model: str         = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """
    Blocking generation. Returns the complete response string.
    Raises RuntimeError on any failure.
    """
    if not is_online():
        raise RuntimeError(
            "Ollama server not found.\n"
            "  1. Install: https://ollama.com\n"
            f"  2. Start:   ollama serve\n"
            f"  3. Pull:    ollama pull {model}"
        )

    payload = _base_payload(prompt, model, temperature, streaming=False)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        _invalidate()
        return text

    except requests.Timeout:
        raise RuntimeError(
            f"Ollama timed out ({REQUEST_TIMEOUT}s). "
            "Try: ollama pull phi  — it's fast on 8 GB RAM."
        )
    except requests.ConnectionError as exc:
        _invalidate()
        raise RuntimeError(f"Ollama connection lost: {exc}") from exc
    except requests.HTTPError as exc:
        code = exc.response.status_code
        body = exc.response.text[:200]
        if code == 404:
            raise RuntimeError(
                f"Model '{model}' not found in Ollama.\n"
                f"Pull it with: ollama pull {model}"
            ) from exc
        raise RuntimeError(f"Ollama HTTP {code}: {body}") from exc


def stream(
    prompt: str,
    model: str         = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Generator[str, None, None]:
    """
    Streaming generation. Yields string tokens as they arrive.
    Raises RuntimeError if Ollama is unreachable before the first token.
    """
    if not is_online():
        raise RuntimeError(
            "Ollama server not found. Start it with: ollama serve"
        )

    payload = _base_payload(prompt, model, temperature, streaming=True)
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                try:
                    data  = json.loads(raw_line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except requests.ConnectionError as exc:
        _invalidate()
        raise RuntimeError(f"Ollama connection dropped: {exc}") from exc
