"""Prompt loader & query formatter for the VLM."""

from __future__ import annotations

from pathlib import Path

_PROMPT_CACHE: dict[str, str] = {}


def load_system_prompt(path: str | Path) -> str:
    key = str(path)
    if key not in _PROMPT_CACHE:
        _PROMPT_CACHE[key] = Path(path).read_text(encoding="utf-8")
    return _PROMPT_CACHE[key]


def format_user_query(query_vi: str, fallback_en: str = "Describe what you see.") -> str:
    """Translate or pass-through the user query for the VLM input.

    For models without VI support (Moondream/SmolVLM) we either translate VI→EN
    upstream (recommended) or fall back to a generic English query.
    """
    if not query_vi.strip():
        return fallback_en
    return query_vi.strip()
