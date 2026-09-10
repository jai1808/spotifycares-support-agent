"""Shared OpenAI-compatible client (DeepSeek / OpenAI) + env loading."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load project .env once on import
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a cached OpenAI-compatible client (reads OPENAI_API_KEY / BASE_URL)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env or export it in your shell."
        )
    base_url = os.getenv("OPENAI_BASE_URL") or None
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
