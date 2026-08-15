"""Authenticate Hugging Face Hub downloads using HF_TOKEN from .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH, override=True)


def hf_token() -> Optional[str]:
    raw = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or ""
    )
    token = raw.strip().strip('"').strip("'")
    return token or None


def _export_token(token: str) -> None:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token


@lru_cache(maxsize=1)
def ensure_hf_login() -> Optional[str]:
    """Set env aliases so transformers / open_clip send the token."""
    token = hf_token()
    if not token:
        return None
    _export_token(token)
    try:
        from huggingface_hub import get_token, login

        if get_token() != token:
            login(token=token, add_to_git_credential=False)
    except Exception:
        pass
    return token
