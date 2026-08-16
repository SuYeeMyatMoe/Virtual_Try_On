"""Authenticate Hugging Face Hub downloads using HF_TOKEN from .env."""

from __future__ import annotations

import builtins
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH, override=True)

_HF_DOC_NOISE = (
    "but not documented",
    "Make sure to add it to the docstring",
)


def silence_transformers_autodoc() -> None:
    """Hide Transformers 5.x autodoc print() noise for unused VL models."""
    current = builtins.print
    if getattr(current, "_vesture_quiet_hf", False):
        return

    def _print(*args, **kwargs):
        if args:
            msg = str(args[0])
            if any(bit in msg for bit in _HF_DOC_NOISE):
                return
        return current(*args, **kwargs)

    _print._vesture_quiet_hf = True  # type: ignore[attr-defined]
    builtins.print = _print
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


silence_transformers_autodoc()


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
