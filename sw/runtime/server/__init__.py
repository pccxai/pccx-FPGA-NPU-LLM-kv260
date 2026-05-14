"""aiohttp daemon surface for the PCCX KV260 Gemma runtime."""
from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
