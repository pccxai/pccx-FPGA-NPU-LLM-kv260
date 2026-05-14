"""Tokenizer wrapper for Gemma 3N E4B.

Depends on Core Secrets 3 and 7 only indirectly: token IDs drive per-token
AltUp/PLE initialization without changing block ordering.
"""
from __future__ import annotations

from typing import List


class _ByteFallbackTokenizer:
    """Small local fallback so the daemon can start before tokenizer install."""

    available = False

    def __call__(self, text: str, **_kwargs) -> dict:
        return {"input_ids": [self.encode(text)]}

    def encode(self, text: str, **_kwargs) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int], **_kwargs) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="ignore")


class GemmaTokenizer:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.load_error: Exception | None = None
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
            self.available = True
        except Exception as exc:
            self._tokenizer = _ByteFallbackTokenizer()
            self.available = False
            self.load_error = exc

    def encode(self, text: str) -> List[int]:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "encode"):
            return [int(x) for x in tokenizer.encode(text)]
        encoded = tokenizer(text, return_tensors=None)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(x) for x in encoded]

    def decode(self, ids: list[int]) -> str:
        return str(self._tokenizer.decode([int(x) for x in ids], skip_special_tokens=True))

    def decode_piece(self, ids: list[int]) -> str:
        return str(self._tokenizer.decode([int(x) for x in ids], skip_special_tokens=False))
