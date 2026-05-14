"""Gemma 3N E4B architecture constants.

Depends on Core Secrets 4-5: attention has no scale factor, and RoPE follows
the Local/Local/Local/Local/Global theta cycle.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core_secrets import GLOBAL, LOCAL, ROPE_GLOBAL_THETA, ROPE_LOCAL_THETA, ROPE_PATTERN


GEMMA_3N_E4B_MODEL_ID = "google/gemma-3n-e4b"


@dataclass(frozen=True)
class GemmaArch:
    """Static architecture values for the Gemma 3N E4B text path."""

    model_id: str = GEMMA_3N_E4B_MODEL_ID
    num_layers: int = 35
    hidden_dim: int = 2048
    intermediate_dim: int = 16384
    num_q_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 262400
    sliding_kv_src_layer: int = 18
    global_kv_src_layer: int = 19
    kv_shared_first_layer: int = 20

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def gqa_ratio(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    def rope_kind(self, layer_idx: int) -> str:
        return ROPE_PATTERN[layer_idx % len(ROPE_PATTERN)]

    def rope_theta(self, layer_idx: int) -> float:
        return ROPE_GLOBAL_THETA if self.rope_kind(layer_idx) == GLOBAL else ROPE_LOCAL_THETA

    def kv_source_layer(self, layer_idx: int) -> int:
        if layer_idx < self.kv_shared_first_layer:
            return layer_idx
        if self.rope_kind(layer_idx) == GLOBAL:
            return self.global_kv_src_layer
        return self.sliding_kv_src_layer

    def validate(self) -> None:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("Gemma GQA requires num_q_heads to divide by num_kv_heads")
        if self.q_dim != self.hidden_dim:
            raise ValueError("Gemma Q projection dimension must equal hidden_dim")
        if self.kv_dim != 512:
            raise ValueError("Gemma E4B KV projection dimension must be 512")
        if LOCAL not in ROPE_PATTERN or GLOBAL not in ROPE_PATTERN:
            raise ValueError("RoPE pattern must include local and global layers")


GEMMA_3N_E4B_DEFAULTS = GemmaArch()
GEMMA_3N_E4B_DEFAULTS.validate()
