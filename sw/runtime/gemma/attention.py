"""Attention block for Gemma 3N E4B.

Depends on Core Secrets 1, 4, and 5: Q/K RMSNorm uses raw scale weights,
attention logits are not divided or softcapped, and RoPE theta follows the
five-layer Local/Global cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .arch import GemmaArch
from .ops import rms_norm, rope_rotate, softmax
from .weights import matmul_weight, weight_to_kmajor

try:  # pragma: no cover - exercised when the board runtime module lands.
    from sw.runtime.npu import npu_gemm as _runtime_npu_gemm
except Exception:  # pragma: no cover - import fallback is tested indirectly.
    _runtime_npu_gemm = None


def _gemm(weight: Any, x: np.ndarray, *, layer_idx: int, use_npu: bool) -> np.ndarray:
    if not use_npu or _runtime_npu_gemm is None:
        return matmul_weight(x, weight)
    matrix = weight_to_kmajor(weight)
    x2 = np.asarray(x, dtype=np.float32)
    was_vector = x2.ndim == 1
    if was_vector:
        x2 = x2.reshape(1, -1)
    out = _runtime_npu_gemm(matrix, x2, layer_idx=layer_idx)
    out = np.asarray(out, dtype=np.float32)
    return out.reshape(-1) if was_vector else out


@dataclass
class KVCache:
    """Per-session K/V cache with Gemma 3N shared-layer source routing."""

    arch: GemmaArch
    max_seq_len: int
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        np_dtype = np.float16 if self.dtype == "fp16" else np.float32
        shape = (self.arch.num_layers, self.max_seq_len, self.arch.kv_dim)
        self.k = np.zeros(shape, dtype=np_dtype)
        self.v = np.zeros(shape, dtype=np_dtype)

    def store(self, layer_idx: int, pos: int, k: np.ndarray, v: np.ndarray) -> None:
        if pos >= self.max_seq_len:
            raise IndexError(f"position {pos} exceeds max_seq_len {self.max_seq_len}")
        self.k[layer_idx, pos, :] = k.astype(self.k.dtype, copy=False)
        self.v[layer_idx, pos, :] = v.astype(self.v.dtype, copy=False)

    def read_for_layer(self, layer_idx: int, pos: int) -> tuple[np.ndarray, np.ndarray]:
        if pos >= self.max_seq_len:
            raise IndexError(f"position {pos} exceeds max_seq_len {self.max_seq_len}")
        src_layer = self.arch.kv_source_layer(layer_idx)
        return (
            self.k[src_layer, : pos + 1, :].astype(np.float32),
            self.v[src_layer, : pos + 1, :].astype(np.float32),
        )


class GemmaAttention:
    """Single-token Gemma attention with GQA and v002 dispatch hooks."""

    def __init__(self, arch: GemmaArch, *, use_npu: bool = True) -> None:
        self.arch = arch
        self.use_npu = use_npu

    def project_qkv(
        self,
        x: np.ndarray,
        layer: dict[str, Any],
        *,
        layer_idx: int,
        pos: int,
        kv_cache: KVCache,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = _gemm(layer["W_q"], x, layer_idx=layer_idx, use_npu=self.use_npu)
        q = rms_norm(q.reshape(self.arch.num_q_heads, self.arch.head_dim), layer["gamma_q"])
        q = rope_rotate(q, layer_idx=layer_idx, pos=pos, arch=self.arch).reshape(-1)

        if layer_idx < self.arch.kv_shared_first_layer:
            k = _gemm(layer["W_k"], x, layer_idx=layer_idx, use_npu=self.use_npu)
            v = _gemm(layer["W_v"], x, layer_idx=layer_idx, use_npu=self.use_npu)
            k = rms_norm(k.reshape(self.arch.num_kv_heads, self.arch.head_dim), layer["gamma_k"])
            k = rope_rotate(k, layer_idx=layer_idx, pos=pos, arch=self.arch).reshape(-1)
            kv_cache.store(layer_idx, pos, k, v)

        k_cache, v_cache = kv_cache.read_for_layer(layer_idx, pos)
        return q, k_cache, v_cache

    def combine_gqa(
        self,
        q: np.ndarray,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
    ) -> np.ndarray:
        q_heads = q.reshape(self.arch.num_q_heads, self.arch.head_dim)
        k_heads = k_cache.reshape(-1, self.arch.num_kv_heads, self.arch.head_dim)
        v_heads = v_cache.reshape(-1, self.arch.num_kv_heads, self.arch.head_dim)
        out = np.empty_like(q_heads, dtype=np.float32)

        for q_head in range(self.arch.num_q_heads):
            kv_head = q_head // self.arch.gqa_ratio
            scores = np.matmul(k_heads[:, kv_head, :], q_heads[q_head])
            probs = softmax(scores)
            out[q_head] = np.matmul(probs, v_heads[:, kv_head, :])
        return out.reshape(-1)

    def forward(
        self,
        x: np.ndarray,
        layer: dict[str, Any],
        *,
        layer_idx: int,
        pos: int,
        kv_cache: KVCache,
    ) -> np.ndarray:
        q, k_cache, v_cache = self.project_qkv(
            x,
            layer,
            layer_idx=layer_idx,
            pos=pos,
            kv_cache=kv_cache,
        )
        context = self.combine_gqa(q, k_cache, v_cache)
        return _gemm(layer["W_o"], context, layer_idx=layer_idx, use_npu=self.use_npu)
