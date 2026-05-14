"""Feed-forward network for Gemma 3N E4B.

Depends on Core Secrets 1 and 6: RMSNorm callers use raw scale weights, and
layers 0-9 apply Gaussian Top-K sparsity before GELU.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .core_secrets import layer_uses_gaussian_topk
from .ops import gaussian_topk, gelu
from .weights import matmul_weight, weight_to_kmajor

try:  # pragma: no cover - exercised when the board runtime module lands.
    from sw.runtime.npu import npu_cvo as _runtime_npu_cvo
    from sw.runtime.npu import npu_gemm as _runtime_npu_gemm
except Exception:  # pragma: no cover - import fallback is tested indirectly.
    _runtime_npu_cvo = None
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


def _gelu_via_cvo(x: np.ndarray, *, use_npu: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if use_npu and _runtime_npu_cvo is not None:
        try:
            return np.asarray(_runtime_npu_cvo("GELU", x), dtype=np.float32)
        except Exception:
            pass
    return gelu(x)


class GemmaFFN:
    """Gate/up/down FFN with v002 dispatch hooks."""

    def __init__(self, *, use_npu: bool = True) -> None:
        self.use_npu = use_npu

    def forward(self, x: np.ndarray, layer: dict[str, Any], *, layer_idx: int) -> np.ndarray:
        gate = _gemm(layer["W_gate"], x, layer_idx=layer_idx, use_npu=self.use_npu)
        up = _gemm(layer["W_up"], x, layer_idx=layer_idx, use_npu=self.use_npu)
        if layer_uses_gaussian_topk(layer_idx):
            gate = gaussian_topk(gate)
        activated = _gelu_via_cvo(gate, use_npu=self.use_npu)
        hidden = activated * up
        return _gemm(layer["W_down"], hidden, layer_idx=layer_idx, use_npu=self.use_npu)
