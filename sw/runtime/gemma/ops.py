"""NumPy Gemma 3N E4B primitive operations.

Depends on Core Secrets 1-7: no RMSNorm plus-one, tanh AltUp routing,
unscaled attention helpers, RoPE theta cycling, Gaussian Top-K FFN sparsity,
and LAuReL/PLE scaling rules.
"""
from __future__ import annotations

import numpy as np

from .arch import GemmaArch
from .core_secrets import (
    ALTUP_ROUTER_SCALE_DIM,
    GAUSSIAN_TOPK_SIGMA,
    LAUREL_SCALE,
)


def rms_norm(x: np.ndarray, gamma: np.ndarray | None = None, *, eps: float = 1e-6) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    out = x_f32 / rms
    if gamma is not None:
        out = out * np.asarray(gamma, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    return 0.5 * x_f32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_f32 + 0.044715 * x_f32 ** 3)))


def gaussian_topk(x: np.ndarray) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    threshold = float(np.mean(x_f32) + GAUSSIAN_TOPK_SIGMA * np.std(x_f32))
    return np.where(x_f32 >= threshold, x_f32, 0.0).astype(np.float32)


def gaussian_topk_gelu(x: np.ndarray) -> np.ndarray:
    return gelu(gaussian_topk(x))


def softmax(x: np.ndarray) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    shifted = x_f32 - np.max(x_f32, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def rope_rotate(x: np.ndarray, *, layer_idx: int, pos: int, arch: GemmaArch) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    original_shape = x_f32.shape
    heads = x_f32.reshape(-1, arch.head_dim)
    even = heads[:, 0::2]
    odd = heads[:, 1::2]
    inv_freq = 1.0 / (
        arch.rope_theta(layer_idx)
        ** (np.arange(0, arch.head_dim, 2, dtype=np.float32) / float(arch.head_dim))
    )
    angle = float(pos) * inv_freq
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotated = np.empty_like(heads)
    rotated[:, 0::2] = even * cos - odd * sin
    rotated[:, 1::2] = even * sin + odd * cos
    return rotated.reshape(original_shape)


def laurel_add(attn_output: np.ndarray, laurel_output: np.ndarray) -> np.ndarray:
    return (np.asarray(attn_output, dtype=np.float32) + np.asarray(laurel_output, dtype=np.float32)) * LAUREL_SCALE


def altup_route(x: np.ndarray, w_norm: np.ndarray, w_router: np.ndarray) -> np.ndarray:
    x_n = rms_norm(x, w_norm) / ALTUP_ROUTER_SCALE_DIM
    return np.tanh(np.dot(x_n, np.asarray(w_router, dtype=np.float32))).astype(np.float32)
