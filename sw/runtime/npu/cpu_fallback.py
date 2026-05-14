"""NumPy golden reference implementations for NPU-dispatch tests and fallback."""
from __future__ import annotations

import numpy as np


def cpu_gemm(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return ``X @ W`` for W [K_in, M_out], X [B, K_in]."""
    Wf = _as_float32(W, "W")
    Xf = _as_float32(X, "X")
    if Wf.ndim != 2 or Xf.ndim != 2:
        raise ValueError("cpu_gemm expects W and X to be rank-2 arrays")
    if Xf.shape[1] != Wf.shape[0]:
        raise ValueError(
            f"shape mismatch: X has K={Xf.shape[1]} but W has K={Wf.shape[0]}"
        )
    return np.dot(Xf, Wf).astype(np.float32, copy=False)


def cpu_gemv(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``x @ W`` for W [K_in, M_out], x [K_in]."""
    Wf = _as_float32(W, "W")
    xf = _as_float32(x, "x")
    if Wf.ndim != 2 or xf.ndim != 1:
        raise ValueError("cpu_gemv expects W rank-2 and x rank-1")
    if xf.shape[0] != Wf.shape[0]:
        raise ValueError(
            f"shape mismatch: x has K={xf.shape[0]} but W has K={Wf.shape[0]}"
        )
    return np.dot(xf, Wf).astype(np.float32, copy=False)


def cpu_cvo(op: str, x: np.ndarray) -> np.ndarray:
    """Reference CVO operation using NumPy float32 arithmetic."""
    xf = _as_float32(x, "x")
    op_norm = op.upper()
    if op_norm == "EXP":
        result = np.exp(xf)
    elif op_norm == "SQRT":
        result = np.sqrt(xf)
    elif op_norm == "GELU":
        result = _gelu(xf)
    elif op_norm == "SIN":
        result = np.sin(xf)
    elif op_norm == "COS":
        result = np.cos(xf)
    elif op_norm == "REDUCE_SUM":
        result = np.asarray(np.sum(xf, dtype=np.float32), dtype=np.float32)
    elif op_norm == "SCALE":
        result = xf
    elif op_norm == "RECIP":
        result = np.reciprocal(xf)
    else:
        raise ValueError(
            "unsupported CVO op; expected one of "
            "EXP, SQRT, GELU, SIN, COS, REDUCE_SUM, SCALE, RECIP"
        )
    return np.asarray(result, dtype=np.float32)


def _gelu(x: np.ndarray) -> np.ndarray:
    coeff = np.float32(0.7978845608028654)
    cubic = np.float32(0.044715) * np.power(x, 3, dtype=np.float32)
    return np.float32(0.5) * x * (
        np.float32(1.0) + np.tanh(coeff * (x + cubic))
    )


def _as_float32(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    return arr.astype(np.float32, copy=False)
