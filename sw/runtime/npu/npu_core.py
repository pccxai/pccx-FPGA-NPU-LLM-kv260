"""High-level NPU dispatch API with NumPy fallback."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import time

import numpy as np

from sw.runtime import isa
from sw.runtime.uio import NpuMmio

from .address_map import (
    AXIL_CMD_IN,
    AXIL_CMD_KICK,
    AXIL_STAT_OUT,
    BITSTREAM_PATH_DEFAULT,
    bitstream_sha256_is_expected,
)
from .cpu_fallback import cpu_cvo, cpu_gemm, cpu_gemv


LOG = logging.getLogger(__name__)

NPU_AVAILABLE: bool | None = None
UIO_DEVICE_DEFAULT = "/dev/uio4"
NPU_TIMEOUT_SEC = 5.0

_TRUTHY = {"1", "true", "yes", "on"}
_CVO_FUNC_BY_NAME = {
    "EXP": isa.CvoFunc.EXP,
    "SQRT": isa.CvoFunc.SQRT,
    "GELU": isa.CvoFunc.GELU,
    "SIN": isa.CvoFunc.SIN,
    "COS": isa.CvoFunc.COS,
    "REDUCE_SUM": isa.CvoFunc.REDUCE_SUM,
    "SCALE": isa.CvoFunc.SCALE,
    "RECIP": isa.CvoFunc.RECIP,
}


def npu_gemm(
    W: np.ndarray,
    X: np.ndarray,
    *,
    layer_idx: int | None = None,
) -> np.ndarray:
    """W [K_in, M_out], X [B, K_in] -> [B, M_out] float32."""
    Wf = np.asarray(W, dtype=np.float32)
    Xf = np.asarray(X, dtype=np.float32)
    if not _can_attempt_hardware():
        return cpu_gemm(Wf, Xf)

    try:
        _submit_experimental_program([
            isa.encode_gemm(
                dest_reg=0,
                src_addr=0,
                size_ptr_addr=_shape_ptr_for_layer(layer_idx),
                shape_ptr_addr=_shape_ptr_for_layer(layer_idx),
                parallel_lane=_parallel_lane_for_shape(Wf.shape),
            )
        ])
    except Exception as exc:
        LOG.warning("NPU GEMM dispatch failed; using CPU fallback: %s", exc)
    return _tiled_gemm_fallback(Wf, Xf)


def npu_gemv(
    W: np.ndarray,
    x: np.ndarray,
    *,
    layer_idx: int | None = None,
) -> np.ndarray:
    """W [K_in, M_out], x [K_in] -> [M_out] float32."""
    Wf = np.asarray(W, dtype=np.float32)
    xf = np.asarray(x, dtype=np.float32)
    if not _can_attempt_hardware():
        return cpu_gemv(Wf, xf)

    try:
        _submit_experimental_program([
            isa.encode_gemv(
                dest_reg=0,
                src_addr=0,
                size_ptr_addr=_shape_ptr_for_layer(layer_idx),
                shape_ptr_addr=_shape_ptr_for_layer(layer_idx),
                parallel_lane=_parallel_lane_for_shape(Wf.shape),
            )
        ])
    except Exception as exc:
        LOG.warning("NPU GEMV dispatch failed; using CPU fallback: %s", exc)
    return cpu_gemv(Wf, xf)


def npu_cvo(op: str, x: np.ndarray) -> np.ndarray:
    """op in {EXP, SQRT, GELU, SIN, COS, REDUCE_SUM, SCALE, RECIP}."""
    xf = np.asarray(x, dtype=np.float32)
    op_norm = op.upper()
    if op_norm not in _CVO_FUNC_BY_NAME:
        return cpu_cvo(op_norm, xf)
    if not _can_attempt_hardware():
        return cpu_cvo(op_norm, xf)

    try:
        _submit_experimental_program([
            isa.encode_cvo(
                _CVO_FUNC_BY_NAME[op_norm],
                src_addr=0,
                dst_addr=0,
                length=min(int(xf.size), 0xFFFF),
            )
        ])
    except Exception as exc:
        LOG.warning("NPU CVO dispatch failed; using CPU fallback: %s", exc)
    return cpu_cvo(op_norm, xf)


def npu_status() -> dict:
    """Return status bits exposed through AXIL_STAT_OUT."""
    available = npu_available()
    status_word = 0
    if available:
        try:
            with NpuMmio(uio=UIO_DEVICE_DEFAULT) as mmio:
                status_word = mmio.read32(AXIL_STAT_OUT)
        except Exception as exc:
            LOG.warning("could not read NPU status; reporting unavailable: %s", exc)
            available = False
            status_word = 0

    return {
        "mmio_hex": f"0x{status_word & 0xFFFF_FFFF:08x}",
        "busy": isa.status_busy(status_word),
        "done": isa.status_done(status_word),
        "available": available,
        "last_cycle_count": 0,
    }


def npu_available() -> bool:
    """True iff /dev/uio4 opens and the loaded bitstream hash is expected."""
    global NPU_AVAILABLE
    if _fallback_requested():
        return False
    if NPU_AVAILABLE is not None:
        return bool(NPU_AVAILABLE)
    if not bitstream_sha256_is_expected(BITSTREAM_PATH_DEFAULT):
        return False
    if not Path(UIO_DEVICE_DEFAULT).exists():
        return False
    try:
        with NpuMmio(uio=UIO_DEVICE_DEFAULT):
            pass
    except OSError:
        return False
    NPU_AVAILABLE = True
    return True


def _can_attempt_hardware() -> bool:
    if not npu_available():
        return False
    if _env_truthy("PCCX_NPU_EXPERIMENTAL_DISPATCH"):
        return True
    LOG.info(
        "NPU is present but high-level dispatch is using CPU fallback until "
        "DMA buffer ownership is golden-vector gated"
    )
    return False


def _submit_experimental_program(words: list[int]) -> None:
    if len(words) > 8:
        raise ValueError("AXIL_CMD_IN FIFO accepts at most 8 words per program")
    with NpuMmio(uio=UIO_DEVICE_DEFAULT) as mmio:
        for word in words:
            mmio.write64(AXIL_CMD_IN, word)
        mmio.write64(AXIL_CMD_KICK, 0x1)
        deadline = time.monotonic() + NPU_TIMEOUT_SEC
        while time.monotonic() < deadline:
            status = mmio.read32(AXIL_STAT_OUT)
            if isa.status_done(status):
                return
            time.sleep(0.001)
    raise TimeoutError("NPU program did not report DONE")


def _tiled_gemm_fallback(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Wf = np.asarray(W, dtype=np.float32)
    Xf = np.asarray(X, dtype=np.float32)
    if Wf.ndim != 2 or Xf.ndim != 2:
        return cpu_gemm(Wf, Xf)
    batch, kin = Xf.shape
    if Wf.shape[0] != kin:
        return cpu_gemm(Wf, Xf)

    tile_m = int(os.getenv("PCCX_NPU_TILE_M", "32"))
    tile_k = int(os.getenv("PCCX_NPU_TILE_K", "128"))
    tile_n = int(os.getenv("PCCX_NPU_TILE_N", "128"))
    if batch <= tile_m and kin <= tile_k and Wf.shape[1] <= tile_n:
        return cpu_gemm(Wf, Xf)

    out = np.zeros((batch, Wf.shape[1]), dtype=np.float32)
    for m0 in range(0, batch, tile_m):
        m1 = min(m0 + tile_m, batch)
        for n0 in range(0, Wf.shape[1], tile_n):
            n1 = min(n0 + tile_n, Wf.shape[1])
            acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float32)
            for k0 in range(0, kin, tile_k):
                k1 = min(k0 + tile_k, kin)
                acc += np.dot(Xf[m0:m1, k0:k1], Wf[k0:k1, n0:n1])
            out[m0:m1, n0:n1] = acc
    return out


def _parallel_lane_for_shape(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return max(1, min(31, int(shape[-1] if len(shape) > 1 else shape[0])))


def _shape_ptr_for_layer(layer_idx: int | None) -> int:
    if layer_idx is None:
        return 0
    return int(layer_idx) & 0x3F


def _fallback_requested() -> bool:
    return (
        _env_truthy("PCCX_NPU_FORCE_FALLBACK")
        or _env_truthy("NPU_FORCE_FALLBACK")
    )


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY
