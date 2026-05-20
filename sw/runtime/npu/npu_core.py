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
    discover_address_map,
)
from .cpu_fallback import cpu_cvo, cpu_gemm, cpu_gemv
from .dma import create_channels
from .dma_buffer import (
    DMA_DEVICE_ENV,
    DMA_PHYS_ENV,
    DMA_SIZE_ENV,
    open_dma_region_from_env,
)


LOG = logging.getLogger(__name__)

NPU_AVAILABLE: bool | None = None
UIO_DEVICE_DEFAULT = "/dev/uio4"
NPU_TIMEOUT_SEC = 5.0
DMA_TIMEOUT_SEC = 5.0
GEMM_READBACK_ENV = "PCCX_NPU_ENABLE_GEMM_READBACK"
GEMM_SHAPE_PTR_ENV = "PCCX_NPU_GEMM_SHAPE_PTR"
GEMM_INPUT_L2_ENV = "PCCX_NPU_GEMM_INPUT_L2_ADDR"
GEMM_OUTPUT_L2_ENV = "PCCX_NPU_GEMM_OUTPUT_L2_ADDR"
FALLBACK_REASON = (
    "NPU UIO/bitstream is available, but high-level Gemma tensor dispatch "
    "still returns NumPy golden results; DMA buffer ownership and result "
    "readback are not enabled"
)

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
    if _can_attempt_gemm_readback():
        try:
            return _dispatch_gemm_readback(Wf, Xf, layer_idx=layer_idx)
        except Exception as exc:
            LOG.warning("NPU GEMM readback failed; using CPU fallback: %s", exc)
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


def npu_backend_readiness() -> dict:
    """Return whether the NPU path can produce Gemma tensor results."""
    available = npu_available()
    experimental_axil = _env_truthy("PCCX_NPU_EXPERIMENTAL_DISPATCH")
    gemm_readback = _gemm_readback_configured()
    hardware_results = available and gemm_readback[0]
    if not available:
        reason = (
            "NPU UIO/bitstream is not available; Gemma tensor dispatch would "
            "use CPU fallback"
        )
    elif not gemm_readback[0]:
        reason = gemm_readback[1]
    else:
        reason = (
            "NPU GEMM DMA buffer ownership and BF16 result readback are "
            "configured; board result-path evidence is still required"
        )
    return {
        "npu_available": available,
        "hardware_results": hardware_results,
        "experimental_axil_dispatch": experimental_axil,
        "supported_ops": ["gemm"] if hardware_results else [],
        "backend_kind": "hybrid" if hardware_results else "cpu",
        "reason": reason,
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


def _can_attempt_gemm_readback() -> bool:
    if not npu_available():
        return False
    ready, reason = _gemm_readback_configured()
    if ready:
        return True
    LOG.info(reason)
    return False


def _submit_experimental_program(words: list[int]) -> None:
    if len(words) > 8:
        raise ValueError("AXIL_CMD_IN FIFO accepts at most 8 words per program")
    with NpuMmio(uio=UIO_DEVICE_DEFAULT) as mmio:
        if _submit_program_on_mmio(mmio, words):
            return
    raise TimeoutError("NPU program did not report DONE")


def _submit_program_on_mmio(mmio: NpuMmio, words: list[int]) -> bool:
    for word in words:
        mmio.write64(AXIL_CMD_IN, word)
    mmio.write64(AXIL_CMD_KICK, 0x1)
    deadline = time.monotonic() + NPU_TIMEOUT_SEC
    while time.monotonic() < deadline:
        status = mmio.read32(AXIL_STAT_OUT)
        if isa.status_done(status):
            return True
        time.sleep(0.001)
    return False


def _dispatch_gemm_readback(
    W: np.ndarray,
    X: np.ndarray,
    *,
    layer_idx: int | None,
) -> np.ndarray:
    if W.ndim != 2 or X.ndim != 2:
        raise ValueError("hardware GEMM readback requires 2D W and X")
    batch, kin = X.shape
    if W.shape[0] != kin:
        raise ValueError("GEMM shape mismatch")
    nout = int(W.shape[1])
    _check_memset_dim(batch, "batch")
    _check_memset_dim(kin, "kin")
    _check_memset_dim(nout, "nout")

    input_bytes = _pad_128b(_float32_to_bf16_bytes(X))
    hp0_bytes, hp1_bytes = _pack_int4_weight_lanes(W)
    output_elements = batch * nout
    output_bytes = _padded_bf16_nbytes(output_elements)

    shape_ptr = (
        int(os.getenv(GEMM_SHAPE_PTR_ENV, str(_shape_ptr_for_layer(layer_idx) or 3)))
        & 0x3F
    )
    out_shape_ptr = (shape_ptr + 1) & 0x3F
    l2_input_addr = _env_int(GEMM_INPUT_L2_ENV, 0)
    l2_output_addr = _env_int(GEMM_OUTPUT_L2_ENV, 512)

    with open_dma_region_from_env() as region:
        input_buf = region.allocate("gemm_input_bf16", len(input_bytes), alignment=64)
        hp0_buf = region.allocate("gemm_weight_hp0", len(hp0_bytes), alignment=64)
        hp1_buf = region.allocate("gemm_weight_hp1", len(hp1_bytes), alignment=64)
        output_buf = region.allocate("gemm_output_bf16", output_bytes, alignment=64)

        input_buf.write(input_bytes)
        hp0_buf.write(hp0_bytes)
        hp1_buf.write(hp1_bytes)
        output_buf.write(bytes(output_bytes))
        input_buf.sync_for_device()
        hp0_buf.sync_for_device()
        hp1_buf.sync_for_device()
        output_buf.sync_for_device()

        with NpuMmio(uio=UIO_DEVICE_DEFAULT) as mmio:
            channels = create_channels(mmio, discover_address_map())
            result_token = channels["acp_result"].issue_command(
                output_buf.phys_addr,
                0x4,
                output_buf.size,
            )
            fmap_token = channels["acp_fmap"].issue_command(
                input_buf.phys_addr,
                0x3,
                input_buf.size,
            )
            hp0_token = channels["hp0"].issue_command(
                hp0_buf.phys_addr,
                0x1,
                hp0_buf.size,
            )
            hp1_token = channels["hp1"].issue_command(
                hp1_buf.phys_addr,
                0x2,
                hp1_buf.size,
            )

            words = [
                isa.encode_memset(0, shape_ptr, int(batch), int(kin), 1),
                isa.encode_memset(0, out_shape_ptr, int(batch), int(nout), 1),
                isa.encode_memcpy(
                    isa.FROM_HOST,
                    isa.TO_NPU,
                    dest_addr=l2_input_addr,
                    src_addr=0,
                    shape_ptr_addr=shape_ptr,
                ),
                isa.encode_gemm(
                    dest_reg=l2_output_addr,
                    src_addr=l2_input_addr,
                    size_ptr_addr=min(int(kin), 0x3F),
                    shape_ptr_addr=shape_ptr,
                    parallel_lane=_parallel_lane_for_shape(W.shape),
                ),
                isa.encode_memcpy(
                    isa.FROM_NPU,
                    isa.TO_HOST,
                    dest_addr=0,
                    src_addr=l2_output_addr,
                    shape_ptr_addr=out_shape_ptr,
                ),
            ]
            if not _submit_program_on_mmio(mmio, words):
                raise TimeoutError("NPU GEMM program did not report DONE")

            channels["acp_fmap"].poll_status(fmap_token, DMA_TIMEOUT_SEC)
            channels["hp0"].poll_status(hp0_token, DMA_TIMEOUT_SEC)
            channels["hp1"].poll_status(hp1_token, DMA_TIMEOUT_SEC)
            channels["acp_result"].poll_status(result_token, DMA_TIMEOUT_SEC)

        output_buf.sync_for_cpu()
        return _bf16_bytes_to_float32(output_buf.read(output_bytes), (batch, nout))


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


def _gemm_readback_configured() -> tuple[bool, str]:
    if not _env_truthy(GEMM_READBACK_ENV):
        return False, (
            f"{FALLBACK_REASON}; {GEMM_READBACK_ENV}=1 is required because "
            "the current source tree still needs board evidence for the GEMM "
            "result producer to L2/ACP path"
        )
    if not _dma_provider_configured():
        return False, (
            "GEMM readback requested, but no DMA-safe PS buffer provider is "
            f"configured; set {DMA_DEVICE_ENV}, {DMA_PHYS_ENV}, and {DMA_SIZE_ENV}"
        )
    return True, "GEMM readback configured"


def _dma_provider_configured() -> bool:
    if (
        os.getenv(DMA_DEVICE_ENV)
        and os.getenv(DMA_PHYS_ENV)
        and os.getenv(DMA_SIZE_ENV)
    ):
        return True
    try:
        dev = Path("/dev")
        return any(dev.glob("udmabuf*")) or any(dev.glob("u-dma-buf*"))
    except OSError:
        return False


def _float32_to_bf16_bytes(value: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(value, dtype=np.float32)
    bits = arr.view(np.uint32)
    rounded = bits + (((bits >> 16) & 1) + 0x7FFF)
    bf16 = (rounded >> 16).astype("<u2", copy=False)
    return bf16.tobytes()


def _bf16_bytes_to_float32(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    raw = np.frombuffer(data[: count * 2], dtype="<u2").astype(np.uint32)
    bits = raw << 16
    return bits.view(np.float32).reshape(shape).astype(np.float32, copy=False)


def _padded_bf16_nbytes(elements: int) -> int:
    return _align_up(int(elements) * 2, 16)


def _pad_128b(data: bytes) -> bytes:
    padded = _align_up(len(data), 16)
    if padded == len(data):
        return data
    return data + bytes(padded - len(data))


def _pack_int4_weight_lanes(W: np.ndarray) -> tuple[bytes, bytes]:
    quant = np.rint(np.asarray(W, dtype=np.float32))
    if not np.allclose(quant, W, atol=0.0):
        raise ValueError(
            "hardware GEMM readback currently requires already-quantized int4 weights"
        )
    if np.any(quant < -8) or np.any(quant > 7):
        raise ValueError("hardware GEMM readback int4 weights must be in [-8, 7]")

    flat = np.ascontiguousarray(quant.astype(np.int8)).reshape(-1)
    hp0 = _pack_int4_values(flat[0::2])
    hp1 = _pack_int4_values(flat[1::2])
    return _pad_128b(hp0), _pad_128b(hp1)


def _pack_int4_values(values: np.ndarray) -> bytes:
    vals = np.asarray(values, dtype=np.int8).reshape(-1)
    if vals.size % 2:
        vals = np.concatenate([vals, np.zeros(1, dtype=np.int8)])
    nibbles = vals.astype(np.uint8) & 0x0F
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return packed.astype(np.uint8, copy=False).tobytes()


def _check_memset_dim(value: int, name: str) -> None:
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"{name}={value} does not fit in the MEMSET shape field")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)), 0)


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)
