"""High-level NPU token-step runtime with NumPy fallback helpers."""
from __future__ import annotations

from contextlib import nullcontext
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable

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
from .sim_mmio import TokenSimMmio


LOG = logging.getLogger(__name__)

NPU_AVAILABLE: bool | None = None
UIO_DEVICE_DEFAULT = "/dev/uio4"
NPU_TIMEOUT_SEC = 5.0

_TRUTHY = {"1", "true", "yes", "on"}
_TOKEN_RUNTIME: "NpuTokenRuntime | None" = None


class NpuTokenRuntime:
    """Host control for the self-contained NPU next-token path.

    The only value read back for generated text is the 32-bit token embedded in
    the completion status word of NEXT_TOKEN.  Intermediate activations, layer
    outputs, accumulators, and logits remain NPU-resident.
    """

    def __init__(
        self,
        *,
        uio_device: str = UIO_DEVICE_DEFAULT,
        timeout_sec: float = NPU_TIMEOUT_SEC,
        poll_interval_sec: float = 0.001,
        mmio_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.uio_device = uio_device
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self._mmio_factory = mmio_factory
        self._sim_mmio: TokenSimMmio | None = None
        self.command_count = 0
        self.last_status_word = 0
        self.weights_loaded = False

    def load_weights_to_l2(self, weights: Any = None) -> None:
        descriptor_count = _descriptor_count(weights)
        manifest_id = _manifest_id(weights)
        word = isa.encode_load_weight(
            descriptor_count=descriptor_count,
            manifest_id=manifest_id,
        )
        self._submit_command(word)
        self.weights_loaded = True

    def reset_kv_cache(self, *, session_id: int = 0) -> None:
        self._submit_command(isa.encode_reset_kv_cache(session_id=session_id))

    def init_activation(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        self.reset_kv_cache()
        self.load_prompt_tokens(prompt_tokens)

    def load_prompt_tokens(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        for position, token_id in enumerate(prompt_tokens):
            self._submit_command(
                isa.encode_load_prompt(position=position, token_id=int(token_id))
            )

    def run_one_token_step(
        self,
        *,
        request_id: int | None = None,
        sampling: isa.SamplingMode = isa.SamplingMode.ARGMAX,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        rid = self.command_count & 0xFFFF if request_id is None else int(request_id)
        word = isa.encode_next_token(
            request_id=rid,
            sampling=sampling,
            temperature_q8_8=isa.q8_8(temperature),
            top_p_q8_8=isa.q8_8(top_p),
        )
        status = self._submit_command(word)
        token = isa.status_token(status)
        if token is None:
            raise RuntimeError("NEXT_TOKEN completed without a token-valid status word")
        return token

    def _submit_command(self, word64: int) -> int:
        with self._open_mmio() as mmio:
            mmio.write64(AXIL_CMD_IN, word64)
            mmio.write64(AXIL_CMD_KICK, 0x1)
            self.command_count += 1
            deadline = time.monotonic() + self.timeout_sec
            while time.monotonic() < deadline:
                status = _read_status_word(mmio)
                self.last_status_word = status
                if isa.status_error(status):
                    raise RuntimeError(f"NPU command failed: status=0x{status:016x}")
                if isa.status_done(status):
                    return status
                time.sleep(self.poll_interval_sec)
        raise TimeoutError("NPU command did not report DONE")

    def _open_mmio(self):
        if self._mmio_factory is not None:
            return self._mmio_factory()
        if _env_truthy("PCCX_NPU_SIM_BACKEND"):
            if self._sim_mmio is None:
                self._sim_mmio = TokenSimMmio.from_env()
            return nullcontext(self._sim_mmio)
        return NpuMmio(uio=self.uio_device)


def load_weights_to_l2(weights: Any = None) -> None:
    token_runtime().load_weights_to_l2(weights)


def reset_kv_cache(*, session_id: int = 0) -> None:
    token_runtime().reset_kv_cache(session_id=session_id)


def init_activation(prompt_tokens: list[int] | tuple[int, ...]) -> None:
    token_runtime().init_activation(prompt_tokens)


def load_prompt_tokens(prompt_tokens: list[int] | tuple[int, ...]) -> None:
    token_runtime().load_prompt_tokens(prompt_tokens)


def run_one_token_step(
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    sampling: isa.SamplingMode = isa.SamplingMode.ARGMAX,
) -> int:
    return token_runtime().run_one_token_step(
        temperature=temperature,
        top_p=top_p,
        sampling=sampling,
    )


def token_runtime() -> NpuTokenRuntime:
    global _TOKEN_RUNTIME
    if _TOKEN_RUNTIME is None:
        _TOKEN_RUNTIME = NpuTokenRuntime()
    return _TOKEN_RUNTIME


def npu_gemm(
    W: np.ndarray,
    X: np.ndarray,
    *,
    layer_idx: int | None = None,
) -> np.ndarray:
    """CPU fallback for legacy tensor-call sites.

    Hardware Gemma generation uses run_one_token_step(), not per-matrix
    dispatch.  This wrapper remains so CPU-mode code can keep sharing the
    Gemma math modules.
    """
    del layer_idx
    return cpu_gemm(np.asarray(W, dtype=np.float32), np.asarray(X, dtype=np.float32))


def npu_gemv(
    W: np.ndarray,
    x: np.ndarray,
    *,
    layer_idx: int | None = None,
) -> np.ndarray:
    del layer_idx
    return cpu_gemv(np.asarray(W, dtype=np.float32), np.asarray(x, dtype=np.float32))


def npu_cvo(op: str, x: np.ndarray) -> np.ndarray:
    return cpu_cvo(str(op).upper(), np.asarray(x, dtype=np.float32))


def npu_status() -> dict:
    """Return the last observed token-step status without popping a FIFO."""
    runtime = _TOKEN_RUNTIME
    status_word = runtime.last_status_word if runtime is not None else 0
    token = isa.status_token(status_word)
    return {
        "mmio_hex": f"0x{status_word & 0xFFFF_FFFF:08x}",
        "busy": isa.status_busy(status_word),
        "done": isa.status_done(status_word),
        "error": isa.status_error(status_word),
        "token_valid": token is not None,
        "last_token": token,
        "available": npu_available(),
        "last_cycle_count": 0,
        "axil_command_count": runtime.command_count if runtime is not None else 0,
        "readback_bytes_last": 4 if token is not None else 0,
    }


def npu_backend_readiness() -> dict:
    """Return whether the high-level token backend can produce token results."""
    available = npu_available()
    token_path_enabled = (
        _env_truthy("PCCX_NPU_TOKEN_BACKEND")
        or _env_truthy("PCCX_NPU_SIM_BACKEND")
    )
    hardware_results = bool(available and token_path_enabled)
    if hardware_results:
        reason = "NPU self-contained token backend is ready"
    elif not token_path_enabled:
        reason = "NPU self-contained token backend is disabled; CPU backend selected"
    else:
        reason = "NPU self-contained token backend is enabled but UIO/sim access is unavailable"
    return {
        "npu_available": available,
        "hardware_results": hardware_results,
        "token_backend": token_path_enabled,
        "experimental_axil_dispatch": False,
        "reason": reason,
    }


def npu_available() -> bool:
    """True when token-step MMIO is available or the local sim backend is enabled."""
    global NPU_AVAILABLE
    if _fallback_requested():
        return False
    if _env_truthy("PCCX_NPU_SIM_BACKEND"):
        return True
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


def _read_status_word(mmio: Any) -> int:
    read64 = getattr(mmio, "read64", None)
    if read64 is not None:
        return int(read64(AXIL_STAT_OUT)) & 0xFFFF_FFFF_FFFF_FFFF
    return int(mmio.read32(AXIL_STAT_OUT)) & 0xFFFF_FFFF


def _descriptor_count(weights: Any) -> int:
    if isinstance(weights, dict):
        descriptors = weights.get("descriptors")
        if descriptors is not None:
            return min(len(descriptors), 0xFF)
    return 0


def _manifest_id(weights: Any) -> int:
    if weights is None:
        return 0
    model_dir = getattr(weights, "model_dir", None)
    if model_dir is not None:
        return isa.manifest_id_from_text(model_dir)
    return isa.manifest_id_from_text(weights)


def _fallback_requested() -> bool:
    return (
        _env_truthy("PCCX_NPU_FORCE_FALLBACK")
        or _env_truthy("NPU_FORCE_FALLBACK")
    )


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def _reset_token_runtime_for_tests() -> None:
    global _TOKEN_RUNTIME, NPU_AVAILABLE
    _TOKEN_RUNTIME = None
    NPU_AVAILABLE = None
