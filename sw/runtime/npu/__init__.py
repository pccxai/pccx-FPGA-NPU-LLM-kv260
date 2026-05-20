"""PS-side helpers for the pccx v002 self-contained NPU path.

The Gemma hardware backend sends high-level 64-bit commands through
AXIL_CMD_IN: reset KV cache, load weights, load prompt tokens, and run one
NEXT_TOKEN step.  Intermediate activations and accumulators stay inside the
NPU.  The host reads back only the 32-bit generated token from the completion
status word.
"""
from __future__ import annotations

from .npu_core import (
    init_activation,
    load_prompt_tokens,
    load_weights_to_l2,
    npu_available,
    npu_backend_readiness,
    npu_cvo,
    npu_gemm,
    npu_gemv,
    npu_status,
    reset_kv_cache,
    run_one_token_step,
)

__all__ = [
    "init_activation",
    "load_prompt_tokens",
    "load_weights_to_l2",
    "npu_available",
    "npu_backend_readiness",
    "npu_cvo",
    "npu_gemm",
    "npu_gemv",
    "npu_status",
    "reset_kv_cache",
    "run_one_token_step",
]
