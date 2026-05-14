"""Gemma 3N E4B runtime package for the v002 NPU path.

Depends on Core Secrets 1-7: RMSNorm scale handling, AltUp routing and
residual bypass, unscaled attention, layer-cycled RoPE, Gaussian FFN sparsity,
and LAuReL/PLE placement.
"""
from __future__ import annotations

from .arch import GEMMA_3N_E4B_DEFAULTS, GemmaArch
from .inference import GemmaInferenceSession

__all__ = [
    "GEMMA_3N_E4B_DEFAULTS",
    "GemmaArch",
    "GemmaInferenceSession",
]
