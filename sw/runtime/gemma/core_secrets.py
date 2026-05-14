"""Executable Gemma 3N E4B architecture invariants.

Depends on Core Secrets 1-7 and centralizes them so attention, FFN, AltUp,
LAuReL, PLE, RoPE, and final logits cannot drift independently.
"""
from __future__ import annotations

import math


LOCAL = "LOCAL"
GLOBAL = "GLOBAL"

RMSNORM_PLUS_ONE = False
ALTUP_ROUTER_ACTIVATION = "tanh"
ALTUP_ROUTER_SCALE_DIM = 2048.0
ALTUP_STREAMS = 4
ALTUP_BYPASS_PRED0_FOR_BLOCKS = True

ATTN_NO_SCALING = True
ATTN_SCALE_FACTOR = None
ATTN_SOFTCAP = None
FINAL_LOGITS_SOFTCAP = None

ROPE_LOCAL_THETA = 10_000.0
ROPE_GLOBAL_THETA = 1_000_000.0
ROPE_PATTERN = (LOCAL, LOCAL, LOCAL, LOCAL, GLOBAL)

GAUSSIAN_TOPK_FIRST_LAYER = 0
GAUSSIAN_TOPK_LAST_LAYER = 9
GAUSSIAN_TOPK_SIGMA = 1.645

LAUREL_SCALE = 1.0 / math.sqrt(2.0)
PLE_SHADOW_STREAMS_ONLY = True


def layer_uses_gaussian_topk(layer_idx: int) -> bool:
    return GAUSSIAN_TOPK_FIRST_LAYER <= layer_idx <= GAUSSIAN_TOPK_LAST_LAYER


def validate_core_secrets() -> None:
    if RMSNORM_PLUS_ONE:
        raise ValueError("Gemma 3N RMSNorm must not add one to scale weights")
    if ALTUP_ROUTER_ACTIVATION != "tanh":
        raise ValueError("Gemma 3N AltUp routing must use tanh")
    if not ALTUP_BYPASS_PRED0_FOR_BLOCKS:
        raise ValueError("Gemma 3N blocks must bypass xs_pred[0]")
    if not ATTN_NO_SCALING or ATTN_SCALE_FACTOR is not None:
        raise ValueError("Gemma 3N attention must remain unscaled")
    if ATTN_SOFTCAP is not None or FINAL_LOGITS_SOFTCAP is not None:
        raise ValueError("Gemma 3N does not use attention or final-logit softcap")
    if ROPE_PATTERN != (LOCAL, LOCAL, LOCAL, LOCAL, GLOBAL):
        raise ValueError("Gemma 3N RoPE pattern must be L/L/L/L/G")
    if not PLE_SHADOW_STREAMS_ONLY:
        raise ValueError("Gemma 3N PLE must update only shadow streams")


validate_core_secrets()
