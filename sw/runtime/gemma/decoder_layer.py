"""Gemma decoder layer wiring.

Depends on Core Secrets 1-3 and 7: RMSNorm uses raw scales, AltUp uses tanh
routing with the xs_pred[0] block bypass, and LAuReL/PLE are injected at their
specific parallel and shadow-stream points.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .arch import GemmaArch
from .attention import GemmaAttention, KVCache
from .core_secrets import ALTUP_STREAMS
from .ffn import GemmaFFN
from .ops import altup_route, gelu, laurel_add, rms_norm
from .weights import matmul_weight


class GemmaDecoderLayer:
    """One single-token Gemma layer."""

    def __init__(self, arch: GemmaArch, *, use_npu: bool = True) -> None:
        self.arch = arch
        self.attention = GemmaAttention(arch, use_npu=use_npu)
        self.ffn = GemmaFFN(use_npu=use_npu)
        self.use_npu = use_npu

    def _altup_predict(self, xs: np.ndarray, layer: dict[str, Any]) -> np.ndarray:
        modalities = altup_route(xs[0], layer["altup_rn"], layer["altup_router"])
        coef_mat = np.dot(layer["altup_pred"], modalities).reshape(ALTUP_STREAMS, ALTUP_STREAMS)
        return xs + np.dot(coef_mat, xs)

    def _altup_correct(
        self,
        xs_pred: np.ndarray,
        activated: np.ndarray,
        layer: dict[str, Any],
    ) -> np.ndarray:
        innovation = activated - xs_pred[0]
        mod_corr = altup_route(activated, layer["altup_rn"], layer["altup_router"])
        corr_coefs = np.dot(layer["altup_corr"], mod_corr) + 1.0
        return xs_pred + corr_coefs[:, np.newaxis] * innovation

    def _laurel_parallel(self, inputs_normalized: np.ndarray, layer: dict[str, Any]) -> np.ndarray:
        laurel_x = matmul_weight(inputs_normalized, layer["laurel_left"])
        laurel_x = matmul_weight(laurel_x, layer["laurel_right"])
        return inputs_normalized + rms_norm(laurel_x, layer["laurel_norm"])

    def _inject_ple(
        self,
        xs_new: np.ndarray,
        activated: np.ndarray,
        layer: dict[str, Any],
        pli: np.ndarray | None,
    ) -> np.ndarray:
        if pli is None:
            return xs_new
        gate_ple = gelu(matmul_weight(activated, layer["ple_gate"])) * pli
        mapped = rms_norm(matmul_weight(gate_ple, layer["ple_proj"]), layer["ple_post_ln"])
        xs_new[1:] += mapped
        return xs_new

    def forward(
        self,
        xs: np.ndarray,
        layer: dict[str, Any],
        *,
        layer_idx: int,
        pos: int,
        kv_cache: KVCache,
        pli: np.ndarray | None = None,
    ) -> np.ndarray:
        xs_pred = self._altup_predict(xs, layer)

        block_input = xs[0].copy()
        inputs_normalized = rms_norm(block_input, layer["input_ln"])

        attn_out = self.attention.forward(
            inputs_normalized,
            layer,
            layer_idx=layer_idx,
            pos=pos,
            kv_cache=kv_cache,
        )
        attn_residual = rms_norm(attn_out, layer["post_attn_ln"]) + block_input
        laurel_out = self._laurel_parallel(inputs_normalized, layer)
        attn_combined = laurel_add(attn_residual, laurel_out)

        ffn_input = rms_norm(attn_combined, layer["pre_ffn_ln"])
        ffn_out = self.ffn.forward(ffn_input, layer, layer_idx=layer_idx)
        outputs = rms_norm(ffn_out, layer["post_ffn_ln"]) + attn_combined

        activated = outputs * layer["altup_scale"]
        xs_new = self._altup_correct(xs_pred, activated, layer)
        return self._inject_ple(xs_new, activated, layer, pli)
