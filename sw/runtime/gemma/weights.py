"""Lazy MMAP weight loader for Gemma 3N E4B.

Depends on Core Secrets 1 and 7: raw scale tensors are exposed without adding
one, and the PLE/AltUp/LAuReL tensors are grouped so callers inject them only
at the required sites.
"""
from __future__ import annotations

import json
import os
import re
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arch import GEMMA_3N_E4B_DEFAULTS, GemmaArch


DEFAULT_MODEL_DIR = "~/models/gemma-3n-e4b-int4/"
_LANGUAGE_PREFIX = "model.language_model."
_LAYER_PATTERN = re.compile(r"model\.language_model\.layers\.(\d+)\.(.*)")


def _raise_nofile_limit(target: int = 65536) -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard == resource.RLIM_INFINITY:
            new_soft = max(soft, target)
        else:
            new_soft = min(max(soft, target), hard)
        if new_soft != soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        pass


_raise_nofile_limit()


LAYER_KEY_MAP = {
    "W_q": "self_attn.q_proj.weight",
    "W_k": "self_attn.k_proj.weight",
    "W_v": "self_attn.v_proj.weight",
    "W_o": "self_attn.o_proj.weight",
    "gamma_q": "self_attn.q_norm.weight",
    "gamma_k": "self_attn.k_norm.weight",
    "input_ln": "input_layernorm.weight",
    "post_attn_ln": "post_attention_layernorm.weight",
    "pre_ffn_ln": "pre_feedforward_layernorm.weight",
    "post_ffn_ln": "post_feedforward_layernorm.weight",
    "W_gate": "mlp.gate_proj.weight",
    "W_up": "mlp.up_proj.weight",
    "W_down": "mlp.down_proj.weight",
    "ple_gate": "per_layer_input_gate.weight",
    "ple_proj": "per_layer_projection.weight",
    "ple_post_ln": "post_per_layer_input_norm.weight",
    "laurel_left": "laurel.linear_left.weight",
    "laurel_right": "laurel.linear_right.weight",
    "laurel_norm": "laurel.post_laurel_norm.weight",
    "altup_rn": "altup.router_norm.weight",
    "altup_router": "altup.modality_router.weight",
    "altup_pred": "altup.prediction_coefs.weight",
    "altup_corr": "altup.correction_coefs.weight",
    "altup_scale": "altup.correct_output_scale",
}


@dataclass
class GemmaGlobalWeights:
    w_embed: Any
    w_ple_packed: np.ndarray
    w_ple_scale: np.ndarray | None
    norm_ple: np.ndarray
    w_ple_proj: Any
    altup_projs: list[Any]
    altup_unprojs: list[Any]
    w_final_norm: np.ndarray
    w_lm_head: Any


class MissingWeightError(FileNotFoundError):
    pass


def _key_from_path(path: Path) -> str:
    name = path.name
    return name[:-4] if name.endswith(".npy") else name


def _dequant_tuple(weight: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    packed, scale = weight
    packed = np.asarray(packed)
    scale_arr = np.asarray(scale, dtype=np.float32)
    if packed.dtype == np.uint8:
        low = (packed & 0x0F).astype(np.int8)
        low = np.where(low > 7, low - 16, low)
        high = (packed >> 4).astype(np.int8)
        high = np.where(high > 7, high - 16, high)
        decoded = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        decoded[:, 0::2] = low
        decoded[:, 1::2] = high
    else:
        decoded = packed.astype(np.float32)
    if scale_arr.ndim == 1:
        decoded = decoded * scale_arr[:, np.newaxis]
    else:
        decoded = decoded * scale_arr
    return np.ascontiguousarray(decoded.T)


def weight_to_kmajor(weight: Any) -> np.ndarray:
    if isinstance(weight, tuple):
        return _dequant_tuple(weight)
    return np.ascontiguousarray(np.asarray(weight, dtype=np.float32))


def matmul_weight(x: np.ndarray, weight: Any) -> np.ndarray:
    matrix = weight_to_kmajor(weight)
    x_f32 = np.asarray(x, dtype=np.float32)
    return np.dot(x_f32, matrix).astype(np.float32)


class GemmaWeights:
    """Index `.npy` split weights and load arrays lazily with mmap."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, *, arch: GemmaArch = GEMMA_3N_E4B_DEFAULTS) -> None:
        self.arch = arch
        self.model_dir = str(Path(os.path.expanduser(model_dir)).resolve())
        self._index: dict[str, Path] | None = None
        self._scale_for: dict[str, str] = {}
        self._cache: dict[str, Any] = {}
        self._layer_cache: dict[int, dict[str, Any]] = {}
        self.mode = self._detect_mode()

    @property
    def available(self) -> bool:
        return Path(self.model_dir).is_dir() and any(Path(self.model_dir).glob("*.npy"))

    def _detect_mode(self) -> str:
        manifest = Path(self.model_dir) / "manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text())
                return str(data.get("mode", "int4"))
            except Exception:
                pass
        return "int4"

    def _scan(self) -> dict[str, Path]:
        if self._index is not None:
            return self._index
        root = Path(self.model_dir)
        if not root.is_dir():
            raise MissingWeightError(self.model_dir)
        index = {_key_from_path(path): path for path in root.glob("*.npy")}
        self._scale_for = {key[:-6]: key for key in index if key.endswith(".scale")}
        self._index = index
        return index

    def list_keys(self) -> list[str]:
        return sorted(self._scan())

    def load_key(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        index = self._scan()
        if key not in index:
            raise MissingWeightError(key)
        value = np.load(index[key], mmap_mode="r")
        scale_key = self._scale_for.get(key)
        if scale_key is not None:
            value = (value, np.load(index[scale_key], mmap_mode="r"))
        self._cache[key] = value
        return value

    def get_layer(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx in self._layer_cache:
            return self._layer_cache[layer_idx]
        if not 0 <= layer_idx < self.arch.num_layers:
            raise IndexError(layer_idx)
        prefix = f"{_LANGUAGE_PREFIX}layers.{layer_idx}."
        layer = {
            public_key: self.load_key(prefix + tensor_key)
            for public_key, tensor_key in LAYER_KEY_MAP.items()
        }
        self._layer_cache[layer_idx] = layer
        return layer

    def load_globals(self) -> GemmaGlobalWeights:
        w_embed = self.load_key(_LANGUAGE_PREFIX + "embed_tokens.weight")
        w_ple = self.load_key(_LANGUAGE_PREFIX + "embed_tokens_per_layer.weight")
        if isinstance(w_ple, tuple):
            w_ple_packed, w_ple_scale = w_ple
        else:
            w_ple_packed, w_ple_scale = w_ple, None
        return GemmaGlobalWeights(
            w_embed=w_embed,
            w_ple_packed=w_ple_packed,
            w_ple_scale=w_ple_scale,
            norm_ple=self.load_key(_LANGUAGE_PREFIX + "per_layer_projection_norm.weight"),
            w_ple_proj=self.load_key(_LANGUAGE_PREFIX + "per_layer_model_projection.weight"),
            altup_projs=[
                self.load_key(_LANGUAGE_PREFIX + f"altup_projections.{idx}.weight")
                for idx in range(3)
            ],
            altup_unprojs=[
                self.load_key(_LANGUAGE_PREFIX + f"altup_unembed_projections.{idx}.weight")
                for idx in range(3)
            ],
            w_final_norm=self.load_key(_LANGUAGE_PREFIX + "norm.weight"),
            w_lm_head=w_embed,
        )

    def embed_row(self, token_id: int, weight: Any) -> np.ndarray:
        if isinstance(weight, tuple):
            packed, scale = weight
            return self.embed_row_split(token_id, packed, scale)
        return np.asarray(weight[token_id], dtype=np.float32)

    def embed_row_split(
        self,
        token_id: int,
        packed: np.ndarray,
        scale: np.ndarray | None,
    ) -> np.ndarray:
        if scale is None:
            return np.asarray(packed[token_id], dtype=np.float32)
        row = np.asarray(packed[token_id])
        row_scale = np.asarray(scale[token_id], dtype=np.float32)
        if row.dtype == np.uint8:
            low = (row & 0x0F).astype(np.int8)
            low = np.where(low > 7, low - 16, low)
            high = (row >> 4).astype(np.int8)
            high = np.where(high > 7, high - 16, high)
            out = np.empty(row.size * 2, dtype=np.float32)
            out[0::2] = low
            out[1::2] = high
            return out * row_scale
        return row.astype(np.float32) * row_scale

    def ple_vocab_size(self, globals_: GemmaGlobalWeights) -> int:
        return int(globals_.w_ple_packed.shape[0])

    def close(self) -> None:
        self._cache.clear()
        self._layer_cache.clear()
