"""Streaming Gemma 3N E4B inference session.

Depends on Core Secrets 1-7 by composing the invariant-backed ops, attention,
FFN, decoder layer, tokenizer, and MMAP weight loader.
"""
from __future__ import annotations

import math
import time
from typing import Iterator, Tuple

import numpy as np

from .arch import GEMMA_3N_E4B_DEFAULTS, GemmaArch
from .attention import KVCache
from .decoder_layer import GemmaDecoderLayer
from .ops import rms_norm, softmax
from .tokenizer import GemmaTokenizer
from .weights import DEFAULT_MODEL_DIR, GemmaGlobalWeights, GemmaWeights, matmul_weight

try:  # pragma: no cover - exercised when the board runtime module lands.
    from sw.runtime.npu import npu_status as _runtime_npu_status
except Exception:  # pragma: no cover - import fallback is tested indirectly.
    _runtime_npu_status = None


def _default_npu_status(available: bool = False) -> dict:
    return {
        "mmio_hex": "0x0",
        "busy": False,
        "done": False,
        "available": available,
    }


def _sample(logits: np.ndarray, temperature: float, top_p: float) -> int:
    logits = np.asarray(logits, dtype=np.float32)
    if temperature == 0.0:
        return int(np.argmax(logits))
    probs = softmax(logits / max(float(temperature), 1e-6))
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        keep = cumsum - probs[sorted_idx] < top_p
        filtered = np.zeros_like(probs)
        filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
        if filtered.sum() == 0.0:
            filtered[sorted_idx[0]] = 1.0
        probs = filtered / filtered.sum()
    return int(np.random.choice(len(probs), p=probs))


class GemmaInferenceSession:
    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        *,
        use_npu: bool = True,
        max_seq_len: int = 1024,
        dtype: str = "fp16",
    ) -> None:
        self.arch: GemmaArch = GEMMA_3N_E4B_DEFAULTS
        self.arch.validate()
        self.model_dir = model_dir
        self.use_npu = use_npu
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.weights = GemmaWeights(model_dir, arch=self.arch)
        self.tokenizer = GemmaTokenizer(model_dir)
        self.decoder = GemmaDecoderLayer(self.arch, use_npu=use_npu)
        self._globals: GemmaGlobalWeights | None = None
        self._sessions: dict[str, dict] = {}
        self._tokens_total = 0
        self._tokens_per_sec_last = 0.0

    def _npu_status(self) -> dict:
        if not self.use_npu or _runtime_npu_status is None:
            return _default_npu_status(False)
        try:
            return dict(_runtime_npu_status())
        except Exception:
            return _default_npu_status(False)

    def _global_weights(self) -> GemmaGlobalWeights:
        if self._globals is None:
            self._globals = self.weights.load_globals()
        return self._globals

    def _new_state(self) -> dict:
        return {
            "pos": 0,
            "xs": None,
            "kv_cache": KVCache(self.arch, self.max_seq_len, self.dtype),
        }

    def _initial_streams(self, token_id: int, globals_: GemmaGlobalWeights) -> tuple[np.ndarray, np.ndarray]:
        safe_token_id = int(min(token_id, self.weights.ple_vocab_size(globals_) - 1))
        x0 = self.weights.embed_row(safe_token_id, globals_.w_embed)
        x0 = x0 * math.sqrt(float(self.arch.hidden_dim))

        xs = np.zeros((4, self.arch.hidden_dim), dtype=np.float32)
        xs[0] = x0
        for idx, projection in enumerate(globals_.altup_projs):
            xs[idx + 1] = matmul_weight(x0, projection)

        x_proj = matmul_weight(x0, globals_.w_ple_proj) / math.sqrt(float(self.arch.hidden_dim))
        x_proj = x_proj.reshape(self.arch.num_layers, self.arch.head_dim)
        x_proj = rms_norm(x_proj, globals_.norm_ple)
        ple_row = self.weights.embed_row_split(safe_token_id, globals_.w_ple_packed, globals_.w_ple_scale)
        ple_row = ple_row.reshape(self.arch.num_layers, self.arch.head_dim) * math.sqrt(
            float(self.arch.head_dim)
        )
        pli_all = (x_proj + ple_row) * (1.0 / math.sqrt(2.0))
        return xs, pli_all.astype(np.float32)

    def _forward_one_token(self, token_id: int, state: dict) -> np.ndarray:
        globals_ = self._global_weights()
        xs, pli_all = self._initial_streams(token_id, globals_)
        pos = state["pos"]
        kv_cache: KVCache = state["kv_cache"]
        for layer_idx in range(self.arch.num_layers):
            layer = self.weights.get_layer(layer_idx)
            xs = self.decoder.forward(
                xs,
                layer,
                layer_idx=layer_idx,
                pos=pos,
                kv_cache=kv_cache,
                pli=pli_all[layer_idx],
            )
        state["pos"] = pos + 1
        state["xs"] = xs
        return xs

    def _decode_logits(self, xs: np.ndarray) -> np.ndarray:
        globals_ = self._global_weights()
        target_mag = float(np.mean(xs[0] ** 2) ** 0.5)
        unembedded = [xs[0]]
        for idx, projection in enumerate(globals_.altup_unprojs):
            proj_x = matmul_weight(xs[idx + 1], projection)
            new_mag = float(np.mean(proj_x ** 2) ** 0.5)
            proj_x = proj_x * (target_mag / max(new_mag, 1e-12))
            unembedded.append(proj_x)
        x_final = np.mean(np.stack(unembedded, axis=0), axis=0)
        x_final = rms_norm(x_final, globals_.w_final_norm)
        return matmul_weight(x_final, globals_.w_lm_head)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        session_id: str = "default",
    ) -> Iterator[Tuple[str, dict]]:
        if not self.weights.available:
            raise FileNotFoundError(self.weights.model_dir)

        state = self._new_state()
        self._sessions[session_id] = state
        input_tokens = self.tokenizer.encode(prompt)
        if not input_tokens:
            input_tokens = [0]

        for token_id in input_tokens:
            self._forward_one_token(token_id, state)

        generated: list[int] = []
        stop_tokens = {1, 106}
        for _ in range(max_new_tokens):
            start = time.perf_counter()
            logits = self._decode_logits(state["xs"])
            next_token = _sample(logits, temperature, top_p)
            if next_token in stop_tokens:
                break
            generated.append(next_token)
            self._forward_one_token(next_token, state)
            elapsed = max(time.perf_counter() - start, 1e-9)
            self._tokens_total += 1
            self._tokens_per_sec_last = 1.0 / elapsed
            piece = self.tokenizer.decode_piece([next_token])
            event = {
                "token_id": next_token,
                "tok_per_sec": self._tokens_per_sec_last,
                "npu_status": self._npu_status(),
                "layer_progress": {
                    "current": self.arch.num_layers,
                    "total": self.arch.num_layers,
                    "stage": "decode",
                    "session_id": session_id,
                },
            }
            yield piece, event

    def status(self) -> dict:
        return {
            "model_id": self.arch.model_id,
            "loaded": self.weights.available and self._globals is not None,
            "sessions": sorted(self._sessions.keys()),
            "tokens_total": self._tokens_total,
            "tokens_per_sec_last": self._tokens_per_sec_last,
        }

    def close(self) -> None:
        self._sessions.clear()
        self._globals = None
        self.weights.close()
