from __future__ import annotations

import numpy as np


def test_gemma_package_imports_and_defaults() -> None:
    from sw.runtime.gemma import GEMMA_3N_E4B_DEFAULTS, GemmaInferenceSession

    assert GEMMA_3N_E4B_DEFAULTS.num_layers == 35
    assert GEMMA_3N_E4B_DEFAULTS.hidden_dim == 2048
    assert GEMMA_3N_E4B_DEFAULTS.intermediate_dim == 16384
    assert GEMMA_3N_E4B_DEFAULTS.num_q_heads == 8
    assert GEMMA_3N_E4B_DEFAULTS.num_kv_heads == 2
    assert GEMMA_3N_E4B_DEFAULTS.head_dim == 256
    assert GEMMA_3N_E4B_DEFAULTS.vocab_size == 262400

    session = GemmaInferenceSession("/tmp/no-such-gemma-model", use_npu=False)
    assert session.status()["loaded"] is False
    session.close()


def test_core_secret_values_are_locked() -> None:
    from sw.runtime.gemma import core_secrets as cs

    assert cs.RMSNORM_PLUS_ONE is False
    assert cs.ALTUP_ROUTER_ACTIVATION == "tanh"
    assert cs.ALTUP_ROUTER_SCALE_DIM == 2048.0
    assert cs.ALTUP_BYPASS_PRED0_FOR_BLOCKS is True
    assert cs.ATTN_NO_SCALING is True
    assert cs.ATTN_SCALE_FACTOR is None
    assert cs.ATTN_SOFTCAP is None
    assert cs.FINAL_LOGITS_SOFTCAP is None
    assert cs.ROPE_PATTERN == (cs.LOCAL, cs.LOCAL, cs.LOCAL, cs.LOCAL, cs.GLOBAL)
    assert cs.ROPE_LOCAL_THETA == 10_000.0
    assert cs.ROPE_GLOBAL_THETA == 1_000_000.0
    assert cs.layer_uses_gaussian_topk(0) is True
    assert cs.layer_uses_gaussian_topk(9) is True
    assert cs.layer_uses_gaussian_topk(10) is False
    assert cs.PLE_SHADOW_STREAMS_ONLY is True


def test_weight_loader_uses_mmap_lazily(tmp_path, monkeypatch) -> None:
    from sw.runtime.gemma.weights import GemmaWeights

    key = "model.language_model.norm.weight"
    (tmp_path / f"{key}.npy").write_bytes(b"stub")
    calls: list[tuple[str, str | None]] = []

    def fake_load(path, mmap_mode=None):
        calls.append((str(path), mmap_mode))
        return np.ones((4,), dtype=np.float32)

    monkeypatch.setattr(np, "load", fake_load)

    weights = GemmaWeights(str(tmp_path))
    assert weights.available is True
    assert weights.list_keys() == [key]
    assert weights.load_key(key).shape == (4,)
    assert calls == [(str(tmp_path / f"{key}.npy"), "r")]
