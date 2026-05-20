from __future__ import annotations

from sw.runtime.gemma import inference
from sw.runtime.gemma.arch import GEMMA_3N_E4B_DEFAULTS
from sw.runtime.gemma.inference import BACKEND_NPU, GemmaInferenceSession


class FakeWeights:
    available = True
    model_dir = "/tmp/fake-gemma"


class FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        assert prompt == "annyeong"
        return [101, 102]

    def decode_piece(self, tokens: list[int]) -> str:
        return f"<{tokens[0]}>"


def test_generate_npu_path_loads_prompt_and_reads_one_token(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    token_iter = iter([777, 1])

    def fake_load_weights(weights) -> None:
        calls.append(("load_weights", weights.model_dir))

    def fake_init_activation(tokens) -> None:
        calls.append(("init_activation", list(tokens)))

    def fake_run_one_token_step(**kwargs) -> int:
        calls.append(("next_token", kwargs))
        return next(token_iter)

    monkeypatch.setattr(inference, "_runtime_npu_load_weights_to_l2", fake_load_weights)
    monkeypatch.setattr(inference, "_runtime_npu_init_activation", fake_init_activation)
    monkeypatch.setattr(inference, "_runtime_npu_run_one_token_step", fake_run_one_token_step)
    monkeypatch.setattr(
        inference,
        "_runtime_npu_status",
        lambda: {
            "available": True,
            "done": True,
            "token_valid": True,
            "last_token": 777,
            "readback_bytes_last": 4,
        },
    )

    session = GemmaInferenceSession.__new__(GemmaInferenceSession)
    session.arch = GEMMA_3N_E4B_DEFAULTS
    session.model_dir = "/tmp/fake-gemma"
    session.backend_requested = BACKEND_NPU
    session.npu_backend_readiness = {"hardware_results": True}
    session.use_npu = True
    session.backend = BACKEND_NPU
    session.backend_reason = "test"
    session.max_seq_len = 1024
    session.dtype = "fp16"
    session.weights = FakeWeights()
    session.tokenizer = FakeTokenizer()
    session.decoder = None
    session._globals = None
    session._sessions = {}
    session._tokens_total = 0
    session._tokens_per_sec_last = 0.0
    session._npu_weights_loaded = False

    generated = list(
        session.generate(
            "annyeong",
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            session_id="s",
        )
    )

    assert generated[0][0] == "<777>"
    assert generated[0][1]["token_id"] == 777
    assert generated[0][1]["readback_bytes"] == 4
    assert calls[0] == ("load_weights", "/tmp/fake-gemma")
    assert calls[1] == ("init_activation", [101, 102])
    assert calls[2][0] == "next_token"
    assert calls[3][0] == "next_token"
    assert session._sessions["s"]["backend"] == BACKEND_NPU
