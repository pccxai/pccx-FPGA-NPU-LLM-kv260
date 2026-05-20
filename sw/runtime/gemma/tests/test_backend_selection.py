from __future__ import annotations

import pytest

from sw.runtime.gemma.inference import _resolve_backend


def _readiness(*, hardware_results: bool, reason: str = "not ready") -> dict:
    return {
        "npu_available": True,
        "hardware_results": hardware_results,
        "experimental_axil_dispatch": False,
        "reason": reason,
    }


def test_auto_backend_uses_cpu_when_npu_results_are_not_ready() -> None:
    use_npu, backend, reason = _resolve_backend(
        "auto",
        None,
        _readiness(hardware_results=False, reason="token backend disabled"),
    )

    assert use_npu is False
    assert backend == "cpu"
    assert reason == "token backend disabled"


def test_strict_npu_backend_errors_when_results_are_not_ready() -> None:
    with pytest.raises(RuntimeError, match="NPU backend requested but unavailable"):
        _resolve_backend(
            "npu",
            None,
            _readiness(hardware_results=False, reason="token backend unavailable"),
        )


def test_strict_npu_backend_enables_npu_when_results_are_ready() -> None:
    use_npu, backend, reason = _resolve_backend(
        "npu",
        None,
        _readiness(hardware_results=True),
    )

    assert use_npu is True
    assert backend == "npu"
    assert reason == "NPU backend requested"


def test_cpu_backend_overrides_available_npu_results() -> None:
    use_npu, backend, reason = _resolve_backend(
        "cpu",
        None,
        _readiness(hardware_results=True),
    )

    assert use_npu is False
    assert backend == "cpu"
    assert reason == "CPU backend requested"
