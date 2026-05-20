from __future__ import annotations

from unittest import mock

from sw.runtime.server import app as server_app


def test_status_dict_accepts_runtime_mmio_hex_and_token_metadata() -> None:
    class FakeNpuModule:
        @staticmethod
        def npu_status():
            return {
                "mmio_hex": "0x0000000a",
                "available": True,
                "token_valid": True,
                "last_token": 1043,
                "readback_bytes_last": 4,
            }

    with mock.patch.object(server_app.importlib, "import_module", return_value=FakeNpuModule):
        payload = server_app.read_npu_status()

    assert payload["npu_mmio_stat_hex"] == "0x0000000a"
    assert payload["npu_done"] is True
    assert payload["npu_token_valid"] is True
    assert payload["npu_last_token"] == 1043
    assert payload["npu_readback_bytes_last"] == 4
