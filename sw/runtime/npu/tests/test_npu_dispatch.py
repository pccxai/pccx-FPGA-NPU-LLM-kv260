from __future__ import annotations

import struct

import numpy as np

from sw.runtime.npu import address_map
from sw.runtime.npu import npu_core
from sw.runtime.npu.cpu_fallback import cpu_gemm
from sw.runtime.npu.dma import (
    CMD_EXT,
    CMD_HI,
    CMD_LO,
    CMD_PUSH,
    FLAGS,
    PSDataMoverChannel,
    pack_datamover_command,
)


def test_npu_gemm_matches_cpu_fallback(monkeypatch):
    monkeypatch.setattr(npu_core, "NPU_AVAILABLE", False)
    W = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    X = np.array(
        [
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(npu_core.npu_gemm(W, X), cpu_gemm(W, X))


def test_discover_address_map_uses_mocked_uio_sysfs(tmp_path):
    sysfs_root = tmp_path / "sys" / "class" / "uio"
    map0 = sysfs_root / "uio4" / "maps" / "map0"
    map0.mkdir(parents=True)
    (map0 / "addr").write_text("0xA1000000\n", encoding="ascii")
    (map0 / "size").write_text("0x00010000\n", encoding="ascii")

    amap = address_map.discover_address_map(
        sysfs_uio_root=sysfs_root,
        proc_device_tree_root=tmp_path / "missing-device-tree",
    )

    assert amap.npu_base == 0xA1000000
    assert amap.uio_map_base == 0xA1000000
    assert amap.cmdsts_bases["hp0"] == 0xA1001000
    assert amap.cmdsts_bases["acp_result"] == 0xA1006000


def test_discover_address_map_scans_mocked_device_tree_ranges(tmp_path):
    dt_root = tmp_path / "proc" / "device-tree" / "fabric"
    dt_root.mkdir(parents=True)
    (dt_root / "ranges").write_bytes(
        struct.pack(">IIII", 0x00000000, 0xA0200000, 0x00000000, 0x00010000)
    )

    amap = address_map.discover_address_map(
        sysfs_uio_root=tmp_path / "missing-sysfs",
        proc_device_tree_root=tmp_path / "proc" / "device-tree",
    )

    assert amap.npu_base == 0xA0200000
    assert amap.cmdsts_bases["hp3"] == 0xA0204000


def test_datamover_command_word_packing():
    word = pack_datamover_command(
        addr=0x12345678,
        tag=0xA,
        length_bytes=0x40,
        eof=True,
    )

    expected = (
        (0xA << 68)
        | (0x12345678 << 32)
        | (1 << 30)
        | (1 << 23)
        | 0x40
    )
    assert word == expected


def test_datamover_issue_command_writes_three_words_then_push():
    class FakeMmio:
        def __init__(self) -> None:
            self.writes: list[tuple[int, int]] = []

        def write32(self, offset: int, value: int) -> None:
            self.writes.append((offset, value))

        def read32(self, offset: int) -> int:
            assert offset == 0x1000 + FLAGS
            return 0

    fake = FakeMmio()
    chan = PSDataMoverChannel("hp0", fake, base_addr=0x1000, uio_map_base=0)
    token = chan.issue_command(0x12345678, 0xA, 0x40)
    word = pack_datamover_command(
        addr=0x12345678,
        tag=0xA,
        length_bytes=0x40,
    )

    assert token == 0xA
    assert fake.writes == [
        (0x1000 + CMD_LO, word & 0xFFFF_FFFF),
        (0x1000 + CMD_HI, (word >> 32) & 0xFFFF_FFFF),
        (0x1000 + CMD_EXT, (word >> 64) & 0xFFFF_FFFF),
        (0x1000 + CMD_PUSH, 0x1),
    ]
