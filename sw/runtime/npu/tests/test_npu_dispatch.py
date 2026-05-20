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
    DATAMOVER_STATUS_ERROR_MASK,
    FLAGS,
    PSDataMoverChannel,
    datamover_status_error_bits,
    pack_datamover_command,
)
from sw.runtime.npu.dma_buffer import MappedDmaRegion


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


def test_datamover_status_error_bits():
    assert datamover_status_error_bits(0xA0) == 0
    assert datamover_status_error_bits(0xAF) == DATAMOVER_STATUS_ERROR_MASK


def test_mapped_dma_region_allocates_aligned_physical_slices(tmp_path):
    backing = tmp_path / "dma.bin"
    backing.write_bytes(bytes(4096))

    with MappedDmaRegion(device_path=backing, phys_addr=0x10000000, size=4096) as region:
        first = region.allocate("first", 3, alignment=64)
        second = region.allocate("second", 5, alignment=64)

        assert first.phys_addr == 0x10000000
        assert second.phys_addr == 0x10000040
        first.write(b"abc")
        second.write(b"12345")
        assert first.read(3) == b"abc"
        assert second.read(5) == b"12345"


def test_bf16_codec_roundtrip_is_float32_shape_preserving():
    values = np.array([[1.0, -2.5, 0.125]], dtype=np.float32)
    encoded = npu_core._float32_to_bf16_bytes(values)
    decoded = npu_core._bf16_bytes_to_float32(encoded, values.shape)

    assert decoded.dtype == np.float32
    assert decoded.shape == values.shape
    np.testing.assert_allclose(decoded, values, rtol=0.01, atol=0.01)


def test_int4_weight_lanes_are_signed_nibble_packed_and_padded():
    W = np.array([[-1, 2], [3, -4]], dtype=np.float32)

    hp0, hp1 = npu_core._pack_int4_weight_lanes(W)

    assert hp0[:1] == bytes([0x3F])
    assert hp1[:1] == bytes([0xC2])
    assert len(hp0) == 16
    assert len(hp1) == 16


def test_gemm_readback_path_stages_dma_and_reads_bf16_result(monkeypatch):
    class FakeSlice:
        def __init__(self, name: str, offset: int, size: int) -> None:
            self.name = name
            self.offset = offset
            self.size = size
            self.phys_addr = 0x10000000 + offset
            self.data = bytearray(size)
            self.device_synced = False
            self.cpu_synced = False

        def write(self, data: bytes) -> None:
            if self.name == "gemm_output_bf16" and not any(data):
                return
            self.data[: len(data)] = data

        def read(self, size: int | None = None) -> bytes:
            return bytes(self.data[: self.size if size is None else size])

        def sync_for_device(self) -> None:
            self.device_synced = True

        def sync_for_cpu(self) -> None:
            self.cpu_synced = True

    class FakeRegion:
        def __init__(self) -> None:
            self.next_offset = 0
            self.slices: dict[str, FakeSlice] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

        def allocate(self, name: str, size: int, *, alignment: int = 64) -> FakeSlice:
            offset = ((self.next_offset + alignment - 1) // alignment) * alignment
            self.next_offset = offset + size
            item = FakeSlice(name, offset, size)
            if name == "gemm_output_bf16":
                item.write(npu_core._float32_to_bf16_bytes(np.array([[1.5, -2.0]], dtype=np.float32)))
            self.slices[name] = item
            return item

    class FakeMmio:
        def __init__(self, *args, **kwargs) -> None:
            self.write64_values: list[tuple[int, int]] = []
            self.write32_values: list[tuple[int, int]] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

        def write64(self, offset: int, value: int) -> None:
            self.write64_values.append((offset, value))

        def write32(self, offset: int, value: int) -> None:
            self.write32_values.append((offset, value))

        def read32(self, offset: int) -> int:
            local = offset & 0xFFF
            page = offset & 0xFFFFF000
            if offset == address_map.AXIL_STAT_OUT:
                return 0x2
            if local == FLAGS:
                return 0
            if local == 0x010:
                tag_by_page = {
                    0x1000: 0x1,
                    0x2000: 0x2,
                    0x5000: 0x3,
                    0x6000: 0x4,
                }
                return tag_by_page[page] << 4
            return 0

    fake_region = FakeRegion()
    monkeypatch.setattr(npu_core, "open_dma_region_from_env", lambda: fake_region)
    monkeypatch.setattr(npu_core, "NpuMmio", FakeMmio)
    monkeypatch.setattr(
        npu_core,
        "discover_address_map",
        address_map.compiled_default_address_map,
    )

    W = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X = np.array([[5, 6]], dtype=np.float32)
    out = npu_core._dispatch_gemm_readback(W, X, layer_idx=None)

    np.testing.assert_allclose(out, np.array([[1.5, -2.0]], dtype=np.float32))
    assert fake_region.slices["gemm_input_bf16"].device_synced
    assert fake_region.slices["gemm_output_bf16"].cpu_synced
