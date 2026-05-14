"""Address-map constants and discovery helpers for the v12d NPU BD.

Compiled defaults mirror ``hw/vivado/system_bd.tcl`` at the v12d handoff:
the NPU AXI-Lite control page starts at ``0xA0000000`` and six DataMover
command/status helper pages occupy the next 4 KiB pages.  Runtime discovery
uses the UIO map and device-tree ranges when available, then falls back to
these defaults with a warning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import struct
from typing import Mapping


LOG = logging.getLogger(__name__)

UIO_INDEX_DEFAULT = 4
UIO_MAP_SIZE_DEFAULT = 0x00010000
NPU_AXIL_BASE_DEFAULT = 0xA0000000
HELPER_PAGE_SIZE = 0x1000

AXIL_CMD_IN = 0x000
AXIL_STAT_OUT = 0x004
AXIL_CMD_KICK = 0x008

CMDSTS_HP0_BASE_DEFAULT = NPU_AXIL_BASE_DEFAULT + 0x1000
CMDSTS_HP1_BASE_DEFAULT = NPU_AXIL_BASE_DEFAULT + 0x2000
CMDSTS_HP2_BASE_DEFAULT = NPU_AXIL_BASE_DEFAULT + 0x3000
CMDSTS_HP3_BASE_DEFAULT = NPU_AXIL_BASE_DEFAULT + 0x4000
CMDSTS_ACP_FMAP_BASE_DEFAULT = NPU_AXIL_BASE_DEFAULT + 0x5000
CMDSTS_ACP_RESULT_BASE_DEFAULT = NPU_AXIL_BASE_DEFAULT + 0x6000

CMDSTS_HELPER_BASES_DEFAULT = {
    "hp0": CMDSTS_HP0_BASE_DEFAULT,
    "hp1": CMDSTS_HP1_BASE_DEFAULT,
    "hp2": CMDSTS_HP2_BASE_DEFAULT,
    "hp3": CMDSTS_HP3_BASE_DEFAULT,
    "acp_fmap": CMDSTS_ACP_FMAP_BASE_DEFAULT,
    "acp_result": CMDSTS_ACP_RESULT_BASE_DEFAULT,
}

BITSTREAM_PATH_DEFAULT = (
    "/lib/firmware/xilinx/pccx_npu_bd/pccx_npu_bd.bit.bin"
)
EXPECTED_BITSTREAM_SHA256 = {
    "59558c5f86968be2cd968212be3519afeb7afd148809079a314af29a50cf0c6c": "v12d",
}


@dataclass(frozen=True)
class AddressMap:
    """Resolved physical addresses for the NPU AXI-Lite aperture."""

    npu_base: int = NPU_AXIL_BASE_DEFAULT
    uio_map_base: int = NPU_AXIL_BASE_DEFAULT
    uio_map_size: int = UIO_MAP_SIZE_DEFAULT
    axil_cmd_in: int = AXIL_CMD_IN
    axil_stat_out: int = AXIL_STAT_OUT
    axil_cmd_kick: int = AXIL_CMD_KICK
    cmdsts_bases: Mapping[str, int] = field(
        default_factory=lambda: dict(CMDSTS_HELPER_BASES_DEFAULT)
    )

    def mmio_offset(self, physical_addr: int, register_offset: int = 0) -> int:
        """Return mmap-relative offset for a physical register address."""
        offset = physical_addr + register_offset - self.uio_map_base
        if offset < 0 or offset >= self.uio_map_size:
            raise ValueError(
                f"physical address 0x{physical_addr + register_offset:08x} "
                f"is outside UIO map 0x{self.uio_map_base:08x}"
                f"+0x{self.uio_map_size:x}"
            )
        return offset


def compiled_default_address_map() -> AddressMap:
    """Return the v12d address map compiled into this package."""
    return AddressMap(cmdsts_bases=dict(CMDSTS_HELPER_BASES_DEFAULT))


def discover_address_map(
    *,
    sysfs_uio_root: str | os.PathLike[str] = "/sys/class/uio",
    proc_device_tree_root: str | os.PathLike[str] = "/proc/device-tree",
    uio_index: int = UIO_INDEX_DEFAULT,
) -> AddressMap:
    """Discover the BD AXI-Lite base, falling back to compiled defaults.

    The primary source is ``/sys/class/uio/uio4/maps/map0/addr`` because the
    UIO device exposes the mmap base used by ``/dev/uio4``.  As a secondary
    heuristic this scans device-tree ``ranges`` blobs for an address near the
    known ``0xA0000000`` fabric aperture.  If both fail, the v12d defaults are
    returned and a warning is logged.
    """
    default = compiled_default_address_map()
    sysfs_root = Path(sysfs_uio_root)
    dt_root = Path(proc_device_tree_root)
    map_base = _read_hex_file(
        sysfs_root / f"uio{uio_index}" / "maps" / "map0" / "addr"
    )
    map_size = _read_hex_file(
        sysfs_root / f"uio{uio_index}" / "maps" / "map0" / "size"
    )
    ranges_base = _discover_ranges_base(dt_root)

    discovered_base = map_base if map_base is not None else ranges_base
    if discovered_base is None:
        LOG.warning(
            "could not discover NPU UIO/device-tree address map; "
            "using compiled v12d defaults"
        )
        return default

    helper_bases = {
        name: discovered_base + (base - NPU_AXIL_BASE_DEFAULT)
        for name, base in CMDSTS_HELPER_BASES_DEFAULT.items()
    }
    return AddressMap(
        npu_base=discovered_base,
        uio_map_base=discovered_base,
        uio_map_size=map_size or default.uio_map_size,
        cmdsts_bases=helper_bases,
    )


def bitstream_sha256(
    path: str | os.PathLike[str] = BITSTREAM_PATH_DEFAULT,
) -> str | None:
    """Return the SHA-256 of the loaded bitstream file, or ``None``."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def bitstream_sha256_is_expected(
    path: str | os.PathLike[str] = BITSTREAM_PATH_DEFAULT,
    expected: Mapping[str, str] = EXPECTED_BITSTREAM_SHA256,
) -> bool:
    """Return True when the bitstream hash is in the expected v12d list."""
    digest = bitstream_sha256(path)
    return digest in expected if digest is not None else False


def _read_hex_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        LOG.debug("ignoring non-hex address file %s", path)
        return None


def _discover_ranges_base(root: Path) -> int | None:
    if not root.exists():
        return None
    try:
        ranges_files = list(root.rglob("ranges"))
    except OSError:
        return None
    for path in ranges_files[:128]:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        base = _parse_ranges_base(data)
        if base is not None:
            return base
    return None


def _parse_ranges_base(data: bytes) -> int | None:
    text = data.strip()
    if b"0x" in text.lower():
        for token in text.replace(b",", b" ").split():
            try:
                value = int(token, 0)
            except ValueError:
                continue
            if _looks_like_npu_base(value):
                return value

    if len(data) >= 4:
        cells = []
        usable = len(data) - (len(data) % 4)
        for i in range(0, usable, 4):
            cells.append(struct.unpack(">I", data[i:i + 4])[0])
        for value in cells:
            if _looks_like_npu_base(value):
                return value
    return None


def _looks_like_npu_base(value: int) -> bool:
    return 0xA0000000 <= value <= 0xA0FF0000 and value % HELPER_PAGE_SIZE == 0
