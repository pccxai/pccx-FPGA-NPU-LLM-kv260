"""PS-issued AXI DataMover command/status helpers."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .address_map import AddressMap, compiled_default_address_map


CMD_LO = 0x000
CMD_HI = 0x004
CMD_EXT = 0x008
CMD_PUSH = 0x00C
STS_POP = 0x010
FLAGS = 0x014
CMD_LVL = 0x018
STS_LVL = 0x01C
ERR_W1C = 0x020

FLAG_CMD_EMPTY = 1 << 0
FLAG_CMD_FULL = 1 << 1
FLAG_STS_EMPTY = 1 << 2
FLAG_STS_FULL = 1 << 3
DATAMOVER_STATUS_ERROR_MASK = 0x0F

BTT_MASK = (1 << 23) - 1
ADDR_MASK = (1 << 32) - 1
TAG_MASK = (1 << 4) - 1

CHANNEL_NAMES = ("hp0", "hp1", "hp2", "hp3", "acp_fmap", "acp_result")


@dataclass
class PSDataMoverChannel:
    """One PS-controlled AXI DataMover command/status port."""

    name: str
    mmio: Any
    base_addr: int
    uio_map_base: int = 0

    @classmethod
    def from_address_map(
        cls,
        name: str,
        mmio: Any,
        address_map: AddressMap | None = None,
    ) -> "PSDataMoverChannel":
        amap = address_map or compiled_default_address_map()
        if name not in amap.cmdsts_bases:
            raise KeyError(f"unknown DataMover channel {name!r}")
        return cls(
            name=name,
            mmio=mmio,
            base_addr=amap.cmdsts_bases[name],
            uio_map_base=amap.uio_map_base,
        )

    def issue_command(
        self,
        src_addr: int,
        dst_axis_tag: int,
        length_bytes: int,
        *,
        eof: bool = True,
    ) -> int:
        """Stage and push one 72-bit DataMover command.

        ``src_addr`` is the source address for MM2S channels and destination
        address for the S2MM result channel.  ``dst_axis_tag`` becomes the
        4-bit DataMover tag, which is returned as the command token.
        """
        command = pack_datamover_command(
            addr=src_addr,
            tag=dst_axis_tag,
            length_bytes=length_bytes,
            eof=eof,
        )
        if self._read32(FLAGS) & FLAG_CMD_FULL:
            raise RuntimeError(f"{self.name} command FIFO is full")

        self._write32(CMD_LO, command & 0xFFFF_FFFF)
        self._write32(CMD_HI, (command >> 32) & 0xFFFF_FFFF)
        self._write32(CMD_EXT, (command >> 64) & 0xFFFF_FFFF)
        self._write32(CMD_PUSH, 0x1)
        return dst_axis_tag & TAG_MASK

    def poll_status(self, cmd_token: int, timeout_sec: float) -> int:
        """Wait for and pop one 8-bit DataMover status word."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            flags = self._read32(FLAGS)
            if not (flags & FLAG_STS_EMPTY) or self._read32(STS_LVL) > 0:
                status = self._read32(STS_POP) & 0xFF
                status_tag = (status >> 4) & TAG_MASK
                if status_tag != (cmd_token & TAG_MASK):
                    raise RuntimeError(
                        f"{self.name} status tag 0x{status_tag:x} "
                        f"did not match command token 0x{cmd_token & TAG_MASK:x}"
                    )
                error_bits = datamover_status_error_bits(status)
                if error_bits:
                    raise RuntimeError(
                        f"{self.name} DataMover status error bits 0x{error_bits:x}"
                    )
                return status
            time.sleep(0.0005)
        raise TimeoutError(
            f"timed out waiting {timeout_sec:.3f}s for {self.name} status"
        )

    def _offset(self, register_offset: int) -> int:
        return self.base_addr - self.uio_map_base + register_offset

    def _write32(self, register_offset: int, value: int) -> None:
        self.mmio.write32(self._offset(register_offset), value)

    def _read32(self, register_offset: int) -> int:
        return self.mmio.read32(self._offset(register_offset))


def create_channels(
    mmio: Any,
    address_map: AddressMap | None = None,
) -> dict[str, PSDataMoverChannel]:
    """Create all six PS DataMover channel helpers."""
    amap = address_map or compiled_default_address_map()
    return {
        name: PSDataMoverChannel.from_address_map(name, mmio, amap)
        for name in CHANNEL_NAMES
    }


def pack_datamover_command(
    *,
    addr: int,
    tag: int,
    length_bytes: int,
    eof: bool = True,
) -> int:
    """Pack the v12d 72-bit AXI DataMover command word."""
    if addr < 0 or addr > ADDR_MASK:
        raise ValueError("DataMover address must fit in 32 bits")
    if tag < 0 or tag > TAG_MASK:
        raise ValueError("DataMover tag must fit in 4 bits")
    if length_bytes <= 0 or length_bytes > BTT_MASK:
        raise ValueError("DataMover length must be in the 23-bit BTT range")

    return (
        ((tag & TAG_MASK) << 68)
        | ((addr & ADDR_MASK) << 32)
        | ((1 if eof else 0) << 30)
        | (1 << 23)
        | (length_bytes & BTT_MASK)
    )


def datamover_status_error_bits(status: int) -> int:
    """Return the low DataMover status bits that signal transfer errors."""

    return int(status) & DATAMOVER_STATUS_ERROR_MASK
