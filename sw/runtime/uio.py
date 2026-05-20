"""KV260 NPU UIO driver — user-space mmap helpers for /dev/uioN fabric@A0000000.

Pre-conditions:
- Bitstream loaded via `xmutil loadapp pccx_npu_bd`
- /dev/uio4 (or whichever UIO node points at fabric@A0000000) exists
- Process has read/write permission (typically requires sudo)

NPU register layout (mirrors hw/rtl/NPU_Controller/NPU_frontend):
- 0x000  ADDR_INST    — write only.  Push a 64-bit instruction word into CMD FIFO.
                        Two 32-bit writes form one 64-bit push: the RTL slave is
                        AXI4-Lite 64-bit so a single `struct.pack_into('<Q', ...)`
                        is the right call.
- 0x008  ADDR_KICK    — write only.  Any word triggers the hardware to push
                        0x8000_0000_0000_0000 into the FIFO (decoder kick marker).
- any    AXIL_STAT_OUT — read returns the head of the status FIFO (64-bit).
                        bits [31:0] carry status flags. For NEXT_TOKEN,
                        bits [63:32] carry exactly one generated token when
                        TOKEN_VALID is set.
                        Reads before the FIFO has data block forever — only safe
                        once status backflow RTL is in the loaded bitstream.

The FT4232H board cabling on host means losing the PS over an AXI hang requires
a manual KV260 power cycle, so all primitives time out aggressively and the
caller should always check :func:`NpuMmio.alive` between batches.
"""
from __future__ import annotations

import glob
import mmap
import os
import struct
import time
from typing import Optional


WINDOW_SIZE = 0x10000           # 64 KiB AXIL window
ADDR_INST   = 0x000
ADDR_KICK   = 0x008
ADDR_STAT   = 0x000              # any address; AXIL_STAT_OUT returns FIFO head


def find_fabric_uio() -> str:
    """Return /dev/uioN whose /sys/class/uio/uioN/name contains 'fabric'."""
    for entry in sorted(glob.glob("/sys/class/uio/uio*")):
        try:
            with open(os.path.join(entry, "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        if "fabric" in name.lower():
            return entry.replace("/sys/class/uio/", "/dev/")
    raise FileNotFoundError(
        "no /sys/class/uio/uio*/name contained 'fabric'. "
        "Did you `xmutil loadapp pccx_npu_bd` first?"
    )


class NpuMmio:
    """Open and mmap the fabric UIO node. Use as a context manager."""

    def __init__(self, uio: Optional[str] = None, size: int = WINDOW_SIZE):
        self.path = uio or find_fabric_uio()
        self.size = size
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None

    def __enter__(self) -> "NpuMmio":
        self._fd = os.open(self.path, os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(self._fd, self.size,
                             mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # ---- low-level writes (NEVER call from outside without thinking) ----

    def write32(self, offset: int, value: int) -> None:
        assert self._mm is not None
        struct.pack_into("<I", self._mm, offset, value & 0xFFFF_FFFF)

    def write64(self, offset: int, value: int) -> None:
        assert self._mm is not None
        struct.pack_into("<Q", self._mm, offset, value & 0xFFFF_FFFF_FFFF_FFFF)

    def read32(self, offset: int) -> int:
        """WARNING: blocks forever if the AXIL_STAT_OUT FIFO is empty.

        Requires the status-backflow-connected RTL (commit c3fea5e or later) in
        the loaded bitstream, otherwise the ARM CPU stalls and the KV260 needs
        a manual power cycle.
        """
        assert self._mm is not None
        return struct.unpack_from("<I", self._mm, offset)[0]

    def read64(self, offset: int) -> int:
        assert self._mm is not None
        return struct.unpack_from("<Q", self._mm, offset)[0]

    # ---- instruction-stream primitives ----

    def push_inst(self, word64: int) -> None:
        """Push one 64-bit ISA word into AXIL_CMD_IN's FIFO."""
        self.write64(ADDR_INST, word64)

    def push_kick(self) -> None:
        """Trigger the decoder kick marker (hw forces 0x80_00_00_00_00_00_00_00)."""
        self.write64(ADDR_KICK, 0x1)

    def read_status(self) -> int:
        """Return the lower 32 bits of the head AXIL_STAT_OUT word.

        Bit 0 BUSY, Bit 1 DONE. Token-step callers that need the generated
        token should read the full 64-bit status word instead.
        """
        return self.read32(ADDR_STAT)

    # ---- convenience submit-and-wait orchestration ----

    def submit_program(self, words: list[int]) -> None:
        """Push a list of 64-bit instructions followed by one KICK.

        Caller should not exceed AXIL_CMD_IN FIFO depth (8) without first
        ensuring the decoder has drained, otherwise older words are silently
        dropped by the RTL.
        """
        if len(words) > 8:
            raise ValueError(f"program has {len(words)} words; AXIL_CMD_IN FIFO depth is 8")
        for w in words:
            self.push_inst(w)
        self.push_kick()

    def wait_done(self, timeout_s: float = 5.0, poll_interval_s: float = 0.001) -> bool:
        """Poll mmio_npu_stat.DONE for up to timeout_s. Return True on DONE."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            s = self.read_status()
            if s & 0x2:
                return True
            time.sleep(poll_interval_s)
        return False
