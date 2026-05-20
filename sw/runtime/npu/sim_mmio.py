"""Small MMIO-compatible token-step simulator for local runtime tests."""
from __future__ import annotations

from collections import deque
import os

from sw.runtime import isa

from .address_map import AXIL_CMD_IN, AXIL_CMD_KICK, AXIL_STAT_OUT


class TokenSimMmio:
    """MMIO shim that accepts high-level NPU commands and returns tokens.

    The class is intentionally narrow: it validates the host command sequence
    and returns one 32-bit token only after a NEXT_TOKEN command.  It does not
    expose tensor result buffers.
    """

    def __init__(self, tokens: list[int] | None = None) -> None:
        self.tokens = deque(int(token) & isa.STAT_TOKEN_MASK for token in (tokens or [1]))
        self.commands: list[int] = []
        self.command_count = 0
        self.status_word = isa.encode_status(done=True)

    @classmethod
    def from_env(cls) -> "TokenSimMmio":
        raw = os.getenv("PCCX_NPU_SIM_TOKENS", "1")
        tokens = [int(part.strip(), 0) for part in raw.split(",") if part.strip()]
        return cls(tokens or [1])

    def __enter__(self) -> "TokenSimMmio":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def write64(self, offset: int, value: int) -> None:
        if offset == AXIL_CMD_IN:
            self.commands.append(value & 0xFFFF_FFFF_FFFF_FFFF)
            return
        if offset == AXIL_CMD_KICK:
            if not self.commands:
                self.status_word = isa.encode_status(done=True, error=True)
                return
            self._execute(self.commands[-1])

    def read64(self, offset: int) -> int:
        if offset != AXIL_STAT_OUT:
            return 0
        return self.status_word

    def read32(self, offset: int) -> int:
        return self.read64(offset) & 0xFFFF_FFFF

    def _execute(self, word64: int) -> None:
        opcode, _operand = isa.decode_command(word64)
        self.command_count += 1
        if opcode == isa.Opcode.NEXT_TOKEN:
            token = self.tokens.popleft() if self.tokens else 1
            self.status_word = isa.encode_status(done=True, token=token)
            return
        if opcode in (
            isa.Opcode.RESET_KV_CACHE,
            isa.Opcode.LOAD_WEIGHT,
            isa.Opcode.LOAD_PROMPT,
        ):
            self.status_word = isa.encode_status(done=True)
            return
        self.status_word = isa.encode_status(done=True, error=True)
