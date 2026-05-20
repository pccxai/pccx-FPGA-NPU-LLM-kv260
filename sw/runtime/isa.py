"""pccx v002 high-level NPU command encoders.

The host-visible SW contract is intentionally coarse grained: the host starts
weight load, prompt load, KV reset, and next-token commands.  It does not issue
per-layer GEMM/GEMV/CVO programs or read intermediate activations.

SystemVerilog representation:

    typedef struct packed {
        logic [7:0]  opcode;
        logic [55:0] operand;
    } pccx_host_cmd_t;

This follows the packed-structure shape used by IEEE Std 1800-2023: the
structure is a contiguous vector with no gaps, so Python packs the same fields
as an unsigned 64-bit word.
"""
from __future__ import annotations

from enum import IntEnum
from zlib import crc32


class Opcode(IntEnum):
    RESET_KV_CACHE = 0x01
    LOAD_WEIGHT = 0x02
    LOAD_PROMPT = 0x03
    NEXT_TOKEN = 0x04


class SamplingMode(IntEnum):
    ARGMAX = 0x00
    RANDOM = 0x01


KICK_MARKER = 0x8000_0000_0000_0000

STAT_BUSY_MASK = 0x1
STAT_DONE_MASK = 0x2
STAT_ERROR_MASK = 0x4
STAT_TOKEN_VALID_MASK = 0x8
STAT_TOKEN_SHIFT = 32
STAT_TOKEN_MASK = 0xFFFF_FFFF


def _mask(width: int) -> int:
    return (1 << width) - 1


def _check(val: int, width: int, name: str) -> int:
    m = _mask(width)
    if val < 0 or val > m:
        raise ValueError(f"{name}={val:#x} does not fit in {width} bits (max {m:#x})")
    return val & m


def _pack_command(opcode: Opcode, operand: int = 0) -> int:
    return (_check(int(opcode), 8, "opcode") << 56) | _check(operand, 56, "operand")


def decode_command(word64: int) -> tuple[int, int]:
    word64 = _check(word64, 64, "word64")
    return (word64 >> 56) & 0xFF, word64 & _mask(56)


def encode_reset_kv_cache(*, session_id: int = 0) -> int:
    """Reset NPU-resident KV cache and token-step activation state."""
    return _pack_command(Opcode.RESET_KV_CACHE, _check(session_id, 32, "session_id"))


def decode_reset_kv_cache(word64: int) -> dict[str, int]:
    opcode, operand = decode_command(word64)
    if opcode != Opcode.RESET_KV_CACHE:
        raise ValueError(
            f"opcode {opcode:#x} is not RESET_KV_CACHE ({int(Opcode.RESET_KV_CACHE):#x})"
        )
    return {"session_id": operand & _mask(32)}


def encode_load_weight(
    *,
    weight_slot: int = 0,
    descriptor_count: int = 0,
    manifest_id: int = 0,
    flags: int = 0,
) -> int:
    """Start one host-to-L2 weight-load phase.

    Operand layout:
      weight_slot[55:48] | flags[47:40] | descriptor_count[39:32] |
      manifest_id[31:0]

    Bulk bytes are described by the DMA command stream.  This command only
    starts the NPU-side weight-cache acceptance phase and never requests a
    result buffer from PS memory.
    """
    operand = (
        (_check(weight_slot, 8, "weight_slot") << 48)
        | (_check(flags, 8, "flags") << 40)
        | (_check(descriptor_count, 8, "descriptor_count") << 32)
        | _check(manifest_id, 32, "manifest_id")
    )
    return _pack_command(Opcode.LOAD_WEIGHT, operand)


def decode_load_weight(word64: int) -> dict[str, int]:
    opcode, operand = decode_command(word64)
    if opcode != Opcode.LOAD_WEIGHT:
        raise ValueError(f"opcode {opcode:#x} is not LOAD_WEIGHT ({int(Opcode.LOAD_WEIGHT):#x})")
    return {
        "weight_slot": (operand >> 48) & _mask(8),
        "flags": (operand >> 40) & _mask(8),
        "descriptor_count": (operand >> 32) & _mask(8),
        "manifest_id": operand & _mask(32),
    }


def encode_load_prompt(*, position: int, token_id: int) -> int:
    """Load one prompt token into NPU-resident activation/KV state."""
    operand = (_check(position, 24, "position") << 32) | _check(token_id, 32, "token_id")
    return _pack_command(Opcode.LOAD_PROMPT, operand)


def decode_load_prompt(word64: int) -> dict[str, int]:
    opcode, operand = decode_command(word64)
    if opcode != Opcode.LOAD_PROMPT:
        raise ValueError(f"opcode {opcode:#x} is not LOAD_PROMPT ({int(Opcode.LOAD_PROMPT):#x})")
    return {
        "position": (operand >> 32) & _mask(24),
        "token_id": operand & _mask(32),
    }


def q8_8(value: float) -> int:
    """Quantize a non-negative scalar into unsigned Q8.8."""
    return _check(int(round(float(value) * 256.0)), 16, "q8_8")


def encode_next_token(
    *,
    request_id: int = 0,
    sampling: SamplingMode = SamplingMode.ARGMAX,
    temperature_q8_8: int = 0,
    top_p_q8_8: int = 0x0100,
) -> int:
    """Run all NPU-resident layers and return one 32-bit token in status."""
    operand = (
        (_check(request_id, 16, "request_id") << 40)
        | (_check(int(sampling), 8, "sampling") << 32)
        | (_check(temperature_q8_8, 16, "temperature_q8_8") << 16)
        | _check(top_p_q8_8, 16, "top_p_q8_8")
    )
    return _pack_command(Opcode.NEXT_TOKEN, operand)


def decode_next_token(word64: int) -> dict[str, int | SamplingMode]:
    opcode, operand = decode_command(word64)
    if opcode != Opcode.NEXT_TOKEN:
        raise ValueError(f"opcode {opcode:#x} is not NEXT_TOKEN ({int(Opcode.NEXT_TOKEN):#x})")
    sampling_raw = (operand >> 32) & _mask(8)
    return {
        "request_id": (operand >> 40) & _mask(16),
        "sampling": SamplingMode(sampling_raw),
        "temperature_q8_8": (operand >> 16) & _mask(16),
        "top_p_q8_8": operand & _mask(16),
    }


def status_busy(stat: int) -> bool:
    return bool(stat & STAT_BUSY_MASK)


def status_done(stat: int) -> bool:
    return bool(stat & STAT_DONE_MASK)


def status_error(stat: int) -> bool:
    return bool(stat & STAT_ERROR_MASK)


def status_token_valid(stat: int) -> bool:
    return bool(stat & STAT_TOKEN_VALID_MASK)


def status_token(stat: int) -> int | None:
    stat = _check(stat, 64, "stat")
    if not status_token_valid(stat):
        return None
    return (stat >> STAT_TOKEN_SHIFT) & STAT_TOKEN_MASK


def encode_status(
    *,
    busy: bool = False,
    done: bool = False,
    error: bool = False,
    token: int | None = None,
) -> int:
    flags = (
        (STAT_BUSY_MASK if busy else 0)
        | (STAT_DONE_MASK if done else 0)
        | (STAT_ERROR_MASK if error else 0)
    )
    if token is None:
        return flags
    return ((_check(token, 32, "token") << STAT_TOKEN_SHIFT)
            | STAT_TOKEN_VALID_MASK
            | flags)


def manifest_id_from_text(value: object) -> int:
    return crc32(str(value).encode("utf-8")) & 0xFFFF_FFFF
