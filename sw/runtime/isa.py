"""pccx v002 ISA word encoders.

Layout (mirrors hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv):

  bits [63:60] = 4-bit opcode
  bits [59:0]  = 60-bit instruction body, format depends on opcode

KICK is special — pushed by writing any word to ADDR_KICK = 0x008; the RTL
overrides the wdata with 0x8000_0000_0000_0000 internally, so KICK_MARKER is
provided here only for symmetry / asserts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Opcode(IntEnum):
    GEMV   = 0x0
    GEMM   = 0x1
    MEMCPY = 0x2
    MEMSET = 0x3
    CVO    = 0x4


class CvoFunc(IntEnum):
    EXP        = 0x0
    SQRT       = 0x1
    GELU       = 0x2
    SIN        = 0x3
    COS        = 0x4
    REDUCE_SUM = 0x5
    SCALE      = 0x6
    RECIP      = 0x7


# Address direction enums (memcpy)
FROM_NPU, FROM_HOST = 0, 1
TO_NPU,   TO_HOST   = 0, 1
SYNC, ASYNC         = 0, 1


KICK_MARKER = 0x8000_0000_0000_0000


def _mask(width: int) -> int:
    return (1 << width) - 1


def _check(val: int, width: int, name: str) -> int:
    m = _mask(width)
    if val < 0 or val > m:
        raise ValueError(f"{name}={val:#x} does not fit in {width} bits (max {m:#x})")
    return val & m


def _pack_opcode(opcode: Opcode, body: int) -> int:
    """body must already fit in 60 bits."""
    body = _check(body, 60, "body")
    return (int(opcode) & 0xF) << 60 | body


@dataclass(frozen=True)
class GemmFlags:
    """6-bit GEMV/GEMM flags struct (isa_pkg.sv flags_t)."""
    findemax: bool = False
    accm:     bool = False
    w_scale:  bool = False

    def to_bits(self) -> int:
        # bits: findemax[5] | accm[4] | w_scale[3] | reserved[2:0]
        return ((int(self.findemax) << 5)
                | (int(self.accm)     << 4)
                | (int(self.w_scale)  << 3))


def encode_gemm(
    dest_reg: int,
    src_addr: int,
    flags: GemmFlags = GemmFlags(),
    size_ptr_addr: int = 0,
    shape_ptr_addr: int = 0,
    parallel_lane: int = 0,
) -> int:
    """GEMM_op_x64_t — same layout as GEMV.

    Body (60 bits, MSB first):
      dest_reg[16:0] (17) | src_addr[16:0] (17) | flags (6) |
      size_ptr (6) | shape_ptr (6) | parallel_lane (5) | reserved (3)
    """
    body = ((_check(dest_reg,        17, "dest_reg")       << 43)
            | (_check(src_addr,       17, "src_addr")       << 26)
            | (flags.to_bits()                              << 20)
            | (_check(size_ptr_addr,   6, "size_ptr_addr")  << 14)
            | (_check(shape_ptr_addr,  6, "shape_ptr_addr") <<  8)
            | (_check(parallel_lane,   5, "parallel_lane")  <<  3))
    return _pack_opcode(Opcode.GEMM, body)


def encode_gemv(
    dest_reg: int,
    src_addr: int,
    flags: GemmFlags = GemmFlags(),
    size_ptr_addr: int = 0,
    shape_ptr_addr: int = 0,
    parallel_lane: int = 0,
) -> int:
    """GEMV_op_x64_t — identical layout to GEMM, opcode differs."""
    body = ((_check(dest_reg,        17, "dest_reg")       << 43)
            | (_check(src_addr,       17, "src_addr")       << 26)
            | (flags.to_bits()                              << 20)
            | (_check(size_ptr_addr,   6, "size_ptr_addr")  << 14)
            | (_check(shape_ptr_addr,  6, "shape_ptr_addr") <<  8)
            | (_check(parallel_lane,   5, "parallel_lane")  <<  3))
    return _pack_opcode(Opcode.GEMV, body)


def encode_memcpy(
    from_device: int,
    to_device: int,
    dest_addr: int,
    src_addr: int,
    aux_addr: int = 0,
    shape_ptr_addr: int = 0,
    async_op: int = SYNC,
) -> int:
    """MEMCPY layout (60 bits): from(1) to(1) dest(17) src(17) aux(17) shape(6) async(1)."""
    body = ((_check(from_device, 1, "from_device") << 59)
            | (_check(to_device,   1, "to_device")   << 58)
            | (_check(dest_addr,  17, "dest_addr")   << 41)
            | (_check(src_addr,   17, "src_addr")    << 24)
            | (_check(aux_addr,   17, "aux_addr")    <<  7)
            | (_check(shape_ptr_addr, 6, "shape_ptr_addr") << 1)
            | (_check(async_op,    1, "async_op")    <<  0))
    return _pack_opcode(Opcode.MEMCPY, body)


def encode_memset(
    dest_cache: int,
    dest_addr: int,
    a_value: int,
    b_value: int = 0,
    c_value: int = 0,
) -> int:
    """MEMSET layout (60 bits): dest_cache(2) dest_addr(6) a(16) b(16) c(16) reserved(4)."""
    body = ((_check(dest_cache, 2, "dest_cache") << 58)
            | (_check(dest_addr,  6, "dest_addr")  << 52)
            | (_check(a_value,   16, "a_value")    << 36)
            | (_check(b_value,   16, "b_value")    << 20)
            | (_check(c_value,   16, "c_value")    <<  4))
    return _pack_opcode(Opcode.MEMSET, body)


def encode_cvo(
    cvo_func: CvoFunc,
    src_addr: int,
    dst_addr: int,
    length: int,
    flags: int = 0,
    async_op: int = SYNC,
) -> int:
    """CVO layout (60 bits): func(4) src(17) dst(17) length(16) flags(5) async(1)."""
    body = ((_check(int(cvo_func), 4, "cvo_func")  << 56)
            | (_check(src_addr,   17, "src_addr")    << 39)
            | (_check(dst_addr,   17, "dst_addr")    << 22)
            | (_check(length,     16, "length")      <<  6)
            | (_check(flags,       5, "flags")       <<  1)
            | (_check(async_op,    1, "async_op")    <<  0))
    return _pack_opcode(Opcode.CVO, body)


# Status word (read from AXIL_STAT_OUT, 64-bit but only [31:0] is mmio_npu_stat)
STAT_BUSY_MASK = 0x1
STAT_DONE_MASK = 0x2


def status_busy(stat: int) -> bool:
    return bool(stat & STAT_BUSY_MASK)


def status_done(stat: int) -> bool:
    return bool(stat & STAT_DONE_MASK)


# ============================================================================
# Decoders — inverse of the encode_* functions. Used by test_isa.py round-trip
# coverage and host-side debug tools that inspect the raw 64-bit words pulled
# off /dev/mem traces.
#
# Each decoder takes a 64-bit ISA word and returns a dict of the named fields
# (matching the encode_* signature). decode_x64() returns just the opcode and
# raw body for callers that want to dispatch themselves.
# ============================================================================


def decode_x64(word64: int) -> tuple[int, int]:
    """Split a 64-bit ISA word into (opcode, 60-bit body)."""
    word64 = _check(word64, 64, "word64")
    return ((word64 >> 60) & 0xF, word64 & ((1 << 60) - 1))


def _decode_field(body: int, lsb: int, width: int) -> int:
    """Extract `width` bits starting at `lsb` from a 60-bit body."""
    return (body >> lsb) & _mask(width)


def decode_gemm(word64: int) -> dict:
    """Inverse of encode_gemm. Returns dict matching encode_gemm kwargs."""
    op, body = decode_x64(word64)
    if op != Opcode.GEMM:
        raise ValueError(f"opcode {op:#x} is not GEMM ({int(Opcode.GEMM):#x})")
    flags_bits = _decode_field(body, 20, 6)
    return {
        "dest_reg":       _decode_field(body, 43, 17),
        "src_addr":       _decode_field(body, 26, 17),
        "flags":          GemmFlags(
                              findemax=bool(flags_bits & (1 << 5)),
                              accm=    bool(flags_bits & (1 << 4)),
                              w_scale= bool(flags_bits & (1 << 3))),
        "size_ptr_addr":  _decode_field(body, 14, 6),
        "shape_ptr_addr": _decode_field(body,  8, 6),
        "parallel_lane":  _decode_field(body,  3, 5),
    }


def decode_gemv(word64: int) -> dict:
    """Inverse of encode_gemv (same layout as GEMM, opcode differs)."""
    op, body = decode_x64(word64)
    if op != Opcode.GEMV:
        raise ValueError(f"opcode {op:#x} is not GEMV ({int(Opcode.GEMV):#x})")
    flags_bits = _decode_field(body, 20, 6)
    return {
        "dest_reg":       _decode_field(body, 43, 17),
        "src_addr":       _decode_field(body, 26, 17),
        "flags":          GemmFlags(
                              findemax=bool(flags_bits & (1 << 5)),
                              accm=    bool(flags_bits & (1 << 4)),
                              w_scale= bool(flags_bits & (1 << 3))),
        "size_ptr_addr":  _decode_field(body, 14, 6),
        "shape_ptr_addr": _decode_field(body,  8, 6),
        "parallel_lane":  _decode_field(body,  3, 5),
    }


def decode_memcpy(word64: int) -> dict:
    op, body = decode_x64(word64)
    if op != Opcode.MEMCPY:
        raise ValueError(f"opcode {op:#x} is not MEMCPY ({int(Opcode.MEMCPY):#x})")
    return {
        "from_device":    _decode_field(body, 59,  1),
        "to_device":      _decode_field(body, 58,  1),
        "dest_addr":      _decode_field(body, 41, 17),
        "src_addr":       _decode_field(body, 24, 17),
        "aux_addr":       _decode_field(body,  7, 17),
        "shape_ptr_addr": _decode_field(body,  1,  6),
        "async_op":       _decode_field(body,  0,  1),
    }


def decode_memset(word64: int) -> dict:
    op, body = decode_x64(word64)
    if op != Opcode.MEMSET:
        raise ValueError(f"opcode {op:#x} is not MEMSET ({int(Opcode.MEMSET):#x})")
    return {
        "dest_cache": _decode_field(body, 58,  2),
        "dest_addr":  _decode_field(body, 52,  6),
        "a_value":    _decode_field(body, 36, 16),
        "b_value":    _decode_field(body, 20, 16),
        "c_value":    _decode_field(body,  4, 16),
    }


def decode_cvo(word64: int) -> dict:
    op, body = decode_x64(word64)
    if op != Opcode.CVO:
        raise ValueError(f"opcode {op:#x} is not CVO ({int(Opcode.CVO):#x})")
    return {
        "cvo_func": CvoFunc(_decode_field(body, 56, 4)),
        "src_addr": _decode_field(body, 39, 17),
        "dst_addr": _decode_field(body, 22, 17),
        "length":   _decode_field(body,  6, 16),
        "flags":    _decode_field(body,  1,  5),
        "async_op": _decode_field(body,  0,  1),
    }
