"""Unit tests for sw/runtime/isa.py encoders.

Validates that each encoder produces the exact bit pattern the RTL decoder
expects per hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv.
"""
from __future__ import annotations

from sw.runtime import isa


def _opcode_of(word64: int) -> int:
    return (word64 >> 60) & 0xF


def _body_of(word64: int) -> int:
    return word64 & ((1 << 60) - 1)


def test_opcode_placement():
    """Opcode is in bits [63:60]; body fills [59:0]."""
    w = isa.encode_gemm(dest_reg=0, src_addr=0)
    assert _opcode_of(w) == isa.Opcode.GEMM
    w = isa.encode_gemv(dest_reg=0, src_addr=0)
    assert _opcode_of(w) == isa.Opcode.GEMV
    w = isa.encode_memcpy(0, 0, 0, 0)
    assert _opcode_of(w) == isa.Opcode.MEMCPY
    w = isa.encode_memset(0, 0, 0)
    assert _opcode_of(w) == isa.Opcode.MEMSET
    w = isa.encode_cvo(isa.CvoFunc.EXP, 0, 0, 0)
    assert _opcode_of(w) == isa.Opcode.CVO


def test_gemm_fields_at_known_offsets():
    """Verify each GEMM field lands at the bit offset isa_pkg.sv declares."""
    # dest_reg [17 bits] at [59:43]
    w = isa.encode_gemm(dest_reg=0x1FFFF, src_addr=0)
    assert (_body_of(w) >> 43) & 0x1FFFF == 0x1FFFF
    # src_addr [17 bits] at [42:26]
    w = isa.encode_gemm(dest_reg=0, src_addr=0x1FFFF)
    assert (_body_of(w) >> 26) & 0x1FFFF == 0x1FFFF
    # size_ptr_addr [6 bits] at [19:14]
    w = isa.encode_gemm(0, 0, size_ptr_addr=0x3F)
    assert (_body_of(w) >> 14) & 0x3F == 0x3F
    # shape_ptr_addr [6 bits] at [13:8]
    w = isa.encode_gemm(0, 0, shape_ptr_addr=0x3F)
    assert (_body_of(w) >> 8) & 0x3F == 0x3F
    # parallel_lane [5 bits] at [7:3]
    w = isa.encode_gemm(0, 0, parallel_lane=0x1F)
    assert (_body_of(w) >> 3) & 0x1F == 0x1F


def test_gemm_flags():
    """flags_t: findemax[5] | accm[4] | w_scale[3] | reserved[2:0]."""
    f = isa.GemmFlags(findemax=True, accm=False, w_scale=False)
    assert f.to_bits() == 0b100_000
    f = isa.GemmFlags(findemax=False, accm=True, w_scale=False)
    assert f.to_bits() == 0b010_000
    f = isa.GemmFlags(findemax=True, accm=True, w_scale=True)
    assert f.to_bits() == 0b111_000
    # When passed to encode_gemm, flags occupies bits [25:20] of body
    w = isa.encode_gemm(0, 0, flags=isa.GemmFlags(findemax=True, accm=True, w_scale=True))
    assert (_body_of(w) >> 20) & 0x3F == 0b111_000


def test_memcpy_fields():
    """memcpy: from(1) to(1) dest(17) src(17) aux(17) shape(6) async(1)."""
    w = isa.encode_memcpy(
        from_device=isa.FROM_HOST,
        to_device=isa.TO_NPU,
        dest_addr=0x1FFFF,
        src_addr=0x12345,
        aux_addr=0x0,
        shape_ptr_addr=0x2A,
        async_op=isa.ASYNC,
    )
    assert _opcode_of(w) == isa.Opcode.MEMCPY
    assert (_body_of(w) >> 59) & 0x1 == 1   # from_device
    assert (_body_of(w) >> 58) & 0x1 == 0   # to_device
    assert (_body_of(w) >> 41) & 0x1FFFF == 0x1FFFF  # dest_addr
    assert (_body_of(w) >> 24) & 0x1FFFF == 0x12345  # src_addr
    assert (_body_of(w) >> 1)  & 0x3F == 0x2A         # shape_ptr_addr
    assert _body_of(w) & 0x1 == 1                     # async


def test_memset_fields():
    """memset: dest_cache(2) dest_addr(6) a(16) b(16) c(16) reserved(4)."""
    w = isa.encode_memset(
        dest_cache=0x2,
        dest_addr=0x3F,
        a_value=0xAAAA,
        b_value=0xBBBB,
        c_value=0xCCCC,
    )
    assert _opcode_of(w) == isa.Opcode.MEMSET
    assert (_body_of(w) >> 58) & 0x3 == 0x2
    assert (_body_of(w) >> 52) & 0x3F == 0x3F
    assert (_body_of(w) >> 36) & 0xFFFF == 0xAAAA
    assert (_body_of(w) >> 20) & 0xFFFF == 0xBBBB
    assert (_body_of(w) >> 4)  & 0xFFFF == 0xCCCC


def test_cvo_fields():
    """cvo: func(4) src(17) dst(17) length(16) flags(5) async(1)."""
    w = isa.encode_cvo(
        cvo_func=isa.CvoFunc.GELU,
        src_addr=0x1ABCD,
        dst_addr=0x12345,
        length=0xBEEF,
        flags=0x1F,
        async_op=isa.ASYNC,
    )
    assert _opcode_of(w) == isa.Opcode.CVO
    assert (_body_of(w) >> 56) & 0xF == int(isa.CvoFunc.GELU)
    assert (_body_of(w) >> 39) & 0x1FFFF == 0x1ABCD
    assert (_body_of(w) >> 22) & 0x1FFFF == 0x12345
    assert (_body_of(w) >> 6)  & 0xFFFF == 0xBEEF
    assert (_body_of(w) >> 1)  & 0x1F == 0x1F
    assert _body_of(w) & 0x1 == 1


def test_out_of_range_raises():
    import pytest
    with pytest.raises(ValueError):
        isa.encode_gemm(dest_reg=0x20000, src_addr=0)   # > 17 bits
    with pytest.raises(ValueError):
        isa.encode_memset(dest_cache=4, dest_addr=0, a_value=0)  # > 2 bits


def test_kick_marker_bit63_only():
    assert isa.KICK_MARKER == 0x8000_0000_0000_0000
    assert (isa.KICK_MARKER >> 63) & 0x1 == 1
    assert isa.KICK_MARKER & ((1 << 63) - 1) == 0


def test_status_helpers():
    assert isa.status_busy(0b00) is False
    assert isa.status_busy(0b01) is True
    assert isa.status_busy(0b11) is True
    assert isa.status_done(0b00) is False
    assert isa.status_done(0b10) is True
    assert isa.status_done(0b11) is True
