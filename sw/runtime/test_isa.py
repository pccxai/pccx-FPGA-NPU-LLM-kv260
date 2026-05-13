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


# ----------------------------------------------------------------------------
# Decoder + round-trip coverage
# ----------------------------------------------------------------------------

def test_decode_x64_split():
    assert isa.decode_x64(0xF000_0000_0000_0000) == (0xF, 0)
    assert isa.decode_x64(0x0000_0000_0000_0001) == (0x0, 1)
    assert isa.decode_x64(0x8000_0000_0000_0000) == (0x8, 0)  # KICK marker
    # Reject out-of-range
    import pytest
    with pytest.raises(ValueError):
        isa.decode_x64(1 << 64)


def test_decode_gemm_round_trip():
    args = dict(
        dest_reg=0x1ABCD,
        src_addr=0x12345,
        flags=isa.GemmFlags(findemax=True, accm=False, w_scale=True),
        size_ptr_addr=0x2A,
        shape_ptr_addr=0x15,
        parallel_lane=0x0F,
    )
    w = isa.encode_gemm(**args)
    decoded = isa.decode_gemm(w)
    assert decoded == args


def test_decode_gemv_round_trip():
    args = dict(
        dest_reg=0x0DEAD,
        src_addr=0x1BEEF,
        flags=isa.GemmFlags(findemax=False, accm=True, w_scale=False),
        size_ptr_addr=0x00,
        shape_ptr_addr=0x3F,
        parallel_lane=0x1F,
    )
    w = isa.encode_gemv(**args)
    decoded = isa.decode_gemv(w)
    assert decoded == args


def test_decode_memcpy_round_trip():
    args = dict(
        from_device=isa.FROM_HOST,
        to_device=isa.TO_NPU,
        dest_addr=0x1FFFF,
        src_addr=0x10000,
        aux_addr=0x05555,
        shape_ptr_addr=0x2A,
        async_op=isa.ASYNC,
    )
    w = isa.encode_memcpy(**args)
    decoded = isa.decode_memcpy(w)
    assert decoded == args


def test_decode_memset_round_trip():
    args = dict(
        dest_cache=0x3,
        dest_addr=0x3F,
        a_value=0xAAAA,
        b_value=0xBBBB,
        c_value=0xCCCC,
    )
    w = isa.encode_memset(**args)
    decoded = isa.decode_memset(w)
    assert decoded == args


def test_decode_cvo_round_trip():
    args = dict(
        cvo_func=isa.CvoFunc.GELU,
        src_addr=0x1ABCD,
        dst_addr=0x12345,
        length=0xBEEF,
        flags=0x1F,
        async_op=isa.ASYNC,
    )
    w = isa.encode_cvo(**args)
    decoded = isa.decode_cvo(w)
    assert decoded == args


def test_decoder_opcode_mismatch_raises():
    import pytest
    gemm_word = isa.encode_gemm(0, 0)
    with pytest.raises(ValueError, match="not GEMV"):
        isa.decode_gemv(gemm_word)
    with pytest.raises(ValueError, match="not MEMCPY"):
        isa.decode_memcpy(gemm_word)
    with pytest.raises(ValueError, match="not MEMSET"):
        isa.decode_memset(gemm_word)
    with pytest.raises(ValueError, match="not CVO"):
        isa.decode_cvo(gemm_word)
    memcpy_word = isa.encode_memcpy(0, 0, 0, 0)
    with pytest.raises(ValueError, match="not GEMM"):
        isa.decode_gemm(memcpy_word)


# ----------------------------------------------------------------------------
# Boundary + adversarial cases
# ----------------------------------------------------------------------------

def test_max_fields_no_overflow_between_neighbors():
    """When every field is at its max, no field bleeds into a neighbor.

    Detects off-by-one bit shifts that would let a saturated field overwrite
    the adjacent one.
    """
    # GEMM: every field max
    w = isa.encode_gemm(
        dest_reg=0x1FFFF,
        src_addr=0x1FFFF,
        flags=isa.GemmFlags(findemax=True, accm=True, w_scale=True),
        size_ptr_addr=0x3F,
        shape_ptr_addr=0x3F,
        parallel_lane=0x1F,
    )
    d = isa.decode_gemm(w)
    assert d["dest_reg"]       == 0x1FFFF
    assert d["src_addr"]       == 0x1FFFF
    assert d["flags"]          == isa.GemmFlags(findemax=True, accm=True, w_scale=True)
    assert d["size_ptr_addr"]  == 0x3F
    assert d["shape_ptr_addr"] == 0x3F
    assert d["parallel_lane"]  == 0x1F

    # MEMCPY: every field max
    w = isa.encode_memcpy(
        from_device=1, to_device=1,
        dest_addr=0x1FFFF, src_addr=0x1FFFF, aux_addr=0x1FFFF,
        shape_ptr_addr=0x3F, async_op=1,
    )
    d = isa.decode_memcpy(w)
    assert d["from_device"]    == 1
    assert d["to_device"]      == 1
    assert d["dest_addr"]      == 0x1FFFF
    assert d["src_addr"]       == 0x1FFFF
    assert d["aux_addr"]       == 0x1FFFF
    assert d["shape_ptr_addr"] == 0x3F
    assert d["async_op"]       == 1


def test_negative_value_rejected():
    """Unsigned fields must reject negatives; the encoder check uses `< 0`."""
    import pytest
    with pytest.raises(ValueError):
        isa.encode_gemm(dest_reg=-1, src_addr=0)
    with pytest.raises(ValueError):
        isa.encode_memcpy(0, 0, dest_addr=-1, src_addr=0)
    with pytest.raises(ValueError):
        isa.encode_memset(dest_cache=0, dest_addr=0, a_value=-1)
    with pytest.raises(ValueError):
        isa.encode_cvo(isa.CvoFunc.EXP, src_addr=-1, dst_addr=0, length=0)


def test_off_by_one_max_plus_one_rejected():
    """Each field exactly one past its max must raise."""
    import pytest
    with pytest.raises(ValueError):
        isa.encode_gemm(dest_reg=0x20000, src_addr=0)            # 17-bit + 1
    with pytest.raises(ValueError):
        isa.encode_gemm(0, 0, size_ptr_addr=0x40)                # 6-bit + 1
    with pytest.raises(ValueError):
        isa.encode_gemm(0, 0, parallel_lane=0x20)                # 5-bit + 1
    with pytest.raises(ValueError):
        isa.encode_memset(dest_cache=0x4, dest_addr=0, a_value=0)  # 2-bit + 1
    with pytest.raises(ValueError):
        isa.encode_memset(0, 0x40, 0)                            # 6-bit + 1
    with pytest.raises(ValueError):
        isa.encode_memset(0, 0, 0x10000)                         # 16-bit + 1
    with pytest.raises(ValueError):
        isa.encode_cvo(isa.CvoFunc.EXP, 0, 0, length=0x10000)    # 16-bit + 1


def test_kick_marker_is_not_a_valid_opcode_body():
    """KICK_MARKER decodes to opcode 0x8 with body 0 - confirms it can't be
    mistaken for any of the 5 real ops (0..4)."""
    op, body = isa.decode_x64(isa.KICK_MARKER)
    assert op == 0x8
    assert body == 0
    assert op not in (int(isa.Opcode.GEMV),
                       int(isa.Opcode.GEMM),
                       int(isa.Opcode.MEMCPY),
                       int(isa.Opcode.MEMSET),
                       int(isa.Opcode.CVO))
