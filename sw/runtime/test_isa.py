"""Unit tests for high-level sw/runtime/isa.py command encoders."""
from __future__ import annotations

import pytest

from sw.runtime import isa


def _opcode_of(word64: int) -> int:
    return (word64 >> 56) & 0xFF


def _operand_of(word64: int) -> int:
    return word64 & ((1 << 56) - 1)


def test_opcode_placement() -> None:
    assert _opcode_of(isa.encode_reset_kv_cache()) == isa.Opcode.RESET_KV_CACHE
    assert _opcode_of(isa.encode_load_weight()) == isa.Opcode.LOAD_WEIGHT
    assert _opcode_of(isa.encode_load_prompt(position=0, token_id=0)) == isa.Opcode.LOAD_PROMPT
    assert _opcode_of(isa.encode_next_token()) == isa.Opcode.NEXT_TOKEN


def test_no_per_matmul_opcode_surface() -> None:
    assert not hasattr(isa.Opcode, "GEMM")
    assert not hasattr(isa.Opcode, "GEMV")
    assert not hasattr(isa, "encode_gemm")
    assert not hasattr(isa, "encode_gemv")


def test_load_weight_fields() -> None:
    word = isa.encode_load_weight(
        weight_slot=0x12,
        flags=0x34,
        descriptor_count=0x56,
        manifest_id=0x89AB_CDEF,
    )

    assert _operand_of(word) == 0x12345689ABCDEF
    assert isa.decode_load_weight(word) == {
        "weight_slot": 0x12,
        "flags": 0x34,
        "descriptor_count": 0x56,
        "manifest_id": 0x89AB_CDEF,
    }


def test_load_prompt_fields() -> None:
    word = isa.encode_load_prompt(position=0x00ABCD, token_id=0x12345678)

    assert _operand_of(word) == 0x00ABCD12345678
    assert isa.decode_load_prompt(word) == {
        "position": 0x00ABCD,
        "token_id": 0x12345678,
    }


def test_next_token_fields() -> None:
    word = isa.encode_next_token(
        request_id=0x1234,
        sampling=isa.SamplingMode.RANDOM,
        temperature_q8_8=isa.q8_8(0.75),
        top_p_q8_8=isa.q8_8(0.95),
    )

    assert isa.decode_next_token(word) == {
        "request_id": 0x1234,
        "sampling": isa.SamplingMode.RANDOM,
        "temperature_q8_8": 0x00C0,
        "top_p_q8_8": 0x00F3,
    }


def test_reset_kv_cache_fields() -> None:
    word = isa.encode_reset_kv_cache(session_id=0xCAFE_BABE)
    assert isa.decode_reset_kv_cache(word) == {"session_id": 0xCAFE_BABE}


def test_decode_command_split_and_kick_marker() -> None:
    assert isa.decode_command(isa.KICK_MARKER) == (0x80, 0)
    assert _opcode_of(isa.KICK_MARKER) not in [int(op) for op in isa.Opcode]


def test_status_token_helpers() -> None:
    empty_done = isa.encode_status(done=True)
    assert isa.status_done(empty_done) is True
    assert isa.status_token(empty_done) is None

    token_done = isa.encode_status(done=True, token=0x1234_5678)
    assert isa.status_done(token_done) is True
    assert isa.status_token_valid(token_done) is True
    assert isa.status_token(token_done) == 0x1234_5678
    assert (token_done >> isa.STAT_TOKEN_SHIFT) & isa.STAT_TOKEN_MASK == 0x1234_5678


def test_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        isa.encode_load_prompt(position=1 << 24, token_id=0)
    with pytest.raises(ValueError):
        isa.encode_load_prompt(position=0, token_id=1 << 32)
    with pytest.raises(ValueError):
        isa.encode_load_weight(descriptor_count=1 << 8)
    with pytest.raises(ValueError):
        isa.encode_next_token(request_id=1 << 16)
    with pytest.raises(ValueError):
        isa.encode_status(token=1 << 32)
