`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===| Module: ctrl_npu_decoder — VLIW opcode → engine valid demux |============
// Purpose      : Receive raw 64-bit VLIW instructions from the frontend FIFO,
//                strip the 4-bit opcode, assert exactly one matching valid
//                pulse for one cycle, and forward the 60-bit body to the
//                Global Scheduler.
// Spec ref     : pccx v002 §3 (ISA), §3.1 (opcode encoding).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// Latency      : 1-cycle registered (raw_instruction_pop_valid → OUT_*_valid).
// Throughput   : 1 instruction/cycle (decoder is purely combinational save
//                for the output register).
// Handshake    : OUT_fetch_PC_ready asserted unconditionally — frontend FIFO
//                provides buffering, decoder is single-cycle.
// Reset state  : All OUT_*_op_x64_valid = 0; OUT_op_x64 = 0.
// Errors       : Unknown opcodes are silently dropped (no valid asserted).
// Assertions   : (Stage C) one-hot of OUT_*_op_x64_valid; valid pulses
//                are exactly one cycle wide.
// Notes        : OP_CVO uses a separate FF (cvo_valid_ff) outside the
//                4-bit OUT_valid bus because it is the 5th opcode.
// ===============================================================================

module ctrl_npu_decoder (
    input logic clk,
    input logic rst_n,

    // ===| From Frontend |=======================================================
    input logic [`ISA_WIDTH-1:0] IN_raw_instruction,
    input logic                  raw_instruction_pop_valid,

    // ===| Flow Control |========================================================
    output logic OUT_fetch_PC_ready,

    // ===| Decoded Valid Pulses (one-hot, one cycle) |===========================
    output logic OUT_GEMV_op_x64_valid,
    output logic OUT_GEMM_op_x64_valid,
    output logic OUT_memcpy_op_x64_valid,
    output logic OUT_memset_op_x64_valid,
    output logic OUT_cvo_op_x64_valid,

    // ===| Instruction Body (60-bit, opcode stripped) |=========================
    output instruction_op_x64_t OUT_op_x64
);

  // ===| Internal |==============================================================
  logic [3:0] OUT_valid;
  assign OUT_GEMV_op_x64_valid   = OUT_valid[0];
  assign OUT_GEMM_op_x64_valid   = OUT_valid[1];
  assign OUT_memcpy_op_x64_valid = OUT_valid[2];
  assign OUT_memset_op_x64_valid = OUT_valid[3];
  // CVO valid uses a separate FF (5th opcode)
  logic cvo_valid_ff;
  assign OUT_cvo_op_x64_valid = cvo_valid_ff;

  // ===| Opcode Decoder |========================================================
  // Top 4 bits are the opcode; bottom 60 bits are the instruction body.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      OUT_valid        <= 4'b0000;
      cvo_valid_ff     <= 1'b0;
      OUT_op_x64       <= '0;
    end else begin
      OUT_valid      <= 4'b0000;
      cvo_valid_ff   <= 1'b0;

      if (raw_instruction_pop_valid) begin
        // Body: bits [59:0] (opcode at [63:60] already stripped by slicing)
        OUT_op_x64.instruction <= IN_raw_instruction[`ISA_BODY_WIDTH-1:0];

        case (IN_raw_instruction[`ISA_WIDTH-1:`ISA_WIDTH-`ISA_OPCODE_WIDTH])
          OP_GEMV:   OUT_valid <= 4'b0001;
          OP_GEMM:   OUT_valid <= 4'b0010;
          OP_MEMCPY: OUT_valid <= 4'b0100;
          OP_MEMSET: OUT_valid <= 4'b1000;
          OP_CVO:    cvo_valid_ff <= 1'b1;
          default:   ;  // unknown opcode: drop silently
        endcase
      end
    end
  end

  // ===| Backpressure |==========================================================
  // Always ready — the frontend FIFO provides buffering; the decoder is single-cycle.
  assign OUT_fetch_PC_ready = 1'b1;

endmodule
