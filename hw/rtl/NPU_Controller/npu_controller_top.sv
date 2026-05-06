`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===| Module: npu_controller_top — control-plane wrapper |=====================
// Purpose      : Aggregate AXIL frontend + opcode decoder behind one boundary.
//                Hides FIFO/handshake/decoding complexity from NPU_top.
// Spec ref     : pccx v002 §3 (ISA), §4 (control plane).
// Clock        : clk (= clk_core, 400 MHz).
// Reset        : rst_n active-low; i_clear synchronous soft-clear.
// Latency      : Decode is 1-cycle registered after AXIL kick (ctrl_npu_decoder).
// Throughput   : Issues at most 1 decoded uop per clock (ISA serial issue).
// Handshake    : One-hot OUT_*_op_x64_valid pulses — exactly one (or none)
//                asserted per cycle. Raw 60-bit body driven on OUT_op_x64.
// Backpressure : Decoder gates pop via fetch_PC_ready; AXIL frontend respects.
// Reset state  : All OUT_*_op_x64_valid = 0; OUT_op_x64 = 0.
// Errors       : none surfaced (illegal opcodes silently dropped — TODO).
// Counters     : none.
// Assertions   : (Stage C) one-hot of decoded valids; raw_instruction stable
//                while pop_valid && !fetch_PC_ready.
// ===============================================================================

module npu_controller_top #() (
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // ===| AXI4-Lite Slave : PS <-> NPU control plane |=========================
    axil_if.slave S_AXIL_CTRL,

    // ===| Encoded Status |======================================================
    input logic [`ISA_WIDTH-1:0] IN_enc_stat,
    input logic                  IN_enc_valid,

    // ===| Decoded Instruction Valids |=========================================
    output logic OUT_GEMV_op_x64_valid,
    output logic OUT_GEMM_op_x64_valid,
    output logic OUT_memcpy_op_x64_valid,
    output logic OUT_memset_op_x64_valid,
    output logic OUT_cvo_op_x64_valid,

    // ===| Raw Instruction Body (60-bit, opcode stripped) |=====================
    output instruction_op_x64_t OUT_op_x64
);

  // ===| Internal Wires |========================================================
  logic [`ISA_WIDTH-1:0] raw_instruction;
  logic                  raw_instruction_pop_valid;
  logic                  fetch_PC_ready;

  // ===| Frontend : AXI-Lite CMD/STAT |==========================================
  ctrl_npu_frontend #() u_npu_frontend (
      .clk     (clk),
      .rst_n   (rst_n),
      .IN_clear(i_clear),

      .S_AXIL_CTRL(S_AXIL_CTRL),

      .OUT_RAW_instruction(raw_instruction),
      .OUT_kick           (raw_instruction_pop_valid),

      .IN_enc_stat (IN_enc_stat),
      .IN_enc_valid(IN_enc_valid),

      .IN_fetch_ready(fetch_PC_ready)
  );

  // ===| Decoder : Opcode -> Engine FIFOs |======================================
  ctrl_npu_decoder u_decoder (
      .clk                    (clk),
      .rst_n                  (rst_n),
      .IN_raw_instruction     (raw_instruction),
      .raw_instruction_pop_valid(raw_instruction_pop_valid),

      .OUT_fetch_PC_ready     (fetch_PC_ready),

      .OUT_GEMV_op_x64_valid  (OUT_GEMV_op_x64_valid),
      .OUT_GEMM_op_x64_valid  (OUT_GEMM_op_x64_valid),
      .OUT_memcpy_op_x64_valid(OUT_memcpy_op_x64_valid),
      .OUT_memset_op_x64_valid(OUT_memset_op_x64_valid),
      .OUT_cvo_op_x64_valid   (OUT_cvo_op_x64_valid),

      .OUT_op_x64(OUT_op_x64)
  );

endmodule
