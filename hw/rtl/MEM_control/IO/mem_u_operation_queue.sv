`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===| Module: mem_u_operation_queue — scheduler↔L2 decoupling FIFOs |==========
// Purpose      : Decouple Global_Scheduler issuing rate from L2 cache
//                controller throughput. Provides two independent FIFO
//                channels (ACP and NPU) so neither blocks the other.
// Spec ref     : pccx v002 §5.4 (op queue), §4.2 (uop semantics).
// Clock        : clk_core @ 400 MHz.
// Reset        : rst_n_core active-low.
// Topology     : 2 × xpm_fifo_sync, depth 128, width 35, BRAM-backed,
//                prog_full threshold 100 (gives the scheduler 28 entries
//                of grace before back-pressuring upstream).
// Uop layout   : acp_uop_t / npu_uop_t are 35-bit packed structs:
//                  {write_en[0], base_addr[16:0], end_addr[16:0]} = 1+17+17 = 35.
// Latency      : 1 BRAM cycle from wr_en → dout (READ_MODE = "std").
// Throughput   : 1 push + 1 pop per channel per cycle (independent channels).
// Handshake    : OUT_*_cmd_valid asserts when (~busy && ~empty); pop fires
//                continuously while consumer is idle.
// Backpressure : OUT_*_cmd_fifo_full asserts at PROG_FULL_THRESH; upstream
//                stops issuing.
// Reset state  : Both FIFOs cleared.
// Counters     : none. (Stage D candidate: max_acp_depth, max_npu_depth.)
// Assertions   : (Stage C) — fifo over/underflow guarded by xpm_fifo_sync.
// ===============================================================================

module mem_u_operation_queue #() (
    input logic clk_core,
    input logic rst_n_core,

    // ===| ACP channel |=========================================================
    input  logic     IN_acp_rdy,
    input  acp_uop_t IN_acp_cmd,
    output acp_uop_t OUT_acp_cmd,
    output logic     OUT_acp_cmd_valid,
    output logic     OUT_acp_cmd_fifo_full,
    input  logic     IN_acp_is_busy,

    // ===| NPU internal channel |================================================
    input  logic     IN_npu_rdy,
    input  npu_uop_t IN_npu_cmd,
    output npu_uop_t OUT_npu_cmd,
    output logic     OUT_npu_cmd_valid,
    output logic     OUT_npu_cmd_fifo_full,
    input  logic     IN_npu_is_busy
);

  localparam int UopWidth = 35;  // 1 + 17 + 17 = write_en + base_addr + end_addr

  logic acp_fifo_empty;
  logic acp_fifo_full;
  logic npu_fifo_empty;
  logic npu_fifo_full;

  assign OUT_acp_cmd_fifo_full = acp_fifo_full;
  assign OUT_npu_cmd_fifo_full = npu_fifo_full;

  always_comb begin
    OUT_acp_cmd_valid = ~IN_acp_is_busy & ~acp_fifo_empty;
    OUT_npu_cmd_valid = ~IN_npu_is_busy & ~npu_fifo_empty;
  end

  // ===| ACP FIFO |==============================================================
  xpm_fifo_sync #(
      .FIFO_WRITE_DEPTH  (128),
      .WRITE_DATA_WIDTH  (UopWidth),
      .READ_DATA_WIDTH   (UopWidth),
      .FIFO_MEMORY_TYPE  ("block"),
      .READ_MODE         ("std"),
      .FULL_RESET_VALUE  (0),
      .PROG_FULL_THRESH  (100)
  ) u_acp_uop_fifo (
      .sleep    (1'b0),
      .rst      (~rst_n_core),
      .wr_clk   (clk_core),
      .wr_en    (IN_acp_rdy & ~acp_fifo_full),
      .din      (IN_acp_cmd),
      .prog_full(acp_fifo_full),
      .rd_en    (~IN_acp_is_busy & ~acp_fifo_empty),
      .dout     (OUT_acp_cmd),
      .empty    (acp_fifo_empty)
  );

  // ===| NPU FIFO |==============================================================
  xpm_fifo_sync #(
      .FIFO_WRITE_DEPTH  (128),
      .WRITE_DATA_WIDTH  (UopWidth),
      .READ_DATA_WIDTH   (UopWidth),
      .FIFO_MEMORY_TYPE  ("block"),
      .READ_MODE         ("std"),
      .FULL_RESET_VALUE  (0),
      .PROG_FULL_THRESH  (100)
  ) u_npu_uop_fifo (
      .sleep    (1'b0),
      .rst      (~rst_n_core),
      .wr_clk   (clk_core),
      .wr_en    (IN_npu_rdy & ~npu_fifo_full),
      .din      (IN_npu_cmd),
      .prog_full(npu_fifo_full),
      .rd_en    (~IN_npu_is_busy & ~npu_fifo_empty),
      .dout     (OUT_npu_cmd),
      .empty    (npu_fifo_empty)
  );

endmodule
