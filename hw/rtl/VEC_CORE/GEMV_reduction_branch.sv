`timescale 1ns / 1ps

`include "GEMV_Vec_Matrix_MUL.svh"
`include "GLOBAL_CONST.svh"

// ===| Module: GEMV_reduction_branch — single muV-Core lane wrapper |===========
// Purpose      : One independent lane of the 4-lane Vector Core. Pairs the
//                weight-indexed reduction stage with the per-lane recurrent
//                accumulator that walks the GEMV inner-product schedule.
// Spec ref     : pccx v002 §2.3.2 (per-lane MAC + reduction tree).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// Sub-units    : GEMV_reduction (5-stage 32 → 1 reduction tree)
//                GEMV_accumulate (recurrent batch accumulator).
// Latency      : reduction = 5 cycles + accumulate scheduling per recur.
// Throughput   : 1 partial sum per cycle while IN_weight_valid &
//                IN_activated_lane both high.
// Handshake    : IN_weight_valid + IN_activated_lane gate the reduction
//                tree; init = fmap_ready opens a new GEMV batch.
// Backpressure : None — push-only lane; upstream HP CDC FIFO must pace.
// Reset state  : All output result valid = 0; result vector zeroed.
// Counters     : none.
// Protected    : Internals untouched (CLAUDE.md §6.2 — reduction math is
//                user-owned).
// ===============================================================================
module GEMV_reduction_branch
  import vec_core_pkg::*;
#(
    parameter gemv_cfg_t param = VecCoreDefaultCfg
) (
    input logic clk,
    input logic rst_n,

    input logic IN_weight_valid,
    input logic [param.weight_width - 1:0] IN_weight[0:param.weight_cnt-1],

    input logic fmap_ready,
    input logic [16:0] IN_num_recur,

    input logic IN_activated_lane,
    input logic signed [param.fixed_mant_width+2:0] IN_fmap_LUT [0:param.fmap_cache_out_cnt-1][0:(1<<param.weight_width)-1],

    output logic [param.fixed_mant_width+2:0] OUT_GEMV_result_vector[0:param.gemv_batch-1],
    output logic OUT_valid
);

  logic [param.fixed_mant_width+2:0] reduction_result_wire;

  logic reduction_res_valid_wire;

  GEMV_reduction #(
      .param(param)
  ) u_GEMV_reduction (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_fmap_LUT(IN_fmap_LUT),
      .IN_valid(IN_weight_valid),

      .IN_is_lane_active(IN_activated_lane),
      .IN_weight(IN_weight),

      .OUT_reduction_result(reduction_result_wire),
      .OUT_reduction_res_valid(reduction_res_valid_wire)
  );

  GEMV_accumulate #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_accumulate (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_reduction_result(reduction_result_wire),
      .init(fmap_ready),

      .IN_valid(reduction_res_valid_wire),
      .IN_num_recur(IN_num_recur),
      .OUT_GEMV_result_vector(OUT_GEMV_result_vector),
      .OUT_acc_valid(OUT_valid)
  );

endmodule
