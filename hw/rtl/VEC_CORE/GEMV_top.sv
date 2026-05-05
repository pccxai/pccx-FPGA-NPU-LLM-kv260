// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
`timescale 1ns / 1ps

`include "GEMV_Vec_Matrix_MUL.svh"
`include "GLOBAL_CONST.svh"

// ===| Module: GEMV_top — Vector Core (4 parallel muV-Core lanes) |==============
// Purpose      : INT4 weight × BF16 fmap GEMV for transformer decode tokens.
//                Each of A/B/C/D lanes runs an independent reduction branch
//                sharing the broadcast fmap LUT generator.
// Spec ref     : pccx v002 §2.3 (Vector Core), §3.2 (GEMV uop).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// Topology     : 1× GEMV_generate_lut (BF16 fmap → fixed-point LUT, shared)
//              + 4× GEMV_reduction_branch (per-lane MAC + reduction tree).
// Data widths  : INT4 weight (param.weight_width), fixed-point mantissa
//                (param.fixed_mant_width), output 1+2+mant width = signed
//                fixed-point (fixed_mant_width + 3 bits).
// Latency      : LUT + reduction tree depth (see GEMV_reduction_branch contract).
// Throughput   : 1 result/cycle per active lane in steady state, after the
//                first num_recur cycles of fmap accumulation.
// Handshake    : Per-lane weight valid+ready; activation gated by
//                IN_activated_lane[lane] so disabled lanes hold weight stream.
// Backpressure : OUT_weight_ready_* asserts only while reduction branch is
//                ready to consume the next weight cycle; upstream (HP2/HP3
//                CDC FIFO) must respect.
// Reset state  : All output result valid = 0; result vectors registered to 0.
// Errors       : none.
// Counters     : none. (Stage D: lane_active_cycles, weight_handshakes.)
// Assertions   : (Stage C) OUT_weight_ready_* never asserted while
//                IN_weight_valid_* low; OUT_result_valid_* one-hot per cycle
//                across activated lanes.
// ===============================================================================
// weight size = 4bit, feature_map size = BF16 (converted to fixed-point upstream).
module GEMV_top
  import vec_core_pkg::*;
#(
    parameter gemv_cfg_t param = VecCoreDefaultCfg,
    parameter A = 0,
    parameter B = 1,
    parameter C = 2,
    parameter D = 3
) (
    input logic clk,
    input logic rst_n,

    input logic IN_weight_valid_A,
    input logic IN_weight_valid_B,
    input logic IN_weight_valid_C,
    input logic IN_weight_valid_D,

    input logic [param.weight_width - 1:0] IN_weight_A[0:param.weight_cnt -1],
    input logic [param.weight_width - 1:0] IN_weight_B[0:param.weight_cnt -1],
    input logic [param.weight_width - 1:0] IN_weight_C[0:param.weight_cnt -1],
    input logic [param.weight_width - 1:0] IN_weight_D[0:param.weight_cnt -1],

    output logic OUT_weight_ready_A,
    output logic OUT_weight_ready_B,
    output logic OUT_weight_ready_C,
    output logic OUT_weight_ready_D,

    input logic [param.fixed_mant_width-1:0] IN_fmap_broadcast      [0:param.fmap_cache_out_cnt-1],
    input logic                              IN_fmap_broadcast_valid,

    input logic [16:0] IN_num_recur,
    // e_max (from Cache for Normalization alignment)
    input logic [dtype_pkg::Bf16ExpWidth-1:0] IN_cached_emax_out[0:param.fmap_cache_out_cnt-1],

    input logic IN_activated_lane[0:param.num_gemv_pipeline-1],

    // Per-lane batch vector. Width tracks `GEMV_reduction_branch`'s signed
    // fixed-point format (fixed_mant_width + 2 headroom + 1 sign).
    output logic [param.fixed_mant_width+2:0] OUT_final_fmap_A [0:param.gemv_batch-1],
    output logic [param.fixed_mant_width+2:0] OUT_final_fmap_B [0:param.gemv_batch-1],
    output logic [param.fixed_mant_width+2:0] OUT_final_fmap_C [0:param.gemv_batch-1],
    output logic [param.fixed_mant_width+2:0] OUT_final_fmap_D [0:param.gemv_batch-1],

    output logic OUT_result_valid_A,
    output logic OUT_result_valid_B,
    output logic OUT_result_valid_C,
    output logic OUT_result_valid_D
);

  logic signed [param.fixed_mant_width+2:0] fmap_LUT_wire[0:param.fmap_cache_out_cnt-1][0:(1<<param.weight_width)-1];

  logic fmap_ready_wire;
  logic fmap_broadcast_valid_q;

  // GEMV_generate_lut is combinational. The recurrent accumulator needs a
  // one-cycle batch-start pulse, so generate it at the clocked GEMV boundary
  // from the level-valid fmap broadcast.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      fmap_broadcast_valid_q <= 1'b0;
      fmap_ready_wire        <= 1'b0;
    end else begin
      fmap_ready_wire        <= IN_fmap_broadcast_valid & ~fmap_broadcast_valid_q;
      fmap_broadcast_valid_q <= IN_fmap_broadcast_valid;
    end
  end

  assign OUT_weight_ready_A = IN_activated_lane[A] & IN_weight_valid_A;
  assign OUT_weight_ready_B = IN_activated_lane[B] & IN_weight_valid_B;
  assign OUT_weight_ready_C = IN_activated_lane[C] & IN_weight_valid_C;
  assign OUT_weight_ready_D = IN_activated_lane[D] & IN_weight_valid_D;

  GEMV_generate_lut #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_generate_lut (
      .IN_fmap_broadcast(IN_fmap_broadcast),
      .IN_fmap_broadcast_valid(IN_fmap_broadcast_valid),
      .IN_cached_emax_out(IN_cached_emax_out),

      .OUT_fmap_LUT(fmap_LUT_wire)
  );


  GEMV_reduction_branch #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_reduction_branch_A (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_weight_valid(IN_weight_valid_A),
      .IN_weight(IN_weight_A),

      .fmap_ready(fmap_ready_wire),
      .IN_num_recur(IN_num_recur),  // shape x * y * z

      .IN_activated_lane(IN_activated_lane[A]),
      .IN_fmap_LUT(fmap_LUT_wire),

      .OUT_GEMV_result_vector(OUT_final_fmap_A),
      .OUT_valid(OUT_result_valid_A)
  );


  GEMV_reduction_branch #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_reduction_branch_B (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_weight_valid(IN_weight_valid_B),
      .IN_weight(IN_weight_B),

      .fmap_ready(fmap_ready_wire),
      .IN_num_recur(IN_num_recur),  // shape x * y * z

      .IN_activated_lane(IN_activated_lane[B]),
      .IN_fmap_LUT(fmap_LUT_wire),

      .OUT_GEMV_result_vector(OUT_final_fmap_B),
      .OUT_valid(OUT_result_valid_B)
  );

  GEMV_reduction_branch #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_reduction_branch_C (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_weight_valid(IN_weight_valid_C),
      .IN_weight(IN_weight_C),

      .fmap_ready(fmap_ready_wire),
      .IN_num_recur(IN_num_recur),  // shape x * y * z

      .IN_activated_lane(IN_activated_lane[C]),
      .IN_fmap_LUT(fmap_LUT_wire),

      .OUT_GEMV_result_vector(OUT_final_fmap_C),
      .OUT_valid(OUT_result_valid_C)
  );

  GEMV_reduction_branch #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_reduction_branch_D (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_weight_valid(IN_weight_valid_D),
      .IN_weight(IN_weight_D),

      .fmap_ready(fmap_ready_wire),
      .IN_num_recur(IN_num_recur),  // shape x * y * z

      .IN_activated_lane(IN_activated_lane[D]),
      .IN_fmap_LUT(fmap_LUT_wire),

      .OUT_GEMV_result_vector(OUT_final_fmap_D),
      .OUT_valid(OUT_result_valid_D)
  );

endmodule
