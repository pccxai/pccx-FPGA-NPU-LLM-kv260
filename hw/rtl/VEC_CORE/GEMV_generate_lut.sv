`timescale 1ns / 1ps

`include "GEMV_Vec_Matrix_MUL.svh"
`include "GLOBAL_CONST.svh"

// ===| Module: GEMV_generate_lut — INT4 weight × fmap product LUT |=============
// Purpose      : For each fmap lane, pre-multiply the broadcast fmap by all
//                16 possible signed INT4 weight values (-8..7) to form a LUT
//                that the per-lane reduction stage can index by weight.
//                Trades 16 multipliers per fmap lane for an O(1) lookup
//                inside the reduction tree.
// Spec ref     : pccx v002 §2.3.1 (GEMV LUT generation).
// Clock        : combinational (no register).
// Reset        : N/A.
// Geometry     : param.fmap_cache_out_cnt fmap lanes × 16 weight values.
//                Each LUT entry is signed (param.fixed_mant_width + 3) bits.
// Latency      : 0 cycles (pure combinational).
// Throughput   : 1 LUT update per IN_fmap_broadcast_valid cycle.
// Reset state  : N/A.
// Notes        : F is sign-extended to 30-bit signed before the multiply
//                with (w − 8); INT4 range is [-8, 7] so the offset places
//                the LUT index at [0, 15].
// ===============================================================================
// LUT entry order: descending (w = 0 .. 15 maps to weight = -8 .. 7).
module GEMV_generate_lut
  import vec_core_pkg::*;
#(
    parameter gemv_cfg_t param = VecCoreDefaultCfg
) (
    input logic [param.fixed_mant_width-1:0] IN_fmap_broadcast      [0:param.fmap_cache_out_cnt-1],
    input logic                              IN_fmap_broadcast_valid,

    // e_max (from Cache for Normalization alignment) — only the 8-bit BF16
    // exponent field is carried here; matches GEMV_top's port width.
    input logic [dtype_pkg::Bf16ExpWidth-1:0] IN_cached_emax_out[0:param.fmap_cache_out_cnt-1],
    // LUT depth = 2^weight_width (one entry per signed INT4 value, -8..+7).
    output logic signed [param.fixed_mant_width+2:0] OUT_fmap_LUT[0:param.fmap_cache_out_cnt-1][0:(1<<param.weight_width)-1],
    // OUT_fmap_ready feeds GEMV_accumulate.init, which the reduction_branch
    // header documents as a one-cycle pulse opening a new GEMV batch. This
    // module is pure-combinational with no clk/rst_n, so it cannot produce
    // such a pulse on its own; fmap_cache.rd_valid is also a level held
    // high across the 2048-cycle broadcast burst. Driving this port from
    // the level-valid signal would gate the accumulator off for the whole
    // burst. Left undriven until a clocked edge-detector lands at the
    // consumer side.
    output logic OUT_fmap_ready
);
  genvar idx, w;
  generate
    for (idx = 0; idx < param.fmap_cache_out_cnt; idx++) begin : fmap_lut_pre_cal
      wire signed [29:0] F;
      assign F = {{3{IN_fmap_broadcast[idx][26]}}, IN_fmap_broadcast[idx]};

      for (w = 0; w < (1<<param.weight_width); w++) begin : lut_entry
        // w - 8 = INT4 range (-8 ~ 7)
        assign OUT_fmap_LUT[idx][w] = F * $signed(5'(w) - 5'd8);
      end
    end
  endgenerate
endmodule
