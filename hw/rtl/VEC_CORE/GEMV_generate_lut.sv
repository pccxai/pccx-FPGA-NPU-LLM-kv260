`timescale 1ns / 1ps

`include "GEMV_Vec_Matrix_MUL.svh"
`include "GLOBAL_CONST.svh"

// Descending order

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
    output logic signed [param.fixed_mant_width+2:0] OUT_fmap_LUT[0:param.fmap_cache_out_cnt-1][0:param.weight_width-1],
    output logic OUT_fmap_ready
);
  genvar idx, w;
  generate
    for (idx = 0; idx < param.fmap_cache_out_cnt; idx++) begin : fmap_lut_pre_cal
      wire signed [29:0] F;
      assign F = {{3{IN_fmap_broadcast[idx][26]}}, IN_fmap_broadcast[idx]};

      for (w = 0; w < 16; w++) begin : lut_entry
        // w - 8 = INT4 range (-8 ~ 7)
        assign OUT_fmap_LUT[idx][w] = F * $signed(5'(w) - 5'd8);
      end
    end
  endgenerate
endmodule
