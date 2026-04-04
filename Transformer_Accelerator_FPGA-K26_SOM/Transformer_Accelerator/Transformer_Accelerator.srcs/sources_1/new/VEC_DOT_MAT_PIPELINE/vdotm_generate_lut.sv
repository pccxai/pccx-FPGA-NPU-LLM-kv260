`timescale 1ns / 1ps

`include "vdotm_Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

// Descending order

module vdotm_generate_lut #(
    parameter fmap_line_length = 32

) (
    input logic [`FIXED_MANT_WIDTH-1:0] IN_fmap_broadcast      [0:`FMAP_CACHE_OUT_SIZE-1],
    input logic                         IN_fmap_broadcast_valid,

    // e_max (from Cache for Normalization alignment)
    input logic [`BF16_EXP_WIDTH-1:0] IN_cached_emax_out[0:`FMAP_CACHE_OUT_SIZE-1],

    output logic [`FIXED_MANT_WIDTH+2:0] OUT_fmap_low_LUT[0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1]
    output logic [`FIXED_MANT_WIDTH+2:0] OUT_fmap_high_LUT[0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1]
    output logic OUT_fmap_ready;
);
  genvar idx, w;
  generate
    for (idx = 0; idx < fmap_line_length; idx++) begin : fmap_lut_pre_cal
      wire signed [29:0] F;
      assign F = {{3{IN_fmap_broadcast[idx][26]}}, IN_fmap_broadcast[idx]};

      for (w = 0; w < 16; w++) begin : lut_entry
        // w - 8 = 실제 INT4 값 (-8 ~ 7)
        assign OUT_fmap_low_LUT[idx][w] = F * $signed(5'(w) - 5'd8);
        assign OUT_fmap_high_LUT[idx][w] = F * $signed(5'(w) - 5'd8);
      end
    end
  endgenerate
endmodule
