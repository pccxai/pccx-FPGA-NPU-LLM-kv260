`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps

// ===============================================================================
// ===| BF16 Barrel Shifter |=====================================================
//
// Converts one BF16 value to a 27-bit fixed-point mantissa aligned to a given
// e_max exponent.  Combinational — no registers.
//
// Output delayLine_in: 27-bit aligned mantissa in 2's complement signed format.
//   bit[26]   = sign (MSB of 2's complement)
//   bit[25:19]= integer part  (max = 2^6 when delta_e = 0)
//   bit[18:0] = fractional part
//
// exp_out  : passes through IN e_max (used by downstream delay line tracking).
// sign_out : BF16 sign bit (positive = 0).
//
// ===============================================================================

module barrel_shifter_BF16 #(
) (
    input logic [15:0] bf16_act,
    input logic [ 7:0] e_max,

    output logic [26:0] delayLine_in,
    output logic [ 7:0] exp_out,
    output logic        sign_out
);

  logic        sign;
  logic [ 7:0] exp;
  logic [ 7:0] mantissa;  // includes hidden bit
  logic [ 7:0] delta_e;
  logic [26:0] base_vec;
  logic [26:0] shifted;

  assign sign = bf16_act[15];
  assign exp = bf16_act[14:7];
  assign mantissa = (exp == 8'd0) ? {1'b0, bf16_act[6:0]}  // denormal: no hidden bit
      : {1'b1, bf16_act[6:0]};  // normal: hidden bit = 1

  // Place mantissa at [26:19] (8-bit mantissa, 19 fractional zeros)
  assign base_vec = {mantissa, 19'b0};

  assign delta_e = e_max - exp;

  // Barrel shift right by delta_e; clamp to zero on overflow
  always_comb begin
    if (delta_e >= 8'd27) shifted = 27'd0;
    else shifted = base_vec >> delta_e[4:0];  // max shift 26, fits in 5-bit
  end

  // Convert to 2's complement
  always_comb begin
    delayLine_in = sign ? (~shifted + 27'd1) : shifted;
  end

  assign exp_out  = e_max;
  assign sign_out = sign;

endmodule
