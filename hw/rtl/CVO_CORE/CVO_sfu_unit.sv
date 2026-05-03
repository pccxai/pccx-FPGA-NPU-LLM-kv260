// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import isa_pkg::*;
import bf16_math_pkg::*;

// ===| Module: CVO_sfu_unit — streaming BF16 special-function unit |============
// Purpose      : Element-wise BF16 non-linear operators routed by IN_func:
//                EXP, SQRT, GELU, RECIP, SCALE, REDUCE_SUM.
// Spec ref     : pccx v002 §2.4.1 (CVO SFU datapath), §3.5 (CVO uop).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low; i_clear synchronous soft-clear.
// BF16 layout  : {sign[15], exp[14:7], mant[6:0]}.
// Pipelines    : Independent per-op pipelines, multiplexed at the output:
//   EXP        :  5 cycles
//   SQRT       :  3 cycles
//   RECIP      :  4 cycles  (Newton-Raphson, 1 step)
//   SCALE      :  3 cycles  (first input word = scalar, rest = data)
//   REDUCE_SUM :  variable  (accumulates for IN_length cycles, then emits scalar)
//   GELU       : 14 cycles  (MUL + NEG + EXP + ADD + RECIP + MUL chain)
// Throughput   : 1 result/cycle for streaming ops; REDUCE_SUM emits 1 scalar
//                per IN_length elements.
// Handshake    : OUT_data_ready tied 1'b1 — combinational ready (sub-units
//                are non-blocking pipelines).
// Reset state  : All per-op pipeline registers cleared; output mux silent.
// Errors       : EXP overflow saturates to +inf (16'h7F80); EXP underflow → 0.
// Counters     : none.
// Protected    : Internals untouched (CLAUDE.md §6.2 — LUT math user-owned).
// ===============================================================================

module CVO_sfu_unit (
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // ===| Operation Select |====================================================
    input cvo_func_e IN_func,
    input logic [15:0] IN_length,
    input cvo_flags_t IN_flags,

    // ===| Streaming Input |=====================================================
    input  logic [15:0] IN_data,
    input  logic        IN_valid,
    output logic        OUT_data_ready,

    // ===| Streaming Output |====================================================
    output logic [15:0] OUT_result,
    output logic        OUT_result_valid
);

  // ===| BF16 Constants |========================================================
  localparam logic [15:0] Bf16One = 16'h3F80;  // 1.0
  localparam logic [15:0] Bf16Two = 16'h4000;  // 2.0
  localparam logic [15:0] Bf16Scale1702 = 16'h3FD9;  // 1.702 (GELU sigmoid scale)
  localparam logic [7:0] Log2EQ17 = 8'hB8;  // log2(e) ≈ 1.4427 in Q1.7

  // ===| BF16 Arithmetic (combinational) |=======================================

  // ===| BF16 Multiply |===
  function automatic logic [15:0] bf16_mul(input logic [15:0] a, input logic [15:0] b);
    logic        s;
    logic [ 8:0] esum;
    logic [15:0] mp;
    logic [ 7:0] er;
    logic [ 6:0] mr;
    if (a[14:0] == 0 || b[14:0] == 0) return 16'd0;
    s    = a[15] ^ b[15];
    esum = {1'b0, a[14:7]} + {1'b0, b[14:7]};
    mp   = {1'b1, a[6:0]} * {1'b1, b[6:0]};
    if (mp[15]) begin
      er = 8'(esum - 9'd127 + 9'd1);
      mr = mp[14:8];
    end else begin
      er = 8'(esum - 9'd127);
      mr = mp[13:7];
    end
    return {s, er, mr};
  endfunction

  // ===| BF16 Add |===
  function automatic logic [15:0] bf16_add(input logic [15:0] a, input logic [15:0] b);
    logic [7:0] ea, eb, elarge;
    logic [7:0] diff;
    logic [8:0] ma, mb;
    logic [9:0] sum;
    logic [7:0] eout;
    logic [6:0] mout;
    logic       sout;
    if (a[14:0] == 0) return b;
    if (b[14:0] == 0) return a;
    ea = a[14:7];
    eb = b[14:7];
    ma = {1'b0, 1'b1, a[6:0]};
    mb = {1'b0, 1'b1, b[6:0]};
    if (ea >= eb) begin
      elarge = ea;
      diff = ea - eb;
      mb = 9'(mb >> diff);
    end else begin
      elarge = eb;
      diff = eb - ea;
      ma = 9'(ma >> diff);
    end
    if (a[15] == b[15]) begin
      sout = a[15];
      sum  = {1'b0, ma} + {1'b0, mb};
    end else if (ma >= mb) begin
      sout = a[15];
      sum  = {1'b0, ma} - {1'b0, mb};
    end else begin
      sout = b[15];
      sum  = {1'b0, mb} - {1'b0, ma};
    end
    if (sum == 0) return 16'd0;
    if (sum[9]) begin
      eout = elarge + 8'd1;
      mout = sum[9:3];
    end else if (sum[8]) begin
      eout = elarge;
      mout = sum[8:2];
    end else begin
      eout = elarge - 8'd1;
      mout = sum[7:1];
    end
    return {sout, eout, mout};
  endfunction

  // ===| EXP mantissa LUT: mantissa bits of 2^(k/128), k=0..127 |===
  function automatic logic [6:0] exp_mant_lut(input logic [6:0] k);
    // 2^(k/128) - 1 ≈ k*ln2/128; includes curvature correction
    logic [8:0] v;
    v = {2'b0, k} + ({2'b0, k} * {2'b0, k} >> 9);
    return v[6:0];
  endfunction

  // ===| SQRT mantissa LUTs |===
  function automatic logic [6:0] sqrt_mant_even(input logic [6:0] k);
    // sqrt(1 + k/128) - 1; range [0, 0.414] → mant bits [0, 53]
    logic [8:0] v;
    v = ({2'b0, k} + ({2'b0, k} * {2'b0, 7'(128 - k)} >> 9)) >> 1;
    return v[6:0];
  endfunction

  function automatic logic [6:0] sqrt_mant_odd(input logic [6:0] k);
    // sqrt(2*(1+k/128)) - 1; range [0.414, 0.848] → mant bits [53, 108]
    logic [8:0] v;
    v = 9'd53 + ({2'b0, k} * 9'd91 >> 7);
    return v[6:0];
  endfunction

  // ===| EXP Pipeline (5 stages) |===============================================

  // Stage 1 — unpack BF16
  logic       exp_s1_valid;
  logic       exp_s1_sign;
  logic [7:0] exp_s1_exp;
  logic [6:0] exp_s1_mant;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s1_valid <= 1'b0;
    end else begin
      exp_s1_valid <= IN_valid && (IN_func == CVO_EXP || IN_func == CVO_GELU);
      exp_s1_sign  <= IN_data[15];
      exp_s1_exp   <= IN_data[14:7];
      exp_s1_mant  <= IN_data[6:0];
    end
  end

  // Stage 2 — BF16 → Q8.7 fixed-point
  logic               exp_s1f_valid;
  logic signed [15:0] exp_s1f_xfixed;

  logic signed [15:0] exp_s1_xfixed_wire;  // Q8.7 signed

  always_comb begin : comb_exp_convert
    logic [15:0] mag;
    int          sh;
    sh  = int'(exp_s1_exp) - 127;
    mag = 16'd0;
    if (exp_s1_exp == 8'd0) mag = 16'd0;
    else if (sh >= 8) mag = 16'h7FFF;
    else if (sh >= -7) mag = 16'({1'b1, exp_s1_mant, 7'b0} << (sh + 7));
    else mag = 16'({1'b1, exp_s1_mant, 7'b0} >> -(sh + 7));
    exp_s1_xfixed_wire = exp_s1_sign ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s1f_valid  <= 1'b0;
      exp_s1f_xfixed <= '0;
    end else begin
      exp_s1f_valid  <= exp_s1_valid;
      exp_s1f_xfixed <= exp_s1_xfixed_wire;
    end
  end

  // Stage 3 — multiply by log2(e)
  logic               exp_s2_valid;
  logic        [ 8:0] exp_s2_n;  // integer part of x*log2e
  logic        [ 6:0] exp_s2_frac;  // fractional 7-bit index for LUT

  logic signed [23:0] exp_s1_y_wire;  // Q9.14: x*log2e
  assign exp_s1_y_wire = $signed(exp_s1f_xfixed) * $signed({1'b0, Log2EQ17});

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s2_valid <= 1'b0;
    end else begin
      exp_s2_valid <= exp_s1f_valid;
      exp_s2_n     <= 9'(exp_s1_y_wire[23:14]);
      exp_s2_frac  <= exp_s1_y_wire[13:7];
    end
  end

  // Stage 4 — assemble output BF16
  logic        exp_s3_valid;
  logic [15:0] exp_s3_result;

  logic [ 8:0] exp_s2_out_exp_wire;

  always_comb begin
    exp_s2_out_exp_wire = 9'd127 + exp_s2_n;
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s3_valid <= 1'b0;
    end else begin
      exp_s3_valid <= exp_s2_valid;
      if (exp_s2_out_exp_wire[8] || exp_s2_out_exp_wire == 0)
        exp_s3_result <= (exp_s2_n[8] == 0) ? 16'h7F80 : 16'd0;  // +inf or 0
      else exp_s3_result <= {1'b0, exp_s2_out_exp_wire[7:0], exp_mant_lut(exp_s2_frac)};
    end
  end

  // ===| SQRT Pipeline (3 stages) |===============================================

  logic       sqrt_s1_valid;
  logic [7:0] sqrt_s1_exp;
  logic [6:0] sqrt_s1_mant;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      sqrt_s1_valid <= 1'b0;
    end else begin
      sqrt_s1_valid <= IN_valid && (IN_func == CVO_SQRT);
      sqrt_s1_exp   <= IN_data[14:7];
      sqrt_s1_mant  <= IN_data[6:0];
    end
  end

  logic        sqrt_s2_valid;
  logic [15:0] sqrt_s2_result;

  logic [ 7:0] sqrt_s1_unbiased_wire;
  logic [ 7:0] sqrt_s1_out_exp_wire;
  logic [ 6:0] sqrt_s1_out_mant_wire;

  always_comb begin
    sqrt_s1_unbiased_wire = sqrt_s1_exp - 8'd127;
    sqrt_s1_out_exp_wire = 8'd127 + {1'b0, sqrt_s1_unbiased_wire[7:1]};
    sqrt_s1_out_mant_wire = sqrt_s1_unbiased_wire[0] ? sqrt_mant_odd(sqrt_s1_mant) :
        sqrt_mant_even(sqrt_s1_mant);
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      sqrt_s2_valid <= 1'b0;
    end else begin
      sqrt_s2_valid  <= sqrt_s1_valid;
      sqrt_s2_result <= {1'b0, sqrt_s1_out_exp_wire, sqrt_s1_out_mant_wire};
    end
  end

  logic        sqrt_s3_valid;
  logic [15:0] sqrt_s3_result;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      sqrt_s3_valid <= 1'b0;
    end else begin
      sqrt_s3_valid  <= sqrt_s2_valid;
      sqrt_s3_result <= sqrt_s2_result;
    end
  end

  // ===| RECIP Pipeline (4 stages) |=============================================
  // 1/x via 1 Newton-Raphson step: r1 = r0 * (2 - x*r0)
  // Initial estimate: exp flipped around 254, mantissa roughly inverted.

  logic        recip_s1_valid;
  logic        recip_s1_sign;
  logic [15:0] recip_s1_r0;
  logic [15:0] recip_s1_x;

  logic [ 7:0] recip_in_inv_exp_wire;
  logic [ 6:0] recip_in_inv_mant_wire;

  always_comb begin
    recip_in_inv_exp_wire  = 8'd254 - IN_data[14:7];
    recip_in_inv_mant_wire = 7'd127 - {1'b0, IN_data[6:1]};
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s1_valid <= 1'b0;
    end else begin
      recip_s1_valid <= IN_valid && (IN_func == CVO_RECIP);
      recip_s1_sign  <= IN_data[15];
      recip_s1_x     <= {1'b0, IN_data[14:0]};  // |x|
      recip_s1_r0    <= {1'b0, recip_in_inv_exp_wire, recip_in_inv_mant_wire};
    end
  end

  logic        recip_s2_valid;
  logic [15:0] recip_s2_xr0;
  logic [15:0] recip_s2_r0;
  logic        recip_s2_sign;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s2_valid <= 1'b0;
    end else begin
      recip_s2_valid <= recip_s1_valid;
      recip_s2_xr0   <= bf16_mul(recip_s1_x, recip_s1_r0);
      recip_s2_r0    <= recip_s1_r0;
      recip_s2_sign  <= recip_s1_sign;
    end
  end

  logic        recip_s3_valid;
  logic [15:0] recip_s3_corr;
  logic [15:0] recip_s3_r0;
  logic        recip_s3_sign;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s3_valid <= 1'b0;
    end else begin
      recip_s3_valid <= recip_s2_valid;
      recip_s3_corr  <= bf16_add(Bf16Two, {1'b1, recip_s2_xr0[14:0]});
      recip_s3_r0    <= recip_s2_r0;
      recip_s3_sign  <= recip_s2_sign;
    end
  end

  logic        recip_s4_valid;
  logic [15:0] recip_s4_result;

  logic [15:0] recip_s3_mag_wire;
  always_comb begin
    recip_s3_mag_wire = bf16_mul(recip_s3_r0, recip_s3_corr);
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s4_valid <= 1'b0;
    end else begin
      recip_s4_valid  <= recip_s3_valid;
      recip_s4_result <= {recip_s3_sign, recip_s3_mag_wire[14:0]};
    end
  end

  // ===| SCALE Pipeline (3 stages) |=============================================
  // First element received = scalar (or 1/scalar if FLAG_RECIP_SCALE).

  logic        scale_scalar_loaded;
  logic [15:0] scale_scalar;

  logic        scale_s1_valid;
  logic [15:0] scale_s1_product;

  logic        scale_s2_valid;
  logic [15:0] scale_s2_result;

  logic [15:0] scale_scalar_next_wire;

  always_comb begin
    // Approximate 1/scalar for recip_scale mode: flip exponent + invert mantissa
    scale_scalar_next_wire = IN_flags.recip_scale
      ? {IN_data[15], 8'(8'd254 - IN_data[14:7]), 7'(7'd127 - {1'b0, IN_data[6:1]})}
      : IN_data;
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      scale_scalar_loaded <= 1'b0;
      scale_scalar        <= 16'd0;
      scale_s1_valid      <= 1'b0;
      scale_s2_valid      <= 1'b0;
    end else begin
      if (IN_valid && IN_func == CVO_SCALE) begin
        if (!scale_scalar_loaded) begin
          scale_scalar        <= scale_scalar_next_wire;
          scale_scalar_loaded <= 1'b1;
          scale_s1_valid      <= 1'b0;
        end else begin
          scale_s1_valid   <= 1'b1;
          scale_s1_product <= bf16_mul(IN_data, scale_scalar);
        end
      end else begin
        if (IN_func != CVO_SCALE) scale_scalar_loaded <= 1'b0;
        scale_s1_valid <= 1'b0;
      end

      scale_s2_valid  <= scale_s1_valid;
      scale_s2_result <= scale_s1_product;
    end
  end

  // ===| REDUCE_SUM (sequential BF16 accumulation) |=============================

  logic [15:0] rsum_count;
  logic [15:0] rsum_accum;
  logic        rsum_out_valid;
  logic [15:0] rsum_out;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      rsum_count     <= 16'd0;
      rsum_accum     <= 16'd0;
      rsum_out_valid <= 1'b0;
      rsum_out       <= 16'd0;
    end else begin
      rsum_out_valid <= 1'b0;
      if (IN_valid && IN_func == CVO_REDUCE_SUM) begin
        rsum_count <= rsum_count + 16'd1;
        rsum_accum <= bf16_add(rsum_accum, IN_data);
        if (rsum_count == IN_length - 16'd1) begin
          rsum_out       <= bf16_add(rsum_accum, IN_data);
          rsum_out_valid <= 1'b1;
          rsum_count     <= 16'd0;
          rsum_accum     <= 16'd0;
        end
      end else if (IN_func != CVO_REDUCE_SUM) begin
        rsum_count <= 16'd0;
        rsum_accum <= 16'd0;
      end
    end
  end

  // ===| GELU Pipeline (14 stages) |=============================================
  // GELU(x) ≈ x * sigmoid(1.702*x),  sigmoid(y) = 1/(1+exp(-y))
  // Chain: MUL(1.702)[1] -> NEG[0] -> EXP[3] -> ADD(1)[1] ->
  // RECIP seed/Newton[4] -> MUL(x)[1]. The Newton correction and
  // multiply are separated to keep the 400 MHz core timing boundary real.
  // x is preserved in a 12-cycle delay chain.

  localparam int GeluDelay = 12;

  logic [15:0] gelu_x_pipe   [GeluDelay];

  // Stage g1: 1.702 * x
  logic        gelu_g1_valid;
  logic [15:0] gelu_g1_y;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_g1_valid <= 1'b0;
      for (int d = 0; d < GeluDelay; d++) gelu_x_pipe[d] <= 16'd0;
    end else begin
      gelu_x_pipe[0] <= IN_data;
      for (int d = 1; d < GeluDelay; d++) gelu_x_pipe[d] <= gelu_x_pipe[d-1];
      gelu_g1_valid <= IN_valid && (IN_func == CVO_GELU);
      gelu_g1_y     <= bf16_mul(IN_data, Bf16Scale1702);
    end
  end

  // Stage g2: negate → -1.702x
  logic        gelu_g2_valid;
  logic [15:0] gelu_g2_neg_y;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_g2_valid <= 1'b0;
    end else begin
      gelu_g2_valid <= gelu_g1_valid;
      gelu_g2_neg_y <= {~gelu_g1_y[15], gelu_g1_y[14:0]};
    end
  end

  // Stages g3-g6: EXP(-y) — split BF16->fixed conversion from log2(e) multiply
  logic               gelu_e1_valid;
  logic        [ 8:0] gelu_e1_n;
  logic        [ 6:0] gelu_e1_frac;

  logic               gelu_e0_valid;
  logic signed [15:0] gelu_e0_xf;
  logic signed [15:0] gelu_g2_xf_wire;
  logic signed [23:0] gelu_e0_y_wire;

  always_comb begin : comb_gelu_exp_convert
    logic [15:0] mag;
    int          sh;
    sh  = int'(gelu_g2_neg_y[14:7]) - 127;
    mag = 16'd0;
    if (gelu_g2_neg_y[14:7] == 8'd0) mag = 16'd0;
    else if (sh >= 8) mag = 16'h7FFF;
    else if (sh >= -7) mag = 16'({1'b1, gelu_g2_neg_y[6:0], 7'b0} << (sh + 7));
    else mag = 16'({1'b1, gelu_g2_neg_y[6:0], 7'b0} >> -(sh + 7));
    gelu_g2_xf_wire = gelu_g2_neg_y[15] ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_e0_valid <= 1'b0;
      gelu_e0_xf    <= '0;
    end else begin
      gelu_e0_valid <= gelu_g2_valid;
      gelu_e0_xf    <= gelu_g2_xf_wire;
    end
  end

  assign gelu_e0_y_wire = $signed(gelu_e0_xf) * $signed({1'b0, Log2EQ17});

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_e1_valid <= 1'b0;
    end else begin
      gelu_e1_valid <= gelu_e0_valid;
      gelu_e1_n     <= 9'(gelu_e0_y_wire[23:14]);
      gelu_e1_frac  <= gelu_e0_y_wire[13:7];
    end
  end

  logic        gelu_e2_valid;
  logic [15:0] gelu_e2_expval;

  logic [ 8:0] gelu_e1_out_exp_wire;
  always_comb begin
    gelu_e1_out_exp_wire = 9'd127 + gelu_e1_n;
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_e2_valid <= 1'b0;
    end else begin
      gelu_e2_valid <= gelu_e1_valid;
      if (gelu_e1_out_exp_wire[8] || gelu_e1_out_exp_wire == 0)
        gelu_e2_expval <= (gelu_e1_n[8] == 0) ? 16'h7F80 : 16'd0;
      else gelu_e2_expval <= {1'b0, gelu_e1_out_exp_wire[7:0], exp_mant_lut(gelu_e1_frac)};
    end
  end

  // Stage g6: 1 + exp(-y)
  logic        gelu_g6_valid;
  logic [15:0] gelu_g6_denom;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_g6_valid <= 1'b0;
    end else begin
      gelu_g6_valid <= gelu_e2_valid;
      gelu_g6_denom <= bf16_add(Bf16One, gelu_e2_expval);
    end
  end

  // Stages g7-g9: RECIP(1 + exp(-y)) = sigmoid
  logic        gelu_r1_valid;
  logic [15:0] gelu_r1_r0;
  logic [15:0] gelu_r1_x;

  logic [ 7:0] gelu_recip_inv_exp_wire;
  logic [ 6:0] gelu_recip_inv_mant_wire;
  always_comb begin
    gelu_recip_inv_exp_wire  = 8'd254 - gelu_g6_denom[14:7];
    gelu_recip_inv_mant_wire = 7'd127 - {1'b0, gelu_g6_denom[6:1]};
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r1_valid <= 1'b0;
    end else begin
      gelu_r1_valid <= gelu_g6_valid;
      gelu_r1_x     <= gelu_g6_denom;
      gelu_r1_r0    <= {1'b0, gelu_recip_inv_exp_wire, gelu_recip_inv_mant_wire};
    end
  end

  logic        gelu_r2_valid;
  logic [15:0] gelu_r2_xr0;
  logic [15:0] gelu_r2_r0;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r2_valid <= 1'b0;
    end else begin
      gelu_r2_valid <= gelu_r1_valid;
      gelu_r2_xr0   <= bf16_mul(gelu_r1_x, gelu_r1_r0);
      gelu_r2_r0    <= gelu_r1_r0;
    end
  end

  logic        gelu_r3a_valid;
  logic [15:0] gelu_r3a_r0;
  logic [15:0] gelu_r3a_corr;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r3a_valid <= 1'b0;
      gelu_r3a_r0    <= 16'd0;
      gelu_r3a_corr  <= 16'd0;
    end else begin
      gelu_r3a_valid <= gelu_r2_valid;
      gelu_r3a_r0    <= gelu_r2_r0;
      gelu_r3a_corr  <= bf16_add(Bf16Two, {1'b1, gelu_r2_xr0[14:0]});
    end
  end

  logic        gelu_r3_valid;
  logic [15:0] gelu_r3_sigmoid;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r3_valid <= 1'b0;
    end else begin
      gelu_r3_valid   <= gelu_r3a_valid;
      gelu_r3_sigmoid <= bf16_mul(gelu_r3a_r0, gelu_r3a_corr);
    end
  end

  // Stage g10: x * sigmoid = GELU(x)
  logic        gelu_out_valid;
  logic [15:0] gelu_out;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_out_valid <= 1'b0;
    end else begin
      gelu_out_valid <= gelu_r3_valid;
      gelu_out       <= bf16_mul(gelu_x_pipe[GeluDelay-1], gelu_r3_sigmoid);
    end
  end

  // ===| Output Mux |============================================================
  always_comb begin
    OUT_data_ready   = 1'b1;
    OUT_result       = 16'd0;
    OUT_result_valid = 1'b0;
    case (IN_func)
      CVO_EXP: begin
        OUT_result = exp_s3_result;
        OUT_result_valid = exp_s3_valid;
      end
      CVO_SQRT: begin
        OUT_result = sqrt_s3_result;
        OUT_result_valid = sqrt_s3_valid;
      end
      CVO_GELU: begin
        OUT_result = gelu_out;
        OUT_result_valid = gelu_out_valid;
      end
      CVO_RECIP: begin
        OUT_result = recip_s4_result;
        OUT_result_valid = recip_s4_valid;
      end
      CVO_SCALE: begin
        OUT_result = scale_s2_result;
        OUT_result_valid = scale_s2_valid;
      end
      CVO_REDUCE_SUM: begin
        OUT_result = rsum_out;
        OUT_result_valid = rsum_out_valid;
      end
      default: ;
    endcase
  end

endmodule
