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
//   EXP        :  7 cycles
//   SQRT       :  3 cycles
//   RECIP      :  9 cycles  (Newton-Raphson, 1 step)
//   SCALE      :  5 cycles  (first input word = scalar, rest = data)
//   REDUCE_SUM :  variable  (ready-gated BF16 add pipeline, emits scalar)
//   GELU       : 22 cycles  (MUL + NEG + EXP + ADD + RECIP + MUL chain)
// Throughput   : 1 result/cycle for streaming ops; REDUCE_SUM emits 1 scalar
//                per IN_length elements.
// Handshake    : OUT_data_ready is 1'b1 for streaming ops; REDUCE_SUM applies
//                local backpressure while the accumulator pipeline is busy.
// Reset state  : All per-op pipeline registers cleared; output mux silent.
// Errors       : EXP overflow saturates to +inf (16'h7F80); EXP underflow → 0.
// Counters     : none.
// Ownership    : LUT math internals are timing-sensitive and kept local to this SFU.
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

  // ===| EXP Pipeline (7 stages) |===============================================

  // Stage 1 — unpack BF16
  logic       exp_s1_valid;
  logic       exp_s1_sign;
  logic [7:0] exp_s1_exp;
  logic [6:0] exp_s1_mant;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s1_valid <= 1'b0;
    end else begin
      exp_s1_valid <= IN_valid && (IN_func == CVO_EXP);
      exp_s1_sign  <= IN_data[15];
      exp_s1_exp   <= IN_data[14:7];
      exp_s1_mant  <= IN_data[6:0];
    end
  end

  // Stages 2-4 — BF16 → Q8.7 fixed-point.
  // The conversion is intentionally split before the DSP multiply boundary:
  // exponent classify, variable shift, and sign handling were previously one
  // setup path into the DSP A input.
  logic        exp_s1a_valid;
  logic        exp_s1a_sign;
  logic        exp_s1a_zero;
  logic        exp_s1a_sat;
  logic        exp_s1a_shift_left;
  logic [ 4:0] exp_s1a_shift_amt;
  logic [15:0] exp_s1a_mant_ext;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s1a_valid      <= 1'b0;
      exp_s1a_sign       <= 1'b0;
      exp_s1a_zero       <= 1'b0;
      exp_s1a_sat        <= 1'b0;
      exp_s1a_shift_left <= 1'b0;
      exp_s1a_shift_amt  <= 5'd0;
      exp_s1a_mant_ext   <= 16'd0;
    end else begin
      int sh;
      sh = int'(exp_s1_exp) - 127;

      exp_s1a_valid      <= exp_s1_valid;
      exp_s1a_sign       <= exp_s1_sign;
      exp_s1a_mant_ext   <= {1'b1, exp_s1_mant, 7'b0};
      exp_s1a_zero       <= (exp_s1_exp == 8'd0) || (sh <= -23);
      exp_s1a_sat        <= (sh >= 8);
      exp_s1a_shift_left <= (sh >= -7);
      if (sh >= -7) begin
        exp_s1a_shift_amt <= 5'(sh + 7);
      end else begin
        exp_s1a_shift_amt <= 5'((-sh) - 7);
      end
    end
  end

  logic        exp_s1b_valid;
  logic        exp_s1b_sign;
  logic [15:0] exp_s1b_mag;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s1b_valid <= 1'b0;
      exp_s1b_sign  <= 1'b0;
      exp_s1b_mag   <= 16'd0;
    end else begin
      exp_s1b_valid <= exp_s1a_valid;
      exp_s1b_sign  <= exp_s1a_sign;
      if (exp_s1a_zero) begin
        exp_s1b_mag <= 16'd0;
      end else if (exp_s1a_sat) begin
        exp_s1b_mag <= 16'h7FFF;
      end else if (exp_s1a_shift_left) begin
        exp_s1b_mag <= exp_s1a_mant_ext << exp_s1a_shift_amt;
      end else begin
        exp_s1b_mag <= exp_s1a_mant_ext >> exp_s1a_shift_amt;
      end
    end
  end

  logic               exp_s1f_valid;
  logic signed [15:0] exp_s1f_xfixed;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      exp_s1f_valid  <= 1'b0;
      exp_s1f_xfixed <= '0;
    end else begin
      exp_s1f_valid  <= exp_s1b_valid;
      exp_s1f_xfixed <= exp_s1b_sign ? -$signed({1'b0, exp_s1b_mag}) : $signed({1'b0, exp_s1b_mag});
    end
  end

  // Stage 5 — multiply by log2(e)
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

  // Stage 6 — assemble output BF16
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

  // ===| RECIP Pipeline (8 stages) |=============================================
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

  logic        recip_s2a_valid;
  logic        recip_s2a_zero;
  logic        recip_s2a_sign;
  logic [ 8:0] recip_s2a_esum;
  logic [11:0] recip_s2a_mp_lo;
  logic [11:0] recip_s2a_mp_hi;
  logic [15:0] recip_s2a_r0;
  logic        recip_s2a_out_sign;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s2a_valid    <= 1'b0;
      recip_s2a_zero     <= 1'b1;
      recip_s2a_sign     <= 1'b0;
      recip_s2a_esum     <= 9'd0;
      recip_s2a_mp_lo    <= 12'd0;
      recip_s2a_mp_hi    <= 12'd0;
      recip_s2a_r0       <= 16'd0;
      recip_s2a_out_sign <= 1'b0;
    end else begin
      recip_s2a_valid    <= recip_s1_valid;
      recip_s2a_zero     <= (recip_s1_x[14:0] == 15'd0) || (recip_s1_r0[14:0] == 15'd0);
      recip_s2a_sign     <= recip_s1_x[15] ^ recip_s1_r0[15];
      recip_s2a_esum     <= {1'b0, recip_s1_x[14:7]} + {1'b0, recip_s1_r0[14:7]};
      recip_s2a_mp_lo    <= 12'({1'b1, recip_s1_x[6:0]} * recip_s1_r0[3:0]);
      recip_s2a_mp_hi    <= 12'({1'b1, recip_s1_x[6:0]} * {1'b1, recip_s1_r0[6:4]});
      recip_s2a_r0       <= recip_s1_r0;
      recip_s2a_out_sign <= recip_s1_sign;
    end
  end

  logic        recip_s2m_valid;
  logic        recip_s2m_zero;
  logic        recip_s2m_sign;
  logic [ 8:0] recip_s2m_esum;
  logic [15:0] recip_s2m_mp;
  logic [15:0] recip_s2m_r0;
  logic        recip_s2m_out_sign;

  logic [15:0] recip_s2m_mp_wire;
  logic [ 7:0] recip_s2m_er_wire;
  logic [ 6:0] recip_s2m_mr_wire;

  always_comb begin : comb_recip_xr0_partial_sum
    recip_s2m_mp_wire = {4'd0, recip_s2a_mp_lo} + {recip_s2a_mp_hi, 4'd0};
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s2m_valid    <= 1'b0;
      recip_s2m_zero     <= 1'b1;
      recip_s2m_sign     <= 1'b0;
      recip_s2m_esum     <= 9'd0;
      recip_s2m_mp       <= 16'd0;
      recip_s2m_r0       <= 16'd0;
      recip_s2m_out_sign <= 1'b0;
    end else begin
      recip_s2m_valid    <= recip_s2a_valid;
      recip_s2m_zero     <= recip_s2a_zero;
      recip_s2m_sign     <= recip_s2a_sign;
      recip_s2m_esum     <= recip_s2a_esum;
      recip_s2m_mp       <= recip_s2m_mp_wire;
      recip_s2m_r0       <= recip_s2a_r0;
      recip_s2m_out_sign <= recip_s2a_out_sign;
    end
  end

  always_comb begin : comb_recip_xr0_pack
    if (recip_s2m_mp[15]) begin
      recip_s2m_er_wire = 8'(recip_s2m_esum - 9'd127 + 9'd1);
      recip_s2m_mr_wire = recip_s2m_mp[14:8];
    end else begin
      recip_s2m_er_wire = 8'(recip_s2m_esum - 9'd127);
      recip_s2m_mr_wire = recip_s2m_mp[13:7];
    end
  end

  logic        recip_s2_valid;
  logic [15:0] recip_s2_xr0;
  logic [15:0] recip_s2_r0;
  logic        recip_s2_sign;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s2_valid <= 1'b0;
      recip_s2_xr0   <= 16'd0;
      recip_s2_r0    <= 16'd0;
      recip_s2_sign  <= 1'b0;
    end else begin
      recip_s2_valid <= recip_s2m_valid;
      recip_s2_xr0   <= recip_s2m_zero ? 16'd0 : {recip_s2m_sign, recip_s2m_er_wire, recip_s2m_mr_wire};
      recip_s2_r0    <= recip_s2m_r0;
      recip_s2_sign  <= recip_s2m_out_sign;
    end
  end

  logic        recip_s3a_valid;
  logic        recip_s3a_bypass;
  logic [15:0] recip_s3a_bypass_value;
  logic [ 7:0] recip_s3a_elarge;
  logic [ 8:0] recip_s3a_ma;
  logic [ 8:0] recip_s3a_mb;
  logic [15:0] recip_s3a_r0;
  logic        recip_s3a_sign;

  logic [ 7:0] recip_corr_elarge_wire;
  logic [ 8:0] recip_corr_ma_wire;
  logic [ 8:0] recip_corr_mb_wire;

  always_comb begin : comb_recip_corr_align
    if (8'd128 >= recip_s2_xr0[14:7]) begin
      recip_corr_elarge_wire = 8'd128;
      recip_corr_ma_wire     = {1'b0, 1'b1, 7'd0};
      recip_corr_mb_wire     = 9'({1'b0, 1'b1, recip_s2_xr0[6:0]} >>
                                   (8'd128 - recip_s2_xr0[14:7]));
    end else begin
      recip_corr_elarge_wire = recip_s2_xr0[14:7];
      recip_corr_ma_wire     = 9'({1'b0, 1'b1, 7'd0} >>
                                   (recip_s2_xr0[14:7] - 8'd128));
      recip_corr_mb_wire     = {1'b0, 1'b1, recip_s2_xr0[6:0]};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s3a_valid        <= 1'b0;
      recip_s3a_bypass       <= 1'b0;
      recip_s3a_bypass_value <= 16'd0;
      recip_s3a_elarge       <= 8'd0;
      recip_s3a_ma           <= 9'd0;
      recip_s3a_mb           <= 9'd0;
      recip_s3a_r0           <= 16'd0;
      recip_s3a_sign         <= 1'b0;
    end else begin
      recip_s3a_valid        <= recip_s2_valid;
      recip_s3a_bypass       <= (recip_s2_xr0[14:0] == 15'd0);
      recip_s3a_bypass_value <= Bf16Two;
      recip_s3a_elarge       <= recip_corr_elarge_wire;
      recip_s3a_ma           <= recip_corr_ma_wire;
      recip_s3a_mb           <= recip_corr_mb_wire;
      recip_s3a_r0           <= recip_s2_r0;
      recip_s3a_sign         <= recip_s2_sign;
    end
  end

  logic        recip_s3_valid;
  logic [15:0] recip_s3_corr;
  logic [15:0] recip_s3_r0;
  logic        recip_s3_sign;

  logic [15:0] recip_corr_packed_wire;
  logic        recip_corr_sout_wire;
  logic [ 9:0] recip_corr_sum_wire;
  logic        recip_s3b_valid;
  logic        recip_s3b_bypass;
  logic [15:0] recip_s3b_bypass_value;
  logic [ 7:0] recip_s3b_elarge;
  logic        recip_s3b_sout;
  logic [ 9:0] recip_s3b_sum;
  logic [15:0] recip_s3b_r0;
  logic        recip_s3b_sign;

  always_comb begin : comb_recip_corr_pack
    if (recip_s3a_ma >= recip_s3a_mb) begin
      recip_corr_sout_wire = 1'b0;
      recip_corr_sum_wire  = {1'b0, recip_s3a_ma} - {1'b0, recip_s3a_mb};
    end else begin
      recip_corr_sout_wire = 1'b1;
      recip_corr_sum_wire  = {1'b0, recip_s3a_mb} - {1'b0, recip_s3a_ma};
    end

    if (recip_s3b_bypass) begin
      recip_corr_packed_wire = recip_s3b_bypass_value;
    end else if (recip_s3b_sum == 10'd0) begin
      recip_corr_packed_wire = 16'd0;
    end else if (recip_s3b_sum[9]) begin
      recip_corr_packed_wire = {recip_s3b_sout,
                                8'(recip_s3b_elarge + 8'd1),
                                recip_s3b_sum[9:3]};
    end else if (recip_s3b_sum[8]) begin
      recip_corr_packed_wire = {recip_s3b_sout,
                                recip_s3b_elarge,
                                recip_s3b_sum[8:2]};
    end else begin
      recip_corr_packed_wire = {recip_s3b_sout,
                                8'(recip_s3b_elarge - 8'd1),
                                recip_s3b_sum[7:1]};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s3b_valid        <= 1'b0;
      recip_s3b_bypass       <= 1'b0;
      recip_s3b_bypass_value <= 16'd0;
      recip_s3b_elarge       <= 8'd0;
      recip_s3b_sout         <= 1'b0;
      recip_s3b_sum          <= 10'd0;
      recip_s3b_r0           <= 16'd0;
      recip_s3b_sign         <= 1'b0;
    end else begin
      recip_s3b_valid        <= recip_s3a_valid;
      recip_s3b_bypass       <= recip_s3a_bypass;
      recip_s3b_bypass_value <= recip_s3a_bypass_value;
      recip_s3b_elarge       <= recip_s3a_elarge;
      recip_s3b_sout         <= recip_corr_sout_wire;
      recip_s3b_sum          <= recip_corr_sum_wire;
      recip_s3b_r0           <= recip_s3a_r0;
      recip_s3b_sign         <= recip_s3a_sign;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s3_valid <= 1'b0;
      recip_s3_corr  <= 16'd0;
      recip_s3_r0    <= 16'd0;
      recip_s3_sign  <= 1'b0;
    end else begin
      recip_s3_valid <= recip_s3b_valid;
      recip_s3_corr  <= recip_corr_packed_wire;
      recip_s3_r0    <= recip_s3b_r0;
      recip_s3_sign  <= recip_s3b_sign;
    end
  end

  logic        recip_s4_valid;
  logic        recip_s4_zero;
  logic        recip_s4_sign;
  logic [ 8:0] recip_s4_esum;
  logic [15:0] recip_s4_mp;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s4_valid <= 1'b0;
      recip_s4_zero  <= 1'b1;
      recip_s4_sign  <= 1'b0;
      recip_s4_esum  <= 9'd0;
      recip_s4_mp    <= 16'd0;
    end else begin
      recip_s4_valid <= recip_s3_valid;
      recip_s4_zero  <= (recip_s3_r0[14:0] == 15'd0) || (recip_s3_corr[14:0] == 15'd0);
      recip_s4_sign  <= recip_s3_sign;
      recip_s4_esum  <= {1'b0, recip_s3_r0[14:7]} + {1'b0, recip_s3_corr[14:7]};
      recip_s4_mp    <= {1'b1, recip_s3_r0[6:0]} * {1'b1, recip_s3_corr[6:0]};
    end
  end

  logic        recip_s5_valid;
  logic [15:0] recip_s5_result;

  logic [ 7:0] recip_s4_er_wire;
  logic [ 6:0] recip_s4_mr_wire;

  always_comb begin : comb_recip_mul_pack
    if (recip_s4_mp[15]) begin
      recip_s4_er_wire = 8'(recip_s4_esum - 9'd127 + 9'd1);
      recip_s4_mr_wire = recip_s4_mp[14:8];
    end else begin
      recip_s4_er_wire = 8'(recip_s4_esum - 9'd127);
      recip_s4_mr_wire = recip_s4_mp[13:7];
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      recip_s5_valid  <= 1'b0;
      recip_s5_result <= 16'd0;
    end else begin
      recip_s5_valid  <= recip_s4_valid;
      recip_s5_result <= recip_s4_zero ? 16'd0 : {recip_s4_sign, recip_s4_er_wire, recip_s4_mr_wire};
    end
  end

  // ===| SCALE Pipeline (5 stages) |=============================================
  // First element received = scalar (or 1/scalar if FLAG_RECIP_SCALE).

  logic        scale_scalar_loaded;
  logic [15:0] scale_scalar;

  logic        scale_s0_valid;
  logic [15:0] scale_s0_data;
  logic [15:0] scale_s0_scalar;

  logic        scale_s1_valid;
  logic        scale_s1_zero;
  logic        scale_s1_sign;
  logic [ 8:0] scale_s1_esum;
  logic [15:0] scale_s1_mp;

  logic        scale_s2_valid;
  logic [15:0] scale_s2_result;

  logic [15:0] scale_scalar_next_wire;
  logic [ 7:0] scale_s1_er_wire;
  logic [ 6:0] scale_s1_mr_wire;

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
      scale_s0_valid      <= 1'b0;
      scale_s0_data       <= 16'd0;
      scale_s0_scalar     <= 16'd0;
      scale_s1_valid      <= 1'b0;
      scale_s1_zero       <= 1'b1;
      scale_s1_sign       <= 1'b0;
      scale_s1_esum       <= 9'd0;
      scale_s1_mp         <= 16'd0;
      scale_s2_valid      <= 1'b0;
      scale_s2_result     <= 16'd0;
    end else begin
      if (IN_valid && IN_func == CVO_SCALE) begin
        if (!scale_scalar_loaded) begin
          scale_scalar        <= scale_scalar_next_wire;
          scale_scalar_loaded <= 1'b1;
          scale_s0_valid      <= 1'b0;
        end else begin
          scale_s0_valid  <= 1'b1;
          scale_s0_data   <= IN_data;
          scale_s0_scalar <= scale_scalar;
        end
      end else begin
        if (IN_func != CVO_SCALE) scale_scalar_loaded <= 1'b0;
        scale_s0_valid <= 1'b0;
      end

      scale_s1_valid <= scale_s0_valid;
      scale_s1_zero  <= (scale_s0_data[14:0] == 15'd0) || (scale_s0_scalar[14:0] == 15'd0);
      scale_s1_sign  <= scale_s0_data[15] ^ scale_s0_scalar[15];
      scale_s1_esum  <= {1'b0, scale_s0_data[14:7]} + {1'b0, scale_s0_scalar[14:7]};
      scale_s1_mp    <= {1'b1, scale_s0_data[6:0]} * {1'b1, scale_s0_scalar[6:0]};

      scale_s2_valid  <= scale_s1_valid;
      scale_s2_result <= scale_s1_zero ? 16'd0 : {scale_s1_sign, scale_s1_er_wire, scale_s1_mr_wire};
    end
  end

  always_comb begin : comb_scale_mul_pack
    if (scale_s1_mp[15]) begin
      scale_s1_er_wire = 8'(scale_s1_esum - 9'd127 + 9'd1);
      scale_s1_mr_wire = scale_s1_mp[14:8];
    end else begin
      scale_s1_er_wire = 8'(scale_s1_esum - 9'd127);
      scale_s1_mr_wire = scale_s1_mp[13:7];
    end
  end

  // ===| REDUCE_SUM (sequential BF16 accumulation) |=============================

  logic [15:0] rsum_count;
  logic [15:0] rsum_accum;
  logic        rsum_active;
  logic        rsum_out_valid;
  logic [15:0] rsum_out;

  logic        rsum_in_valid;
  logic [15:0] rsum_in_data;
  logic        rsum_in_last;

  logic        rsum_a_valid;
  logic        rsum_a_last;
  logic        rsum_a_bypass;
  logic [15:0] rsum_a_bypass_value;
  logic        rsum_a_sa;
  logic        rsum_a_sb;
  logic [ 7:0] rsum_a_elarge;
  logic [ 8:0] rsum_a_ma;
  logic [ 8:0] rsum_a_mb;

  logic        rsum_b_valid;
  logic        rsum_b_last;
  logic        rsum_b_bypass;
  logic [15:0] rsum_b_bypass_value;
  logic        rsum_b_sout;
  logic [ 7:0] rsum_b_elarge;
  logic [ 9:0] rsum_b_sum;

  logic        rsum_p_valid;
  logic        rsum_p_last;
  logic [15:0] rsum_p_result;

  logic        rsum_data_ready;
  logic [15:0] rsum_packed_wire;

  assign rsum_data_ready = !rsum_in_valid && !rsum_a_valid && !rsum_b_valid && !rsum_p_valid;

  always_comb begin : comb_rsum_pack
    rsum_packed_wire = 16'd0;
    if (rsum_b_bypass) begin
      rsum_packed_wire = rsum_b_bypass_value;
    end else if (rsum_b_sum == 10'd0) begin
      rsum_packed_wire = 16'd0;
    end else if (rsum_b_sum[9]) begin
      rsum_packed_wire = {rsum_b_sout, 8'(rsum_b_elarge + 8'd1), rsum_b_sum[9:3]};
    end else if (rsum_b_sum[8]) begin
      rsum_packed_wire = {rsum_b_sout, rsum_b_elarge, rsum_b_sum[8:2]};
    end else begin
      rsum_packed_wire = {rsum_b_sout, 8'(rsum_b_elarge - 8'd1), rsum_b_sum[7:1]};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      rsum_count     <= 16'd0;
      rsum_accum     <= 16'd0;
      rsum_active    <= 1'b0;
      rsum_out_valid <= 1'b0;
      rsum_out       <= 16'd0;
      rsum_in_valid  <= 1'b0;
      rsum_in_data   <= 16'd0;
      rsum_in_last   <= 1'b0;
      rsum_a_valid   <= 1'b0;
      rsum_a_last    <= 1'b0;
      rsum_a_bypass  <= 1'b0;
      rsum_a_bypass_value <= 16'd0;
      rsum_a_sa      <= 1'b0;
      rsum_a_sb      <= 1'b0;
      rsum_a_elarge  <= 8'd0;
      rsum_a_ma      <= 9'd0;
      rsum_a_mb      <= 9'd0;
      rsum_b_valid   <= 1'b0;
      rsum_b_last    <= 1'b0;
      rsum_b_bypass  <= 1'b0;
      rsum_b_bypass_value <= 16'd0;
      rsum_b_sout    <= 1'b0;
      rsum_b_elarge  <= 8'd0;
      rsum_b_sum     <= 10'd0;
      rsum_p_valid   <= 1'b0;
      rsum_p_last    <= 1'b0;
      rsum_p_result  <= 16'd0;
    end else begin
      rsum_out_valid <= 1'b0;

      // Stage 0: accept exactly one input while the dependent accumulator
      // pipeline is empty. CVO_top already honors OUT_data_ready.
      if (IN_valid && (IN_func == CVO_REDUCE_SUM) && rsum_data_ready) begin
        rsum_in_valid <= 1'b1;
        rsum_in_data  <= IN_data;
        if (!rsum_active) begin
          rsum_in_last <= (IN_length <= 16'd1);
          rsum_count   <= (IN_length <= 16'd1) ? 16'd0 : (IN_length - 16'd1);
          rsum_active  <= (IN_length > 16'd1);
        end else begin
          rsum_in_last <= (rsum_count == 16'd1);
          rsum_count   <= rsum_count - 16'd1;
          rsum_active  <= (rsum_count != 16'd1);
        end
      end else begin
        rsum_in_valid <= 1'b0;
        if ((IN_func != CVO_REDUCE_SUM) && rsum_data_ready) begin
          rsum_active <= 1'b0;
          rsum_accum  <= 16'd0;
        end
      end

      // Stage 1: unpack, exponent-align mantissas, and carry zero bypasses.
      rsum_a_valid <= rsum_in_valid;
      rsum_a_last  <= rsum_in_last;
      if (rsum_accum[14:0] == 0 || rsum_in_data[14:0] == 0) begin
        rsum_a_bypass       <= 1'b1;
        rsum_a_bypass_value <= (rsum_accum[14:0] == 0) ? rsum_in_data : rsum_accum;
        rsum_a_sa           <= 1'b0;
        rsum_a_sb           <= 1'b0;
        rsum_a_elarge       <= 8'd0;
        rsum_a_ma           <= 9'd0;
        rsum_a_mb           <= 9'd0;
      end else begin
        rsum_a_bypass       <= 1'b0;
        rsum_a_bypass_value <= 16'd0;
        rsum_a_sa           <= rsum_accum[15];
        rsum_a_sb           <= rsum_in_data[15];
        if (rsum_accum[14:7] >= rsum_in_data[14:7]) begin
          rsum_a_elarge <= rsum_accum[14:7];
          rsum_a_ma     <= {1'b0, 1'b1, rsum_accum[6:0]};
          rsum_a_mb     <= 9'({1'b0, 1'b1, rsum_in_data[6:0]} >> (rsum_accum[14:7] - rsum_in_data[14:7]));
        end else begin
          rsum_a_elarge <= rsum_in_data[14:7];
          rsum_a_ma     <= 9'({1'b0, 1'b1, rsum_accum[6:0]} >> (rsum_in_data[14:7] - rsum_accum[14:7]));
          rsum_a_mb     <= {1'b0, 1'b1, rsum_in_data[6:0]};
        end
      end

      // Stage 2: signed mantissa add/subtract.
      rsum_b_valid        <= rsum_a_valid;
      rsum_b_last         <= rsum_a_last;
      rsum_b_bypass       <= rsum_a_bypass;
      rsum_b_bypass_value <= rsum_a_bypass_value;
      rsum_b_elarge       <= rsum_a_elarge;
      if (rsum_a_bypass) begin
        rsum_b_sout <= 1'b0;
        rsum_b_sum  <= 10'd0;
      end else if (rsum_a_sa == rsum_a_sb) begin
        rsum_b_sout <= rsum_a_sa;
        rsum_b_sum  <= {1'b0, rsum_a_ma} + {1'b0, rsum_a_mb};
      end else if (rsum_a_ma >= rsum_a_mb) begin
        rsum_b_sout <= rsum_a_sa;
        rsum_b_sum  <= {1'b0, rsum_a_ma} - {1'b0, rsum_a_mb};
      end else begin
        rsum_b_sout <= rsum_a_sb;
        rsum_b_sum  <= {1'b0, rsum_a_mb} - {1'b0, rsum_a_ma};
      end

      // Stage 3: normalize and pack BF16.
      rsum_p_valid  <= rsum_b_valid;
      rsum_p_last   <= rsum_b_last;
      rsum_p_result <= rsum_packed_wire;

      // Stage 4: commit accumulator or emit final scalar.
      if (rsum_p_valid) begin
        if (rsum_p_last) begin
          rsum_out       <= rsum_p_result;
          rsum_out_valid <= 1'b1;
          rsum_accum     <= 16'd0;
        end else begin
          rsum_accum <= rsum_p_result;
        end
      end
    end
  end

  // ===| GELU Pipeline (22 stages) |=============================================
  // GELU(x) ≈ x * sigmoid(1.702*x),  sigmoid(y) = 1/(1+exp(-y))
  // Chain: MUL(1.702)[1] -> NEG[0] -> EXP convert[3] -> EXP lut[2] ->
  // ADD(1)[2] -> RECIP seed/Newton[5] -> MUL(x)[1]. The BF16-to-fixed
  // conversion, denominator add, and Newton correction are split to keep
  // 400 MHz boundaries real. x is preserved in a 19-cycle delay chain.

  localparam int GeluDelay = 19;

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

  // Stages g3-g8: EXP(-y) — classify, shift, sign, then log2(e) multiply
  logic               gelu_e1_valid;
  logic        [ 8:0] gelu_e1_n;
  logic        [ 6:0] gelu_e1_frac;

  logic        gelu_e0a_valid;
  logic        gelu_e0a_sign;
  logic        gelu_e0a_zero;
  logic        gelu_e0a_sat;
  logic        gelu_e0a_shift_left;
  logic [ 4:0] gelu_e0a_shift_amt;
  logic [15:0] gelu_e0a_mant_ext;

  logic        gelu_e0b_valid;
  logic        gelu_e0b_sign;
  logic [15:0] gelu_e0b_mag;

  logic               gelu_e0_valid;
  logic signed [15:0] gelu_e0_xf;
  logic signed [23:0] gelu_e0_y_wire;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_e0a_valid      <= 1'b0;
      gelu_e0a_sign       <= 1'b0;
      gelu_e0a_zero       <= 1'b0;
      gelu_e0a_sat        <= 1'b0;
      gelu_e0a_shift_left <= 1'b0;
      gelu_e0a_shift_amt  <= 5'd0;
      gelu_e0a_mant_ext   <= 16'd0;
      gelu_e0b_valid      <= 1'b0;
      gelu_e0b_sign       <= 1'b0;
      gelu_e0b_mag        <= 16'd0;
      gelu_e0_valid       <= 1'b0;
      gelu_e0_xf          <= '0;
    end else begin
      // Stage e0a: classify the BF16 exponent and register shift controls.
      automatic int sh;
      sh = int'(gelu_g2_neg_y[14:7]) - 127;
      gelu_e0a_valid      <= gelu_g2_valid;
      gelu_e0a_sign       <= gelu_g2_neg_y[15];
      gelu_e0a_mant_ext   <= {1'b0, 1'b1, gelu_g2_neg_y[6:0], 7'b0};
      gelu_e0a_zero       <= (gelu_g2_neg_y[14:7] == 8'd0) || (sh <= -23);
      gelu_e0a_sat        <= (sh >= 8);
      gelu_e0a_shift_left <= (sh >= -7);
      if (sh >= -7) gelu_e0a_shift_amt <= 5'(sh + 7);
      else gelu_e0a_shift_amt <= 5'(-(sh + 7));

      // Stage e0b: perform only the variable shift / saturation.
      gelu_e0b_valid <= gelu_e0a_valid;
      gelu_e0b_sign  <= gelu_e0a_sign;
      if (gelu_e0a_zero) begin
        gelu_e0b_mag <= 16'd0;
      end else if (gelu_e0a_sat) begin
        gelu_e0b_mag <= 16'h7FFF;
      end else if (gelu_e0a_shift_left) begin
        gelu_e0b_mag <= 16'(gelu_e0a_mant_ext << gelu_e0a_shift_amt);
      end else begin
        gelu_e0b_mag <= 16'(gelu_e0a_mant_ext >> gelu_e0a_shift_amt);
      end

      // Stage e0c: apply sign before the DSP multiply input boundary.
      gelu_e0_valid <= gelu_e0b_valid;
      gelu_e0_xf    <= gelu_e0b_sign ? -$signed({1'b0, gelu_e0b_mag}) :
                                      $signed({1'b0, gelu_e0b_mag});
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

  // Stages g6a-g6b: 1 + exp(-y)
  logic        gelu_g6a_valid;
  logic        gelu_g6a_bypass;
  logic [15:0] gelu_g6a_bypass_value;
  logic [ 7:0] gelu_g6a_elarge;
  logic [ 8:0] gelu_g6a_ma;
  logic [ 8:0] gelu_g6a_mb;

  logic        gelu_g6_valid;
  logic [15:0] gelu_g6_denom;

  logic [ 9:0] gelu_g6_sum_wire;
  logic [15:0] gelu_g6_pack_wire;

  assign gelu_g6_sum_wire = {1'b0, gelu_g6a_ma} + {1'b0, gelu_g6a_mb};

  always_comb begin : comb_gelu_denom_pack
    gelu_g6_pack_wire = 16'd0;
    if (gelu_g6a_bypass) begin
      gelu_g6_pack_wire = gelu_g6a_bypass_value;
    end else if (gelu_g6_sum_wire == 10'd0) begin
      gelu_g6_pack_wire = 16'd0;
    end else if (gelu_g6_sum_wire[9]) begin
      gelu_g6_pack_wire = {1'b0, 8'(gelu_g6a_elarge + 8'd1), gelu_g6_sum_wire[9:3]};
    end else if (gelu_g6_sum_wire[8]) begin
      gelu_g6_pack_wire = {1'b0, gelu_g6a_elarge, gelu_g6_sum_wire[8:2]};
    end else begin
      gelu_g6_pack_wire = {1'b0, 8'(gelu_g6a_elarge - 8'd1), gelu_g6_sum_wire[7:1]};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_g6a_valid        <= 1'b0;
      gelu_g6a_bypass       <= 1'b0;
      gelu_g6a_bypass_value <= 16'd0;
      gelu_g6a_elarge       <= 8'd0;
      gelu_g6a_ma           <= 9'd0;
      gelu_g6a_mb           <= 9'd0;
      gelu_g6_valid         <= 1'b0;
      gelu_g6_denom         <= 16'd0;
    end else begin
      gelu_g6a_valid <= gelu_e2_valid;
      if (gelu_e2_expval[14:0] == 15'd0) begin
        gelu_g6a_bypass       <= 1'b1;
        gelu_g6a_bypass_value <= Bf16One;
        gelu_g6a_elarge       <= 8'd0;
        gelu_g6a_ma           <= 9'd0;
        gelu_g6a_mb           <= 9'd0;
      end else if (8'd127 >= gelu_e2_expval[14:7]) begin
        gelu_g6a_bypass       <= 1'b0;
        gelu_g6a_bypass_value <= 16'd0;
        gelu_g6a_elarge       <= 8'd127;
        gelu_g6a_ma           <= {1'b0, 1'b1, Bf16One[6:0]};
        gelu_g6a_mb           <= 9'({1'b0, 1'b1, gelu_e2_expval[6:0]} >>
                                     (8'd127 - gelu_e2_expval[14:7]));
      end else begin
        gelu_g6a_bypass       <= 1'b0;
        gelu_g6a_bypass_value <= 16'd0;
        gelu_g6a_elarge       <= gelu_e2_expval[14:7];
        gelu_g6a_ma           <= 9'({1'b0, 1'b1, Bf16One[6:0]} >>
                                     (gelu_e2_expval[14:7] - 8'd127));
        gelu_g6a_mb           <= {1'b0, 1'b1, gelu_e2_expval[6:0]};
      end
      gelu_g6_valid <= gelu_g6a_valid;
      gelu_g6_denom <= gelu_g6_pack_wire;
    end
  end

  // Stages g7-g12: RECIP(1 + exp(-y)) = sigmoid
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

  logic        gelu_r2a_valid;
  logic        gelu_r2a_zero;
  logic        gelu_r2a_sign;
  logic [ 8:0] gelu_r2a_esum;
  logic [15:0] gelu_r2a_mp;
  logic [15:0] gelu_r2a_r0;

  logic [ 7:0] gelu_r2a_er_wire;
  logic [ 6:0] gelu_r2a_mr_wire;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r2a_valid <= 1'b0;
      gelu_r2a_zero  <= 1'b1;
      gelu_r2a_sign  <= 1'b0;
      gelu_r2a_esum  <= 9'd0;
      gelu_r2a_mp    <= 16'd0;
      gelu_r2a_r0    <= 16'd0;
    end else begin
      gelu_r2a_valid <= gelu_r1_valid;
      gelu_r2a_zero  <= (gelu_r1_x[14:0] == 15'd0) || (gelu_r1_r0[14:0] == 15'd0);
      gelu_r2a_sign  <= gelu_r1_x[15] ^ gelu_r1_r0[15];
      gelu_r2a_esum  <= {1'b0, gelu_r1_x[14:7]} + {1'b0, gelu_r1_r0[14:7]};
      gelu_r2a_mp    <= {1'b1, gelu_r1_x[6:0]} * {1'b1, gelu_r1_r0[6:0]};
      gelu_r2a_r0    <= gelu_r1_r0;
    end
  end

  always_comb begin : comb_gelu_xr0_pack
    if (gelu_r2a_mp[15]) begin
      gelu_r2a_er_wire = 8'(gelu_r2a_esum - 9'd127 + 9'd1);
      gelu_r2a_mr_wire = gelu_r2a_mp[14:8];
    end else begin
      gelu_r2a_er_wire = 8'(gelu_r2a_esum - 9'd127);
      gelu_r2a_mr_wire = gelu_r2a_mp[13:7];
    end
  end

  logic        gelu_r2_valid;
  logic [15:0] gelu_r2_xr0;
  logic [15:0] gelu_r2_r0;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r2_valid <= 1'b0;
      gelu_r2_xr0   <= 16'd0;
      gelu_r2_r0    <= 16'd0;
    end else begin
      gelu_r2_valid <= gelu_r2a_valid;
      gelu_r2_xr0   <= gelu_r2a_zero ? 16'd0 : {gelu_r2a_sign, gelu_r2a_er_wire, gelu_r2a_mr_wire};
      gelu_r2_r0    <= gelu_r2a_r0;
    end
  end

  logic        gelu_r3aa_valid;
  logic        gelu_r3aa_bypass;
  logic [15:0] gelu_r3aa_bypass_value;
  logic [ 7:0] gelu_r3aa_elarge;
  logic [ 8:0] gelu_r3aa_ma;
  logic [ 8:0] gelu_r3aa_mb;
  logic [15:0] gelu_r3aa_r0;

  logic [ 7:0] gelu_corr_elarge_wire;
  logic [ 8:0] gelu_corr_ma_wire;
  logic [ 8:0] gelu_corr_mb_wire;

  always_comb begin : comb_gelu_corr_align
    if (8'd128 >= gelu_r2_xr0[14:7]) begin
      gelu_corr_elarge_wire = 8'd128;
      gelu_corr_ma_wire     = {1'b0, 1'b1, 7'd0};
      gelu_corr_mb_wire     = 9'({1'b0, 1'b1, gelu_r2_xr0[6:0]} >>
                                  (8'd128 - gelu_r2_xr0[14:7]));
    end else begin
      gelu_corr_elarge_wire = gelu_r2_xr0[14:7];
      gelu_corr_ma_wire     = 9'({1'b0, 1'b1, 7'd0} >>
                                  (gelu_r2_xr0[14:7] - 8'd128));
      gelu_corr_mb_wire     = {1'b0, 1'b1, gelu_r2_xr0[6:0]};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r3aa_valid        <= 1'b0;
      gelu_r3aa_bypass       <= 1'b0;
      gelu_r3aa_bypass_value <= 16'd0;
      gelu_r3aa_elarge       <= 8'd0;
      gelu_r3aa_ma           <= 9'd0;
      gelu_r3aa_mb           <= 9'd0;
      gelu_r3aa_r0           <= 16'd0;
    end else begin
      gelu_r3aa_valid        <= gelu_r2_valid;
      gelu_r3aa_bypass       <= (gelu_r2_xr0[14:0] == 15'd0);
      gelu_r3aa_bypass_value <= Bf16Two;
      gelu_r3aa_elarge       <= gelu_corr_elarge_wire;
      gelu_r3aa_ma           <= gelu_corr_ma_wire;
      gelu_r3aa_mb           <= gelu_corr_mb_wire;
      gelu_r3aa_r0           <= gelu_r2_r0;
    end
  end

  logic        gelu_r3a_valid;
  logic [15:0] gelu_r3a_r0;
  logic [15:0] gelu_r3a_corr;

  logic [15:0] gelu_corr_packed_wire;
  logic        gelu_corr_sout_wire;
  logic [ 9:0] gelu_corr_sum_wire;
  logic        gelu_r3ab_valid;
  logic        gelu_r3ab_bypass;
  logic [15:0] gelu_r3ab_bypass_value;
  logic [ 7:0] gelu_r3ab_elarge;
  logic        gelu_r3ab_sout;
  logic [ 9:0] gelu_r3ab_sum;
  logic [15:0] gelu_r3ab_r0;

  always_comb begin : comb_gelu_corr_pack
    if (gelu_r3aa_ma >= gelu_r3aa_mb) begin
      gelu_corr_sout_wire = 1'b0;
      gelu_corr_sum_wire  = {1'b0, gelu_r3aa_ma} - {1'b0, gelu_r3aa_mb};
    end else begin
      gelu_corr_sout_wire = 1'b1;
      gelu_corr_sum_wire  = {1'b0, gelu_r3aa_mb} - {1'b0, gelu_r3aa_ma};
    end

    if (gelu_r3ab_bypass) begin
      gelu_corr_packed_wire = gelu_r3ab_bypass_value;
    end else if (gelu_r3ab_sum == 10'd0) begin
      gelu_corr_packed_wire = 16'd0;
    end else if (gelu_r3ab_sum[9]) begin
      gelu_corr_packed_wire = {gelu_r3ab_sout,
                               8'(gelu_r3ab_elarge + 8'd1),
                               gelu_r3ab_sum[9:3]};
    end else if (gelu_r3ab_sum[8]) begin
      gelu_corr_packed_wire = {gelu_r3ab_sout,
                               gelu_r3ab_elarge,
                               gelu_r3ab_sum[8:2]};
    end else begin
      gelu_corr_packed_wire = {gelu_r3ab_sout,
                               8'(gelu_r3ab_elarge - 8'd1),
                               gelu_r3ab_sum[7:1]};
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r3ab_valid        <= 1'b0;
      gelu_r3ab_bypass       <= 1'b0;
      gelu_r3ab_bypass_value <= 16'd0;
      gelu_r3ab_elarge       <= 8'd0;
      gelu_r3ab_sout         <= 1'b0;
      gelu_r3ab_sum          <= 10'd0;
      gelu_r3ab_r0           <= 16'd0;
    end else begin
      gelu_r3ab_valid        <= gelu_r3aa_valid;
      gelu_r3ab_bypass       <= gelu_r3aa_bypass;
      gelu_r3ab_bypass_value <= gelu_r3aa_bypass_value;
      gelu_r3ab_elarge       <= gelu_r3aa_elarge;
      gelu_r3ab_sout         <= gelu_corr_sout_wire;
      gelu_r3ab_sum          <= gelu_corr_sum_wire;
      gelu_r3ab_r0           <= gelu_r3aa_r0;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r3a_valid <= 1'b0;
      gelu_r3a_r0    <= 16'd0;
      gelu_r3a_corr  <= 16'd0;
    end else begin
      gelu_r3a_valid <= gelu_r3ab_valid;
      gelu_r3a_r0    <= gelu_r3ab_r0;
      gelu_r3a_corr  <= gelu_corr_packed_wire;
    end
  end

  logic        gelu_r3_valid;
  logic [15:0] gelu_r3_sigmoid;
  logic        gelu_r3m_valid;
  logic        gelu_r3m_zero;
  logic        gelu_r3m_sign;
  logic [ 8:0] gelu_r3m_esum;
  logic [15:0] gelu_r3m_mp;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r3m_valid <= 1'b0;
      gelu_r3m_zero  <= 1'b0;
      gelu_r3m_sign  <= 1'b0;
      gelu_r3m_esum  <= 9'd0;
      gelu_r3m_mp    <= 16'd0;
    end else begin
      gelu_r3m_valid <= gelu_r3a_valid;
      if (gelu_r3a_valid) begin
        gelu_r3m_zero <= (gelu_r3a_r0[14:0] == 15'd0) ||
                         (gelu_r3a_corr[14:0] == 15'd0);
        gelu_r3m_sign <= gelu_r3a_r0[15] ^ gelu_r3a_corr[15];
        gelu_r3m_esum <= {1'b0, gelu_r3a_r0[14:7]} +
                         {1'b0, gelu_r3a_corr[14:7]};
        gelu_r3m_mp   <= {1'b1, gelu_r3a_r0[6:0]} *
                         {1'b1, gelu_r3a_corr[6:0]};
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_r3_valid <= 1'b0;
    end else begin
      gelu_r3_valid <= gelu_r3m_valid;
      if (gelu_r3m_valid) begin
        if (gelu_r3m_zero) begin
          gelu_r3_sigmoid <= 16'd0;
        end else if (gelu_r3m_mp[15]) begin
          gelu_r3_sigmoid <= {gelu_r3m_sign, 8'(gelu_r3m_esum - 9'd127 + 9'd1), gelu_r3m_mp[14:8]};
        end else begin
          gelu_r3_sigmoid <= {gelu_r3m_sign, 8'(gelu_r3m_esum - 9'd127), gelu_r3m_mp[13:7]};
        end
      end
    end
  end

  // Stages g10a-g10b: x * sigmoid = GELU(x)
  logic        gelu_m1_valid;
  logic        gelu_m1_zero;
  logic        gelu_m1_sign;
  logic [ 8:0] gelu_m1_esum;
  logic [15:0] gelu_m1_mp;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_m1_valid <= 1'b0;
      gelu_m1_zero  <= 1'b0;
      gelu_m1_sign  <= 1'b0;
      gelu_m1_esum  <= 9'd0;
      gelu_m1_mp    <= 16'd0;
    end else begin
      gelu_m1_valid <= gelu_r3_valid;
      if (gelu_r3_valid) begin
        gelu_m1_zero <= (gelu_x_pipe[GeluDelay-1][14:0] == 15'd0) ||
                        (gelu_r3_sigmoid[14:0] == 15'd0);
        gelu_m1_sign <= gelu_x_pipe[GeluDelay-1][15] ^ gelu_r3_sigmoid[15];
        gelu_m1_esum <= {1'b0, gelu_x_pipe[GeluDelay-1][14:7]} +
                        {1'b0, gelu_r3_sigmoid[14:7]};
        gelu_m1_mp   <= {1'b1, gelu_x_pipe[GeluDelay-1][6:0]} *
                        {1'b1, gelu_r3_sigmoid[6:0]};
      end
    end
  end

  logic        gelu_out_valid;
  logic [15:0] gelu_out;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      gelu_out_valid <= 1'b0;
    end else begin
      gelu_out_valid <= gelu_m1_valid;
      if (gelu_m1_valid) begin
        if (gelu_m1_zero) begin
          gelu_out <= 16'd0;
        end else if (gelu_m1_mp[15]) begin
          gelu_out <= {gelu_m1_sign, 8'(gelu_m1_esum - 9'd127 + 9'd1), gelu_m1_mp[14:8]};
        end else begin
          gelu_out <= {gelu_m1_sign, 8'(gelu_m1_esum - 9'd127), gelu_m1_mp[13:7]};
        end
      end
    end
  end

  // ===| Output Mux |============================================================
  always_comb begin
    OUT_data_ready   = (IN_func == CVO_REDUCE_SUM) ? rsum_data_ready : 1'b1;
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
        OUT_result = recip_s5_result;
        OUT_result_valid = recip_s5_valid;
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
