`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import bf16_math_pkg::*;

// ===| Module: CVO_cordic_unit — pipelined BF16 sin/cos CORDIC |================
// Purpose      : Rotation-mode CORDIC computing sin(θ) and cos(θ) for a
//                BF16 input angle (radians).
// Spec ref     : pccx v002 §2.4.2 (CVO CORDIC datapath).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// Internal fmt : Q4.12 signed fixed-point (16-bit). 1.0 = 0x1000 = 4096,
//                π ≈ 0x3244 = 12868.
// CORDIC gain K: 14 iterations ≈ 0.60725. Pre-baked into x0 = K * 4096 = 0x09B8.
// Pipeline     : 16 clocks  (1 convert-in + 14 CORDIC + 1 convert-out).
// Latency      : 16 cycles total.
// Throughput   : 1 sin/cos pair per cycle in steady state.
// Handshake    : Push-only; OUT_valid mirrors the input pipeline.
// Reset state  : All pipeline registers cleared; OUT_valid = 0.
// Errors       : Overflow saturates to 0x7FFF; underflow rounds to 0;
//                zero/denormal exp_raw treated as 0.
// Counters     : none.
// Protected    : Internals untouched (CLAUDE.md §6.2 — CORDIC math is
//                user-owned).
// ===============================================================================

module CVO_cordic_unit (
    input  logic        clk,
    input  logic        rst_n,

    // ===| Input |===============================================================
    input  logic [15:0] IN_angle_bf16,    // BF16 angle in radians
    input  logic        IN_valid,

    // ===| Output |==============================================================
    output logic [15:0] OUT_sin_bf16,
    output logic [15:0] OUT_cos_bf16,
    output logic        OUT_valid
);

  // ===| CORDIC Constants (Q4.12) |==============================================
  // atan(2^-i) * 4096, i = 0..13
  localparam logic signed [15:0] AtanLut [14] = '{
    16'sh0C91,  // atan(2^0 ) = π/4     ≈ 0.7854
    16'sh076B,  // atan(2^-1)           ≈ 0.4636
    16'sh03EB,  // atan(2^-2)           ≈ 0.2450
    16'sh01FD,  // atan(2^-3)           ≈ 0.1244
    16'sh0100,  // atan(2^-4)           ≈ 0.0624
    16'sh0080,  // atan(2^-5)           ≈ 0.0312
    16'sh0040,  // atan(2^-6)           ≈ 0.0156
    16'sh0020,  // atan(2^-7)           ≈ 0.0078
    16'sh0010,  // atan(2^-8)           ≈ 0.0039
    16'sh0008,  // atan(2^-9)           ≈ 0.0020
    16'sh0004,  // atan(2^-10)          ≈ 0.0010
    16'sh0002,  // atan(2^-11)          ≈ 0.0005
    16'sh0001,  // atan(2^-12)          ≈ 0.0002
    16'sh0001   // atan(2^-13)          ≈ 0.0001
  };

  // CORDIC gain K pre-scaled: K * 4096 = 0x09B8
  localparam logic signed [15:0] CordicKQ412 = 16'sh09B8;

  // ===| Stage 0: BF16 → Q4.12 Conversion |=====================================
  logic signed [15:0] s0_angle_fixed;
  logic               s0_valid;

  always_comb begin : bf16_to_q412
    automatic logic        sign_bit;
    automatic logic [ 7:0] exp_raw;
    automatic logic [ 6:0] mant_raw;
    automatic logic [15:0] magnitude;
    automatic int          shift_amt;

    sign_bit = IN_angle_bf16[15];
    exp_raw  = IN_angle_bf16[14:7];
    mant_raw = IN_angle_bf16[ 6:0];

    // (1.mant) * 2^(exp-127) in Q4.12 → multiply by 2^12 / 2^(exp-127)
    // = {1, mant} * 2^(exp - 122)   ({1,mant} is an 8-bit value representing 1.mant * 128)
    shift_amt = int'(exp_raw) - 122;

    if (exp_raw == 8'd0) begin
      // denormal / zero → treat as 0
      magnitude = 16'd0;
    end else if (shift_amt >= 15) begin
      // overflow → saturate to max Q4.12 (π ≈ 3.14, max safe = 3.9999)
      magnitude = 16'h7FFF;
    end else if (shift_amt < -7) begin
      // underflow → rounds to 0
      magnitude = 16'd0;
    end else if (shift_amt >= 0) begin
      magnitude = 16'({1'b1, mant_raw, 7'b0} << shift_amt);
    end else begin
      magnitude = 16'({1'b1, mant_raw, 7'b0} >> (-shift_amt));
    end

    s0_angle_fixed = sign_bit ? -$signed(magnitude) : $signed(magnitude);
  end

  // Register stage 0
  logic signed [15:0] s0_angle_ff;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s0_angle_ff <= 16'sh0;
      s0_valid    <= 1'b0;
    end else begin
      s0_angle_ff <= s0_angle_fixed;
      s0_valid    <= IN_valid;
    end
  end

  // ===| Stages 1-14: CORDIC Iterations |========================================
  // Each stage i: x_{i} → x_{i+1}, y_{i} → y_{i+1}, z_{i} → z_{i+1}
  // d_i = sign(z_i)
  // x_{i+1} = x_i - d_i * (y_i >>> i)
  // y_{i+1} = y_i + d_i * (x_i >>> i)
  // z_{i+1} = z_i - d_i * ATAN_LUT[i]

  logic signed [15:0] cx [0:14];  // CORDIC x pipeline
  logic signed [15:0] cy [0:14];  // CORDIC y pipeline
  logic signed [15:0] cz [0:14];  // CORDIC z pipeline
  logic               cv [0:14];  // valid pipeline

  // Initialize iteration 0 from stage-0 output
  assign cx[0] = CordicKQ412;
  assign cy[0] = 16'sh0;
  assign cz[0] = s0_angle_ff;
  assign cv[0] = s0_valid;

  genvar gi;
  generate
    for (gi = 0; gi < 14; gi++) begin : gen_cordic_iter
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          cx[gi+1] <= 16'sh0;
          cy[gi+1] <= 16'sh0;
          cz[gi+1] <= 16'sh0;
          cv[gi+1] <= 1'b0;
        end else begin
          cv[gi+1] <= cv[gi];
          if (cz[gi] >= 0) begin
            // d_i = +1: rotate counter-clockwise
            cx[gi+1] <= cx[gi] - (cy[gi] >>> gi);
            cy[gi+1] <= cy[gi] + (cx[gi] >>> gi);
            cz[gi+1] <= cz[gi] - AtanLut[gi];
          end else begin
            // d_i = -1: rotate clockwise
            cx[gi+1] <= cx[gi] + (cy[gi] >>> gi);
            cy[gi+1] <= cy[gi] - (cx[gi] >>> gi);
            cz[gi+1] <= cz[gi] + AtanLut[gi];
          end
        end
      end
    end
  endgenerate

  // ===| Stage 15: Q4.12 → BF16 Conversion |=====================================
  // cos result = cx[14], sin result = cy[14]

  function automatic logic [15:0] q412_to_bf16(input logic signed [15:0] val);
    automatic logic        sign_out;
    automatic logic [14:0] mag;
    automatic logic [ 3:0] leading;
    automatic logic [ 7:0] exp_out;
    automatic logic [ 6:0] mant_out;

    sign_out = val[15];
    mag      = sign_out ? 15'(-$signed(val)) : 15'(val);

    // Find leading 1 position (bit 14 = highest in 15-bit magnitude)
    leading = 4'd0;
    for (int b = 14; b >= 0; b--) begin
      if (mag[b]) begin
        leading = 4'(b);
        break;
      end
    end

    if (mag == 0) begin
      return 16'd0;
    end else begin
      // biased exponent: value_in_Q412 = mag * 2^(leading-12)
      // BF16 exponent bias = 127; real exp = leading - 12
      exp_out = 8'd127 + leading - 8'd12;

      // 7 mantissa bits below the leading 1
      if (leading >= 7)
        mant_out = mag[leading-1 -: 7];
      else
        // SV disallows variable part-selects, so shift the full magnitude
        // left to align its `leading` significant bits into the top 7
        // positions, then truncate to 7 bits.
        mant_out = 7'(mag << (7 - leading));

      return {sign_out, exp_out, mant_out};
    end
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      OUT_cos_bf16 <= 16'd0;
      OUT_sin_bf16 <= 16'd0;
      OUT_valid    <= 1'b0;
    end else begin
      OUT_valid    <= cv[14];
      OUT_cos_bf16 <= q412_to_bf16(cx[14]);
      OUT_sin_bf16 <= q412_to_bf16(cy[14]);
    end
  end

endmodule
