`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "GEMM_Array.svh"

// ===| Module: gemm_result_normalizer — INT48 → BF16 4-stage pipeline |=========
// Purpose      : Convert one column's 48-bit accumulated 2's-complement
//                result into a BF16 word, re-using the column's delayed
//                e_max for exponent alignment.
// Spec ref     : pccx v002 §2.2.6 (output normalisation).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// Pipeline     : 4 stages, fully registered.
//                  S1 — sign-magnitude (2's comp → |abs|, sign bit).
//                  S2 — leading-one detect (LOD, priority encoder).
//                  S3 — barrel shift mantissa + exp adjust (e_max + LOD - 26).
//                  S4 — pack {sign[1], exp[8], mantissa[7]} = BF16.
// Latency      : 4 cycles.
// Throughput   : 1 BF16 word per cycle in steady state.
// Handshake    : Push-only (valid_in propagates through registered valid).
// Reset state  : All stage valids = 0; pipeline registers zeroed.
// Errors       : Zero input → exp = 0, mantissa = 0 (denormal-flush style).
// Counters     : none.
// Notes        : ExpAlignBias is sourced from dtype_pkg::FixedMantWidth-1 so
//                the S3 exponent fix-up tracks the upstream fixed-point
//                mantissa width automatically.
// ===============================================================================
module gemm_result_normalizer (
    input logic clk,
    input logic rst_n,

    input logic [47:0] data_in,  // 48-bit Accumulator Result
    input logic [ 7:0] e_max,    // Original delayed exponent for this column
    input logic        valid_in,

    output logic [15:0] data_out,  // 1:Sign, 8:Exp, 7:Mantissa (BF16 format)
    output logic        valid_out
);

  // ===| Exponent alignment bias |================================================
  // The fixed-point mantissa carries the implicit-1 at bit (FixedMantWidth - 1).
  // Stage S3 must subtract that position from the original e_max so the BF16
  // exponent reflects the true magnitude of the leading-one. Tying the bias to
  // dtype_pkg keeps this in lock-step with FixedMantWidth changes.
  localparam logic [7:0] ExpAlignBias = 8'(dtype_pkg::FixedMantWidth - 1);

  // ===| Stage 1: Sign-Magnitude Conversion |=======
  // Converting from 2's complement to absolute value (Sign + Magnitude)
  logic [47:0] s1_abs_data;
  logic        s1_sign;
  logic [ 7:0] s1_emax;
  logic        s1_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s1_valid <= 1'b0;
      s1_sign <= 1'b0;
      s1_emax <= 8'd0;
      s1_abs_data <= 48'd0;
    end else begin
      s1_valid <= valid_in;
      s1_sign <= data_in[47];
      s1_emax <= e_max;

      // If negative, invert and add 1 (2's complement to absolute)
      s1_abs_data <= (data_in[47]) ? (~data_in + 1'b1) : data_in;
    end
  end

  // ===| Stage 2: Leading One Detection (LOD) |=======
  // Finding the position of the most significant '1' bit
  logic [ 5:0] s2_first_one_pos;
  logic [47:0] s2_abs_data;
  logic        s2_sign;
  logic [ 7:0] s2_emax;
  logic        s2_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s2_valid <= 1'b0;
      s2_sign <= 1'b0;
      s2_abs_data <= 48'd0;
      s2_emax <= 8'd0;
      s2_first_one_pos <= 6'd0;
    end else begin
      s2_valid <= s1_valid;
      s2_sign <= s1_sign;
      s2_abs_data <= s1_abs_data;
      s2_emax <= s1_emax;

      // Simple Priority Encoder for LOD
      // In 400MHz, this might need further pipelining if timing fails,
      // but starting with a basic loop since Vivado is good at tree extraction.
      s2_first_one_pos <= 6'd0;  // Default to 0
      for (int i = 46; i >= 0; i--) begin
        if (s1_abs_data[i]) begin
          s2_first_one_pos <= i[5:0];
          break;
        end
      end
    end
  end

  // ===| Stage 3: Normalization Barrel Shift & Exponent Update |=======
  // Shifting the mantissa so that the leading '1' sits right before the 7-bit fractional part.
  logic [6:0] s3_mantissa;
  logic [7:0] s3_new_exp;
  logic       s3_sign;
  logic       s3_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s3_valid <= 1'b0;
      s3_sign <= 1'b0;
      s3_new_exp <= 8'd0;
      s3_mantissa <= 7'd0;
    end else begin
      s3_valid <= s2_valid;
      s3_sign  <= s2_sign;

      if (s2_abs_data == 0) begin
        s3_new_exp  <= 8'd0;
        s3_mantissa <= 7'd0;
      end else begin
        // Update exponent: original e_max + current bit position - alignment bias.
        // ExpAlignBias = dtype_pkg::FixedMantWidth - 1; one place to update if
        // the upstream fixed-point format ever widens.
        s3_new_exp <= s2_emax + s2_first_one_pos - ExpAlignBias;

        // Align mantissa to BF16 (7 bits of fraction)
        if (s2_first_one_pos >= 7)
          // Take the 7 bits immediately below the first '1'
          s3_mantissa <= s2_abs_data[s2_first_one_pos-1-:7];
        else
          // Shift left to pad with zeros
          s3_mantissa <= s2_abs_data[6:0] << (7 - s2_first_one_pos);
      end
    end
  end

  // ===| Stage 4: Final Packing |=======
  // Constructing the final 16-bit word
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      valid_out <= 1'b0;
      data_out  <= 16'd0;
    end else begin
      valid_out <= s3_valid;
      data_out  <= {s3_sign, s3_new_exp, s3_mantissa};
    end
  end
endmodule
