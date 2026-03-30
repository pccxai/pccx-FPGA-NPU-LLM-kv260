`timescale 1ns / 1ps

`include "Vec_Matric_MUL.svh"

// Descending order

module multiplier_shift_odd_lane #(
    parameter in_fmap_e_size = `BF16_EXP,
    parameter in_fmap_m_size = `BF16_MANTISSA
) (
    input logic clk,
    input logic rst_n,
    input logic DELAY_SIGN,
    input logic [IN_FMAP_E_SIZE:0] UN_SAFE_EXP_Q2,
    input logic [IN_FMAP_M_SIZE + 1:0] IN_UN_SAFE_MANTISSA,
    input logic DELAY_IS_EVEN,
    output logic is_out_vaild,
    output logic OUT_sign,
    output logic [in_fmap_e_size - 1:0] OUT_EXPONENT,
    output logic [in_fmap_m_size - 1:0] OUT_MANTISSA
);




  logic [7:0] temp_SAFE_EXPONENT;
  logic [7:0] SAFE_MANTISSA;
  logic [8:0] UN_SAFE_EXP_Q3;
  logic lane_odd_Q1;

  // odd LANE
  // ===| 2 clk |===
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      next_step   <= 1'b0;
      lane_odd_Q1 <= 1'b0;
    end else begin
      if (~DELAY_IS_EVEN) begin
        //SAFE_EXP
        if (IN_UN_SAFE_MANTISSA[8]) begin
          UN_SAFE_EXP_Q3 <= UN_SAFE_EXP_Q2 + 1;
          SAFE_MANTISSA  <= IN_UN_SAFE_MANTISSA >> 1;
        end
        lane_odd_Q1 <= ~DELAY_IS_EVEN;
      end

      if (lane_odd_Q1) begin
        is_out_vaild <= lane_odd_Q1;
        OUT_sign     <= DELAY_SIGN;
        //safe exp
        OUT_EXPONENT <= (UN_SAFE_EXP_Q2[8]) ? 8'hFF : UN_SAFE_EXP_Q2[7:0];
        OUT_MANTISSA <= SAFE_MANTISSA;
      end

    end
  end
endmodule
