`timescale 1ns / 1ps

`include "Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

// Descending order

module multiplier_shift_even_lane #(
    parameter in_fmap_e_size = `BF16_EXP,
    parameter in_fmap_m_size = `BF16_MANTISSA
) (
    input logic clk,
    input logic rst_n,
    input logic DELAY_SIGN,
    input logic [IN_FMAP_E_SIZE:0] UN_SAFE_EXP_Q2,
    input logic [IN_FMAP_M_SIZE - 1:0] DELAY_MANTISSA,
    input logic DELAY_IS_EVEN,
    output logic is_out_vaild,
    output logic OUT_sign,
    output logic [in_fmap_e_size - 1:0] OUT_EXPONENT,
    output logic [in_fmap_m_size - 1:0] OUT_MANTISSA
);


  // even LANE, just delay
  // ===| 2 clk |===
  logic even_delay_S;
  logic [7:0] even_delay_E;
  //logic [6:0] even_delay_M [0:2];
  logic lane_even_Q1;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      even_delay_S <= 1'b0;
      even_delay_E <= 1'b0;
      even_delay_M <= 1'b0;
    end else begin
      if (DELAY_IS_EVEN) begin
        even_delay_S <= DELAY_SIGN;
        SAFE_EXP <= (UN_SAFE_EXP_Q2[8]) ? 8'hFF : UN_SAFE_EXP_Q2[7:0];
        even_delay_M <= DELAY_MANTISSA;
        lane_even_Q1 <= DELAY_IS_EVEN;
      end

      if (lane_even_Q1) begin
        is_out_vaild <= lane_even_Q1;
        OUT_sign     <= even_delay_S;
        OUT_EXPONENT <= SAFE_EXP;
        OUT_MANTISSA <= even_delay_M;
      end
    end
  end

endmodule
