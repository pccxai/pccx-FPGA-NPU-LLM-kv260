`timescale 1ns / 1ps

`include "Vec_Matric_MUL.svh"

// Descending order

module multiplier_shift_default_lane #(
    parameter in_fmap_e_size = `BF16_EXP,
    parameter in_fmap_m_size = `BF16_MANTISSA,
    parameter delay_length = 4
) (
    input logic clk,
    input logic rst_n,
    input logic [1:0] default_lane_weight,
    input logic IN_sign,
    input logic [IN_FMAP_E_SIZE - 1:0] IN_exp,
    input logic [IN_FMAP_M_SIZE - 1:0] IN_mantissa,
    input logic goto_default_lane,
    output logic is_out_vaild,
    output logic OUT_sign,
    output logic [in_fmap_e_size - 1:0] OUT_EXPONENT,
    output logic [in_fmap_m_size - 1:0] OUT_MANTISSA
);



// even LANE, just delay
// ===| 2 clk + 2clk(valid_q1 & q2)|===
logic delay_line_s[0:delay_length -1];
logic [in_fmap_e_size - 1:0] delay_line_e[0:delay_length -1];
logic [in_fmap_m_size - 1:0] delay_line_m[0:delay_length -1];
logic delay_line_out_ready[0:delay_length -1];

always_ff @(posedge clk) begin
    if (!rst_n) begin
        for (int i = 0; i < delay_length; i++) begin
        delay_line_out_ready[i] <= 1'b0;
        delay_line_s[i] <= 1'b0;
        delay_line_e[i] <= 1'b0;
        delay_line_m[i] <= 1'b0;
        end
    end else begin
        for (int i = 0; i < delay_length; i++) begin
            delay_line_out_ready[i+1] <= delay_line_out_ready[i];
            delay_line_s[i+1] <= delay_line_s[i];
            delay_line_e[i+1] <= delay_line_e[i];
            delay_line_m[i+1] <= delay_line_m[i];
        end

        if(IN_sign) begin
                // 1 or -1
            if(default_lane_weight[0]) begin
                delay_line_s[0] <= IN_sign ^ default_lane_weight[1];
                delay_line_e[0] <= IN_exp;
                delay_line_m[0] <= IN_mantissa;
            end else begin
                // 0
                delay_line_s[0] <= 0;
                delay_line_e[0] <= 0;
                delay_line_m[0] <= 0;
            end
            delay_line_out_ready[i+1] <= `TRUE;
        end else begin
            delay_line_out_ready[i+1] <= ;
        end
    end
end

assign OUT_sign <= delay_line_s[delay_length - 1];
assign OUT_EXPONENT <= delay_line_e[delay_length - 1];
assign OUT_MANTISSA <= delay_line_m[delay_length - 1];
assign is_out_vaild <= delay_line_out_ready[delay_length - 1];

endmodule
