`timescale 1ns / 1ps

`include "Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

// weight size = 4bit
// feature_map size =  bf16
module Vec_Matrix_MUL(
    parameter   line_length = 32,
    parameter   line_cnt = 64,
    parameter   reduction_rate = 4, //4:1
    parameter   in_weight_size = `INT4,
    parameter   in_fmap_size = `BF16,
    parameter   in_fmap_e_size = `BF16_EXP,
    parameter   in_fmap_m_size = `BF16_MANTISSA

)(
    input logic  clk,
    input logic  rst_n,

    input logic i_valid [0:line_cnt - 1],
    input logic [in_weight_size - 1:0] IN_weight[0:line_cnt -1],
    input logic [in_fmap_size - 1:0] IN_feature_map[0:line_cnt - 1]
);
    //bf16 [sign-1][e-8][mantissa-7]
    logic  SIGN_Reduction_wire[0:line_cnt];
    logic [in_fmap_e_size-1:0] EXP_Reduction_wire[0:line_cnt];
    logic [in_fmap_m_size-1:0] MANTISSA_Reduction_wire[0:line_cnt];

    logic [`FP32 - 1:0] OUT_final_fp32;
    logic OUT_final_valid;

    // input 64 -> multiplier 64
    // batch reduction 4:1 = 16 -> 4 -> 1.
    genvar i, k;
    generate
        for (i = 0; i < line_cnt; i++) begin : weight_fifos
            multiplier_shift_BF16_INT4 #(
                .in_weight_size(in_weight_size),
                .in_fmap_size(in_fmap_size),
                .in_fmap_e_size(in_fmap_e_size),
                .in_fmap_m_size(in_fmap_m_size)
            ) multiplier_shift (
                .IN_weight(IN_weight[i]),
                .IN_feature_map(IN_feature_map[i]),
                .i_valid(i_valid[i]),
                .OUT_sign(SIGN_Reduction_wire),
                .OUT_EXPONENT(EXP_Reduction_wire),
                .OUT_MANTISSA(MANTISSA_Reduction_wire)
            );
        end

        BF16_FP32_Reduction #(
            .line_length(line_length),
            .line_cnt(line_cnt),
            .exp_size(in_fmap_e_size),
            .mantissa_size(in_fmap_m_size),
            .reduction_rate(reduction_rate)
        ) reduction (
            .clk(clk),
            .rst_n(rst_n),
            .i_valid(i_valid),
            .IN_sign(SIGN_Reduction_wire),
            .IN_EXPONENT(EXP_Reduction_wire),
            .IN_MANTISSA(MANTISSA_Reduction_wire),
            .OUT_final_fp32(OUT_final_fp32),
            .OUT_final_valid(OUT_final_valid)
        )
    endgenerate
endmodule
