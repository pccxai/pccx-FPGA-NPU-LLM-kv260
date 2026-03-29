`timescale 1ns / 1ps

`include "Vec_Matric_MUL.svh"

// weight size = 4bit
// feature_map size =  bf16
module Vec_Matrix_MUL(
    parameter line_length = 32,
    parameter line_cnt = 64,
    parameter sorting_depth = 4

)(
    input logic  clk,
    input logic  rst_n,
);

    logic [line_cnt:0] Sorting_Net_wire[0:sorting_depth];


    genvar i, k;
    generate
        for (i = 0; i < line_cnt; i++) begin : weight_fifos
            multiplier_branch #(
                .IN_WEIGHT_SIZ(`WEIGHT_SIZE),
                .IN_FEATURE_MAP_SIZE(`FEATURE_MAP_SIZE)
            ) u_w_fifo (
                IN_weight(),
                IN_feature_map(),
                input i_valid(),
                output OUT_sign(),
                OUT_EXPONENT(),
                OUT_MANTISSA()
            );
        end
    endgenerate

endmodule
