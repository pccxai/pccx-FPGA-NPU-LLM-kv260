`timescale 1ns / 1ps

`include "Vec_Matric_MUL.svh"

// Descending order

module Spatial_Sorting_Net #(
    parameter line_length = 32,
    parameter sort_depth = 5
)(
    input logic  clk,
    input logic  rst_n,

    //input OUT_IS_LAST,
    input logic IN_sign[0:63],
    input logic [7:0] IN_EXPONENT[0:63],
    input logic [6:0] IN_MANTISSA[0:63]
);

    //logic distance
    //generate
    //0 1 / 2 3 / 4 5
    //0 2 / 1 3
    //0 4

    // 64개들이 각각 1클럭 마다 총 32번? 쏟아져 나온다.

    // float32 타입?
    logic  Sorting_Net_sign [sort_depth - 1:0][63:0];
    logic [7:0] Sorting_Net_exp [sort_depth - 1:0][63:0];
    logic [6:0] Sorting_Net_mantissa [sort_depth - 1:0][63:0];

    always_ff @(posedge clk) begin
        if(!rst_n) begin
        end else begin
            for(int i = 0; i < 32; i++) begin
                if(OUT_EXPONENT[i * 2] < IN_EXPONENT[(i * 2) + 1]);
                    Sorting_Net_exp[i * 2] <= IN_EXPONENT[(i * 2) + 1];
                    Sorting_Net_exp[(i * 2) + 1] <= IN_EXPONENT[i * 2];
                else begin
                    Sorting_Net_exp[i * 2] <= IN_EXPONENT[i * 2];
                    Sorting_Net_exp[(i * 2) + 1] <= IN_EXPONENT[(i * 2) + 1];
                end
            end
                //상위4비트만 보고 그룹화.
            for
        end
    end
endmodule