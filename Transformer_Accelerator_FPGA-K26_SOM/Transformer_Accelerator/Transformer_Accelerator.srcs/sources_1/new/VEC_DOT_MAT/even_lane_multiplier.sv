`timescale 1ns / 1ps

//`include "stlc_Array.svh"
`define WEIGHT_HP_PORT_SIZE 512
`define FEATURE_MAP_HPC_PORT_SIZE 256


// weight size = 4bit
// feature_map size =  bf16
module even_lane_multiplier(
    parameter   line_length = 32
)(
    input   clk,
    input   rst_n,
    input [3:0]     IN_weight      [line_length:0],
    input [15:0]    IN_feature_map [line_length:0],

    output [3:0]     OUT_weight,      
    output [15:0]    OUT_feature_map 
    );

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            for(int i = 0; i < line_length; i++) begin
                IN_weight[i] = '0;
                IN_feature_map[i] = '0;
            end
        end else begin
            for (int i=0; i<line_length; ++i) begin
                weight[i+1] <= weight[i];
                IN_feature_map[i+1] <= IN_feature_map[i];
            end
        end

        assign OUT_feature_map = IN_feature_map[line_length - 1];
        assign OUT_weight = IN_weight[line_length - 1];

    end
endmodule
