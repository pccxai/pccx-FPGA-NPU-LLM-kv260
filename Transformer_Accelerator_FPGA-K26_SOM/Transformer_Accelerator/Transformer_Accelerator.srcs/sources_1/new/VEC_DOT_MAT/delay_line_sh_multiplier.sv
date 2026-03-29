`timescale 1ns / 1ps

//`include "stlc_Array.svh"
`define WEIGHT_HP_PORT_SIZE 512
`define FEATURE_MAP_HPC_PORT_SIZE 256


// weight size = 4bit
// feature_map size =  bf16
module delay_line_sh_multiplier(
    parameter   line_length = 32
)(
    input  logic clk,
    input  logic rst_n,
    input logic [3:0]     IN_weight      [line_length:0],
    input logic [15:0]    IN_feature_map [line_length:0],

    output logic [3:0]     OUT_weight,
    output logic [15:0]    OUT_feature_map
    );

    // 부호 | 지수 | 가수
    //  1  |  7  |  8
    // weight   = [1,2048]
    // fmap     = [1,2048]

    // 64개 필요 (1개당 32 index를 가진 delay line)
    // 데이지 체인화.
    //
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
