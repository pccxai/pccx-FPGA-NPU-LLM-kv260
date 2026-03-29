`timescale 1ns / 1ps

//`include "stlc_Array.svh"
`define WEIGHT_HP_PORT_SIZE 512
`define FEATURE_MAP_HPC_PORT_SIZE 256


// weight size = 4bit
// feature_map size =  bf16
module Four_bit_multiplier_ppLine(
    input logic [3:0]     weight      [`WEIGHT_HP_PORT_SIZE:0],
    input logic [15:0]    feature_map [`FEATURE_MAP_HPC_PORT_SIZE:0]


    );

    // 부호 | 지수 | 가수
    //  1  |  7  |  8
    // weight   = [1,2048]
    // fmap     = [1,2048]

    // 64개 필요 (1개당 32 index를 가진 delay line)
    // 데이지 체인화.
    //
    always_comb begin
        case(weight)


    end
endmodule
