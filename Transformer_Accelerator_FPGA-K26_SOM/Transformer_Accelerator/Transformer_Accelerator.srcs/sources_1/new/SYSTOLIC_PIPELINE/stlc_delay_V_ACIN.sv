
`timescale 1ns / 1ps

`include "stlc_Array.svh"



module stlc_delay_V_ACIN #(
    parameter DELAY_LINE_LENGTH = `MINIMUM_DELAY_LINE_LENGTH,
    parameter DELAY_LINE_WIDTH = `STLC_MAC_UNIT_IN_V,
    parameter DSP_INSTRUCTION_CNT = $clog2(`DSP_INSTRUCTION_CNT)
)(
    input logic clk,    
    input logic rst_n,
    
    input logic [DELAY_LINE_WIDTH-1:0] D_Line_in_value,
    input logic [DSP_INSTRUCTION_CNT-1:0] DSP_instruction_in, 
    input logic i_valid,

    output logic [DELAY_LINE_WIDTH-1:0] D_Line_out_value,
    output logic [DSP_INSTRUCTION_CNT-1:0] DSP_instruction_out,
    output logic o_valid
);
    
    logic [DELAY_LINE_WIDTH - 1:0] delay_line [0:DELAY_LINE_LENGTH-1];
    logic [DSP_INSTRUCTION_CNT - 1:0] delay_DSP_instruction_line [0:DELAY_LINE_LENGTH-1];
    logic delay_i_valid_line [0:DELAY_LINE_LENGTH-1];

    //logic valid_line [0:DELAY_LINE_LENGTH-1];

    always_ff @(posedge clk) begin
        if(!rst_n) begin 
            for (int i = 0; i < DELAY_LINE_LENGTH; i++) begin
                delay_line[i] <= 0;
                delay_DSP_instruction_line[i] <= 0;
                delay_i_valid_line[i] <= `FALSE;
            end
        end else begin
            if(i_valid) begin     
                delay_line[0] <= D_Line_in_value;
                delay_DSP_instruction_line[0] <= DSP_instruction_in;
                delay_i_valid_line[0] <= `TRUE;

                for (int i = 1; i < DELAY_LINE_LENGTH; i++) begin
                    delay_line[i] <= delay_line[i-1];
                    delay_DSP_instruction_line[i] <= delay_DSP_instruction_line[i-1];
                    delay_i_valid_line[i] <= delay_i_valid_line[i-1];
                end
            end
        end
    end

    assign D_Line_out_value = delay_line[DELAY_LINE_LENGTH-1];
    assign DSP_instruction_out = delay_DSP_instruction_line[DELAY_LINE_LENGTH-1];
    assign o_valid = delay_i_valid_line[DELAY_LINE_LENGTH-1];


endmodule