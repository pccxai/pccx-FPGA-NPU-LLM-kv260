`timescale 1ns / 1ps

`include "stlc_Array.svh"

module stlc_instruction_dispatcher(
    input logic     clk,
    input logic     rst_n,

    input logic     is_last_op,
    input logic     i_valid,

    output logic    [$clog2(`DSP_INSTRUCTION_CNT) - 1:0] instruction,
    output logic    o_valid
    );

    
    // TODO.. What if the input is? ?
    // 1. Matrix x Matrix, Matrix x Vector
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            instruction <= `DSP_IDLE_MOD;
            o_valid     <= `FALSE;
        end else begin
            if (is_last_op) begin
                instruction <= `DSP_SHIFT_RESULT_MOD;
                o_valid     <= `TRUE;
            end else if (i_valid) begin
                instruction <= `DSP_SYSTOLIC_MOD_P;
                o_valid     <= `TRUE; 
            end
        end
    end

endmodule
