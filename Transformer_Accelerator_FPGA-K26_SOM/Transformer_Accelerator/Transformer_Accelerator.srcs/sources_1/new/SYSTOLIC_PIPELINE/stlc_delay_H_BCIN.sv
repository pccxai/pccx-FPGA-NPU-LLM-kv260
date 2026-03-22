`timescale 1ns / 1ps

`include "stlc_Array.svh"


module stlc_delay_H_BCIN #(
    parameter DELAY_LINE_LENGTH = ` stlc_instruction_dispatcher_CLOCK_CONSUMPTION + `MINIMUM_DELAY_LINE_LENGTH,
    parameter DELAY_LINE_BANDWIDTH = `STLC_MAC_UNIT_IN_H
)(
    input logic clk,    
    input logic rst_n,
    

    input logic [DELAY_LINE_BANDWIDTH-1:0] D_Line_in_value,
    input logic i_valid,


    output logic [DELAY_LINE_BANDWIDTH-1:0] D_Line_out_value
);
    
    logic [DELAY_LINE_BANDWIDTH - 1:0] delay_line [0:DELAY_LINE_LENGTH-1];
    
    //logic valid_line [0:DELAY_LINE_LENGTH-1];

    always_ff @(posedge clk) begin
        if(!rst_n) begin 
            for (int i = 0; i < DELAY_LINE_LENGTH; i++) begin
                delay_line[i] <= 0;
            end
        end else begin
            if(i_valid) begin
                delay_line[0] <= D_Line_in_value;
                
                for (int i = 1; i < DELAY_LINE_LENGTH; i++) begin
                    delay_line[i] <= delay_line[i-1];
                end    
            end
        end
    end

    assign D_Line_out_value = delay_line[DELAY_LINE_LENGTH-1];

endmodule
