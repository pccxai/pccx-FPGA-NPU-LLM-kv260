`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/19/2026 12:13:39 AM
// Design Name: 
// Module Name: tb_systolic_dsp_unit
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
/*
`include "stlc_Array.svh"

module tb_systolic_dsp_unit();
    logic        clk;
    logic        rst_n;

    logic i_clear;
    logic i_valid;

    logic [1:0] mode;

    // horizontal wire
    logic [`STLC_MAC_UNIT_IN_H - 1:0] wire_in_A;
    logic [`STLC_MAC_UNIT_IN_H - 1:0] wire_out_A;

    // vertical wire
    logic [`STLC_MAC_UNIT_IN_V - 1:0] wire_in_B;
    logic [`STLC_MAC_UNIT_IN_V - 1:0] wire_out_B;
    logic [1:0] wire_mode_B;

    logic [47:0] result;

    systolic_dsp_unit uut (
        .clk   (clk),
        .rst_n (rst_n),
        .i_clear (i_clear),
        
        .wire_in_A (wire_in_A),
        .wire_out_A (wire_out_A),
        .i_valid (i_valid),
        
        .wire_in_B (wire_in_B),
        .wire_out_B (wire_out_B),
        .wire_mode_B (wire_mode_B),
        .result (result)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 0; i_clear = 0; i_valid = 0;
        #20;
        rst_n = 1;
        
        #20;

        
        #10; i_clear = 1;
        #20;

        @(posedge clk);
        i_valid = 1; mode=`DSP_SYSTOLIC_MOD_P; wire_in_A = 8'd2; wire_in_B = 8'd3;
        
        //@(posedge clk);
        //i_valid = 0;

        #50; 

        @(posedge clk);
        i_valid = 1; mode=`DSP_SYSTOLIC_MOD_P; wire_in_A = 8'd4; wire_in_B = 8'd5;

        #50; 

        @(posedge clk);
        i_valid = 1; mode=`DSP_SYSTOLIC_MOD_P; wire_in_A = 8'd10; wire_in_B = 8'd10;

        #50;
        $display("Final Accumulation: %d", result);
        $finish;
    end

endmodule

*/