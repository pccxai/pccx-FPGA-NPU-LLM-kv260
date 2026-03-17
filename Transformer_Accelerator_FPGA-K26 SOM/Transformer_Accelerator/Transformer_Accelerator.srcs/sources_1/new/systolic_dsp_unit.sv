`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/17/2026 09:09:56 PM
// Design Name: 
// Module Name: systolic_dsp_unit
// Project Name: TINYNPU-RTL
// Target Devices: KV260
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

`define DSP48E2_MAXIN_A = 27
`define DSP48E2_MAXIN_B = 16
`define DSP48E2_MAXOUT = 48

`define DSP_SYSTOLIC_MOD = 2'b00
`define DSP_MULT_MOD = 2'b01
`define DSP_ADD_MOD = 2'b10
/*
`define DSP_SUB_MOD = 2'b11
`define DSP_INV_DIV_MOD 
*/

module systolic_dsp_unit(
    input   logic clk,
    input   logic rst_n,

    // [2-bit mode selector] --------------------------------------------
    // 2'b00 = ADD Mode, 
    // 2'b01 = MULT Mode, 
    // 2'b10 = SYSTOLIC (MAC) Mode
    input   logic [1:0]mode,
    
    input   logic i_clear,
    input   logic i_valid,

    // horizontal wire
    input   logic [`DSP48E2_MAXIN_A - 1:0] wire_in_A,
    output  logic [`DSP48E2_MAXIN_A - 1:0] wire_out_A,

    // vertical wire
    input   logic [`define DSP48E2_MAXIN_B - 1:0] wire_in_B,
    output  logic [`define DSP48E2_MAXIN_B - 1:0] wire_out_B,

    output  logic [47:0] result,
    
    // Internal registers to pipeline the inputs and outputs
    logic signed [`DSP48E2_MAXIN_A - 1:0]  reg_a;
    logic signed [`DSP48E2_MAXIN_B - 1:0]  reg_b;
    logic signed [31:0] reg_acc;
    logic signed [31:0] reg_local;

    alawys_ff @(posedge clk) begin 
        if(rst_n) begin
            
        end else begin
            reg_a <= wire_in_A;
            reg_b <= wire_in_B;

            case (mode)
                DSP_SYSTOLIC_MOD: begin
                    result <= result + (wire_in_A * wire_in_B);
                    wire_in_B <= wire_in_A;
                    wire_out_B <= wire_in_B;
                end

                DSP_ADD_MOD: begin
                    // preAdder
                    result = 
                    // ALU(Post Adder)
                    result = 
                end

                DSP_MULT_MOD: begin
                    // todo...
                end

                default: begin
                    // Default fallback to prevent latches
                    reg_acc   <= '0;
                    reg_local <= '0;
                end

            endcase
        end
    end
    );

endmodule
