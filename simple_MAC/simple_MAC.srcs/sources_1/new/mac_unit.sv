`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/02/23 03:29:56
// Design Name: 
// Module Name: mac_unit
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

module pe_unit (
    input  logic        clk,     
    input  logic        rst_n,   
    input  logic        i_valid, // Input valid signal
    input  logic [7:0]  i_a,     // Data from left PE
    input  logic [7:0]  i_b,     // Data from top PE
    
    output logic [7:0]  o_a,     // Forwarded data to right PE
    output logic [7:0]  o_b,     // Forwarded data to bottom PE
    output logic        o_valid, // Output valid signal
    output logic [15:0] o_acc    // Accumulated MAC result
);

    logic [15:0] mul_result;
    assign mul_result = i_a * i_b; // Combinational multiplication

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Asynchronous active-low reset
            o_acc   <= 16'd0;
            o_valid <= 1'b0;
        o_a     <= 8'd0;
            o_b     <= 8'd0;
        end else if (i_valid) begin 
            // Accumulate and forward data pipeline
            o_acc   <= o_acc + mul_result;
            o_valid <= 1'b1;
            o_a     <= i_a; 
            o_b     <= i_b;
        end else begin  
            // Pipeline stall on invalid data
            o_valid <= 1'b0;
        end
    end
endmodule