`timescale 1ns / 1ps

(* use_dsp = "yes" *)
module pe_unit (
    input  logic               clk,     
    input  logic               rst_n,   
    input  logic               i_clear, 
    input  logic               i_valid, 
    // Expanded to 16-bit signed input
    input  logic signed [15:0] i_a,     
    input  logic signed [15:0] i_b, 
    // The relay output passed to the next PE is also unified to 16 bits.
    output logic signed [15:0] o_a,     
    output logic signed [15:0] o_b,     
    output logic               o_valid, 
    // Overflow prevention when accumulating 32 times and 48-bit accumulator tailored to DSP48E2 native size
    output logic signed [47:0] o_acc 
);
    // 4 int4 x 4 int4 = 32bit (carry bitX <- overflow = 0)
    '''
    Pre-Adder: Receives two inputs and performs 27-bit addition/subtraction

    Multiplier: Multiplies the Pre-Adder result (27 bits) with another input (18 bits)

    ALU (Accumulator/Adder/Subtractor): Receives multiplication result (up to 45 bits) and performs 48-bit accumulation/operation
    '''
    
    always_ff @(posedge clk) begin
        if (!rst_n || i_clear) begin
            o_acc   <= 48'd0;    // 48-bit initialization
            o_valid <= 1'b0;
            o_a     <= 16'd0;    
            o_b     <= 16'd0;    
        end else if (i_valid) begin 
            // 16-bit * 16-bit multiplication result (32 bits) is added to 48-bit accumulator
            // Result 32bit = [8bit,8bit,8bit,8bit]
            o_acc   <= o_acc + (i_a * i_b);
            o_valid <= 1'b1;
            o_a     <= i_a;
            o_b     <= i_b;
        end else begin  
            o_valid <= 1'b0;
        end
    end
endmodule