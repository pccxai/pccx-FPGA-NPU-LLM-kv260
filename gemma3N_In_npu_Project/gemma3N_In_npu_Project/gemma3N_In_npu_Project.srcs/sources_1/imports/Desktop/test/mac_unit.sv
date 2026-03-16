`timescale 1ns / 1ps

(* use_dsp = "yes" *)
module pe_unit (
    input  logic        clk,     
    input  logic        rst_n,   
    input  logic        i_clear, 
    input  logic        i_valid, 
    input  logic signed [7:0]  i_a,     
    input  logic signed [7:0]  i_b, 
    output logic [7:0]  o_a,     
    output logic [7:0]  o_b,     
    output logic        o_valid, 
    output logic [31:0] o_acc  
);
    // We drive the DSP48E2 internal fusion by writing the multiplication and accumulation in one line.
    always_ff @(posedge clk) begin
        if (!rst_n || i_clear) begin
            o_acc   <= 32'd0;
            o_valid <= 1'b0;
            o_a     <= 8'd0;
            o_b     <= 8'd0;
        end else if (i_valid) begin 
            o_acc   <= o_acc + ($signed(i_a) * $signed(i_b));
            o_valid <= 1'b1;
            o_a     <= i_a; 
            o_b     <= i_b;
        end else begin  
            o_valid <= 1'b0;
        end
    end
endmodule