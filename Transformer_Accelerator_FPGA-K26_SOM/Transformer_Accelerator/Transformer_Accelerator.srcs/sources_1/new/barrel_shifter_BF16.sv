`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/20/2026 05:19:31 PM
// Design Name: 
// Module Name: barrel_shifter_BF16
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


module barrel_shifter_BF16 #()( 
    input  logic [15:0] bf16_act, 
    input  logic [7:0]  e_max,    
    
    output logic [26:0] delayLine_in, 
    output logic [7:0]  exp_out,
    output logic        sign_out  
    );

    logic       sign;
    logic [7:0] exp;
    logic [7:0] mantissa; 

    assign sign = bf16_act[15];
    assign exp  = bf16_act[14:7];
    
    assign mantissa = (exp == 0) ? {1'b0, bf16_act[6:0]} : {1'b1, bf16_act[6:0]};
            
    // [26:19] mantissa / [18:0] is filled with 0
    logic [26:0] base_vec;
    assign base_vec = {mantissa, 19'b0};
    
    logic [7:0] delta_e;
    assign delta_e = e_max - exp;


    logic [7:0] delta_e_reg; 
    logic [26:0] base_vec_reg; 
    logic sign_reg; 
    logic [7:0] e_max_reg;

    // 4. Barrel shifter (MUX tree Auto synthes) & clamping
    always_comb begin
        if (delta_e >= 8'd27) begin
            // Shifting basevec leads to data loss via truncation.
            // Underflow occurs as bfloat cannot resolve such small magnitudes (rounding will occur).
            delayLine_in = 27'd0;
        end else begin
            // Only the lower 5 bits ([4:0]) 
            // are used to shift up to 26 positions. 
            // (Vivado synthesizes this as a MUX tree.)
            delayLine_in = base_vec >> delta_e[4:0];
        end
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            delayLine_in = 27'd0;     
        end else begin
            delta_e_reg <= e_max - exp;
            assign mantissa = (exp == 0) ? {1'b0, bf16_act[6:0]} : {1'b1, bf16_act[6:0]};
            base_vec_reg<= {mantissa, 19'b0};
            sign_reg    <= bf16_act[15];
            e_max_reg   <= bf16_act[14:7];
        end 
    end
    assign sign_out = sign;
    assign exp_out  = e_max;

endmodule