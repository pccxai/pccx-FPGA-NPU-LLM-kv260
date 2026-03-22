`timescale 1ns / 1ps
`include "stlc_Array.svh"

/**
 * Module: stlc_result_normalizer
 * Description: 
 *   Converts 48-bit 2's complement to Normalized Format (BF16-like).
 *   Pipeline: [1] Sign-Mag -> [2] LOD -> [3] Barrel Shift -> [4] Exp Adj
 */
module stlc_result_normalizer (
    input  logic clk,
    input  logic rst_n,
    
    input  logic [47:0] data_in,     // 48-bit Accumulator Result
    input  logic [7:0]  e_max,       // Original delayed exponent for this column
    input  logic        valid_in,

    output logic [15:0] data_out,    // 1:Sign, 8:Exp, 7:Mantissa (BF16 format)
    output logic        valid_out
);

    // ===| Stage 1: Sign-Magnitude Conversion |=======
    // Converting from 2's complement to absolute value (Sign + Magnitude)
    logic [47:0] s1_abs_data;
    logic        s1_sign;
    logic [7:0]  s1_emax;
    logic        s1_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_sign  <= 1'b0;
            s1_emax  <= 8'd0;
            s1_abs_data <= 48'd0;
        end else begin
            s1_valid <= valid_in;
            s1_sign  <= data_in[47];
            s1_emax  <= e_max;
            
            // If negative, invert and add 1 (2's complement to absolute)
            s1_abs_data <= (data_in[47]) ? (~data_in + 1'b1) : data_in;
        end
    end

    // ===| Stage 2: Leading One Detection (LOD) |=======
    // Finding the position of the most significant '1' bit
    logic [5:0]  s2_first_one_pos;
    logic [47:0] s2_abs_data;
    logic        s2_sign;
    logic [7:0]  s2_emax;
    logic        s2_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            s2_sign  <= 1'b0;
            s2_abs_data <= 48'd0;
            s2_emax  <= 8'd0;
            s2_first_one_pos <= 6'd0;
        end else begin
            s2_valid <= s1_valid;
            s2_sign  <= s1_sign;
            s2_abs_data <= s1_abs_data;
            s2_emax  <= s1_emax;

            // Simple Priority Encoder for LOD
            // In 400MHz, this might need further pipelining if timing fails,
            // but starting with a basic loop since Vivado is good at tree extraction.
            s2_first_one_pos <= 6'd0; // Default to 0
            for (int i = 46; i >= 0; i--) begin
                if (s1_abs_data[i]) begin
                    s2_first_one_pos <= i[5:0];
                    break;
                end
            end
        end
    end

    // ===| Stage 3: Normalization Barrel Shift & Exponent Update |=======
    // Shifting the mantissa so that the leading '1' sits right before the 7-bit fractional part.
    logic [6:0]  s3_mantissa;
    logic [7:0]  s3_new_exp;
    logic        s3_sign;
    logic        s3_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s3_valid <= 1'b0;
            s3_sign  <= 1'b0;
            s3_new_exp <= 8'd0;
            s3_mantissa <= 7'd0;
        end else begin
            s3_valid <= s2_valid;
            s3_sign  <= s2_sign;
            
            if (s2_abs_data == 0) begin
                s3_new_exp  <= 8'd0;
                s3_mantissa <= 7'd0;
            end else begin
                // Update exponent: original e_max + current bit position offset
                // Example bias: Assume our fixed-point format implies the 1.0 bit is at position 26.
                // Depending on your actual Shifter logic, this offset (26) should be matched.
                s3_new_exp <= s2_emax + s2_first_one_pos - 8'd26; 

                // Align mantissa to BF16 (7 bits of fraction)
                if (s2_first_one_pos >= 7)
                    // Take the 7 bits immediately below the first '1'
                    s3_mantissa <= s2_abs_data[s2_first_one_pos - 1 -: 7];
                else
                    // Shift left to pad with zeros
                    s3_mantissa <= s2_abs_data[6:0] << (7 - s2_first_one_pos);
            end
        end
    end

    // ===| Stage 4: Final Packing |=======
    // Constructing the final 16-bit word
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            data_out  <= 16'd0;
        end else begin
            valid_out <= s3_valid;
            data_out  <= {s3_sign, s3_new_exp, s3_mantissa};
        end
    end

endmodule

/*

`timescale 1ns / 1ps

`include "stlc_Array.svh"


module stlc_result_normalizer (
    input  logic clk,
    input  logic rst_n,
    input  logic [47:0] data_in,
    input  logic [7:0]  e_max,
    input  logic        valid_in,
    output logic [15:0] data_out, // 1:Sign, 7:Exp, 8:Mantissa (BF16-like)
    output logic        valid_out
);
    // ===| Stage 1: Sign-Magnitude Conversion |=======
    logic [47:0] s1_abs_data;
    logic        s1_sign;
    logic [7:0]  s1_emax;
    logic        s1_valid;
    always_ff @(posedge clk) begin
        s1_valid <= valid_in;
        s1_sign  <= data_in[47];
        s1_emax  <= e_max;
        // If negative, invert and add 1 (2's complement to absolute)
        s1_abs_data <= (data_in[47]) ? (~data_in + 1'b1) : data_in;
    end
    // ===| Stage 2: Leading One Detection (LOD) |=======
    // Finding the position of the first '1' from MSB
    logic [5:0]  s2_first_one_pos;
    logic [47:0] s2_abs_data;
    logic        s2_sign;
    logic [7:0]  s2_emax;
    logic        s2_valid;
    always_ff @(posedge clk) begin
        s2_valid <= s1_valid;
        s2_sign  <= s1_sign;
        s2_abs_data <= s1_abs_data;
        s2_emax  <= s1_emax;
        // Simple priority encoder for LOD (300-400MHz requires optimized tree)
        s2_first_one_pos <= 0;
        for (int i = 46; i >= 0; i--) begin
            if (s1_abs_data[i]) begin
                s2_first_one_pos <= i;
                break;
            end
        end
    end
    
    // ===| Stage 3: Normalization Barrel Shift |=======
    // Shift the '1' to the 7th bit position (BF16 style)
    logic [7:0]  s3_mantissa;
    logic [7:0]  s3_new_exp;
    logic        s3_sign;
    logic        s3_valid;
    always_ff @(posedge clk) begin
        s3_valid <= s2_valid;
        s3_sign  <= s2_sign;
        
        // Exponent Update Logic: emax + (pos - original_bias)
        // Adjusting based on your formula: emax + shift_count - original_pos
        s3_new_exp <= s2_emax + (s2_first_one_pos - 6'd26); // Example bias 26
        
        // Aligning mantissa to 8-bit output
        if (s2_first_one_pos >= 7)
            s3_mantissa <= s2_abs_data[s2_first_one_pos -: 8];
        else
            s3_mantissa <= s2_abs_data << (7 - s2_first_one_pos);
    end
    // ===| Stage 4: Final Packing |=======
    always_ff @(posedge clk) begin
        valid_out <= s3_valid;
        data_out  <= {s3_sign, s3_new_exp, s3_mantissa[6:0]}; // Packed BF16
    end
endmodule


*/