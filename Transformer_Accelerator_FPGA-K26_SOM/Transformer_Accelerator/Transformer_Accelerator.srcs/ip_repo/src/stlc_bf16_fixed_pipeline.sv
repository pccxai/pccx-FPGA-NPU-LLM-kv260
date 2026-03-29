`timescale 1ns / 1ps
`include "stlc_Array.svh"

/**
 * Module: stlc_bf16_fixed_pipeline
 * Description: 
 *   High-Throughput 16-Lane Pipelined BF16 to Fixed-point Converter.
 *   - Input: 256-bit (16 x BF16 elements) per clock.
 *   - Block Size: 32 elements (Takes 2 clocks to receive one block).
 *   - Operation:
 *       1. Finds the Global e_max among the 32 elements.
 *       2. Shifts the Mantissas (27-bit) to align with Global e_max.
 *   - Output: 432-bit (16 x 27-bit Mantissas) per clock.
 */
module stlc_bf16_fixed_pipeline (
    input  logic clk,
    input  logic rst_n,
    
    // AXI-Stream Slave (Input from 256-bit FIFO)
    input  logic [255:0] s_axis_tdata,
    input  logic         s_axis_tvalid,
    output logic         s_axis_tready,
    
    // AXI-Stream Master (Output to SRAM Cache - 16 x 27-bit = 432-bit)
    output logic [431:0] m_axis_tdata,
    output logic         m_axis_tvalid,
    input  logic         m_axis_tready
);

    // ===| Stage 1: Input Buffering & Local Max Exponent |===
    // We need to buffer the first 16 words while waiting for the next 16.
    logic [255:0] buffer_low;
    logic [7:0]   local_max_low;
    logic         first_half_valid;
    
    logic phase; // 0: Expecting Low 16, 1: Expecting High 16
    
    // Combinational Logic to find the maximum exponent among 16 BF16 elements
    function automatic logic [7:0] find_max_e_16(input logic [255:0] data);
        logic [7:0] max_val = 8'd0;
        for (int i = 0; i < 16; i++) begin
            if (data[(i*16)+7 +: 8] > max_val) begin
                max_val = data[(i*16)+7 +: 8];
            end
        end
        return max_val;
    endfunction

    assign s_axis_tready = 1'b1; // Always ready to sink data in this pipeline design

    // Buffer registers for 32 elements
    logic [255:0] block_data_low;
    logic [255:0] block_data_high;
    logic [7:0]   global_emax;
    logic         block_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            phase <= 1'b0;
            first_half_valid <= 1'b0;
            block_valid <= 1'b0;
        end else if (s_axis_tvalid) begin
            if (phase == 1'b0) begin
                // Store first 16 words and their max exponent
                buffer_low    <= s_axis_tdata;
                local_max_low <= find_max_e_16(s_axis_tdata);
                first_half_valid <= 1'b1;
                block_valid   <= 1'b0;
                phase         <= 1'b1;
            end else begin
                // Second 16 words arrived! Combine to form 32-word block.
                block_data_low  <= buffer_low;
                block_data_high <= s_axis_tdata;
                
                // Compare max of low 16 and high 16 to get GLOBAL e_max
                automatic logic [7:0] local_max_high = find_max_e_16(s_axis_tdata);
                global_emax <= (local_max_low > local_max_high) ? local_max_low : local_max_high;
                
                block_valid <= 1'b1;
                phase       <= 1'b0; // Reset for next block
            end
        end else begin
            block_valid <= 1'b0;
        end
    end

    // ===| Stage 2: Parallel Shifting (16 Lanes at a time) |===
    // To save resources, we will shift the 32 elements over 2 clock cycles.
    // Cycle 1: Shift block_data_low
    // Cycle 2: Shift block_data_high
    
    logic shift_phase; // 0: shifting low, 1: shifting high
    logic [255:0] shift_target_data;
    logic [7:0]   shift_target_emax;
    logic         shift_trigger;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            shift_phase <= 1'b0;
            shift_trigger <= 1'b0;
        end else begin
            if (block_valid) begin
                // Start shifting process
                shift_phase <= 1'b0;
                shift_target_data <= block_data_low;
                shift_target_emax <= global_emax;
                shift_trigger <= 1'b1;
            end else if (shift_trigger && shift_phase == 1'b0) begin
                // Next cycle, shift the high part
                shift_phase <= 1'b1;
                shift_target_data <= block_data_high;
                // keep shift_target_emax same
                shift_trigger <= 1'b1;
            end else begin
                shift_trigger <= 1'b0;
            end
        end
    end

    // The 16 Parallel Shifters
    logic [431:0] shifted_mantissas; // 16 * 27-bit

    genvar i;
    generate
        for (i = 0; i < 16; i++) begin : gen_shifters
            logic [15:0] word;
            logic [7:0]  e_val;
            logic [6:0]  m_val;
            logic [26:0] base_vec;
            logic [7:0]  delta_e;
            
            assign word = shift_target_data[(i*16) +: 16];
            assign e_val = word[14:7];
            assign m_val = word[6:0];
            
            // Add hidden bit: if exponent is 0, hidden bit is 0 (denormal), else 1
            assign base_vec = (e_val == 0) ? {8'h0, m_val, 12'b0} : {8'h1, m_val, 12'b0};
            assign delta_e  = shift_target_emax - e_val;
            
            // The actual shift
            assign shifted_mantissas[(i*27) +: 27] = (delta_e >= 27) ? 27'd0 : (base_vec >> delta_e);
        end
    endgenerate

    // ===| Stage 3: Output Register |===
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tdata  <= 0;
        end else begin
            m_axis_tvalid <= shift_trigger;
            if (shift_trigger) begin
                m_axis_tdata <= shifted_mantissas;
            end
        end
    end

endmodule
