`timescale 1ns / 1ps

`include "stlc_Array.svh"
/*
 * Module: stlc_bf16_fixed_pipeline
 * Description: 
 *   Pipelined architecture for BF16 to Fixed-point conversion.
 *   1. Receives 32-word tiles from AXI-DMA.
 *   2. Finds the maximum exponent (emax) within the tile in real-time.
 *   3. Uses a 3-stage pipelined barrel shifter to align all values to emax.
 *   4. Implements Dual-Bank (Ping-Pong) buffering to hide latency.
*/

module stlc_bf16_fixed_pipeline #(
    parameter TILE_SIZE = 32
)(
    input  logic clk,
    input  logic rst_n,
    
    // AXI-Stream Slave Interface (Input from DMA)
    input  logic [15:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    
    // AXI-Stream Master Interface (Output to BRAM/NPU)
    output logic [26:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready
);

    // --- Internal Signals ---
    logic [15:0] buffer_0 [0:TILE_SIZE-1];
    logic [15:0] buffer_1 [0:TILE_SIZE-1];
    logic [7:0]  emax_0, emax_1;
    
    logic        wr_bank;   // 0: Writing to buffer_0, 1: Writing to buffer_1
    logic [4:0]  wr_ptr;
    logic        rd_bank;   // Reading bank (swapped with wr_bank)
    logic [4:0]  rd_ptr;
    
    logic        tile_full;     // Current bank is ready for shifting
    logic        busy_shifting; // Shifter is pulling data from buffer

    // Flow control: Ready when not waiting for shifter to clear a bank
    assign s_axis_tready = !tile_full;
    // --- 1. Write Logic: AXI-DMA to Buffer + E-max Tracking ---
    logic clear_emax_0, clear_emax_1; // Signals from Read Logic to clear emax

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr    <= 0;
            wr_bank   <= 0;
            emax_0    <= 0;
            emax_1    <= 0;
            tile_full <= 0;
        end else begin
            // Handle emax clearing from Read Logic
            if (clear_emax_0) emax_0 <= 0;
            if (clear_emax_1) emax_1 <= 0;

            if (s_axis_tvalid && s_axis_tready) begin
                // Write to selected bank and update local emax
                if (wr_bank == 0) begin
                    buffer_0[wr_ptr] <= s_axis_tdata;
                    if (s_axis_tdata[14:7] > emax_0) emax_0 <= s_axis_tdata[14:7];
                end else begin
                    buffer_1[wr_ptr] <= s_axis_tdata;
                    if (s_axis_tdata[14:7] > emax_1) emax_1 <= s_axis_tdata[14:7];
                end
                
                // Bank boundary check
                if (wr_ptr == TILE_SIZE - 1) begin
                    wr_ptr    <= 0;
                    wr_bank   <= ~wr_bank;
                    tile_full <= 1; // Bank ready for processing
                end else begin
                    wr_ptr    <= wr_ptr + 1;
                end
            end else if (busy_shifting && rd_ptr == TILE_SIZE - 1) begin
                // Reset tile_full when the current read cycle is completing
                tile_full <= 0; 
            end
        end
    end


    // --- 2. Read Control: Buffer to Shifter Stage ---
    logic [15:0] shifter_in;
    logic [7:0]  shifter_emax;
    logic        shifter_in_valid;
    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rd_ptr           <= 0;
            rd_bank          <= 0;
            busy_shifting    <= 0;
            shifter_in_valid <= 0;
            clear_emax_0     <= 0;
            clear_emax_1     <= 0;
        end else begin
            // Default to not clearing
            clear_emax_0 <= 0;
            clear_emax_1 <= 0;

            if (tile_full && !busy_shifting) begin
                // Start shifting process from the completed bank
                busy_shifting <= 1;
                rd_ptr        <= 0;
                rd_bank       <= ~wr_bank; 
            end else if (busy_shifting) begin
                shifter_in_valid <= 1;
                // Fetch data and current bank's emax
                if (rd_bank == 0) begin
                    shifter_in   <= buffer_0[rd_ptr];
                    shifter_emax <= emax_0;
                end else begin
                    shifter_in   <= buffer_1[rd_ptr];
                    shifter_emax <= emax_1;
                end

                // Reset ptr after reading the last element
                if (rd_ptr == TILE_SIZE - 1) begin
                    rd_ptr        <= 0;
                    busy_shifting <= 0;
                    // Trigger clear signals for the Write Logic to handle
                    if (rd_bank == 0) clear_emax_0 <= 1;
                    else              clear_emax_1 <= 1;
                end else begin
                    rd_ptr        <= rd_ptr + 1;
                end
            end else begin
                shifter_in_valid <= 0;
            end
        end
    end
    
    // --- 3. Pipelined Barrel Shifter (3-Stage Logic) ---
    
    // Stage 1: Mantissa Extraction & Exponent Difference
    logic [7:0]  s1_delta_e;
    logic [26:0] s1_base_vec;
    logic        s1_valid;
    
    always_ff @(posedge clk) begin
        if (!rst_n) s1_valid <= 0;
        else begin
            s1_valid    <= shifter_in_valid;
            s1_delta_e  <= shifter_emax - shifter_in[14:7];
            // Handle hidden bit (BF16: 1.mantissa or 0.mantissa)
            s1_base_vec <= (shifter_in[14:7] == 0) ? 
                           {8'h0, shifter_in[6:0], 12'b0} : 
                           {8'h1, shifter_in[6:0], 12'b0};
        end
    end

    // Stage 2: Coarse Shift (Shift by multiples of 4: 0, 4, 8, ..., 24)
    logic [26:0] s2_mid_vec;
    logic [1:0]  s2_fine_shift;
    logic        s2_valid;
    always_ff @(posedge clk) begin
        if (!rst_n) s2_valid <= 0;
        else begin
            s2_valid      <= s1_valid;
            s2_fine_shift <= s1_delta_e[1:0]; // Pass lower bits to fine shift
            if (s1_delta_e >= 27) 
                s2_mid_vec <= 27'd0; // Underflow case
            else                  
                s2_mid_vec <= s1_base_vec >> (s1_delta_e[4:2] << 2);
        end
    end
    
    // Stage 3: Fine Shift (Shift by 0, 1, 2, 3) & Final Output Mapping
    always_ff @(posedge clk) begin
        if (!rst_n) m_axis_tvalid <= 0;
        else begin
            m_axis_tvalid <= s2_valid;
            m_axis_tdata  <= s2_mid_vec >> s2_fine_shift;
        end
    end
endmodule
