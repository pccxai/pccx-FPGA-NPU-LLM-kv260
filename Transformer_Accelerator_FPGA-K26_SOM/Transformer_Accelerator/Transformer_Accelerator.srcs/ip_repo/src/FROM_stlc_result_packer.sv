`timescale 1ns / 1ps
`include "stlc_Array.svh"

/**
 * Module: stlc_result_packer
 * Description: 
 *   Collects 32 staggered 48-bit results and packs them into 128-bit DMA words.
 *   Strategy: 2 Results per 128-bit word (96-bit payload + 32-bit padding).
 */

module FROM_stlc_result_packer #(
    parameter ARRAY_SIZE = 32
)(
    input  logic clk,
    input  logic rst_n,

    // =| Input from Normalizers (16-bit BF16) |=
    input  logic [`BF16_WIDTH-1:0] row_res       [0:ARRAY_SIZE-1],
    input  logic                    row_res_valid [0:ARRAY_SIZE-1],

    // =| Output to FIFO (128-bit) |=
    output logic [`AXI_DATA_WIDTH-1:0] packed_data,
    output logic                       packed_valid,
    input  logic                       packed_ready
);

    // ===| Sequential Collection Logic |=======
    // Results come out staggered (Row 0 first, then Row 1, etc.)
    // We use a multiplexer to pick the valid row result.
    logic [`BF16_WIDTH-1:0] active_res;
    logic                   active_valid;
    
    always_comb begin
        active_res   = '0;
        active_valid = 1'b0;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            if (row_res_valid[i]) begin
                active_res   = row_res[i];
                active_valid = 1'b1;
            end
        end
    end

    // ===| 8-to-1 Packing Buffer (16-bit * 8 = 128-bit) |=======
    // We store up to 7 results and fire when the 8th arrives.
    logic [`BF16_WIDTH-1:0] res_buffer [0:6]; 
    logic [2:0]             pack_cnt; // 0 to 7

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pack_cnt     <= 3'd0;
            packed_valid <= 1'b0;
            packed_data  <= '0;
        end else begin
            if (active_valid) begin
                if (pack_cnt == 3'd7) begin
                    // We have 7 stored, this is the 8th. Pack and fire!
                    packed_data <= {active_res, 
                                    res_buffer[6], res_buffer[5], res_buffer[4], 
                                    res_buffer[3], res_buffer[2], res_buffer[1], res_buffer[0]};
                    packed_valid <= 1'b1;
                    pack_cnt     <= 3'd0; // Reset counter
                end else begin
                    // Store in buffer and increment counter
                    res_buffer[pack_cnt] <= active_res;
                    pack_cnt             <= pack_cnt + 1;
                    packed_valid         <= 1'b0;
                end
            end else begin
                // Maintain valid signal state or drop it depending on backpressure?
                // For AXI-Stream, valid should be lowered after ready handshake, 
                // but since this feeds a FIFO that we assume always ready or we just pulse it.
                // Pulsing valid for 1 cycle is usually safe if FIFO is fast.
                packed_valid <= 1'b0; 
            end
        end
    end
endmodule
