`timescale 1ns / 1ps

module result_ping_pong_bram #(
    parameter DATA_WIDTH = 48,  
    parameter ARRAY_SIZE = 32,
    parameter ADDR_WIDTH = 5
)(
    input  logic clk,
    input  logic rst_n,
    input  logic switch_buffer, // 0: NPU->B0, DMA->B1 | 1: NPU->B1, DMA->B0

    // NPU Interface (32 channel write simultaneously)
    input  logic                   npu_we,
    input  logic [ADDR_WIDTH-1:0]  npu_addr,
    input  logic [DATA_WIDTH-1:0]  npu_data_in [0:ARRAY_SIZE-1],

    // DMA Interface (Read)
    input  logic [ADDR_WIDTH-1:0]  dma_addr,
    output logic [(ARRAY_SIZE * DATA_WIDTH) - 1 : 0] dma_data_out
);

    // Flatten DATA 
    logic [(ARRAY_SIZE * DATA_WIDTH) - 1 : 0] npu_flat_data;
    always_comb begin
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            npu_flat_data[i * DATA_WIDTH +: DATA_WIDTH] = npu_data_in[i];
        end
    end

    // BRAM output data
    logic [(ARRAY_SIZE * DATA_WIDTH) - 1 : 0] rdata_0, rdata_1;
    logic          we_0, we_1;

    // Ping-Pong MUX
    assign we_0 = (switch_buffer == 1'b0) ? npu_we : 1'b0;
    assign we_1 = (switch_buffer == 1'b1) ? npu_we : 1'b0;

    assign dma_data_out = (switch_buffer == 1'b0) ? rdata_1 : rdata_0;

    // BRAM 0 (1536-bit Wide)
    (* ram_style = "block" *) logic [(ARRAY_SIZE * DATA_WIDTH)-1:0] mem_0 [0:(1<<ADDR_WIDTH)-1];
    always_ff @(posedge clk) begin
        if (we_0) mem_0[npu_addr] <= npu_flat_data;
        rdata_0 <= mem_0[dma_addr];
    end

    // BRAM 1 (1536-bit Wide)
    (* ram_style = "block" *) logic [(ARRAY_SIZE * DATA_WIDTH)-1:0] mem_1 [0:(1<<ADDR_WIDTH)-1];
    always_ff @(posedge clk) begin
        if (we_1) mem_1[npu_addr] <= npu_flat_data;
        rdata_1 <= mem_1[dma_addr];
    end

endmodule
