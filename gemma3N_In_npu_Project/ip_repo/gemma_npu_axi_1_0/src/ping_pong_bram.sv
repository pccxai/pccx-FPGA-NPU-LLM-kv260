`timescale 1ns / 1ps

module ping_pong_bram #(
    parameter DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 9     
)(
    input  logic       clk,
    input  logic       rst_n,
    input  logic       switch_buffer,
    
    // AXI DMA Interface (Write Only)
    input  logic                  dma_we,
    input  logic [ADDR_WIDTH-1:0] dma_addr,
    input  logic [DATA_WIDTH-1:0] dma_write_data,
    
    // NPU interface (two-lane highway: read A and B simultaneously! )
    input  logic [ADDR_WIDTH-1:0] npu_addr_a,
    input  logic [ADDR_WIDTH-1:0] npu_addr_b,
    output logic [DATA_WIDTH-1:0] npu_read_data_a,
    output logic [DATA_WIDTH-1:0] npu_read_data_b
);

    logic                  we_0_a, we_0_b, we_1_a, we_1_b;
    logic [ADDR_WIDTH-1:0] addr_0_a, addr_0_b, addr_1_a, addr_1_b;
    logic [DATA_WIDTH-1:0] rdata_0_a, rdata_0_b, rdata_1_a, rdata_1_b;

    // --------------------------------------------------------
    // [MUX Logic] Switch paths according to the pointer (switch_buffer)
    // --------------------------------------------------------
    always_comb begin
        if (switch_buffer == 1'b0) begin
            // [State 0] DMA -> Write BRAM_0 / NPU <- Read BRAM_1
            we_0_a = dma_we;     addr_0_a = dma_addr;   // BRAM 0 Port A is occupied by DMA
            we_0_b = 1'b0;       addr_0_b = '0;         // BRAM 0 Port B is resting
            
            we_1_a = 1'b0;       addr_1_a = npu_addr_a; // BRAM 1 Port A reads NPU A
            we_1_b = 1'b0;       addr_1_b = npu_addr_b; // BRAM 1 Port B reads NPU B
            
            npu_read_data_a = rdata_1_a;
            npu_read_data_b = rdata_1_b;
        end else begin
            // [State 1] DMA -> Write BRAM_1 / NPU <- Read BRAM_0
            we_1_a = dma_we;     addr_1_a = dma_addr;   // BRAM 1 Port A is occupied by DMA
            we_1_b = 1'b0;       addr_1_b = '0;         // BRAM 1 Port B is resting
            
            we_0_a = 1'b0;       addr_0_a = npu_addr_a; // BRAM 0 Port A reads NPU A
            we_0_b = 1'b0;       addr_0_b = npu_addr_b; // BRAM 0 Port B reads NPU B
            
            npu_read_data_a = rdata_0_a;
            npu_read_data_b = rdata_0_b;
        end
    end

    // --------------------------------------------------------
    // [BRAM Instantiation] Two Dual-Port BRAMs, boom.
    // --------------------------------------------------------
    simple_bram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) bram_0 (
        .clk(clk),
        .we_a(we_0_a), .addr_a(addr_0_a), .data_in_a(dma_write_data), .data_out_a(rdata_0_a),
        .we_b(we_0_b), .addr_b(addr_0_b), .data_in_b('0),             .data_out_b(rdata_0_b)
    );

    simple_bram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) bram_1 (
        .clk(clk),
        .we_a(we_1_a), .addr_a(addr_1_a), .data_in_a(dma_write_data), .data_out_a(rdata_1_a),
        .we_b(we_1_b), .addr_b(addr_1_b), .data_in_b('0),             .data_out_b(rdata_1_b)
    );

endmodule