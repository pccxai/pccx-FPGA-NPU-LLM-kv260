`timescale 1ns / 1ps

module simple_bram #(
    parameter DATA_WIDTH = 512, // 64 Bytes (data consumed by NPU at once)
    parameter ADDR_WIDTH = 9 // 512 depth
)(
    input  logic clk,

    // Port A (DMA writes data, NPU reads data in A)
    input  logic                  we_a,
    input  logic [ADDR_WIDTH-1:0] addr_a,
    input  logic [DATA_WIDTH-1:0] data_in_a,
    output logic [DATA_WIDTH-1:0] data_out_a,

    // Port B (DMA writes data, NPU reads data in B)
    input  logic                  we_b,
    input  logic [ADDR_WIDTH-1:0] addr_b,
    input  logic [DATA_WIDTH-1:0] data_in_b,
    output logic [DATA_WIDTH-1:0] data_out_b
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram [0:(1<<ADDR_WIDTH)-1];

    // Port A control (Write & Read)
    always_ff @(posedge clk) begin
        if (we_a) ram[addr_a] <= data_in_a;
        data_out_a <= ram[addr_a];
    end

    // Port B control (Write & Read)
    always_ff @(posedge clk) begin
        if (we_b) ram[addr_b] <= data_in_b;
        data_out_b <= ram[addr_b];
    end

endmodule