`timescale 1ns / 1ps

module simple_bram #(
    parameter DATA_WIDTH = 512, // 64 Bytes (NPU가 한 번에 먹는 데이터)
    parameter ADDR_WIDTH = 9 // 512 depth
)(
    input  logic clk,
    input  logic we,      // Write Enable

    // 🚪 Port A (DMA가 데이터를 쓰거나, NPU가 A행렬을 읽을 때 사용)
    input  logic                  we_a,      
    input  logic [ADDR_WIDTH-1:0] addr_a,    
    input  logic [DATA_WIDTH-1:0] data_in_a,     
    output logic [DATA_WIDTH-1:0] data_out_a,    

    // 🚪 Port B (NPU가 B행렬을 읽을 때 전용으로 사용)
    input  logic                  we_b,      
    input  logic [ADDR_WIDTH-1:0] addr_b,    
    input  logic [DATA_WIDTH-1:0] data_in_b,     
    output logic [DATA_WIDTH-1:0] data_out_b
);
    // 실제 메모리 배열 선언 (C++의 배열이랑 똑같아!)
    // Xilinx Vivado에게 "BRAM으로 합성해라"라고 강제하는 속성
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram [0:(1<<ADDR_WIDTH)-1];

    // Port A 제어 (Write & Read)
    always_ff @(posedge clk) begin
        if (we_a) ram[addr_a] <= data_in_a; 
        data_out_a <= ram[addr_a];    
    end

    // Port B 제어 (Write & Read)
    always_ff @(posedge clk) begin
        if (we_b) ram[addr_b] <= data_in_b; 
        data_out_b <= ram[addr_b];    
    end

endmodule