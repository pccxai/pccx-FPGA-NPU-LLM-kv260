`timescale 1ns / 1ps

module simple_bram #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 8 // 256 depth
)(
    input  logic                  clk,
    input  logic                  we,      // Write Enable
    input  logic [ADDR_WIDTH-1:0] addr,    // Address
    input  logic [DATA_WIDTH-1:0] din,     // Write Data
    output logic [DATA_WIDTH-1:0] dout     // Read Data
);

    // 실제 메모리 배열 선언 (C++의 배열이랑 똑같아!)
    logic [DATA_WIDTH-1:0] ram [0:(1<<ADDR_WIDTH)-1];

    always_ff @(posedge clk) begin
        if (we) begin
            ram[addr] <= din; // 쓰기
        end
        dout <= ram[addr];    // 읽기 (동기식)
    end 

endmodule