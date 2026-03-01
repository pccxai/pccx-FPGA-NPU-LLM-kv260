`timescale 1ns / 1ps

module gelu_lut (
    input  logic               clk,
    input  logic               valid_in,
    input  logic signed [15:0] data_in,   // Systolic Array에서 뿜어져 나온 16비트 값
    
    output logic               valid_out,
    output logic signed [7:0]  data_out   // 다음 파이프라인(레이어)을 위해 8비트로 재양자화!
);

    // 65536개의 8비트 데이터를 담는 하드웨어 ROM (64KB 용량)
    // Vivado야, 이거 무조건 BRAM으로 맵핑해라! (명시적 지시어)
    (* rom_style = "block" *) logic signed [7:0] rom_table [0:65535];

    initial begin
        // 파이썬으로 미리 계산해둔 GeLU 헥사(Hex) 코드를 부팅 시점에 촥! 구워넣기
        $readmemh("gelu_table.mem", rom_table);
    end

    // 1클럭 Latency 파이프라인
    always_ff @(posedge clk) begin
        // 부호 있는 16비트 값을 0~65535 인덱스로 캐스팅해서 읽어옴
        // (Verilog는 내부적으로 unsigned처럼 주소를 찾아감)
        data_out  <= rom_table[data_in];
        valid_out <= valid_in;
    end

endmodule