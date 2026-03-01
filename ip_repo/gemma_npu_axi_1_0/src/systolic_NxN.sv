`timescale 1ns / 1ps

module systolic_NxN #(
    parameter ARRAY_SIZE = 32 // 32x32 아키텍처!
)(
    input  logic clk,
    input  logic rst_n,
    input  logic i_clear, // 🔥 외부(Top FSM)에서 받는 글로벌 클리어 신호

    // npu_core_top에서 '동시'에 쏟아지는 64개 데이터
    input  logic [7:0] in_a [0:ARRAY_SIZE-1], 
    input  logic [7:0] in_b [0:ARRAY_SIZE-1],
    input  logic       in_valid,

    output logic [31:0] out_acc [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    // 각 PE들 사이를 연결할 내부 전선
    logic [7:0] wire_a [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    logic [7:0] wire_b [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    logic       wire_v [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]; 

    // -----------------------------------------------------------------
    // Delay Line을 이용한 입력 데이터 계단식 지연 (Wavefront Skewing)
    // -----------------------------------------------------------------
    genvar i;
    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : delay_skewing
            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_a (
                .clk(clk), .rst_n(rst_n),
                .in_data(in_a[i]), .out_data(wire_a[i][0])
            );

            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_b (
                .clk(clk), .rst_n(rst_n),
                .in_data(in_b[i]), .out_data(wire_b[0][i])
            );
        end
    endgenerate

    // -----------------------------------------------------------------
    // 2D PE Array 생성 (mac_unit 4,096개 자동 복사)
    // -----------------------------------------------------------------
    genvar row, col;
    generate
        for (row = 0; row < ARRAY_SIZE; row++) begin : row_loop
            for (col = 0; col < ARRAY_SIZE; col++) begin : col_loop
                
                // 🚨 클로드 지적 반영: 변수 선언을 if 밖으로 빼서 합성 호환성 100% 확보!
                logic current_i_valid;
                
                if (row == 0 && col == 0) begin
                    assign current_i_valid = in_valid;
                end else if (col > 0) begin
                    assign current_i_valid = wire_v[row][col-1]; 
                end else begin
                    assign current_i_valid = wire_v[row-1][col]; 
                end

                pe_unit u_pe (
                    .clk(clk), .rst_n(rst_n),
                    .i_clear(i_clear),                // 🔥 4096개 PE에 동시 클리어 배선!
                    .i_valid(current_i_valid),
                    .i_a(wire_a[row][col]),
                    .i_b(wire_b[row][col]),
                    .o_a(wire_a[row][col+1]),
                    .o_b(wire_b[row+1][col]),
                    .o_valid(wire_v[row][col]),
                    .o_acc(out_acc[row][col])
                );
            end
        end
    endgenerate

endmodule