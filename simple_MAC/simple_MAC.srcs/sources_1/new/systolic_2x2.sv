`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/02/23 18:53:40
// Design Name: 
// Module Name: systolic_2x2
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module systolic_2x2 (
    input  logic clk,
    input  logic rst_n,
    
    // 외부에서 어레이로 들어오는 입력 (위쪽 2개, 왼쪽 2개)
    input  logic [7:0] in_a_0, in_a_1, // 왼쪽에서 들어오는 Feature Map
    input  logic [7:0] in_b_0, in_b_1, // 위에서 내려오는 Weight
    input  logic       in_valid,
    
    // 최종 누적 결과값 4개
    output logic [15:0] out_acc_00, out_acc_01,
    output logic [15:0] out_acc_10, out_acc_11
);

    // 내부 모듈끼리 연결할 '전선(Wire)' 역할의 변수들 선언
    logic [7:0] wire_a_00_to_01, wire_a_10_to_11;
    logic [7:0] wire_b_00_to_10, wire_b_01_to_11;
    logic valid_00, valid_01, valid_10;

    // PE (0,0) - 좌측 상단
    pe_unit pe_00 (
        .clk(clk), .rst_n(rst_n), .i_valid(in_valid),
        .i_a(in_a_0),          .i_b(in_b_0),
        .o_a(wire_a_00_to_01), .o_b(wire_b_00_to_10), // 토스!
        .o_valid(valid_00),    .o_acc(out_acc_00)
    );

    // PE (0,1) - 우측 상단
    pe_unit pe_01 (
        .clk(clk), .rst_n(rst_n), .i_valid(valid_00),
        .i_a(wire_a_00_to_01), .i_b(in_b_1),          // 왼쪽 친구한테 받음
        .o_a(),                .o_b(wire_b_01_to_11), // 오른쪽은 비워둠 (끝이니까)
        .o_valid(valid_01),    .o_acc(out_acc_01)
    );

    // PE (1,0) - 좌측 하단
    pe_unit pe_10 (
        .clk(clk), .rst_n(rst_n), .i_valid(valid_00),
        .i_a(in_a_1),          .i_b(wire_b_00_to_10), // 위쪽 친구한테 받음
        .o_a(wire_a_10_to_11), .o_b(),                // 아래쪽은 비워둠
        .o_valid(valid_10),    .o_acc(out_acc_10)
    );

    // PE (1,1) - 우측 하단
    pe_unit pe_11 (
        .clk(clk), .rst_n(rst_n), .i_valid(valid_01), // valid 조건은 상황에 맞게 조정 가능
        .i_a(wire_a_10_to_11), .i_b(wire_b_01_to_11), // 위, 왼쪽 친구한테 다 받음
        .o_a(),                .o_b(),
        .o_valid(),            .o_acc(out_acc_11)
    );

endmodule
