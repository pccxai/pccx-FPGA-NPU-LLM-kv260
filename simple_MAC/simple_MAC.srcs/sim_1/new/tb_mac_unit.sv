`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/02/23 03:41:37
// Design Name: 
// Module Name: tb_mac_unit
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


`timescale 1ns / 1ps  // 시간 단위: 1ns, 정밀도: 1ps

module tb_mac_unit();

    // 1. Defining dummy signals
    logic        clk;
    logic        rst_n;
    logic [7:0]  i_a;
    logic [7:0]  i_b;
    logic [15:0] o_acc;

    // 2. Instantiate the mac_unit on the testbench
    pe_unit uut (
        .clk   (clk),
        .rst_n (rst_n), 
        .i_a   (i_a),
        .i_b   (i_b),
        .o_acc (o_acc)
    );

    // 3. Generating a dummy clock (100MHz)
    // Toggle the signal every 5ns to create a 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // 4. 테스트 시나리오 입력
    initial begin
        // 초기화
        rst_n = 0; i_a = 0; i_b = 0;
        #20;            // 20ns 대기
        rst_n = 1;      // 리셋 해제
        
        // 첫 번째 연산: 2 * 3 = 6
        #10; i_a = 8'd2; i_b = 8'd3;
        
        // 두 번째 연산: 4 * 5 = 20 (누적 결과 26 예상)
        #10; i_a = 8'd4; i_b = 8'd5;
        
        // 세 번째 연산: 10 * 10 = 100 (누적 결과 126 예상)
        #10; i_a = 8'd10; i_b = 8'd10;

        #50;
        $display("Final Accumulation: %d", o_acc); // 콘솔에 결과 출력
        $finish; // 시뮬레이션 종료
    end

endmodule
