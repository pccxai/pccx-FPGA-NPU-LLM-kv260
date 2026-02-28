`timescale 1ns / 1ps

module pe_unit (
    input  logic        clk,     
    input  logic        rst_n,   
    input  logic        i_clear, // 🔥 [핵심] 타일 연산 전 누산기를 비우는 클리어 핀!
    input  logic        i_valid, 
    
    // 🔥 포트 자체를 signed로 선언!
    input  logic signed [7:0]  i_a,     
    input  logic signed [7:0]  i_b, 
    
    output logic [7:0]  o_a,     
    output logic [7:0]  o_b,     
    output logic        o_valid, 
    output logic [31:0] o_acc  
);

    // [mac_unit.sv 수정본]
    logic signed [31:0] mul_result;

    // 이제 포트가 signed니까 $signed 매크로 없이도 완벽한 2의 보수 곱셈 수행됨!
    assign mul_result = $signed(i_a) * $signed(i_b); // 🔥 완벽한 INT8 곱셈!

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 하드웨어 전체 리셋
            o_acc   <= 16'd0;
            o_valid <= 1'b0;
            o_a     <= 8'd0;
            o_b     <= 8'd0;
        end else if (i_clear) begin 
            // 🔥 소프트웨어(FSM) 명령에 의한 누산기 초기화! (변기 물 내림)
            o_acc   <= 16'd0;
            o_valid <= 1'b0;
            o_a     <= 8'd0;
            o_b     <= 8'd0;
        end else if (i_valid) begin 
            // 정상 MAC 파이프라인 가동
            o_acc   <= o_acc + mul_result;
            o_valid <= 1'b1;
            o_a     <= i_a; 
            o_b     <= i_b;
        end else begin  
            // Valid가 없으면 쉬기
            o_valid <= 1'b0;
        end
    end
endmodule