`timescale 1ns / 1ps

// 🛡️ [마법의 부적] Vivado야, 이 모듈 안의 곱셈은 묻지도 따지지도 말고 무조건 DSP48E2를 써라!!
(* use_dsp = "yes" *)
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

    // 곱셈 결과용 내부 선언 (여기도 부적 한 번 더 발라줌!)
    (* use_dsp = "yes" *) logic signed [31:0] mul_result;
    
    // 이제 포트가 signed니까 $signed 매크로 없이도 완벽한 2의 보수 곱셈 수행됨!
    assign mul_result = $signed(i_a) * $signed(i_b);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // 하드웨어 전체 리셋
            o_acc   <= 32'd0; // 16'd0 이었던 거 32비트 사이즈에 맞게 수정!
            o_valid <= 1'b0;
            o_a     <= 8'd0;
            o_b     <= 8'd0;
        end else if (i_clear) begin 
            // 🔥 소프트웨어(FSM) 명령에 의한 누산기 초기화! (변기 물 내림)
            o_acc   <= 32'd0;
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