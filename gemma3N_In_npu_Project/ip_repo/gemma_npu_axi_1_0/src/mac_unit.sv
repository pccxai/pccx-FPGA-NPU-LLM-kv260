`timescale 1ns / 1ps

(* use_dsp = "yes" *)
module pe_unit (
    input  logic               clk,     
    input  logic               rst_n,   
    input  logic               i_clear, 
    input  logic               i_valid, 
    // 16비트 Signed 입력으로 확장
    input  logic signed [15:0] i_a,     
    input  logic signed [15:0] i_b, 
    // 다음 PE로 넘겨주는 릴레이 출력도 16비트로 통일
    output logic signed [15:0] o_a,     
    output logic signed [15:0] o_b,     
    output logic               o_valid, 
    // 32번 누적 시 오버플로우 방지 및 DSP48E2 네이티브 크기에 맞춘 48비트 누산기
    output logic signed [47:0] o_acc 
);
    // 곱셈과 누산을 한 줄에 작성하여 DSP48E2 내부 퓨전을 유도
    
    '''
    Pre-Adder: 두 입력을 받아서 27비트 덧셈/뺄셈을 수행

    Multiplier: Pre-Adder 결과(27비트)와 다른 입력(18비트)을 곱함

    ALU (Accumulator/Adder/Subtractor): 곱셈 결과(최대 45비트)를 받아서 48비트 누적/연산 수행
    '''
    
    always_ff @(posedge clk) begin
        if (!rst_n || i_clear) begin
            o_acc   <= 48'd0;    // 48비트 초기화
            o_valid <= 1'b0;
            o_a     <= 16'd0;    // 기존 8'd0에서 16'd0으로 버그 수정 완료
            o_b     <= 16'd0;    // 기존 8'd0에서 16'd0으로 버그 수정 완료
        end else if (i_valid) begin 
            // 16비트 * 16비트 곱셈 결과(32비트)를 48비트 누산기에 더함
            o_acc   <= o_acc + (i_a * i_b);
            o_valid <= 1'b1;
            o_a     <= i_a; 
            o_b     <= i_b;
        end else begin  
            o_valid <= 1'b0;
        end
    end
endmodule