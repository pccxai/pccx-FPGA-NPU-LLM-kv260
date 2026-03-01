`timescale 1ns / 1ps

module delay_line #(
    parameter WIDTH = 8,
    parameter DELAY = 1 // 몇 클럭 지연시킬지 결정하는 파라미터!
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [WIDTH-1:0] in_data,
    output logic [WIDTH-1:0] out_data
);

    // DELAY가 0이면 그냥 전선(다이렉트) 연결
    generate
        if (DELAY == 0) begin : gen_no_delay
            assign out_data = in_data;
        end 
        // DELAY가 1 이상이면 그 개수만큼 플립플롭(컨베이어 벨트) 생성
        else begin : gen_delay
            // C++의 배열(큐) 역할
            logic [WIDTH-1:0] shift_reg [0:DELAY-1];
            
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < DELAY; i++) shift_reg[i] <= 0;
                end else begin
                    // 1. 컨베이어 벨트 맨 앞에 새 데이터 올리기
                    shift_reg[0] <= in_data; 
                    
                    // 2. 나머지 데이터들은 한 칸씩 뒤로 밀어주기 (i=1부터 시작)
                    for (int i = 1; i < DELAY; i++) begin
                        shift_reg[i] <= shift_reg[i-1]; 
                    end
                end
            end
            
            // 맨 마지막 칸에 도달한 녀석이 밖으로 튀어나감
            assign out_data = shift_reg[DELAY-1]; 
        end
    endgenerate

endmodule