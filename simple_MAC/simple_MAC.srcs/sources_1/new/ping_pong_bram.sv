`timescale 1ns / 1ps

module ping_pong_bram (
    input  logic       clk,
    input  logic       rst_n,
    
    // 0: DMA -> BRAM_0 쓰기 / Systolic <- BRAM_1 읽기
    // 1: DMA -> BRAM_1 쓰기 / Systolic <- BRAM_0 읽기
    input  logic       ping_pong_sel, 
    
    // AXI DMA 인터페이스
    input  logic       dma_we,
    input  logic [7:0] dma_addr,
    input  logic [7:0] dma_wdata,
    
    // Systolic Array 인터페이스
    input  logic [7:0] sys_addr,
    output logic [7:0] sys_rdata
);

    // BRAM_0 과 BRAM_1 에 연결할 내부 전선들
    logic       we_0, we_1;
    logic [7:0] addr_0, addr_1;
    logic [7:0] rdata_0, rdata_1;

    // --------------------------------------------------------
    // [MUX 로직] 포인터(ping_pong_sel)에 따라 길을 바꿔줌
    // --------------------------------------------------------
    
    // BRAM 0번 제어
    assign we_0   = (ping_pong_sel == 1'b0) ? dma_we   : 1'b0;       // NPU가 쓸 땐 Write 금지
    assign addr_0 = (ping_pong_sel == 1'b0) ? dma_addr : sys_addr;

    // BRAM 1번 제어
    assign we_1   = (ping_pong_sel == 1'b1) ? dma_we   : 1'b0;       // NPU가 쓸 땐 Write 금지
    assign addr_1 = (ping_pong_sel == 1'b1) ? dma_addr : sys_addr;

    // Systolic 쪽으로 나가는 데이터 (Demux)
    assign sys_rdata = (ping_pong_sel == 1'b0) ? rdata_1 : rdata_0;


    // --------------------------------------------------------
    // [BRAM 인스턴시에이션] 실제 메모리 2개 박기
    // --------------------------------------------------------
    simple_bram bram_0 (
        .clk(clk),
        .we(we_0),
        .addr(addr_0),
        .din(dma_wdata), // 데이터 들어가는 선은 묶어놔도 we가 막아줌
        .dout(rdata_0)
    );

    simple_bram bram_1 (
        .clk(clk),
        .we(we_1),
        .addr(addr_1),
        .din(dma_wdata),
        .dout(rdata_1)
    );

endmodule