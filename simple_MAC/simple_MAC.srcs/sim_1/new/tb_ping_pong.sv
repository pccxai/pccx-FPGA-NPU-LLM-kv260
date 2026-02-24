`timescale 1ns / 1ps

module tb_ping_pong;
    logic clk;
    logic rst_n;

    // 핑퐁 스위치
    logic ping_pong_sel;

    // DMA (데이터 밀어넣는 쪽) 포트
    logic       dma_we;
    logic [7:0] dma_addr;
    logic [7:0] dma_wdata;

    // NPU (데이터 빼먹는 쪽) 포트
    logic [7:0] sys_addr;
    logic [7:0] sys_rdata;

    // 우리가 만든 핑퐁 BRAM 모듈 생성 (C++의 객체 생성 느낌)
    ping_pong_bram dut (
        .clk(clk), .rst_n(rst_n),
        .ping_pong_sel(ping_pong_sel),
        .dma_we(dma_we), .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .sys_addr(sys_addr), .sys_rdata(sys_rdata)
    );

    // 클럭 생성 (10ns 주기)
    always #5 clk = ~clk;

    initial begin
        // 1. 초기화 (전원 켜기)
        clk = 0; rst_n = 0;
        ping_pong_sel = 0;
        dma_we = 0; dma_addr = 0; dma_wdata = 0;
        sys_addr = 0;

        #20 rst_n = 1; // 리셋 해제

        // =========================================================
        // [Phase 1] ping_pong_sel = 0
        // DMA가 BRAM_0에 데이터 두 개(10, 20)를 쓴다.
        // =========================================================
        @(posedge clk);
        dma_we <= 1; dma_addr <= 8'd0; dma_wdata <= 8'd10; // 0번 주소에 10 쓰기

        @(posedge clk);
        dma_we <= 1; dma_addr <= 8'd1; dma_wdata <= 8'd20; // 1번 주소에 20 쓰기

        @(posedge clk);
        dma_we <= 0; // 쓰기 끝! 휴식.

        #20;

        // =========================================================
        // [Phase 2] ping_pong_sel = 1 (스위치 교체!)
        // NPU는 방금 쓴 10, 20을 읽어가고, 
        // 동시에 DMA는 BRAM_1에 새로운 데이터(30)를 쓴다.
        // =========================================================
        ping_pong_sel = 1; 

        @(posedge clk);
        // NPU야, 0번 주소 읽어봐! (다음 클럭에 sys_rdata로 10이 나와야 함)
        sys_addr <= 8'd0; 
        
        // 그와 동시에 DMA야, 넌 놀지 말고 0번 주소에 30 써놔! (BRAM_1에 들어감)
        dma_we <= 1; dma_addr <= 8'd0; dma_wdata <= 8'd30;

        @(posedge clk);
        // NPU야, 1번 주소 읽어봐! (다음 클럭에 20이 나와야 함)
        sys_addr <= 8'd1; 
        dma_we <= 0; // DMA 쓰기 끝

        #50;
        $finish;
    end
endmodule