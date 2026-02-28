`timescale 1ns / 1ps

module tb_npu_core_top();

    // 파라미터 셋업
    parameter ARRAY_SIZE = 32;
    parameter DATA_WIDTH = 512;
    parameter ADDR_WIDTH = 9;

    // 포트 연결용 신호
    logic clk;
    logic rst_n;
    
    logic                  dma_we;
    logic [ADDR_WIDTH-1:0] dma_addr;
    logic [DATA_WIDTH-1:0] dma_write_data;
    logic                  start_mac;
    
    logic signed [31:0] out_acc  [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic signed [7:0]  out_gelu [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]; // 🔥 배선 추가
    // NPU 탑 모듈 인스턴스화
    npu_core_top_NxN #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) uut (
        .clk(clk), .rst_n(rst_n),
        .dma_we(dma_we), .dma_addr(dma_addr), .dma_write_data(dma_write_data),
        .start_mac(start_mac),
        .out_acc(out_acc),
        .out_gelu(out_gelu)  // 🔥 포트 연결
    );

    // 100MHz 클럭 생성 (주기 10ns)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 파이썬에서 만든 512비트 타일 데이터를 담을 임시 메모리
    logic [DATA_WIDTH-1:0] test_mem [0:ARRAY_SIZE-1];

    // 시뮬레이션 시나리오 시작!
    initial begin
        // 1. 초기화 및 리셋
        $display("🚀 [Time: %0t] 시뮬레이션 시작! 리셋 가동...", $time);
        rst_n = 0;
        dma_we = 0;
        dma_addr = 0;
        dma_write_data = 0;
        start_mac = 0;
        
        // 메모리 파일 로드 (파이썬에서 만든 gemma 타일)
        $readmemh("gemma_tile.mem", test_mem);
        
        #20 rst_n = 1; // 리셋 해제
        #10;

        // 2. 가상 AXI DMA 전송 시작
        $display("💾 [Time: %0t] AXI DMA: Gemma 타일 데이터 BRAM 전송 시작...", $time);
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            @(posedge clk);
            #1; // 🔥 Race Condition 방지용 딜레이! (이게 핵심)
            dma_we = 1'b1;
            dma_addr = i;
            dma_write_data = test_mem[i]; 
        end
        @(posedge clk);
        #1; // 🔥 여기도 추가!
        dma_we = 1'b0; 
        $display("💾 [Time: %0t] AXI DMA: 전송 완료!", $time);
        #10;

        // 3. NPU 타일 연산 트리거 (발사!)
        $display("🔥 [Time: %0t] NPU 가동! 32x32 Wavefront 파도타기 시작!", $time);
        @(posedge clk);
        #1; // 🔥 여기도 추가!
        start_mac = 1'b1;
        @(posedge clk);
        #1; // 🔥 여기도 추가!
        start_mac = 1'b0;

        // 4. 연산 완료 대기
        // FSM 스트리밍 32클럭 + Systolic Array 파도타기 전파 지연 (Row 31 + Col 31 = 62클럭) + 여유 마진
        // 약 100클럭 대기
        #(150 * 10);
        $display("🌊 [Time: %0t] 파도타기 연산 종료!", $time);

        // 5. 결과 검증 (첫 번째 PE와 마지막 PE 확인)
        // 파이썬 콘솔에 출력된 '파이썬 정답'과 비교해볼 것!
        $display("========================================");
        $display("🎯 PE(0,0) 결과: %d", $signed(out_acc[0][0]));
        $display("🎯 PE(31,31) 결과: %d", $signed(out_acc[31][31]));
        $display("========================================");
        // 5. 결과 검증 (첫 번째 PE와 마지막 PE 확인)
        $display("========================================");
        $display("🎯 PE(0,0)   MAC: %d -> GeLU: %d", $signed(out_acc[0][0]), $signed(out_gelu[0][0]));
        $display("🎯 PE(31,31) MAC: %d -> GeLU: %d", $signed(out_acc[31][31]), $signed(out_gelu[31][31]));
        $display("========================================");
        $finish;
    end

endmodule