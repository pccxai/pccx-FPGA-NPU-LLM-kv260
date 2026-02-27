`timescale 1ns / 1ps

module npu_core_top_NxN #(
    parameter ARRAY_SIZE = 64, // 코어 개수 64 x 64
    parameter DATA_WIDTH = 512, // 64 Bytes (Bram에서 한번에 나오는 양)
    parameter ADDR_WIDTH = 9
)(
    // ==========================================
    // 1. 외부 인터페이스 (Public, Input/Output)
    // ==========================================
    input   logic        clk,
    input   logic        rst_n,

    // AXI DMA 인터페이스
    input   logic           dma_we,     
    input   logic [ADDR_WIDTH - 1 : 0]  dma_addr,
    input   logic [DATA_WIDTH - 1 : 0]  dma_write_data,
    
    // 연산 시작 트리거
    input   logic        start_mac,

    // 최종 연산 결과 (2차원 배열 출력)
    output  logic [15:0] out_acc [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    logic        switch_buffer; 
    logic        i_clear_global


    // Dual-Port BRAM 인터페이스 (A와 B를 동시에 읽기 위한 주소와 데이터)
    logic [ADDR_WIDTH-1:0] read_addr; 
    logic [DATA_WIDTH-1:0] systolic_read_data_a;
    logic [DATA_WIDTH-1:0] systolic_read_data_b;

    // -----------------------------------------------------------------
    // [핵심 1] Dual-Port Vector Unpacking (512비트 -> 8비트 배열 64개 x 2)
    // -----------------------------------------------------------------
    logic [7:0] unpacked_a [0:ARRAY_SIZE-1];
    logic [7:0] unpacked_b [0:ARRAY_SIZE-1];
    
    always_comb begin
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            unpacked_a[i] = systolic_read_data_a[i*8 +: 8];
            unpacked_b[i] = systolic_read_data_b[i*8 +: 8];
        end
    end

    // FSM 상태 레지스터
    logic [2:0]  state;
    logic [7:0]  fire_cnt; // 64번 쏘기 위한 카운터
    logic [7:0]  fire_a [0:ARRAY_SIZE-1];
    logic [7:0]  fire_b [0:ARRAY_SIZE-1];
    logic        fire_valid;

    // -----------------------------------------------------------------
    // [핵심 2] 초고속 장전 FSM (루프 없이 2클럭 만에 장전 완료)
    // -----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state < = 0;
            fire_valid <= 0;
            switch_buffer <= 0;
            read_addr <= 0;
            fire_cnt <= 0;
            i_clear_global <= 0;
            
            for (int i=0; i<ARRAY_SIZE; i++) begin
                fire_a[i] <= 8'd0;
                fire_b[i] <= 8'd0;
            end
        end else begin
            case (state)
                0: begin // [IDLE] CPU의 시작 명령 대기
                    fire_valid <= 0;
                    if (start_mac) begin
                        i_clear_global <= 1'b1; // 💥 연산 시작 전 찌꺼기 초기화 (Clear)
                        read_addr <= 0;         // BRAM 주소 0번지부터 긁어올 준비
                        state <= 1;
                    end
                end

                1: begin // [PRELOAD] BRAM 주소 인가 및 데이터 도착 대기
                    i_clear_global <= 1'b0; // 클리어 펄스 끄기
                    read_addr <= read_addr + 1; // 다음 주소 미리 계산
                    fire_cnt <= 0;
                    state <= 2;
                end

                2: begin // [STREAMING] 🔥 64클럭 연속 데이터 폭포수 발사! 🔥
                    // BRAM에서 갓 도착한 따끈따끈한 A/B 데이터를 레지스터에 찰칵!
                    for (int i = 0; i < ARRAY_SIZE; i++) begin
                        fire_a[i] <= unpacked_a[i];
                        fire_b[i] <= unpacked_b[i];
                    end
                    fire_valid <= 1'b1; // Systolic Array, 일해라!!

                    // 다음 클럭을 위해 주소 증가 (64번까지만)
                    if (read_addr < ARRAY_SIZE) begin
                        read_addr <= read_addr + 1;
                    end

                    // 카운터 체크 (64번 쐈으면 종료)
                    if (fire_cnt == ARRAY_SIZE - 1) begin
                        state <= 3;
                    end else begin
                        fire_cnt <= fire_cnt + 1;
                    end
                end

                3: begin // [COOLDOWN] 발사 중지 및 버퍼 스와핑
                    fire_valid <= 1'b0;
                    switch_buffer <= ~switch_buffer; // 🏓 핑퐁 스위치 토글! (DMA한테 턴 넘기기)
                    state <= 0; // 다음 타일을 위해 IDLE로 복귀
                end
            endcase
        end
    end

    // -----------------------------------------------------------------
    // 하위 모듈 인스턴스화
    // -----------------------------------------------------------------
    
    // 💥 개조된 2차선 (Dual-Port) BRAM
    ping_pong_bram #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_bram (
        .clk(clk),
        .rst_n(rst_n),
        .switch_buffer(switch_buffer),
        
        // DMA는 무조건 Write만
        .dma_we(dma_we),
        .dma_addr(dma_addr),
        .dma_write_data(dma_write_data),
        
        // NPU는 A와 B 행렬을 동시에 Read!
        .npu_addr_a(read_addr),        // A행렬 주소
        .npu_addr_b(read_addr),        // B행렬 주소 (동시에 같은 인덱스 스캔)
        .npu_read_data_a(systolic_read_data_a),
        .npu_read_data_b(systolic_read_data_b)
    );

    // 💥 4096개 코어가 박힌 Systolic Array
    systolic_NxN #(
        .ARRAY_SIZE(ARRAY_SIZE)
    ) u_systolic_array (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(i_clear_global),      // 싹 다 비워라!
        .in_a(fire_a),                 // 왼쪽에서 들어가는 파도
        .in_b(fire_b),                 // 위쪽에서 내려가는 파도
        .in_valid(fire_valid),         // 64클럭 연속 파도타기 신호
        .out_acc(out_acc)
    );

endmodule