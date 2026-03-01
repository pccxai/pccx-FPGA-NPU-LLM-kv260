`timescale 1ns / 1ps

module npu_core_top_NxN #(
    parameter ARRAY_SIZE = 32, 
    parameter DATA_WIDTH = 512, 
    parameter ADDR_WIDTH = 9
)(
    input   logic        clk,
    input   logic        rst_n,

    // AXI DMA 인터페이스
    input   logic                  dma_we,     
    input   logic [ADDR_WIDTH-1:0] dma_addr,
    input   logic [DATA_WIDTH-1:0] dma_write_data,
    
    // 연산 시작 트리거
    input   logic        start_mac,

    // 🔥 [수정 1] 출력 포트에 signed 추가 및 GeLU 출력 포트 신설!
    output  logic signed [31:0] out_acc  [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    output  logic signed [7:0]  out_gelu [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1] 
);

    logic        switch_buffer; 
    logic        i_clear_global;

    logic [ADDR_WIDTH-1:0] read_addr;
    logic [DATA_WIDTH-1:0] systolic_read_data_a;
    logic [DATA_WIDTH-1:0] systolic_read_data_b;

    // -----------------------------------------------------------------
    // [핵심 1] Dual-Port Vector Unpacking (512비트 -> 8비트 배열 32개 x 2)
    // -----------------------------------------------------------------
    logic signed [7:0] unpacked_a [0:ARRAY_SIZE-1];
    logic signed [7:0] unpacked_b [0:ARRAY_SIZE-1];
    
    always_comb begin
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            // 🔥 [수정 2] 하위 256비트는 Matrix A, 상위 256비트는 Matrix B (완벽한 언패킹!)
            unpacked_a[i] = systolic_read_data_a[i*8 +: 8];
            unpacked_b[i] = systolic_read_data_a[(ARRAY_SIZE + i)*8 +: 8]; 
        end
    end

    // FSM 상태 레지스터
    logic [2:0]  state;
    logic [7:0]  fire_cnt; 
    logic signed [7:0]  fire_a [0:ARRAY_SIZE-1];
    logic signed [7:0]  fire_b [0:ARRAY_SIZE-1];
    logic        fire_valid;

    // -----------------------------------------------------------------
    // [핵심 2] 32-Cycle Streaming FSM
    // -----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= 0;
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
                0: begin 
                    fire_valid <= 0;
                    if (start_mac) begin
                        switch_buffer <= ~switch_buffer;
                        i_clear_global <= 1'b1; 
                        read_addr <= 0;
                        state <= 1;
                    end
                end

                1: begin 
                    i_clear_global <= 1'b0;
                    read_addr <= read_addr + 1;
                    fire_cnt <= 0;
                    state <= 2;
                end

                2: begin 
                    for (int i = 0; i < ARRAY_SIZE; i++) begin
                        fire_a[i] <= unpacked_a[i];
                        fire_b[i] <= unpacked_b[i];
                    end
                    fire_valid <= 1'b1;

                    if (read_addr < ARRAY_SIZE) begin
                        read_addr <= read_addr + 1;
                    end

                    if (fire_cnt == ARRAY_SIZE - 1) begin
                        state <= 3;
                    end else begin
                        fire_cnt <= fire_cnt + 1;
                    end
                end

                3: begin 
                    fire_valid <= 1'b0;
                    state <= 0;
                end
            endcase
        end
    end

    // -----------------------------------------------------------------
    // 하위 모듈 인스턴스화
    // -----------------------------------------------------------------
    
    ping_pong_bram #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_bram (
        .clk(clk),
        .rst_n(rst_n),
        .switch_buffer(switch_buffer),
        .dma_we(dma_we),
        .dma_addr(dma_addr),
        .dma_write_data(dma_write_data),
        .npu_addr_a(read_addr), 
        .npu_addr_b(read_addr),        
        .npu_read_data_a(systolic_read_data_a),
        .npu_read_data_b(systolic_read_data_b)
    );

    systolic_NxN #(
        .ARRAY_SIZE(ARRAY_SIZE)
    ) u_systolic_array (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(i_clear_global),      
        .in_a(fire_a),                 
        .in_b(fire_b),      
        .in_valid(fire_valid),         
        .out_acc(out_acc)
    );

    // 🔥 [수정 3] 1,024개의 GeLU LUT 하드웨어 연결!
    genvar r, c;
    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin : gelu_row
            for (c = 0; c < ARRAY_SIZE; c++) begin : gelu_col
                gelu_lut u_gelu (
                    .clk(clk),
                    .valid_in(1'b1), 
                    .data_in(out_acc[r][c]),      
                    .valid_out(), 
                    .data_out(out_gelu[r][c])     
                );
            end
        end
    endgenerate

endmodule