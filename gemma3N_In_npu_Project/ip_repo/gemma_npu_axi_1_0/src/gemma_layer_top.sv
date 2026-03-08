`timescale 1ns / 1ps

module gemma_layer_top #(
    parameter SYSTOLIC_ARRAY_SIZE = 32 // 32x32 아키텍처!
    parameter SYSTOLIC_RESULT_SIZE = 48
    parameter SYSTOLIC_INPUT_SIZE = 16;
)(
    input  logic               clk,
    input  logic               rst_n,

    // -------------------------------------------------------------------
    // [AXI4-Lite MMIO 제어 신호들] (0x00 ~ 0x10)
    // -------------------------------------------------------------------
    input  logic               i_npu_start,     // 0x00 [Bit 0] (Kernel Launch!)
    input  logic               i_acc_clear,     // 0x00 [Bit 1] (누산기 리셋)
    input  logic [31:0]        i_rms_mean_sq,   // 0x08 (RMSNorm 분모)
    input  logic               i_ping_pong_sel, // 0x0C (DMA ↔ NPU 스위치)
    input  logic               i_gelu_en,       // 0x10 [Bit 0] (GeLU 활성화)
    input  logic               i_softmax_en,    // 0x10 [Bit 1] (Softmax 활성화)
    output logic               o_npu_done,      // 0x04 [Bit 0] (연산 완료 깃발)

    // -------------------------------------------------------------------
    // [AXI DMA 스트리밍 인터페이스 (예시)]
    // 실제로는 AXI-Stream (TVALID, TDATA 등) 규격을 사용하겠지만, 개념적으로 표현함!
    // -------------------------------------------------------------------
    input  logic               i_dma_we_token,
    input  logic [7:0]         i_dma_addr_token,
    input  logic [511:0]       i_dma_wdata_token,

    input  logic               i_dma_we_weight,
    input  logic [7:0]         i_dma_addr_weight,
    input  logic [511:0]       i_dma_wdata_weight, // 32x8bit 타일 한 줄

    input  logic [4:0]         i_result_sel,       // [추가] 32개 채널 선택용
    output logic [1023:0]      o_npu_result_all,   // [추가] DMA용 전체 버스
    output logic               o_logic_anchor,     // [필살기] 모든 PE 보존용 닻!
    output logic [15:0]        o_final_result      // DMA를 통해 CPU로 돌아갈 최종 결과
);

    // =========================================================================
    // 1. FSM (Warp Scheduler): 커널의 생명주기 통제
    // =========================================================================
    typedef enum logic [1:0] {
        ST_IDLE     = 2'd0,  // 대기
        ST_WAIT_RMS = 2'd1,  // 0x08로 들어온 mean_sq가 역제곱근으로 변환될 때까지 대기
        ST_RUN      = 2'd2,  // 핑퐁 BRAM에서 32x32 어레이로 데이터 폭격!
        ST_WAIT_MAC = 2'd3   // Systolic 파이프라인(Wavefront) 잔여 연산 완료 대기
    } state_t;

    state_t state, next_state;
    logic [5:0] feed_counter; 
    logic       npu_running;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            feed_counter <= 6'd0;
        end else begin
            state <= next_state;
            if (state == ST_RUN) feed_counter <= feed_counter + 1;
            else                 feed_counter <= 6'd0;
        end
    end

    logic rms_valid_out, mac_valid_out;

    always_comb begin
        next_state  = state;
        o_npu_done  = 1'b0;
        npu_running = 1'b0;

        case (state)
            ST_IDLE: begin
                if (i_npu_start) next_state = ST_WAIT_RMS; 
            end
            ST_WAIT_RMS: begin
                if (rms_valid_out) next_state = ST_RUN; 
            end
            ST_RUN: begin
                npu_running = 1'b1;
                if (feed_counter == 6'd31) next_state = ST_WAIT_MAC; // 32번 데이터 다 넣었음!
            end
            ST_WAIT_MAC: begin
                if (mac_valid_out) begin 
                    o_npu_done = 1'b1;       // CPU한테 0x04 깃발 올려줌
                    next_state = ST_IDLE;
                end
            end
        endcase
    end

    // =========================================================================
    // 2. [Stage 1] Pre-Norm (RMSNorm 계산기)
    // =========================================================================
    logic [15:0] rms_inv_sqrt_val;

    rmsnorm_inv_sqrt u_rmsnorm (
        .clk(clk), .rst_n(rst_n),
        .valid_in(i_npu_start),       // CPU가 Start 때리면 계산 시작!
        .i_mean_sq(i_rms_mean_sq),    // 0x08에서 날아온 스칼라
        .valid_out(rms_valid_out),    // 계산 끝나면 FSM을 RUN으로 넘김
        .o_inv_sqrt(rms_inv_sqrt_val)
    );

    // =========================================================================
    // 3. Ping-Pong BRAM & Vector Scaling (On-the-fly)
    // =========================================================================
    // 핑퐁 버퍼에서 읽어온 원본 데이터 (8비트 배열 32개)
    logic [7:0] raw_token_data  [0:31]; 
    logic [7:0] sys_weight_data [0:31]; 

    // [수정] 512비트 플랫(Flat) 벡터로 BRAM에서 받아서 배열로 풀기
    logic [511:0] flat_token_data;
    logic [511:0] flat_weight_data;

    always_comb begin
        for (int i = 0; i < 32; i++) begin
            raw_token_data[i]  = flat_token_data[i*8 +: 8];
            sys_weight_data[i] = flat_weight_data[i*8 +: 8];
        end
    end

    // [수정] 포트 이름 완벽 일치 및 포맷 캐스팅
    ping_pong_bram #(
        .DATA_WIDTH(512),
        .ADDR_WIDTH(9)
    ) u_bram_token (
        .clk(clk), .rst_n(rst_n),
        .switch_buffer(i_ping_pong_sel),
        .dma_we(i_dma_we_token),
        .dma_addr({1'b0, i_dma_addr_token}),          // 8bit -> 9bit 확장
        .dma_write_data(i_dma_wdata_token), 
        .npu_addr_b(9'd0),                            // 안 씀
        .npu_read_data_a(flat_token_data),            // 512bit 출력
        .npu_read_data_b()                            // 안 씀
    );

    ping_pong_bram #(
        .DATA_WIDTH(512),
        .ADDR_WIDTH(9)
    ) u_bram_weight (
        .clk(clk), .rst_n(rst_n),
        .switch_buffer(i_ping_pong_sel),
        .dma_we(i_dma_we_weight),
        .dma_addr({1'b0, i_dma_addr_weight}),
        .dma_write_data(i_dma_wdata_weight), //  {256'd0, ...} 지우고 직결!        .npu_addr_a({3'd0, feed_counter}),
        .npu_addr_b(9'd0),
        .npu_read_data_a(flat_weight_data),
        .npu_read_data_b()
    );

    // ⚡ BRAM에서 나오는 즉시 RMSNorm 역제곱근을 곱해서 MAC으로 밀어넣기!
    logic [7:0] scaled_token_data [0:31];
    genvar i;
    generate
        for (i = 0; i < 32; i++) begin : gen_scaling
            assign scaled_token_data[i] = (raw_token_data[i] * $signed({1'b0, rms_inv_sqrt_val})) >> 15;
        end
    endgenerate

    // =========================================================================
    // 4. [Stage 2] 32x32 Systolic MAC Array (본체)
    // =========================================================================
    logic [32*32*32-1:0] mac_out_acc_flat; // 2차원 배열 대신 1차원 Flat 버스 (연결성 보장!)

    systolic_NxN #(
        .ARRAY_SIZE(32)
        .OUT_SIZE(48)
        .INPUT_SIZE(16)
    ) u_mac_engine (
        .clk(clk), .rst_n(rst_n),
        .i_clear(i_acc_clear),           
        .in_a(scaled_token_data),        
        .in_b(sys_weight_data),          
        .in_valid(npu_running),
        .out_acc_flat(mac_out_acc_flat)  // Flat 포트로 연결
    );

    // 파이프라인 레이턴시(Wavefront) 추적 shift register
    logic [63:0] shift_reg_valid; 
    always_ff @(posedge clk) begin
        if (!rst_n) shift_reg_valid <= 0;
        else        shift_reg_valid <= {shift_reg_valid[62:0], npu_running};
    end
    assign mac_valid_out = shift_reg_valid[63]; // aproximately 64clocks after 가장 오른쪽 아래 PE 완료

    // =========================================================================
    // 5. [Stage 3] Output Buffering & Activation MUX
    // =========================================================================
    
    // Array to store 48bit from 32 pe units
    logic [SYSTOLIC_RESULT_SIZE-1:0] mac_result_row [0:SYSTOLIC_ARRAY_SIZE-1];
    always_comb begin
        for (int j = 0; j < SYSTOLIC_ARRAY_SIZE; j++) begin
            // 인덱스 계산: ((가장 마지막 줄 번호) * 배열 크기 + 현재 열) * 데이터 폭
            // 32(31)번째 row값 전부 저장 -> 31(30) -> 30(29) -> ... -> 1(0)
            mac_result_row[j] = mac_out_acc_flat[((SYSTOLIC_ARRAY_SIZE-1)*SYSTOLIC_ARRAY_SIZE + j) * SYSTOLIC_RESULT_SIZE +: SYSTOLIC_RESULT_SIZE];
        end
    end

    // result address counter
    logic [4:0] result_addr_cnt;
    always_ff @(posedge clk) begin
        if (!rst_n || i_acc_clear) result_addr_cnt <= 5'd0;
        else if (mac_valid_out)    result_addr_cnt <= result_addr_cnt + 1;
    end

    //DMA output BUS size : 32(column) * 48bit(value) = 1536bit [1535:0]
    logic [(SYSTOLIC_ARRAY_SIZE * SYSTOLIC_RESULT_SIZE) - 1 : 0] dma_result_bus;

    result_ping_pong_bram u_out_buffer (
        .clk(clk), .rst_n(rst_n),
        .switch_buffer(i_ping_pong_sel), 
        .npu_we(mac_valid_out),          
        .npu_addr(result_addr_cnt),      
        .npu_data_in(mac_result_row),
        .dma_addr(5'd0),                 
        .dma_data_out(dma_result_bus)    
    );

    // [정공법] 전체 버스 밖으로 노출 (DMA 준비용)  
    assign o_npu_result_all = dma_result_bus;

    //  [필살기] 모든 PE(1024개)를 강제로 살려두기 위한 논리적 '닻(Anchor)'
    // 1024비트 중 하나만 바뀌어도 이 값이 바뀌므로 Vivado는 PE를 삭제하지 못함!
    assign o_logic_anchor = ^dma_result_bus; 

    // CPU가 MMIO로 읽어갈 채널 선택 MUX (DSP Pruning 방지 핵심!)
    logic [31:0] selected_channel_data;
    assign selected_channel_data = dma_result_bus[i_result_sel * 32 +: 32];

    // 기존 디버그용 변수 이름은 그대로 유지하되 선택된 채널 값을 태움
    logic signed [15:0] mac_attn_score;
    assign mac_attn_score = selected_channel_data[15:0]; 

    logic [15:0] softmax_prob;
    softmax_exp_unit u_softmax (
        .clk(clk), .rst_n(rst_n), .valid_in(mac_valid_out),
        .i_x(mac_attn_score), .valid_out(), .o_exp(softmax_prob)
    );

    //  GeLU 하드웨어(LUT) 꽂기
    logic signed [7:0] gelu_out_8bit;
    logic [15:0]       gelu_out;

    gelu_lut u_gelu_inst (
        .clk(clk),
        .valid_in(mac_valid_out),
        .data_in(mac_attn_score),      
        .valid_out(),
        .data_out(gelu_out_8bit)       
    );

    assign gelu_out = $signed(gelu_out_8bit);

    // 최종 출력 라우팅
    always_comb begin
        if (i_softmax_en)       o_final_result = softmax_prob;   
        else if (i_gelu_en)     o_final_result = gelu_out;       
        else                    o_final_result = mac_attn_score; 
    end

endmodule
