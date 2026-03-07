`timescale 1ns / 1ps

module rmsnorm_inv_sqrt (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               valid_in,
    input  logic [31:0]        i_mean_sq,   // 32비트 입력 (제곱의 평균)
    
    output logic               valid_out,
    output logic [15:0]        o_inv_sqrt   // 16비트 출력 (Q1.15 포맷)
);
    // ----------------------------------------------------------------
    // 1. 구간 인덱스와 소수점 찌꺼기 분리
    // ----------------------------------------------------------------
    logic [9:0]  segment_idx;
    logic [21:0] fractional_x;

    assign segment_idx  = i_mean_sq[31:22]; // 상위 10비트 (BRAM 주소)
    assign fractional_x = i_mean_sq[21:0];  // 하위 22비트 (보간용 x)

    // ----------------------------------------------------------------
    // 2. 1024칸짜리 초정밀 BRAM (기울기 & 절편)
    // ----------------------------------------------------------------
    (* rom_style = "block" *) logic signed [15:0] lut_slope [0:1023];
    (* rom_style = "block" *) logic signed [31:0] lut_inter [0:1023]; // 32비트로 확장!

    initial begin
        // 파이썬으로 만든 1024분할 컨닝페이퍼 2장 로드!
        $readmemh("rmsnorm_slope.mem", lut_slope);
        $readmemh("rmsnorm_inter.mem", lut_inter);
    end

    // ----------------------------------------------------------------
    // 3. 파이프라인 1단계 (BRAM Read & 데이터 지연)
    // ----------------------------------------------------------------
    logic signed [15:0] reg_a;          // 기울기 레지스터 (16bit)
    logic signed [31:0] reg_b;          // 절편 레지스터 (32bit 수정 완료!)
    logic        [21:0] reg_frac_x;     // 소수점 찌꺼기 지연용 레지스터 (22bit)
    logic               reg_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            reg_a <= 0;
            reg_b <= 0;
            reg_frac_x <= 0;
            reg_valid <= 0;
        end else if (valid_in) begin
            reg_a <= lut_slope[segment_idx]; // BRAM 읽기 (1클럭 소요)
            reg_b <= lut_inter[segment_idx]; // BRAM 읽기
            reg_frac_x <= fractional_x;      // BRAM 읽는 동안 타이밍 맞추기 위해 x도 지연!
            reg_valid <= 1'b1;
        end else begin
            reg_valid <= 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // 4. 파이프라인 2단계 (DSP48E2 연산: y = ax + b)
    // ----------------------------------------------------------------
    // DSP48E2는 27x18 곱셈을 지원. a(16비트) * x(23비트 부호확장) 완벽 매칭!
    // 곱셈과 덧셈을 클럭 상관없이 '전선(Wire)'으로 즉시 계산!
    logic signed [47:0] dsp_mult_comb;
    assign dsp_mult_comb = (reg_a * $signed({1'b0, reg_frac_x})) + reg_b; 
    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            o_inv_sqrt <= 0;
            valid_out  <= 0;
        end else if (reg_valid) begin
            // 이미 계산이 끝난 전선(comb)의 값을 안전하게 낚아챔!
            o_inv_sqrt <= dsp_mult_comb[30:15]; 
            valid_out  <= 1'b1;
        end else begin
            valid_out  <= 1'b0;
        end
    end

endmodule