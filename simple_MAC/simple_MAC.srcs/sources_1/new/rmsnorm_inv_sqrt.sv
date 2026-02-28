`timescale 1ns / 1ps

module rmsnorm_inv_sqrt (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               valid_in,
    input  logic [15:0]        i_mean_sq,   // x: 제곱의 평균 (양수니까 unsigned)
    
    output logic               valid_out,
    output logic [15:0]        o_inv_sqrt   // y: 1/sqrt(x) 계산 결과
);

    // ----------------------------------------------------------------
    // 1. 구간(Segment) 판별용 인덱스 추출 (상위 6비트 사용 -> 64등분)
    // ----------------------------------------------------------------
    logic [5:0] segment_idx;
    assign segment_idx = i_mean_sq[15:10]; // 16비트 중 대가리 6비트만 똑! 뗌

    // ----------------------------------------------------------------
    // 2. 미니 컨닝페이퍼 (기울기와 y절편 LUT) - 단 64칸짜리 초미니 롬!
    // ----------------------------------------------------------------
    (* rom_style = "distributed" *) logic signed [15:0] lut_slope [0:63]; // 기울기 (음수)
    (* rom_style = "distributed" *) logic        [15:0] lut_inter [0:63]; // y절편 (양수)

    initial begin
        // 파이썬으로 깎아낼 미니 컨닝페이퍼 2장! (나중에 만들 거임)
        $readmemh("rmsnorm_slope.mem", lut_slope);
        $readmemh("rmsnorm_inter.mem", lut_inter);
    end

    // ----------------------------------------------------------------
    // 3. 파이프라인 1단계 (Data Fetch)
    // ----------------------------------------------------------------
    logic signed [15:0] reg_a;
    logic        [15:0] reg_b;
    logic        [15:0] reg_x;
    logic               reg_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_a <= 0; reg_b <= 0; reg_x <= 0; reg_valid <= 0;
        end else if (valid_in) begin
            reg_a <= lut_slope[segment_idx]; // 구간에 맞는 기울기 픽!
            reg_b <= lut_inter[segment_idx]; // 구간에 맞는 y절편 픽!
            reg_x <= i_mean_sq;              // 입력값도 타이밍 맞추기 위해 전달
            reg_valid <= 1'b1;
        end else begin
            reg_valid <= 1'b0;
        end
    end

    // ----------------------------------------------------------------
    // 4. 파이프라인 2단계 (DSP 연산: y = ax + b)
    // ----------------------------------------------------------------
    // DSP48E2 슬라이스 1개가 이 계산을 단 1클럭에 꿀꺽 해치움!
    logic signed [31:0] dsp_mult;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_inv_sqrt <= 0;
            valid_out  <= 0;
        end else if (reg_valid) begin
            // a(기울기) * x + b(절편)
            // (주의: 실제 하드웨어에선 소수점 처리를 위해 Bit-Shift(>>)가 추가됨. 지금은 개념 뼈대!)
            dsp_mult <= (reg_a * $signed({1'b0, reg_x})) + reg_b; 
            
            // 시프트해서 16비트 스케일로 다시 깎아내기 (임시로 10비트 시프트)
            o_inv_sqrt <= dsp_mult[25:10]; 
            valid_out  <= 1'b1;
        end else begin
            valid_out  <= 1'b0;
        end
    end

endmodule