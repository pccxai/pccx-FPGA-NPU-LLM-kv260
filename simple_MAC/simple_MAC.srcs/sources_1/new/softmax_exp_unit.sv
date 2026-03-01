`timescale 1ns / 1ps

module softmax_exp_unit (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               valid_in,
    input  logic signed [15:0] i_x,        
    
    output logic               valid_out,
    output logic [15:0]        o_exp       
);

    // Q12 포맷: 1.442695 * 4096 = 5909
    localparam logic signed [15:0] LOG2E_Q12 = 16'd5909;

    logic signed [31:0] reg_x_prime; 
    logic               reg_valid_1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            reg_x_prime <= 0;
            reg_valid_1 <= 0;
        end else if (valid_in) begin
            reg_x_prime <= i_x * LOG2E_Q12; 
            reg_valid_1 <= 1'b1;
        end else begin
            reg_valid_1 <= 1'b0;
        end
    end

    // 🔥 [버그 수술 완료] Q12 포맷에 맞춘 완벽한 비트 슬라이싱!
    logic [4:0]  shift_amount;
    logic [9:0]  frac_part;
    
    // 정수부: 32비트 중 상위 20비트 [31:12]. 
    // 음수이므로, 우측 시프트 횟수(+5)로 쓰기 위해 2의 보수를 취함!
    assign shift_amount = ~(reg_x_prime[31:12]) + 1; 
    
    // 소수부: 12비트 소수점 [11:0] 중에서, 1024분할 LUT(10비트)를 쓰니까 [11:2] 픽!
    assign frac_part = reg_x_prime[11:2];      

    (* rom_style = "block" *) logic [15:0] lut_exp_frac [0:1023];

    initial begin
        // 🚨 절대 경로 필수! (네 경로에 맞게 확인)
        $readmemh("softmax_frac.mem", lut_exp_frac);
    end

    logic [15:0] reg_frac_val;
    logic [4:0]  reg_shift_val;
    logic        reg_valid_2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            reg_frac_val  <= 0;
            reg_shift_val <= 0;
            reg_valid_2   <= 0;
        end else if (reg_valid_1) begin
            reg_frac_val  <= lut_exp_frac[frac_part]; 
            reg_shift_val <= shift_amount;                
            reg_valid_2   <= 1'b1;
        end else begin
            reg_valid_2   <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            o_exp     <= 0;
            valid_out <= 0;
        end else if (reg_valid_2) begin
            o_exp     <= reg_frac_val >> reg_shift_val; 
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule