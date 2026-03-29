`timescale 1ns / 1ps

`include "Vec_Matric_MUL.svh"


// weight size = 4bit
// feature_map size =  bf16

// ===| custom type bf25 |=========
// [sign-1bit]
// [exp-8bit]
// [(hidden-1bit)mantissa-16bit]

module BF16_FP32_Reduction(
    parameter   line_length = 32,
    parameter   exp_size = 8,
    parameter   mantissa_size = 7,
    parameter reduction_rate = 4
)(
    input logic  clk,
    input logic  rst_n,

    input logic  i_valid,
    // 4:1 reduction
    input logic IN_sign[0:3],
    input logic [exp_size-1:0] IN_EXPONENT[0:reduction_rate - 1],
    input logic [mantissa_size-1:0] IN_MANTISSA[0:reduction_rate - 1]
);

    // ===| 1-(1) find emax |=====================
    always_comb begin
        logic [exp_size-1:0] max_a, max_b;
        max_a = (IN_EXPONENT[0] > IN_EXPONENT[1]) ? IN_EXPONENT[0] : IN_EXPONENT[1];
        max_b = (IN_EXPONENT[2] > IN_EXPONENT[3]) ? IN_EXPONENT[2] : IN_EXPONENT[3];
        emax  = (max_a > max_b) ? max_a : max_b;
    end

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            for(int i = 0; i < line_length; i++) begin
                IN_weight[i] = '0;
                IN_feature_map[i] = '0;
            end
        end else begin
            // ===| 1-(2) (find emax) and alignt mantissa |=====================
            if(i_valid) begin
                // Add hidden bit: if exponent is 0, hidden bit is 0 (denormal), else 1
                for (int i=0; i<MAX; ++i) begin
                    assign base_vec = (IN_EXPONENT[i] == 0) ? {m_val, 8'h0} : {m_val, 8'h0};
                end
            end


            // ===| 2 pack value for DSP48E2 |============================
            // [A:B]{gaurd Mantissa guard Mantissa} + [C]{gaurd Mantissa guard Mantissa}
            // DSP_res = {gaurd Mantissa guard Mantissa}

            // ===| 3 LUT carray8 fast chain adder |========================
            // new_res_in = DSP_res[high] + DSP_res[low]

            // ===| find emax and alignt mantissa |========================
            // [result] & [new_res_in] emax?? -> align M
            // result <= result + new_res_in
        end

        assign OUT_feature_map = IN_feature_map[line_length - 1];
        assign OUT_weight = IN_weight[line_length - 1];

    end
endmodule
