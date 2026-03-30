`timescale 1ns / 1ps


`include "Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"
// weight size = 4bit
// feature_map size =  bf16
module multiplier_shift_BF16_INT4(
    parameter   in_weight_size = `INT4,
    parameter   in_fmap_size = `BF16,
    parameter   in_fmap_e_size = `BF16_EXP,
    parameter   in_fmap_m_size = `BF16_MANTISSA
)(
    input logic [IN_WEIGHT_SIZE - 1:0]     IN_weight,
    input logic [in_fmap_size - 1:0]    IN_feature_map,
    //input IN_IS_LAST,
    input logic i_valid,


    //output OUT_IS_LAST,
    output logic OUT_sign,
    output logic [6:0]    OUT_EXPONENT,
    output logic [7:0]    OUT_MANTISSA
    );


    logic SIGN;

    assign SIGN = IN_weight[3] ^ IN_feature_map[15];


// ===| [+/-|Exp|M|EVEN]delay line |===========================================================
// ===| mantissa & sign \ 4clk(total 5-clk) |=================
    logic [in_fmap_m_size - 1:0] MANTISSA_DELAY_LINE[0:3];
    logic [in_fmap_e_size - 1:0] EXP_DELAY_LINE[0:3];
    logic       SIGN_DELAY_LINE[0:3];
    logic       IS_EVEN_DELAY_LINE[0:3];

    logic DELAY_IS_EVEN;
    assign DELAY_IS_EVEN = IS_EVEN_DELAY_LINE[3];

    logic [in_fmap_m_size:0] DELAY_MANTISSA;
    assign DELAY_MANTISSA = {1'b1, MANTISSA_DELAY_LINE[3]};

    logic       DELAY_SIGN;
    assign DELAY_SIGN = SIGN_DELAY_LINE[3];

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            for(int i = 0; i < 4; i++) begin
                MANTISSA_DELAY_LINE[i] <= '0;
                SIGN_DELAY_LINE[i] <= '0;
                EXP_DELAY_LINE[i] <= '0;
            end
            DELAY_EVEN_Q1 <= 1'b0;
            DELAY_EVEN_Q2 <= 1'b0;
            DELAY_EVEN_Q3 <= 1'b0;

        end else begin
            MANTISSA_DELAY_LINE[0] <= IN_feature_map[14:7];
            SIGN_DELAY_LINE[0] <= IN_feature_map[15:0];
            EXP_DELAY_LINE[0] <= IN_feature_map[14:7];
            DELAY_IS_EVEN[0] <= (IN_weight[0] == 0) ? 1'b1 : 1'b0;

            for(int i = 0; i < 3; i++) begin
                MANTISSA_DELAY_LINE[i+1] <= MANTISSA_DELAY_LINE[i];
                SIGN_DELAY_LINE[i+1] <= SIGN_DELAY_LINE[i];
                EXP_DELAY_LINE[i+1] <= EXP_DELAY_LINE[i];
                DELAY_IS_EVEN[i+1] <= DELAY_IS_EVEN[i];
            end
        end
    end
// ===| [+/-|Exp|M|EVEN]delay line |===========================================================
    logic [IN_FMAP_M_SIZE + 1:0] temp_UN_SAFE_MANTISSA_s1;
    logic [IN_FMAP_M_SIZE + 1:0] temp_UN_SAFE_MANTISSA_s2;
    logic [IN_FMAP_M_SIZE + 1:0] temp_UN_SAFE_MANTISSA;
    logic [IN_FMAP_M_SIZE + 1:0] IN_UN_SAFE_MANTISSA;

    logic [2:0] weight_to_exp;
    logic [in_fmap_e_size:0] UN_SAFE_EXP;
    logic [in_fmap_e_size:0] UN_SAFE_EXP_Q2;
    logic [in_fmap_e_size - 1:0] SAFE_EXP;

    logic valid_q1;
    logic valid_q2;

    logic IS_NUM_SIX;
    assign IS_NUM_SIX = (IN_weight == 6 || IN_weight == -6) ? 1'b1 : 1'b0;


    logic goto_default_lane;
    logic [1:0] default_lane_weight;

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            next_step <= 1'b0;
            goto_default_lane <= 1'b0;
        end else begin
            if(i_valid) begin
                // delay line idx 0
                weight_to_exp <= (IN_weight[3] == `IS_NEGATIVE_NUMBER) ? ~IN_weight[2:1] : IN_weight[2:1] - 1;

                //check if weight is -1,0,1
                if (IN_weight inside {-1, 0, 1}) begin
                    //4클럭뒤에 내보내기만 하면 ok
                    goto_default_lane <= i_valid;
                    default_lane_weight <= IN_weight[1:0];
                end else begin
                    valid_q1 <= i_valid;
                end

                temp_UN_SAFE_MANTISSA_s1 <= (IN_weight[1]) ? (MANTISSA_DELAY_LINE[0] >> 1) : 0;
                temp_UN_SAFE_MANTISSA_s2 <= (DELAY_MANTISSA[0] >> 2);
            end

            if(valid_q1) begin
                // delay line idx 1
                valid_q2 <= valid_q1;

                if(DELAY_IS_EVEN[1]) begin
                    UN_SAFE_EXP <= EXP_DELAY_LINE[1] + weight_to_exp + 1 - IS_NUM_SIX;

                    // [6] = M + (M>>1)
                    temp_UN_SAFE_MANTISSA <= temp_UN_SAFE_MANTISSA_s1 + 0;
                end else if(weight_to_exp[1]) begin
                    // [5,7] = M + (M >> 1), M + (M>>1) + (M<<2) [! s1 <= (IN_weight[1])]
                    UN_SAFE_EXP <= EXP_DELAY_LINE[1] + 2;
                    temp_UN_SAFE_MANTISSA <= temp_UN_SAFE_MANTISSA_s1 + temp_UN_SAFE_MANTISSA_s2;
                end else begin
                    // [3] = M + (M >> 1)
                    UN_SAFE_EXP <= EXP_DELAY_LINE[1] + 1;
                    temp_UN_SAFE_MANTISSA <= temp_UN_SAFE_MANTISSA_s1 + 0;
                end

            end

            if(valid_q2) begin
                // delay line idx 2
                // prevent overflow
                UN_SAFE_EXP_Q2 <= UN_SAFE_EXP;

                IN_UN_SAFE_MANTISSA <= DELAY_MANTISSA[2] + temp_UN_SAFE_MANTISSA;
                if(DELAY_IS_EVEN[2]) begin

                end begin

                end

            end
        end
    end



    // ===| branch |====================================
    logic is_out_vaild;
    logic OUT_branch_sign;
    logic OUT_branch_EXPONENT;
    logic OUT_branch_MANTISSA;

    multiplier_shift_default_lane #(
        .in_fmap_e_size(in_fmap_e_size),
        .in_fmap_m_size(in_fmap_m_size)
    ) multiplier_shift (
        .clk(clk),
        .rst_n(rst_n),
        .default_lane_weight(default_lane_weight),
        .IN_sign(SIGN_DELAY_LINE[1]),
        .IN_exp(EXP_DELAY_LINE[1]),
        .IN_mantissa(MANTISSA_DELAY_LINE[1]),
        .goto_default_lane(goto_default_lane),
        .is_out_vaild(is_out_vaild),
        .OUT_sign(OUT_branch_sign),
        .OUT_EXPONENT(OUT_branch_EXPONENT),
        .OUT_MANTISSA(OUT_branch_MANTISSA)
    );

    multiplier_shift_even_lane #(
        .in_fmap_e_size(in_fmap_e_size),
        .in_fmap_m_size(in_fmap_m_size)
    ) multiplier_shift (
        .clk(clk),
        .rst_n(rst_n),
        .DELAY_SIGN(DELAY_SIGN),
        .UN_SAFE_EXP_Q2(UN_SAFE_EXP_Q2),
        .DELAY_MANTISSA(DELAY_MANTISSA),
        .DELAY_IS_EVEN(DELAY_IS_EVEN),
        .is_out_vaild(is_out_vaild),
        .OUT_sign(OUT_branch_sign),
        .OUT_EXPONENT(OUT_branch_EXPONENT),
        .OUT_MANTISSA(OUT_branch_MANTISSA)
    );

    multiplier_shift_odd_lane #(
        .in_fmap_e_size(in_fmap_e_size),
        .in_fmap_m_size(in_fmap_m_size)
    ) multiplier_shift (
        .clk(clk),
        .rst_n(rst_n),
        .DELAY_SIGN(DELAY_SIGN),
        .UN_SAFE_EXP_Q2(UN_SAFE_EXP_Q2),
        .IN_UN_SAFE_MANTISSA(IN_UN_SAFE_MANTISSA),
        .DELAY_IS_EVEN(DELAY_IS_EVEN),
        .is_out_vaild(is_out_vaild),
        .OUT_sign(OUT_branch_sign),
        .OUT_EXPONENT(OUT_branch_EXPONENT),
        .OUT_MANTISSA(OUT_branch_MANTISSA)
    );


    always_ff @(posedge clk) begin
        if(!rst_n) begin
            is_out_vaild <= 1'b0;
        end else begin
            if(is_out_vaild) begin
                OUT_sign <= OUT_branch_sign;
                OUT_EXPONENT <= OUT_branch_EXPONENT;
                OUT_MANTISSA <= OUT_branch_MANTISSA;
            end
        end
    end

    // ===| branch end |====================================

    /*
    ===[ integer * Bfloat ]====================================================
    featureMAP [1][8][7] = [sign][exp][mantissa]

    ===[ case1 ]=====| given that integer is a power of 2 |====================
    bf16's exp + integer

    ===[ case2 ]=====| given that integer is odd |=============================
    bf16's exp + integer
    bf16's mantissa * 1.n
    bf16's mantissa * 1.n  is equls to..  mantissa + (mantissa >> 1,2)
    5 = 2^2 * 1.25
    7 = 2^2 * 1.75

    ==[ case3 ]======| given that integer is even but not a power of 2 |=========
    N'6' = 2^1 * 3 = 2^1 * 2^1 * 1.5
    bf16's exp + 2
    bf16's mantissa * 1.5

    weight is 4 bit signed integer
    weight = [1] [2] [3] [4] bits = [sign][2^2][2^1][parity]
    [4] - the sign of bit
    [3] - 2^2 * (1 or 0)
    [2] - 2^1 * (1 or 0)
    [1] - Indicates parity(odd/even)

    bf16's [exp] means 2^([exp])
    so, weight * bf16 == W[3][2] + bf16[exp]

    What if [sign] = 0 ...?
    [All possible combinations]
    -1: |1 1 1| >>> NOT(-1) |0 0 0| + |1| >>> 1 |0 0 1|
    -2: |1 1 0| >>> NOT(-2) |0 0 1| + |1| >>> 2 |0 1 0|
    -3: |1 0 1| >>> NOT(-3) |0 1 0| + |1| >>> 3 |0 1 1|
    -4: |1 0 0| >>> NOT(-4) |0 1 1| + |1| >>> 4 |1 0 0|
    -5: |0 1 1| >>> NOT(-5) |1 0 0| + |1| >>> 5 |1 0 1|
    -6: |0 1 0| >>> NOT(-6) |1 0 1| + |1| >>> 6 |1 1 0|
    -7: |0 0 1| >>> NOT(-7) |1 1 0| + |1| >>> 7 |1 1 1|
    -8: |0 0 0|

    [if we focus on [3][2] bits]
    -1: |S|1 1|P| >>> NOT(-1) |0 0|       >>> 1 |0 0|
    -2: |S|1 1|P| >>> NOT(-2) |0 0| + |1| >>> 2 |0 1|
    -3: |S|1 0|P| >>> NOT(-3) |0 1|       >>> 3 |0 1|
    -4: |S|1 0|P| >>> NOT(-4) |0 1| + |1| >>> 4 |1 0|
    -5: |S|0 1|P| >>> NOT(-5) |1 0|       >>> 5 |1 0|
    -6: |S|0 1|P| >>> NOT(-6) |1 0| + |1| >>> 6 |1 1|
    -7: |S|0 0|P| >>> NOT(-7) |1 1|       >>> 7 |1 1|
    -8: |S|0 0|P|

    */
endmodule
