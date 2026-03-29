`timescale 1ns / 1ps


`include "Vec_Matric_MUL.svh"

// weight size = 4bit
// feature_map size =  bf16
module multiplier_branch(
    parameter   IN_WEIGHT_SIZ = 4,
    parameter   IN_FEATURE_MAP_SIZE = 16
)(
    input logic [IN_WEIGHT_SIZE - 1:0]     IN_weight,
    input logic [IN_FEATURE_MAP_SIZE - 1:0]    IN_feature_map,
    //input IN_IS_LAST,
    input logic i_valid,


    //output OUT_IS_LAST,
    output logic OUT_sign,
    output logic [6:0]    OUT_EXPONENT,
    output logic [7:0]    OUT_MANTISSA
    );


    logic SIGN;

    assign SIGN = IN_weight[3] ^ IN_feature_map[15];


    // ===| delay line |=============================
    // ===| mantissa & sign \ 4clk |=================
    logic [6:0] MANTISSA_DELAY_LINE[0:3];
    logic [7:0] EXP_DELAY_LINE[0:3];
    logic       SIGN_DELAY_LINE[0:3];
    logic       IS_EVEN_DELAY_LINE[0:3];

    //logic DELAY_EVEN_Q1;
    //assign DELAY_EVEN_Q1 = IS_EVEN_DELAY_LINE[1];
    //logic DELAY_EVEN_Q2;
    //logic DELAY_EVEN_Q3;
    //assign DELAY_EVEN_Q2 = IS_EVEN_DELAY_LINE[2];

    logic DELAY_IS_EVEN;
    assign DELAY_IS_EVEN = IS_EVEN_DELAY_LINE[3];

    logic [7:0] DELAY_MANTISSA;
    assign DELAY_MANTISSA = {1'b1,MANTISSA_DELAY_LINE[3]};

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



    logic [2:0] weight_to_exp;
    logic [8:0] UN_SAFE_EXP;
    logic [7:0] SAFE_EXP;
    logic [7:0] UN_SAFE_EXP_Q2;

    logic valid_q1;
    logic valid_q2;

    logic IS_NUM_SIX;
    assign IS_NUM_SIX = (IN_weight == 6 || IN_weight == -6) ? 1'b1 : 1'b0;

    logic goto_even;
    logic goto_odd;
    logic is_even;

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            next_step <= 1'b0;
            goto_even <= 1'b0;
            goto_odd <= 1'b0;
            is_even <= 1'b0;
        end else begin
            if(i_valid) begin
                // delay line idx 0
                weight_to_exp <= (IN_weight[3] == `IS_NEGATIVE_NUMBER) ? ~IN_weight[2:1] : IN_weight[2:1] - 1;
                valid_q1 <= i_valid;

                temp_UN_SAFE_MANTISSA_s1 <= (IN_weight[1]) ? (MANTISSA_DELAY_LINE[0] >> 1) : 0;
                temp_UN_SAFE_MANTISSA_s2 <= (DELAY_MANTISSA[0] >> 2);

                //is_even <= ~IS_NUM_SIX;
            end

            if(valid_q1) begin
                // delay line idx 1
                valid_q2 <= valid_q1;

                if(is_even) begin
                    UN_SAFE_EXP <= EXP_DELAY_LINE[1] + weight_to_exp + 1 - IS_NUM_SIX;

                    //6
                    temp_UN_SAFE_MANTISSA <= temp_UN_SAFE_MANTISSA_s1 + 0;
                end else if(weight_to_exp[1]) begin
                    // 5,7
                    UN_SAFE_EXP <= EXP_DELAY_LINE[1] + 2;
                    temp_UN_SAFE_MANTISSA <= temp_UN_SAFE_MANTISSA_s1 + temp_UN_SAFE_MANTISSA_s2;
                end else begin
                    // 3
                    UN_SAFE_EXP <= EXP_DELAY_LINE[1] + 1;
                    temp_UN_SAFE_MANTISSA <= temp_UN_SAFE_MANTISSA_s1 + 0;
                end

            end

            if(valid_q2) begin
                // delay line idx 2
                // prevent overflow
                UN_SAFE_EXP_Q2 <= UN_SAFE_EXP;

                IN_UN_SAFE_MANTISSA <= DELAY_MANTISSA[2] + temp_UN_SAFE_MANTISSA;
            end
        end
    end


    // even LANE, just delay
    // ===| 2 clk |===
    logic  even_delay_S;
    logic [7:0] even_delay_E;
    //logic [6:0] even_delay_M [0:2];
    logic lane_even_Q1;

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            lane_even_Q1 = 1'b0;
            even_delay_S <= 1'b0;
            even_delay_E <= 1'b0;
            even_delay_M <= 1'b0;
        end else begin
           if(DELAY_IS_EVEN) begin
                even_delay_S <= DELAY_SIGN;
                SAFE_EXP <= (UN_SAFE_EXP_Q2[8]) ? 8'hFF : UN_SAFE_EXP_Q2[7:0];
                even_delay_M <= DELAY_MANTISSA;
                lane_even_Q1 <= DELAY_IS_EVEN;
           end

           if(lane_even_Q1) begin
                OUT_sign     <= even_delay_S;
                OUT_EXPONENT <= SAFE_EXP;
                OUT_MANTISSA <= even_delay_M;
           end
        end
    end


    logic [8:0] temp_UN_SAFE_MANTISSA_s1;
    logic [8:0] temp_UN_SAFE_MANTISSA_s2;
    logic [8:0] temp_UN_SAFE_MANTISSA;
    logic [8:0] IN_UN_SAFE_MANTISSA;

    logic [7:0] temp_SAFE_EXPONENT;
    logic [7:0] SAFE_MANTISSA;
    logic [8:0] UN_SAFE_EXP_Q3;
    logic lane_odd_Q1;

    // odd LANE
    // ===| 2 clk |===
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            next_step <= 1'b0;
            lane_odd_Q1 <= 1'b0;
        end else begin
            if(~DELAY_IS_EVEN) begin
                //SAFE_EXP
                if(IN_UN_SAFE_MANTISSA[8]) begin
                    UN_SAFE_EXP_Q3 <= UN_SAFE_EXP_Q2 + 1;
                    SAFE_MANTISSA <= IN_UN_SAFE_MANTISSA >> 1;
                end
                lane_odd_Q1 <= ~DELAY_IS_EVEN;

            end

            if (lane_odd_Q1) begin
                OUT_sign     <= DELAY_SIGN;
                //safe exp
                OUT_EXPONENT <= (UN_SAFE_EXP_Q2[8]) ? 8'hFF : UN_SAFE_EXP_Q2[7:0];
                OUT_MANTISSA <= SAFE_MANTISSA;
            end

        end
    end

    //Spatial Sorting Network




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
