`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps

`include "vdotm_Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

// weight size = 4bit
// feature_map size =  bf16

// ===| custom type bf25 |=========
// [sign-1bit]
// [exp-8bit]
// [(hidden-1bit)mantissa-16bit]
/*
16bit BF16 input


===| conver to custom type BF25 |===
 Batch in fours -> Find emax & align

[batch=64] 1 group
[batch=32] 2 group
[batch=16] 4 group
[batch=8] 8 group
[batch=4] 16 group
[batch=2] 32 group

===| 64[batch=16] -> 32[batch=8] reduction |===
4:2 group reduction -> DSP[ADD] (use 16)

===| result = A:B + C |===
A:B = | Guard-bit | Mantissa | Guard-bit | Mantissa |
C   = | Guard-bit | Mantissa | Guard-bit | Mantissa |
result = | Guard-bit | Mantissa | Guard-bit | Mantissa |

===| conver custom type BF25 to fp32 |===
Batch in fours -> Find emax & align

===| 32[batch=8] -> 16[batch=4] reduction |===
4:2 group reduction -> DSP[ADD] (use 4)

===| result = A:B + C |===
A:B = | Guard-bit | Mantissa | Guard-bit | Mantissa |
C   = | Guard-bit | Mantissa | Guard-bit | Mantissa |
result = | Guard-bit | Mantissa | Guard-bit | Mantissa |

===| Final step |===
Batch in fours -> Find emax & align

===| 16[batch=4] -> 4[batch=2] reduction |===
4:2 group reduction -> DSP[ADD] (use 4)

final_Result 1 | Guard-bit | Mantissa | Guard-bit | Mantissa |
+
final_Result 2 | Guard-bit | Mantissa | Guard-bit | Mantissa |
= OUTPUT.

===| reduction finish |===
*/
typedef struct packed {
    logic        sign;
    logic [7:0]  exp;
    logic [15:0] mantissa;
} custom_float_t;

function automatic custom_float_t normalize_to_custom_format (
    input logic [23:0] dsp_result,
    input logic [7:0]  group_emax
);
    custom_float_t res;
    logic        is_negative;
    logic [23:0] abs_val;
    logic [4:0]  leading_zero_count;
    logic [23:0] shifted_val;

    is_negative = dsp_result[23];
    abs_val     = is_negative ? (~dsp_result + 24'd1) : dsp_result;

    res.sign    = is_negative;

    if (abs_val == 24'd0) begin
        res.exp      = 8'd0;
        res.mantissa = 16'd0;
        return res;
    end

    // 3. Leading Zero Count Priority Encoder
    leading_zero_count = 5'd0;
    for (int i = 23; i >= 0; i--) begin
        if (abs_val[i] == 1'b1) begin
            leading_zero_count = 23 - i;
            break;
        end
    end

    shifted_val = abs_val << leading_zero_count;

    res.exp = group_emax + (8'd15 - leading_zero_count);

    res.mantissa = shifted_val[21:6];

    return res;
endfunction

//===| Function: find_emax_and_align (Custom 16-bit Mantissa Version) |==========
typedef struct packed {
    logic [7:0]  emax;
    logic [23:0] aligned_val0;
    logic [23:0] aligned_val1;
    logic [23:0] aligned_val2;
    logic [23:0] aligned_val3;
} align_result_t;


function automatic align_result_t find_emax_and_align (
    input logic        s0, s1, s2, s3,        // Sign 1-bit
    input logic [7:0]  e0, e1, e2, e3,        // Exp 8-bit
    input logic [15:0] m0, m1, m2, m3         // Mantissa 16-bit (Implicit 1 포함)
);
    align_result_t res;
    logic [7:0] emax_01, emax_23;
    logic [7:0] diff0, diff1, diff2, diff3;

    // 23bit space = 16bit Mantissa + 7bit spare space(Guard)
    logic [22:0] mag0, mag1, mag2, mag3;

    // | 1 | find Emax
    emax_01  = (e0 > e1) ? e0 : e1;
    emax_23  = (e2 > e3) ? e2 : e3;
    res.emax = (emax_01 > emax_23) ? emax_01 : emax_23;

    // | 2 | calc amount(difference) to shift
    diff0 = res.emax - e0;
    diff1 = res.emax - e1;
    diff2 = res.emax - e2;
    diff3 = res.emax - e3;

    // | 3 | mantissa Alignment
    mag0 = {m0, 7'd0} >> diff0;
    mag1 = {m1, 7'd0} >> diff1;
    mag2 = {m2, 7'd0} >> diff2;
    mag3 = {m3, 7'd0} >> diff3;

    // | 4 | Convert to 24-bit 2's Complement format for DSP48E2
    // 1bit sign + {8+16(hiddenbit)}23bit = total 24bits
    res.aligned_val0 = s0 ? (~{1'b0, mag0} + 24'd1) : {1'b0, mag0};
    res.aligned_val1 = s1 ? (~{1'b0, mag1} + 24'd1) : {1'b0, mag1};
    res.aligned_val2 = s2 ? (~{1'b0, mag2} + 24'd1) : {1'b0, mag2};
    res.aligned_val3 = s3 ? (~{1'b0, mag3} + 24'd1) : {1'b0, mag3};

    return res;
endfunction

typedef struct packed {
    logic [7:0]  emax;
    logic [31:0] val0;
    logic [31:0] val1;
    logic [31:0] val2;
    logic [31:0] val3;
} align32_res_t;

function automatic align32_res_t find_emax_and_align_32 (
    input custom_float_t in0, in1, in2, in3
);
    align32_res_t res;
    logic [7:0] emax_01, emax_23;
    logic [7:0] diff0, diff1, diff2, diff3;

    // 31-bit Magnitude space (16-bit Mantissa + 15-bit Shift Guard)
    logic [30:0] mag0, mag1, mag2, mag3;

    // 1. find Emax
    emax_01  = (in0.exp > in1.exp) ? in0.exp : in1.exp;
    emax_23  = (in2.exp > in3.exp) ? in2.exp : in3.exp;
    res.emax = (emax_01 > emax_23) ? emax_01 : emax_23;

    // 2. Shift calc diff
    diff0 = res.emax - in0.exp;
    diff1 = res.emax - in1.exp;
    diff2 = res.emax - in2.exp;
    diff3 = res.emax - in3.exp;

    // 3. Align
    mag0 = {in0.mantissa, 15'd0} >> diff0;
    mag1 = {in1.mantissa, 15'd0} >> diff1;
    mag2 = {in2.mantissa, 15'd0} >> diff2;
    mag3 = {in3.mantissa, 15'd0} >> diff3;

    // 4. 32-bit 2's Complement convertion (1 Sign + 31 Mag)
    res.val0 = in0.sign ? (~{1'b0, mag0} + 32'd1) : {1'b0, mag0};
    res.val1 = in1.sign ? (~{1'b0, mag1} + 32'd1) : {1'b0, mag1};
    res.val2 = in2.sign ? (~{1'b0, mag2} + 32'd1) : {1'b0, mag2};
    res.val3 = in3.sign ? (~{1'b0, mag3} + 32'd1) : {1'b0, mag3};

    return res;
endfunction

// ===| Final FP32 normailzation function (32-bit 2's Comp -> Standard FP32) |=============
function automatic logic [31:0] normalize_32_to_fp32 (
    input logic [31:0] sum_in,
    input logic [7:0]  group_emax
);
    logic        sign;
    logic [31:0] abs_val;
    logic [4:0]  lzc; // Leading Zero Count (0~31)
    logic [31:0] shifted_val;
    logic [7:0]  final_exp;
    logic [22:0] final_mantissa;

    sign    = sum_in[31];
    abs_val = sign ? (~sum_in + 32'd1) : sum_in;

    if (abs_val == 32'd0) return 32'd0;

    lzc = 5'd0;
    for (int i = 30; i >= 0; i--) begin
        if (abs_val[i] == 1'b1) begin
            lzc = 30 - i;
            break;
        end
    end

    shifted_val = abs_val << lzc;

    // origin data Mantissa(16bit)was at [30:15]
    // FP32's Mantissa is 23bit, so low 7 bits are padded to zeros.
    final_exp = group_emax + (8'd15 - lzc);
    final_mantissa = {shifted_val[29:14], 7'd0};

    return {sign, final_exp, final_mantissa};
endfunction

// input 64 -> multiplier 64
// batch reduction 4:1 = 16 -> 4 -> 1.
module vdotm_BF16_FP32_Reduction(
    parameter   line_length = 32,
    parameter   line_cnt = 64,
    parameter   first_reduction_cnt = 16,
    parameter   second_reduction_cnt = 4,
    parameter   third_reduction_cnt = 1,
    parameter   exp_size = `BF16_EXP,
    parameter   mantissa_size = `BF16_MANTISSA,
    parameter   reduction_rate = 4
)(
    input logic  clk,
    input logic  rst_n,

    input logic  i_valid,
    // 4:1 reduction
    input logic IN_sign[0:reduction_rate - 1],
    input logic [exp_size-1:0] IN_EXPONENT[0:reduction_rate - 1],
    input logic [mantissa_size-1:0] IN_MANTISSA[0:reduction_rate - 1],

    output logic [`FP32 - 1:0] OUT_final_fp32,
    output logic OUT_final_valid
);
    logic [`DSP48E2_AB_WIDTH-1:0]    DSP_IN_AB[0:first_reduction_cnt-1];
    logic [`DSP48E2_C_WIDTH-1:0]     DSP_IN_C [0:first_reduction_cnt-1];

    align_result_t comb_stage1_res [0:first_reduction_cnt-1];


    // find emax and align instantly regardless of clock
    always_comb begin
        for(int i = 0; i < first_reduction_cnt; i++) begin
            comb_stage1_res[i] = find_emax_and_align(
                IN_sign[i*4], IN_sign[i*4+1], IN_sign[i*4+2], IN_sign[i*4+3],
                IN_EXPONENT[i*4], IN_EXPONENT[i*4+1], IN_EXPONENT[i*4+2], IN_EXPONENT[i*4+3],
                IN_MANTISSA[i*4], IN_MANTISSA[i*4+1], IN_MANTISSA[i*4+2], IN_MANTISSA[i*4+3]
            );
        end
    end

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            for(int i = 0; i < line_length; i++) begin
                IN_weight[i] = '0;
                IN_feature_map[i] = '0;
            end
        end else begin
            if(i_valid) begin
                // ===| 2 pack value for DSP48E2 |============================
                // [A:B]{gaurd Mantissa guard Mantissa} + [C]{gaurd Mantissa guard Mantissa}
                // DSP_res = {gaurd Mantissa guard Mantissa}
                for(int i = 0; i < first_reduction_cnt; i++) begin
                    // Non-blocking (<=) packing into DSP input REG!
                    DSP_IN_AB[i] <= {comb_stage1_res[i].aligned_val0, comb_stage1_res[i].aligned_val1};
                    DSP_IN_C[i]  <= {comb_stage1_res[i].aligned_val2, comb_stage1_res[i].aligned_val3};
                end
            end


            // ===| find emax and alignt mantissa |========================

            // [result] & [new_res_in] emax?? -> align M
            // result <= result + new_res_in

        end

        //assign OUT_feature_map =
        //assign OUT_weight =

    end


    logic [47:0] dsp_out_p [0:first_reduction_cnt-1];

    // We need a delayed valid signal to match the DSP's PREG=1 (1 clock delay)
    logic valid_q1;
    always_ff @(posedge clk) begin
        if (!rst_n) valid_q1 <= 1'b0;
        else        valid_q1 <= i_valid; // i_valid was used to latch DSP_IN_AB/C
    end

    // Use generate for-loop to instantiate 16 DSP48E2 primitives
    generate
        for (genvar i = 0; i < first_reduction_cnt; i++) begin : DSP_STAGE1
            DSP48E2 #(
                // SPLIT MAGIC: Divide 48-bit ALU into two 24-bit ALUs
                .USE_SIMD("TWO24"),

                // --- Register Control (0 = Comb, 1 = Registered) ---
                // Inputs are already registered in the previous always_ff block
                .AREG(0), .BREG(0), .CREG(0),
                .PREG(1)                  // Enable P register (1 clock delay for sum)
            ) u_dsp (
                // --- Clock and Reset ---
                .CLK(clk),
                .RSTP(~rst_n),            // Reset for P register

                // --- Operation Mode (Tie-off to fixed values) ---
                .ALUMODE(4'b0000),        // 0000 = ADD
                .INMODE(5'b00000),        // Default A and B routing
                .OPMODE(9'b000_00_11_11), // Z=0, W=0, Y=C, X=A:B  =>  P = A:B + C

                // --- Clock Enables (Tie to high for continuous pipeline) ---
                .CEP(1'b1),               // Enable P register updates

                // --- Data Inputs ---
                .A(DSP_IN_AB[i][47:18]),  // Upper 30 bits go to A port
                .B(DSP_IN_AB[i][17:0]),   // Lower 18 bits go to B port
                .C(DSP_IN_C[i]),          // 48 bits go to C port

                // --- Data Output ---
                .P(dsp_out_p[i])          // 48-bit Result (Contains two 24-bit partial sums)

                // Note: In a real Vivado environment, you must tie off all other
                // unused CE and RST ports to 0 or 1. Omitted here for readability.
            );
        end
    endgenerate

    // ===| 5. Final 24-bit Merge (LUT Addition) |=============================
    // Register array to store the final 24-bit reduced sum for the 16 groups
    logic [23:0] stage1_final_sum [0:first_reduction_cnt-1];
    logic        stage1_valid_out;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            stage1_valid_out <= 1'b0;
            for(int i = 0; i < first_reduction_cnt; i++) begin
                stage1_final_sum[i] <= 24'd0;
            end
        end else begin
            // Pass the valid signal to the next stage
            stage1_valid_out <= valid_q1;

            if (valid_q1) begin
                for(int i = 0; i < first_reduction_cnt; i++) begin
                    // Extract the two 24-bit partial sums from the DSP's P port
                    // and add them together using FPGA LUTs (Fast CARRY8 chain).
                    // This completes the 4 inputs -> 1 output reduction!
                    stage1_final_sum[i] <= dsp_out_p[i][47:24] + dsp_out_p[i][23:0];
                end
            end
        end
    end

    // ===| Stage 1 Emax Delay Line |==========================================
    // DSP calc hold Stage 1's Emax for(2-clk)
    // then, hand over to Normalizer
    logic [7:0] stage1_emax_q1 [0:first_reduction_cnt-1];
    logic [7:0] stage1_emax_q2 [0:first_reduction_cnt-1];

    always_ff @(posedge clk) begin
        if (i_valid) begin
            for(int i=0; i<first_reduction_cnt; i++) begin
                stage1_emax_q1[i] <= comb_stage1_res[i].emax;
            end
        end
        // after DSP_IN_AB is latched next clk
        stage1_emax_q2 <= stage1_emax_q1;
    end

    // ===| Stage 1 Normalization (24-bit -> Custom 25-bit) |==================
    custom_float_t stage1_norm [0:first_reduction_cnt-1];
    logic          stage1_norm_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) stage1_norm_valid <= 1'b0;
        else        stage1_norm_valid <= stage1_valid_out;

        if (stage1_valid_out) begin
            for(int i = 0; i < first_reduction_cnt; i++) begin
                // call normalize_to_custom_format function
                stage1_norm[i] <= normalize_to_custom_format(
                    stage1_final_sum[i],
                    stage1_emax_q2[i]
                );
            end
        end
    end

    // ===| Stage 2: 16 -> 4 Reduction (LUT Addition) |========================
    align32_res_t comb_stage2_res [0:second_reduction_cnt-1];
    logic [31:0]  stage2_sum      [0:second_reduction_cnt-1];
    logic         stage2_sum_valid;

    // [Comb]: [groupsize=4] 16 -> 4 find Emax and align to 32-bit
    always_comb begin
        for(int i = 0; i < second_reduction_cnt; i++) begin
            comb_stage2_res[i] = find_emax_and_align_32(
                stage1_norm[i*4], stage1_norm[i*4+1],
                stage1_norm[i*4+2], stage1_norm[i*4+3]
            );
        end
    end

    // [FF]: 32-bit LUT ADD (A+B+C+D)
    always_ff @(posedge clk) begin
        if (!rst_n) stage2_sum_valid <= 1'b0;
        else        stage2_sum_valid <= stage1_norm_valid;

        if (stage1_norm_valid) begin
            for(int i = 0; i < second_reduction_cnt; i++) begin
                // Sum four 32-bit using LUT
                stage2_sum[i] <= comb_stage2_res[i].val0 + comb_stage2_res[i].val1 +
                                 comb_stage2_res[i].val2 + comb_stage2_res[i].val3;
            end
        end
    end

    // ===| Stage 2 Normalization |============================================
    logic [7:0] stage2_emax_q1 [0:second_reduction_cnt-1];
    always_ff @(posedge clk) begin
        if (stage1_norm_valid) begin
            for(int i=0; i<second_reduction_cnt; i++) begin
                stage2_emax_q1[i] <= comb_stage2_res[i].emax;
            end
        end
    end

    custom_float_t stage2_norm [0:second_reduction_cnt-1];
    logic          stage2_norm_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) stage2_norm_valid <= 1'b0;
        else        stage2_norm_valid <= stage2_sum_valid;

        if (stage2_sum_valid) begin
            for(int i = 0; i < second_reduction_cnt; i++) begin
                // *notice: overload normalize_to_custom_format function to 32-bit input, or
                // cut low bits then, downsize to 24 bits (we assume it as 32-bit)
                stage2_norm[i] <= normalize_to_custom_format(
                    stage2_sum[i][31:8],
                    stage2_emax_q1[i]
                );
            end
        end
    end

    // ===| Stage 3: 4 -> 1 Final Reduction |==================================
    align32_res_t comb_stage3_res;
    logic [31:0]  stage3_sum;
    logic         stage3_sum_valid;

    // [Comb]: align last 4
    always_comb begin
        comb_stage3_res = find_emax_and_align_32(
            stage2_norm[0], stage2_norm[1],
            stage2_norm[2], stage2_norm[3]
        );
    end

    // [FF]: Sum last 4
    always_ff @(posedge clk) begin
        if (!rst_n) stage3_sum_valid <= 1'b0;
        else        stage3_sum_valid <= stage2_norm_valid;

        if (stage2_norm_valid) begin
            stage3_sum <= comb_stage3_res.val0 + comb_stage3_res.val1 +
                          comb_stage3_res.val2 + comb_stage3_res.val3;
        end
    end

    // Stage 3 Emax Delay
    logic [7:0] stage3_emax_q1;
    always_ff @(posedge clk) begin
        if (stage2_norm_valid) stage3_emax_q1 <= comb_stage3_res.emax;
    end

    // ===| Final Output: FP32 Export |=======================================
    logic [31:0] final_fp32_out;
    logic        final_valid_out;

    always_ff @(posedge clk) begin
        if (!rst_n) final_valid_out <= 1'b0;
        else        final_valid_out <= stage3_sum_valid;

        if (stage3_sum_valid) begin
            // 32-bit Sum to FP32 Standard IEEE754 rules
            final_fp32_out <= normalize_32_to_fp32(stage3_sum, stage3_emax_q1);
        end
    end

    assign OUT_final_fp32  = final_fp32_out;
    assign OUT_final_valid = final_valid_out;

endmodule