`timescale 1ns / 1ps

`include "vdotm_Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

// weight size = 4bit
// feature_map size =  bf16
module vdotm_top #(
    parameter line_length = 32,
    parameter lane_throughput = 32,
    parameter lane_cnt = 2,

    parameter fmap_line_cnt = 32,
    parameter reduction_rate = 4,  //4:1
    parameter in_weight_size = `INT4,
    parameter in_fmap_size = `BF16,
    parameter in_fmap_e_size = `BF16_EXP,
    parameter in_fmap_m_size = `BF16_MANTISSA
) (
    input logic clk,
    input logic rst_n,

    input logic IN_weight_valid[0:lane_cnt-1],
    input logic [in_weight_size - 1:0] IN_weight[0:lane_throughput -1][0:lane_cnt-1],

    input logic [`FIXED_MANT_WIDTH-1:0] IN_fmap_broadcast      [0:`FMAP_CACHE_OUT_SIZE-1],
    input logic                         IN_fmap_broadcast_valid,

    // e_max (from Cache for Normalization alignment)
    input logic [`BF16_EXP_WIDTH-1:0] IN_cached_emax_out[0:`FMAP_CACHE_OUT_SIZE-1],

    input logic activated_lane[0:lane_cnt-1],

    output logic [`FP32 - 1:0] OUT_final_fp32,
    output logic OUT_final_valid
);

  logic [`FIXED_MANT_WIDTH+2:0] fmap_LUT_low_wire[0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1];
  logic [`FIXED_MANT_WIDTH+2:0] fmap_LUT_low_wire[0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1];

  logic [`FIXED_MANT_WIDTH+2:0] fmap_LUT_high_wire[0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1];
  logic [`FIXED_MANT_WIDTH+2:0] fmap_LUT_low_wire[0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1];

  logic OUT_fmap_ready;

  vdotm_generate_lut #(
      .fmap_cache_out_size(`FMAP_CACHE_OUT_SIZE),
      .weight_type(`INT4_RANGE)
  ) u_vdotm_generate_lut (
      .IN_fmap_broadcast(IN_fmap_broadcast),
      .IN_fmap_broadcast_valid(IN_fmap_broadcast_valid),
      .IN_cached_emax_out(IN_cached_emax_out),

      .OUT_fmap_low_LUT(fmap_LUT_low_wire),
      .OUT_fmap_high_LUT(fmap_LUT_high_wire),
      .OUT_fmap_ready(OUT_fmap_ready)
  );

  //activated_lane = is systolic mode? [1,0]
  //activated_lane = is v dot m mode? [1,1]

  vdotm_reduction_branch #(
      .lane_cnt(lane_cnt),
      .lane_throughput(lane_throughput),
      .low(`VDOTM_LOW),
      .high(`VDOTM_HIGH)
  ) u_vdotm_reduction_branch (
      .clk(clk),
      .rst_n(rst_n),
      .i_valid(IN_weight_valid),
      .IN_weight(IN_weight),
      .fmap_ready(OUT_fmap_ready),
      .activated_lane(activated_lane),
      .IN_fmap_LUT_low_wire(),
      .IN_fmap_LUT_high_wire()
  );



  //todo..
  //.reduction_1_fmap_wire() == fp32(mantissa) * 1
  //v dot M = m's 32row & col = 0~2048

endmodule








/*
//bf16 [sign-1][e-8][mantissa-7]
  logic                      SIGN_Reduction_wire    [0:lane_throughput-1];
  logic [in_fmap_e_size-1:0] EXP_Reduction_wire     [  0:lane_throughput];
  logic [in_fmap_m_size-1:0] MANTISSA_Reduction_wire[  0:lane_throughput];

    // input 64 -> multiplier 64
    // batch reduction 4:1 = 16 -> 4 -> 1.
    genvar i, k;
    generate
        for (i = 0; i < lane_throughput; i++) begin : weight_fifos
            vdotm_shift_BF16_INT4 #(
                .in_weight_size(in_weight_size),
                .in_fmap_size(in_fmap_size),
                .in_fmap_e_size(in_fmap_e_size),
                .in_fmap_m_size(in_fmap_m_size)
            ) vdotm_shift (
                .IN_weight(IN_weight[i]),
                .IN_feature_map(IN_feature_map[i]),
                .i_valid(i_valid[i]),
                .OUT_sign(SIGN_Reduction_wire[i]),
                .OUT_EXPONENT(EXP_Reduction_wire[i]),
                .OUT_MANTISSA(MANTISSA_Reduction_wire[i])
            );

        assign OUT_featureMAP_BF16[i] = {
            SIGN_Reduction_wire[i],
            EXP_Reduction_wire[i],
            MANTISSA_Reduction_wire[i]
        };
        end
    endgenerate


    BF16_FP32_Reduction #(
        .line_length(line_length),
        .lane_throughput(lane_throughput),
        .exp_size(in_fmap_e_size),
        .mantissa_size(in_fmap_m_size),
        .reduction_rate(reduction_rate)
    ) reduction (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(i_valid),
        .IN_sign(SIGN_Reduction_wire),
        .IN_EXPONENT(EXP_Reduction_wire),
        .IN_MANTISSA(MANTISSA_Reduction_wire),
        .OUT_final_fp32(OUT_final_fp32),
        .OUT_final_valid(OUT_final_valid)
    )
    */
