`timescale 1ns / 1ps

`include "GEMV_Vec_Matrix_MUL.svh"
`include "GLOBAL_CONST.svh"

module GEMV_reduction_branch
  import vec_core_pkg::*;
#(
    parameter gemv_cfg_t param = VecCoreDefaultCfg
) (
    input logic clk,
    input logic rst_n,

    input logic IN_weight_valid,
    input logic [param.weight_width - 1:0] IN_weight[0:param.weight_cnt-1],

    input logic fmap_ready,
    input logic [16:0] IN_num_recur,

    input logic IN_activated_lane,
    input logic signed [param.fixed_mant_width+2:0] IN_fmap_LUT [0:param.fmap_cache_out_cnt-1][0:param.weight_width-1],

    output logic [param.fixed_mant_width+2:0] OUT_GEMV_result_vector[0:param.gemv_batch-1],
    output logic OUT_valid
);

  logic [param.fixed_mant_width+2:0] reduction_result_wire;

  logic reduction_res_valid_wire;

  GEMV_reduction #(
      .param(param)
  ) u_GEMV_reduction (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_fmap_LUT(IN_fmap_LUT),
      .IN_valid(IN_weight_valid),

      .IN_is_lane_active(IN_activated_lane),
      .IN_weight(IN_weight),

      .OUT_reduction_result(reduction_result_wire),
      .OUT_reduction_res_valid(reduction_res_valid_wire)
  );

  GEMV_accumulate #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_accumulate (
      .clk  (clk),
      .rst_n(rst_n),

      .IN_reduction_result(reduction_result_wire),
      .init(fmap_ready),

      .IN_valid(reduction_res_valid_wire),
      .IN_num_recur(IN_num_recur),
      .OUT_GEMV_result_vector(OUT_GEMV_result_vector),
      .OUT_acc_valid(OUT_valid)
  );

endmodule
