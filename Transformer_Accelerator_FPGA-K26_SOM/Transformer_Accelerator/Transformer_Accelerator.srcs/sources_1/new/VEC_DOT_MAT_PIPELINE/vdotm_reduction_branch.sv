`timescale 1ns / 1ps

`include "vdotm_Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

module vdotm_reduction_branch #(
    parameter lane_cnt = 2,
    parameter lane_throughput = 32,
    parameter low = `VDOTM_LOW,
    parameter high = `VDOTM_HIGH
)(
    input logic clk,
    input logic rst_n,
    input logic IN_weight_valid [0:lane_cnt-1],
    input logic [in_weight_size - 1:0] IN_weight[0:lane_throughput -1][0:lane_cnt-1],

    input logic fmap_ready,

    input logic activated_lane [0:lane_cnt-1],
    input logic [`FIXED_MANT_WIDTH+2:0] IN_fmap_LUT_low_wire [0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1];
    input logic [`FIXED_MANT_WIDTH+2:0] IN_fmap_LUT_high_wire [0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE-1];
);

    logic IN_weight_valid;

    always_comb begin
        if(IN_weight_valid && fmap_ready) begin
            IN_weight_valid[low]  <= activated_lane[low];
            IN_weight_valid[high] <= activated_lane[high];
        end else begin
            IN_weight_valid[low]  <= `FALSE;
            IN_weight_valid[high] <= `FALSE;
        end
    end

  logic [`FIXED_MANT_WIDTH+2:0] reduction_high_result_wire;
  logic [`FIXED_MANT_WIDTH+2:0] reduction_low_result_wire;

  // ===| low lane |===========================================
  // if M dot M mode [low        lane single process]
  // count 0~1023 -> 0~1023
  // if V dot M mode [low & high lane dual process]
  vdotm_reduction #(
      .fmap_cache_out_size(`FMAP_CACHE_OUT_SIZE),
      .weight_type(`INT4_RANGE),
      .line_cnt(line_cnt),
  ) vdotm_low_reduction (
      .clk(clk),
      .rst_n(rst_n),
      .IN_fmap_LUT(IN_fmap_LUT_low_wire),
      .IN_valid(IN_weight_valid[low]),
      .IN_weight(IN_weight[low]),
      .OUT_reduction_result(reduction_low_result_wire)
  );

    Vdotm_accumulate #(
      .fmap_line_length(fmap_line_length)
  ) Vdotm_low_res (
      .clk(clk),
      .rst_n(rst_n),
      .IN_reduction_result(reduction_low_result_wire)
  );
  // ===| low lane - end |=======================================

  // ===| high lane |============================================
  vdotm_reduction #(
      .fmap_cache_out_size(`FMAP_CACHE_OUT_SIZE),
      .weight_type(`INT4_RANGE),
      .line_cnt(line_cnt),
  ) vdotm_high_reduction (
      .clk(clk),
      .rst_n(rst_n),
      .IN_fmap_LUT(IN_fmap_LUT_high_wire),
      .IN_valid(IN_weight_valid[high]),
      .IN_weight(IN_weight[high]),
      .OUT_reduction_result(reduction_high_result_wire)
  );

  Vdotm_accumulate #(
      .fmap_line_length(fmap_line_length)
  ) Vdotm_high_res (
      .clk(clk),
      .rst_n(rst_n),
      .IN_reduction_result(reduction_high_result_wire)
  );
  // ===| high lane - end |============================================

endmodule
