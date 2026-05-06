`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"

import isa_pkg::*;
import vec_core_pkg::*;

// =============================================================================
// Testbench: tb_GEMV_lane_mask
//
// Verifies the GEMV lane-mask policy and reduction-valid gating for the masks
// required by issue #36.
// =============================================================================

module DSP48E2 #(
    parameter string USE_SIMD = "ONE48",
    parameter int AREG = 0,
    parameter int BREG = 0,
    parameter int CREG = 0,
    parameter int PREG = 1
) (
    input  logic        CLK,
    input  logic        RSTP,
    input  logic [3:0]  ALUMODE,
    input  logic [4:0]  INMODE,
    input  logic [8:0]  OPMODE,
    input  logic        CEP,
    input  logic [29:0] A,
    input  logic [17:0] B,
    input  logic [47:0] C,
    output logic [47:0] P
);

  always_ff @(posedge CLK) begin
    if (RSTP) begin
      P <= '0;
    end else if (CEP) begin
      P <= {A, B} + C;
    end
  end

endmodule

module tb_GEMV_lane_mask;

  localparam int LaneCnt = VecCoreDefaultCfg.num_gemv_pipeline;
  localparam int WeightW = VecCoreDefaultCfg.weight_width;
  localparam int WeightCnt = VecCoreDefaultCfg.weight_cnt;
  localparam int FmapCnt = VecCoreDefaultCfg.fmap_cache_out_cnt;
  localparam int LutDepth = (1 << WeightW);
  localparam int ResultW = VecCoreDefaultCfg.fixed_mant_width + 3;
  localparam int ReductionLatency = 5;

  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;

  int cycles = 0;
  always @(posedge clk) begin
    if (rst_n) cycles++;
  end

  parallel_lane_t lane_mask;
  logic           activated_lane[0:LaneCnt-1];
  logic           IN_valid;

  logic [WeightW-1:0] weight_A[0:WeightCnt-1];
  logic [WeightW-1:0] weight_B[0:WeightCnt-1];
  logic [WeightW-1:0] weight_C[0:WeightCnt-1];
  logic [WeightW-1:0] weight_D[0:WeightCnt-1];

  logic signed [ResultW-1:0] fmap_lut[0:FmapCnt-1][0:LutDepth-1];
  logic [ResultW-1:0]        reduction_result[0:LaneCnt-1];
  logic [LaneCnt-1:0]        reduction_valid;

  GEMV_lane_mask_decode #(
      .param(VecCoreDefaultCfg)
  ) u_mask_decode (
      .IN_parallel_lane  (lane_mask),
      .OUT_activated_lane(activated_lane)
  );

  GEMV_reduction #(
      .param(VecCoreDefaultCfg)
  ) u_reduction_A (
      .clk                    (clk),
      .rst_n                  (rst_n),
      .IN_is_lane_active      (activated_lane[0]),
      .IN_valid               (IN_valid),
      .IN_fmap_LUT            (fmap_lut),
      .IN_weight              (weight_A),
      .OUT_reduction_result   (reduction_result[0]),
      .OUT_reduction_res_valid(reduction_valid[0])
  );

  GEMV_reduction #(
      .param(VecCoreDefaultCfg)
  ) u_reduction_B (
      .clk                    (clk),
      .rst_n                  (rst_n),
      .IN_is_lane_active      (activated_lane[1]),
      .IN_valid               (IN_valid),
      .IN_fmap_LUT            (fmap_lut),
      .IN_weight              (weight_B),
      .OUT_reduction_result   (reduction_result[1]),
      .OUT_reduction_res_valid(reduction_valid[1])
  );

  GEMV_reduction #(
      .param(VecCoreDefaultCfg)
  ) u_reduction_C (
      .clk                    (clk),
      .rst_n                  (rst_n),
      .IN_is_lane_active      (activated_lane[2]),
      .IN_valid               (IN_valid),
      .IN_fmap_LUT            (fmap_lut),
      .IN_weight              (weight_C),
      .OUT_reduction_result   (reduction_result[2]),
      .OUT_reduction_res_valid(reduction_valid[2])
  );

  GEMV_reduction #(
      .param(VecCoreDefaultCfg)
  ) u_reduction_D (
      .clk                    (clk),
      .rst_n                  (rst_n),
      .IN_is_lane_active      (activated_lane[3]),
      .IN_valid               (IN_valid),
      .IN_fmap_LUT            (fmap_lut),
      .IN_weight              (weight_D),
      .OUT_reduction_result   (reduction_result[3]),
      .OUT_reduction_res_valid(reduction_valid[3])
  );

  int errors = 0;
  int checks = 0;

  function automatic logic [LaneCnt-1:0] pack_activated();
    for (int lane = 0; lane < LaneCnt; lane++) begin
      pack_activated[lane] = activated_lane[lane];
    end
  endfunction

  task automatic expect_lanes(
      input string tag,
      input logic [LaneCnt-1:0] got,
      input logic [LaneCnt-1:0] exp
  );
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%0b exp=%0b", $time, tag, got, exp);
      end
    end
  endtask

  task automatic init_inputs();
    begin
      lane_mask = '0;
      IN_valid  = 1'b0;

      for (int idx = 0; idx < WeightCnt; idx++) begin
        weight_A[idx] = 4'd8;
        weight_B[idx] = 4'd8;
        weight_C[idx] = 4'd8;
        weight_D[idx] = 4'd8;
      end

      for (int fmap_idx = 0; fmap_idx < FmapCnt; fmap_idx++) begin
        for (int weight_idx = 0; weight_idx < LutDepth; weight_idx++) begin
          fmap_lut[fmap_idx][weight_idx] = '0;
        end
      end
    end
  endtask

  task automatic run_mask_case(
      input string tag,
      input parallel_lane_t mask,
      input logic [LaneCnt-1:0] exp_active
  );
    begin
      lane_mask = mask;
      #1;
      expect_lanes($sformatf("%s.decode", tag), pack_activated(), exp_active);

      IN_valid = 1'b1;
      @(posedge clk);
      #1;
      IN_valid = 1'b0;

      repeat (ReductionLatency - 1) @(posedge clk);
      #1;
      expect_lanes($sformatf("%s.reduction_valid", tag), reduction_valid, exp_active);

      @(posedge clk);
      #1;
      expect_lanes($sformatf("%s.reduction_valid_clear", tag), reduction_valid, '0);
    end
  endtask

  initial begin
    rst_n = 1'b0;
    init_inputs();

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;

    run_mask_case("mask_00001", 5'b00001, 4'b0001);
    run_mask_case("mask_00011", 5'b00011, 4'b0011);
    run_mask_case("mask_01111", 5'b01111, 4'b1111);
    run_mask_case("mask_00000", 5'b00000, 4'b1111);

    if (errors == 0) begin
      $display("PASS: %0d cycles, GEMV lane masks honored over %0d checks.", cycles, checks);
    end else begin
      $display("FAIL: %0d GEMV lane-mask mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #100000 $display("FAIL: GEMV lane-mask timeout"); $finish;
  end

endmodule
