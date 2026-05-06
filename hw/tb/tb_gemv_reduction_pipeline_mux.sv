`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"

import device_pkg::*;
import dtype_pkg::*;
import mem_pkg::*;
import vec_core_pkg::*;
import isa_pkg::*;

// Minimal DSP48E2 add-mode model for the GEMV reduction TB.
module DSP48E2 #(
    parameter string USE_SIMD = "ONE48",
    parameter int AREG = 0,
    parameter int BREG = 0,
    parameter int CREG = 0,
    parameter int PREG = 1
) (
    input  logic        CLK,
    input  logic        RSTP,
    input  logic [ 3:0] ALUMODE,
    input  logic [ 4:0] INMODE,
    input  logic [ 8:0] OPMODE,
    input  logic        CEP,
    input  logic [29:0] A,
    input  logic [17:0] B,
    input  logic [47:0] C,
    output logic [47:0] P
);

  logic signed [47:0] ab_wire;
  logic signed [47:0] c_wire;

  assign ab_wire = $signed({A, B});
  assign c_wire  = $signed(C);

  always_ff @(posedge CLK) begin
    if (RSTP) begin
      P <= '0;
    end else if (CEP) begin
      P <= ab_wire + c_wire;
    end
  end

endmodule

// ===============================================================================
// Testbench: tb_gemv_reduction_pipeline_mux
// Phase : pccx v002 - GEMV reduction and SFU/L2 output mux coverage
//
// Purpose
// -------
//   TB-only coverage for issue #40:
//   - Drives deterministic and pseudo-random INT4-indexed GEMV LUT values
//     through four GEMV_reduction lanes.
//   - Checks active-lane valid timing and bit-exact reduction sums with
//     SystemVerilog golden functions.
//   - Verifies disabled lane masks suppress valid.
//   - White-box checks the CVO_top SFU result mux and the mem_CVO_stream_bridge
//     L2 port-B read/write mux without modifying CVO SFU RTL.
// ===============================================================================

module tb_gemv_reduction_pipeline_mux;

  localparam int LaneCnt = device_pkg::VecPipelineCnt;
  localparam int FmapCnt = mem_pkg::FmapL2CacheOutCnt;
  localparam int WeightCnt = mem_pkg::HpSingleWeightCnt;
  localparam int WeightW = mem_pkg::WeightBitWidth;
  localparam int LutDepth = 1 << WeightW;
  localparam int ResultW = dtype_pkg::FixedMantWidth + 3;
  localparam int ReductionValidLatency = 5;
  localparam int ReductionDataLatency = 6;

  // ===| Clock / Reset |=========================================================
  logic clk;
  logic rst_n;
  logic i_clear;
  int   tb_cycles = 0;

  initial clk = 1'b0;
  always #2 clk = ~clk;
  always @(posedge clk) tb_cycles++;

  // ===| GEMV reduction lanes |==================================================
  logic lane_valid [0:LaneCnt-1];
  logic lane_active[0:LaneCnt-1];

  logic signed [ResultW-1:0] fmap_lut[0:LaneCnt-1][0:FmapCnt-1][0:LutDepth-1];
  logic [WeightW-1:0] weights[0:LaneCnt-1][0:WeightCnt-1];

  logic [ResultW-1:0] reduction_result[0:LaneCnt-1];
  logic reduction_out_valid[0:LaneCnt-1];

  generate
    for (genvar lane = 0; lane < LaneCnt; lane++) begin : gen_reduction_lane
      GEMV_reduction #(
          .param(VecCoreDefaultCfg)
      ) u_reduction (
          .clk  (clk),
          .rst_n(rst_n),

          .IN_is_lane_active(lane_active[lane]),
          .IN_valid(lane_valid[lane]),
          .IN_fmap_LUT(fmap_lut[lane]),
          .IN_weight(weights[lane]),

          .OUT_reduction_result(reduction_result[lane]),
          .OUT_reduction_res_valid(reduction_out_valid[lane])
      );
    end
  endgenerate

  // ===| CVO_top output mux path |==============================================
  cvo_control_uop_t cvo_uop;
  logic             cvo_uop_valid;
  logic             cvo_uop_ready;
  logic [15:0]      cvo_in_data;
  logic             cvo_in_valid;
  logic             cvo_data_ready;
  logic [15:0]      cvo_result;
  logic             cvo_result_valid;
  logic             cvo_result_ready;
  logic [15:0]      cvo_emax;
  logic             cvo_busy;
  logic             cvo_done;
  logic             cvo_accm;

  CVO_top u_cvo_top (
      .clk    (clk),
      .rst_n  (rst_n),
      .i_clear(i_clear),

      .IN_uop       (cvo_uop),
      .IN_uop_valid (cvo_uop_valid),
      .OUT_uop_ready(cvo_uop_ready),

      .IN_data       (cvo_in_data),
      .IN_data_valid (cvo_in_valid),
      .OUT_data_ready(cvo_data_ready),

      .OUT_result      (cvo_result),
      .OUT_result_valid(cvo_result_valid),
      .IN_result_ready (cvo_result_ready),

      .IN_e_max(cvo_emax),

      .OUT_busy(cvo_busy),
      .OUT_done(cvo_done),
      .OUT_accm(cvo_accm)
  );

  // ===| L2 bridge output mux path |============================================
  cvo_control_uop_t bridge_uop;
  logic             bridge_uop_valid;
  logic             bridge_busy;
  logic             bridge_done;
  logic             bridge_l2_we;
  logic [16:0]      bridge_l2_addr;
  logic [127:0]     bridge_l2_wdata;
  logic [127:0]     bridge_l2_rdata;
  logic [15:0]      bridge_cvo_data;
  logic             bridge_cvo_valid;
  logic             bridge_cvo_ready;
  logic [15:0]      bridge_cvo_result;
  logic             bridge_cvo_result_valid;
  logic             bridge_cvo_result_ready;

  mem_CVO_stream_bridge u_l2_bridge (
      .clk              (clk),
      .rst_n            (rst_n),
      .IN_cvo_uop       (bridge_uop),
      .IN_cvo_uop_valid (bridge_uop_valid),
      .OUT_busy         (bridge_busy),
      .OUT_done         (bridge_done),

      .OUT_l2_we   (bridge_l2_we),
      .OUT_l2_addr (bridge_l2_addr),
      .OUT_l2_wdata(bridge_l2_wdata),
      .IN_l2_rdata (bridge_l2_rdata),

      .OUT_cvo_data      (bridge_cvo_data),
      .OUT_cvo_valid     (bridge_cvo_valid),
      .IN_cvo_data_ready (bridge_cvo_ready),

      .IN_cvo_result       (bridge_cvo_result),
      .IN_cvo_result_valid (bridge_cvo_result_valid),
      .OUT_cvo_result_ready(bridge_cvo_result_ready)
  );

  // ===| Scoreboard |============================================================
  int errors = 0;
  int checks = 0;
  int cases = 0;

  function automatic logic [ResultW-1:0] golden_reduce(input int lane);
    logic signed [63:0] sum;
    begin
      sum = '0;
      for (int idx = 0; idx < FmapCnt; idx++) begin
        sum += $signed(fmap_lut[lane][idx][weights[lane][idx]]);
      end
      golden_reduce = sum[ResultW-1:0];
    end
  endfunction

  function automatic logic [15:0] golden_cvo_result_mux(
      input cvo_func_e func,
      input logic [15:0] sfu_result,
      input logic [15:0] cordic_sin,
      input logic [15:0] cordic_cos
  );
    begin
      case (func)
        CVO_SIN: golden_cvo_result_mux = cordic_sin;
        CVO_COS: golden_cvo_result_mux = cordic_cos;
        default: golden_cvo_result_mux = sfu_result;
      endcase
    end
  endfunction

  function automatic logic [16:0] golden_l2_addr(
      input logic [16:0] base,
      input logic [12:0] count
  );
    begin
      golden_l2_addr = 17'(base + 17'(count - 13'd1));
    end
  endfunction

  function automatic logic [127:0] golden_l2_wdata(input logic [127:0] wdata);
    begin
      golden_l2_wdata = wdata;
    end
  endfunction

  task automatic expect_bit(input string tag, input logic got, input logic exp);
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%0b exp=%0b", $time, tag, got, exp);
      end
    end
  endtask

  task automatic expect_bits(input string tag, input logic [ResultW-1:0] got,
                             input logic [ResultW-1:0] exp);
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=0x%0h exp=0x%0h got_s=%0d exp_s=%0d",
                 $time, tag, got, exp, $signed(got), $signed(exp));
      end
    end
  endtask

  task automatic expect16(input string tag, input logic [15:0] got, input logic [15:0] exp);
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=0x%04h exp=0x%04h", $time, tag, got, exp);
      end
    end
  endtask

  task automatic expect17(input string tag, input logic [16:0] got, input logic [16:0] exp);
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%0d exp=%0d", $time, tag, got, exp);
      end
    end
  endtask

  task automatic expect128(input string tag, input logic [127:0] got, input logic [127:0] exp);
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=0x%032h exp=0x%032h", $time, tag, got, exp);
      end
    end
  endtask

  task automatic set_reduction_idle;
    begin
      for (int lane = 0; lane < LaneCnt; lane++) begin
        lane_valid[lane]  = 1'b0;
        lane_active[lane] = 1'b0;
      end
    end
  endtask

  task automatic reset_all;
    begin
      rst_n = 1'b0;
      i_clear = 1'b0;
      set_reduction_idle();

      cvo_uop = '0;
      cvo_uop_valid = 1'b0;
      cvo_in_data = '0;
      cvo_in_valid = 1'b0;
      cvo_result_ready = 1'b1;
      cvo_emax = 16'h3F80;

      bridge_uop = '0;
      bridge_uop_valid = 1'b0;
      bridge_l2_rdata = '0;
      bridge_cvo_ready = 1'b1;
      bridge_cvo_result = '0;
      bridge_cvo_result_valid = 1'b0;

      repeat (4) @(posedge clk);
      rst_n = 1'b1;
      repeat (2) @(posedge clk);
      #1;
    end
  endtask

  task automatic load_reduction_case(input int lane, input int seed);
    int value;
    begin
      for (int idx = 0; idx < FmapCnt; idx++) begin
        for (int lut_idx = 0; lut_idx < LutDepth; lut_idx++) begin
          value = ((idx + 3) * (seed + 5) + (lut_idx * 11) + (seed * 7)) % 127;
          fmap_lut[lane][idx][lut_idx] = ResultW'(value - 63);
        end
        weights[lane][idx] = WeightW'((idx * 5 + seed * 3 + lane) % LutDepth);
      end
    end
  endtask

  task automatic expect_reduction_valids(input string tag, input logic [LaneCnt-1:0] mask);
    begin
      for (int lane = 0; lane < LaneCnt; lane++) begin
        expect_bit($sformatf("%s lane%0d valid", tag, lane),
                   reduction_out_valid[lane], mask[lane]);
      end
    end
  endtask

  task automatic run_reduction_mask_case(
      input string tag,
      input logic [LaneCnt-1:0] mask,
      input int seed
  );
    logic [ResultW-1:0] exp[0:LaneCnt-1];
    begin
      cases++;
      reset_all();

      for (int lane = 0; lane < LaneCnt; lane++) begin
        load_reduction_case(lane, seed + lane * 17);
        exp[lane] = golden_reduce(lane);
      end

      for (int lane = 0; lane < LaneCnt; lane++) begin
        lane_valid[lane]  = 1'b1;
        lane_active[lane] = mask[lane];
      end

      for (int cycle = 1; cycle <= 2; cycle++) begin
        @(posedge clk);
        #1;
        expect_reduction_valids($sformatf("%s pre-valid cycle%0d", tag, cycle), '0);
      end

      set_reduction_idle();

      for (int cycle = 3; cycle <= ReductionDataLatency; cycle++) begin
        @(posedge clk);
        #1;
        if (cycle < ReductionValidLatency) begin
          expect_reduction_valids($sformatf("%s pre-valid cycle%0d", tag, cycle), '0);
        end else begin
          expect_reduction_valids($sformatf("%s valid cycle%0d", tag, cycle), mask);
        end
      end

      for (int lane = 0; lane < LaneCnt; lane++) begin
        if (mask[lane]) begin
          expect_bits($sformatf("%s lane%0d reduction", tag, lane),
                      reduction_result[lane], exp[lane]);
        end
      end

      @(posedge clk);
      #1;
      expect_reduction_valids($sformatf("%s post-valid", tag), '0);
    end
  endtask

  task automatic run_sfu_output_mux_case;
    localparam logic [15:0] SfuResult = 16'h4120;
    localparam logic [15:0] CordicSin = 16'hbeef;
    localparam logic [15:0] CordicCos = 16'hcafe;
    logic [15:0] exp_result;
    begin
      cases++;
      reset_all();

      cvo_uop = '{
          cvo_func : CVO_SCALE,
          src_addr : 17'd0,
          dst_addr : 17'd8,
          length   : 16'd1,
          flags    : '0,
          async    : SYNC_OP
      };
      cvo_uop_valid = 1'b1;
      @(posedge clk);
      #1;
      cvo_uop_valid = 1'b0;

      exp_result = golden_cvo_result_mux(CVO_SCALE, SfuResult, CordicSin, CordicCos);
      force u_cvo_top.sfu_result = SfuResult;
      force u_cvo_top.sfu_result_valid = 1'b1;
      force u_cvo_top.cordic_sin = CordicSin;
      force u_cvo_top.cordic_cos = CordicCos;
      force u_cvo_top.cordic_valid = 1'b1;

      @(posedge clk);
      #1;
      expect16("CVO_top SFU output mux result", cvo_result, exp_result);
      expect_bit("CVO_top SFU output mux valid", cvo_result_valid, 1'b1);

      release u_cvo_top.sfu_result;
      release u_cvo_top.sfu_result_valid;
      release u_cvo_top.cordic_sin;
      release u_cvo_top.cordic_cos;
      release u_cvo_top.cordic_valid;
    end
  endtask

  task automatic run_l2_output_mux_case;
    localparam logic [16:0] ReadBase = 17'd40;
    localparam logic [12:0] ReadCount = 13'd3;
    localparam logic [16:0] WriteBase = 17'd96;
    localparam logic [12:0] WriteCount = 13'd2;
    localparam logic [127:0] WriteData = 128'h0123_4567_89ab_cdef_fedc_ba98_7654_3210;
    begin
      cases++;
      reset_all();

      force u_l2_bridge.state = 2'b01;
      force u_l2_bridge.rd_base = ReadBase;
      force u_l2_bridge.rd_word_cnt = ReadCount;
      force u_l2_bridge.rd_lat_pipe = 3'b001;
      #1;
      expect_bit("mem_CVO_stream_bridge L2 read mux we", bridge_l2_we, 1'b0);
      expect17("mem_CVO_stream_bridge L2 read mux addr",
               bridge_l2_addr, golden_l2_addr(ReadBase, ReadCount));

      release u_l2_bridge.state;
      release u_l2_bridge.rd_base;
      release u_l2_bridge.rd_word_cnt;
      release u_l2_bridge.rd_lat_pipe;

      force u_l2_bridge.state = 2'b10;
      force u_l2_bridge.wr_base = WriteBase;
      force u_l2_bridge.wr_word_cnt = WriteCount;
      force u_l2_bridge.wr_elem_idx = 3'd0;
      force u_l2_bridge.wr_ser_buf = WriteData;
      #1;
      expect_bit("mem_CVO_stream_bridge L2 write mux we", bridge_l2_we, 1'b1);
      expect17("mem_CVO_stream_bridge L2 write mux addr",
               bridge_l2_addr, golden_l2_addr(WriteBase, WriteCount));
      expect128("mem_CVO_stream_bridge L2 write mux data",
                bridge_l2_wdata, golden_l2_wdata(WriteData));

      release u_l2_bridge.state;
      release u_l2_bridge.wr_base;
      release u_l2_bridge.wr_word_cnt;
      release u_l2_bridge.wr_elem_idx;
      release u_l2_bridge.wr_ser_buf;
    end
  endtask

  initial begin
    run_reduction_mask_case("single-lane deterministic", 4'b0001, 3);
    run_reduction_mask_case("two-lane mask", 4'b0011, 19);
    run_reduction_mask_case("four-lane pseudo-random", 4'b1111, 37);
    run_reduction_mask_case("all lanes disabled", 4'b0000, 53);
    run_sfu_output_mux_case();
    run_l2_output_mux_case();

    if (errors == 0) begin
      $display("PASS: %0d cycles, TB_PASSED cases=%0d checks=%0d golden=sv_functions reduction_valid_latency=%0d reduction_data_latency=%0d",
               tb_cycles, cases, checks, ReductionValidLatency, ReductionDataLatency);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks across %0d cases.",
               errors, checks, cases);
    end
    $finish;
  end

  initial begin
    #200000 $display("FAIL: gemv reduction pipeline mux timeout"); $finish;
  end

endmodule
