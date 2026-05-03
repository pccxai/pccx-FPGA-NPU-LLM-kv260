// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_CVO_sfu_reduce_sum
// Phase : pccx v002 — CVO SFU local timing path coverage
//
// Purpose
// -------
//   Drives REDUCE_SUM through the SFU ready/valid boundary and checks that the
//   ready-gated accumulator pipeline preserves the scalar BF16 sum.
// ===============================================================================

`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module tb_CVO_sfu_reduce_sum;

  localparam int N_WORDS = 4;

  // ===| Clock + reset |=========================================================
  logic clk;
  logic rst_n;
  logic i_clear;
  initial clk = 1'b0;
  always #2 clk = ~clk;

  // ===| DUT IO |================================================================
  cvo_func_e   IN_func;
  logic [15:0] IN_length;
  cvo_flags_t  IN_flags;
  logic [15:0] IN_data;
  logic        IN_valid;
  logic        OUT_data_ready;
  logic [15:0] OUT_result;
  logic        OUT_result_valid;

  CVO_sfu_unit u_dut (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_clear         (i_clear),
      .IN_func         (IN_func),
      .IN_length       (IN_length),
      .IN_flags        (IN_flags),
      .IN_data         (IN_data),
      .IN_valid        (IN_valid),
      .OUT_data_ready  (OUT_data_ready),
      .OUT_result      (OUT_result),
      .OUT_result_valid(OUT_result_valid)
  );

  // ===| Scoreboard |============================================================
  int errors = 0;
  bit observed_backpressure = 1'b0;

	  function automatic logic [15:0] model_bf16_add(input logic [15:0] a, input logic [15:0] b);
    logic [7:0] ea, eb, elarge;
    logic [7:0] diff;
    logic [8:0] ma, mb;
    logic [9:0] sum;
    logic [7:0] eout;
    logic [6:0] mout;
    logic       sout;
    if (a[14:0] == 0) return b;
    if (b[14:0] == 0) return a;
    ea = a[14:7];
    eb = b[14:7];
    ma = {1'b0, 1'b1, a[6:0]};
    mb = {1'b0, 1'b1, b[6:0]};
    if (ea >= eb) begin
      elarge = ea;
      diff = ea - eb;
      mb = 9'(mb >> diff);
    end else begin
      elarge = eb;
      diff = eb - ea;
      ma = 9'(ma >> diff);
    end
    if (a[15] == b[15]) begin
      sout = a[15];
      sum  = {1'b0, ma} + {1'b0, mb};
    end else if (ma >= mb) begin
      sout = a[15];
      sum  = {1'b0, ma} - {1'b0, mb};
    end else begin
      sout = b[15];
      sum  = {1'b0, mb} - {1'b0, ma};
    end
    if (sum == 0) return 16'd0;
    if (sum[9]) begin
      eout = elarge + 8'd1;
      mout = sum[9:3];
    end else if (sum[8]) begin
      eout = elarge;
      mout = sum[8:2];
    end else begin
      eout = elarge - 8'd1;
      mout = sum[7:1];
    end
	    return {sout, eout, mout};
	  endfunction

  function automatic logic [15:0] model_bf16_mul(input logic [15:0] a, input logic [15:0] b);
    logic        s;
    logic [ 8:0] esum;
    logic [15:0] mp;
    logic [ 7:0] er;
    logic [ 6:0] mr;
    if (a[14:0] == 0 || b[14:0] == 0) return 16'd0;
    s    = a[15] ^ b[15];
    esum = {1'b0, a[14:7]} + {1'b0, b[14:7]};
    mp   = {1'b1, a[6:0]} * {1'b1, b[6:0]};
    if (mp[15]) begin
      er = 8'(esum - 9'd127 + 9'd1);
      mr = mp[14:8];
    end else begin
      er = 8'(esum - 9'd127);
      mr = mp[13:7];
    end
    return {s, er, mr};
  endfunction

  task automatic push_word(input logic [15:0] word);
    int wait_cycles;
    begin
      wait_cycles = 0;
      while (OUT_data_ready !== 1'b1) begin
        observed_backpressure = 1'b1;
        @(posedge clk);
        wait_cycles++;
        if (wait_cycles > 32) begin
          errors++;
          $display("[%0t] ready timeout before input %h", $time, word);
          return;
        end
      end
      @(negedge clk);
      IN_data  = word;
      IN_valid = 1'b1;
      @(negedge clk);
      IN_valid = 1'b0;
      IN_data  = 16'd0;
    end
  endtask

  initial begin
    logic [15:0] expected;
	    bit          reduce_seen;
	    bit          gelu_seen;
	    bit          exp_seen;
    bit          recip_seen;
    logic [15:0] recip_input;
    logic [15:0] recip_seed;
    logic [15:0] recip_xr0;
    logic [15:0] recip_corr;
    logic [15:0] recip_expected;

    rst_n     = 1'b0;
    i_clear   = 1'b0;
    IN_func   = CVO_REDUCE_SUM;
    IN_length = N_WORDS[15:0];
    IN_flags  = '0;
    IN_data   = 16'd0;
    IN_valid  = 1'b0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    push_word(16'h3F80);  // 1.0
    push_word(16'h3F80);  // 1.0
    push_word(16'h3F80);  // 1.0
    push_word(16'h3F80);  // 1.0

    expected = 16'd0;
    for (int i = 0; i < N_WORDS; i++) expected = model_bf16_add(expected, 16'h3F80);

    reduce_seen = 1'b0;
    for (int cycle = 0; cycle < 80; cycle++) begin
      @(posedge clk); #1;
      if (OUT_data_ready !== 1'b1) observed_backpressure = 1'b1;
      if (OUT_result_valid) begin
        reduce_seen = 1'b1;
        if (OUT_result !== expected) begin
          errors++;
          $display("[%0t] reduce sum mismatch: got=%h exp=%h", $time, OUT_result, expected);
        end
        if (!observed_backpressure) begin
          errors++;
          $display("[%0t] reduce sum did not apply expected local backpressure", $time);
        end
        break;
      end
    end

    if (!reduce_seen) begin
      errors++;
      $display("FAIL: timeout waiting for reduce sum output.");
      $finish;
    end

    @(negedge clk);
    IN_func  = CVO_GELU;
    IN_data  = 16'd0;
    IN_valid = 1'b1;
    @(negedge clk);
    IN_valid = 1'b0;
    IN_data  = 16'd0;

    gelu_seen = 1'b0;
    for (int cycle = 0; cycle < 96; cycle++) begin
      @(posedge clk); #1;
      if (OUT_result_valid) begin
        gelu_seen = 1'b1;
        if (OUT_result !== 16'd0) begin
          errors++;
          $display("[%0t] GELU(0) mismatch: got=%h exp=0000", $time, OUT_result);
        end
        break;
      end
    end

    if (!gelu_seen) begin
      errors++;
      $display("FAIL: timeout waiting for GELU(0) output.");
    end

    @(negedge clk);
    IN_func  = CVO_EXP;
    IN_data  = 16'd0;
    IN_valid = 1'b1;
    @(negedge clk);
    IN_valid = 1'b0;
    IN_data  = 16'd0;

    exp_seen = 1'b0;
    for (int cycle = 0; cycle < 96; cycle++) begin
      @(posedge clk); #1;
      if (OUT_result_valid) begin
        exp_seen = 1'b1;
        if (OUT_result !== 16'h3F80) begin
          errors++;
          $display("[%0t] EXP(0) mismatch: got=%h exp=3f80", $time, OUT_result);
        end
        break;
      end
    end

	    if (!exp_seen) begin
	      errors++;
	      $display("FAIL: timeout waiting for EXP(0) output.");
	    end

    recip_input    = 16'h4000;  // 2.0
    recip_seed     = {1'b0,
                      8'(8'd254 - recip_input[14:7]),
                      7'(7'd127 - {1'b0, recip_input[6:1]})};
    recip_xr0      = model_bf16_mul({1'b0, recip_input[14:0]}, recip_seed);
    recip_corr     = model_bf16_add(16'h4000, {1'b1, recip_xr0[14:0]});
    recip_expected = {recip_input[15], model_bf16_mul(recip_seed, recip_corr)[14:0]};

    @(negedge clk);
    IN_func  = CVO_RECIP;
    IN_data  = recip_input;
    IN_valid = 1'b1;
    @(negedge clk);
    IN_valid = 1'b0;
    IN_data  = 16'd0;

    recip_seen = 1'b0;
    for (int cycle = 0; cycle < 96; cycle++) begin
      @(posedge clk); #1;
      if (OUT_result_valid) begin
        recip_seen = 1'b1;
        if (OUT_result !== recip_expected) begin
          errors++;
          $display("[%0t] RECIP smoke mismatch: got=%h exp=%h", $time, OUT_result, recip_expected);
        end
        break;
      end
    end

    if (!recip_seen) begin
      errors++;
      $display("FAIL: timeout waiting for RECIP output.");
    end

	    if (errors == 0) begin
	      $display("PASS: %0d cycles, CVO SFU reduce sum, GELU, EXP, and RECIP smoke match golden.", N_WORDS);
    end else begin
      $display("FAIL: %0d mismatches over CVO SFU smoke.", errors);
    end
    $finish;
  end

  initial begin
    #100000 $display("TIMEOUT"); $finish;
  end

endmodule
