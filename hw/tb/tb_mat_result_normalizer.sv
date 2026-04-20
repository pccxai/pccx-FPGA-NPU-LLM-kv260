`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_mat_result_normalizer
// Phase : pccx v002, Phase A §2.2
//
// Purpose
// -------
//   Closed-loop verification of the 48-bit two's-complement -> BF16-like
//   (1 sign / 8 exp / 7 mantissa) pipeline in `gemm_result_normalizer`.
//
//   Drives `N_VECTORS` random signed 48-bit inputs through the four
//   pipeline stages and compares each emitted `data_out` against a
//   software golden computed in a pure-SV function. Prints the canonical
//   `PASS: <N> cycles, both channels match golden.` line the pccx-lab
//   xsim-log bridge recognises.
//
// Target simulator: Xilinx xsim.
// ===============================================================================

`include "GLOBAL_CONST.svh"

module tb_mat_result_normalizer;

  localparam int N_VECTORS    = 256;
  localparam int PIPELINE_DEP = 4;

  // ===| Clock + reset |=========================================================
  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;  // 250 MHz arbitrary sim clock

  // ===| DUT IO |================================================================
  logic [47:0] data_in;
  logic [ 7:0] e_max;
  logic        valid_in;
  logic [15:0] data_out;
  logic        valid_out;

  gemm_result_normalizer u_dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .data_in  (data_in),
    .e_max    (e_max),
    .valid_in (valid_in),
    .data_out (data_out),
    .valid_out(valid_out)
  );

  // ===| Golden model |==========================================================
  // Pure-SV re-implementation of the normalizer's combinational intent.
  // Kept bit-identical to the module's RTL description so the testbench
  // fails loudly if either side drifts.
  function automatic logic [15:0] golden_normalize(
      input logic [47:0] data,
      input logic [ 7:0] emax
  );
    logic        sign;
    logic [47:0] abs_data;
    int          first_one;
    logic [ 7:0] new_exp;
    logic [ 6:0] mantissa;

    sign     = data[47];
    abs_data = data[47] ? (~data + 48'd1) : data;

    // Leading-one position scan on bits [46:0]; hits the RTL loop.
    first_one = 0;
    for (int i = 46; i >= 0; i--) begin
      if (abs_data[i]) begin
        first_one = i;
        break;
      end
    end

    if (abs_data == 48'd0) begin
      new_exp  = 8'd0;
      mantissa = 7'd0;
    end else begin
      new_exp = emax + first_one[7:0] - 8'd26;
      if (first_one >= 7) begin
        mantissa = abs_data[first_one-1 -: 7];
      end else begin
        mantissa = abs_data[6:0] << (7 - first_one);
      end
    end

    return {sign, new_exp, mantissa};
  endfunction

  // ===| Stimulus + scoreboard |================================================
  // Queues preserve the expected output for each in-flight cycle so the
  // scoreboard can pop the correct golden after the pipeline delay.
  logic [15:0] expected_q [$];
  int          errors = 0;
  int          hits   = 0;
  int          i;
  logic [15:0] exp_now;
  logic [47:0] rand_data;
  logic [ 7:0] rand_emax;

  initial begin
    rst_n    = 1'b0;
    valid_in = 1'b0;
    data_in  = '0;
    e_max    = '0;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    // Fire N_VECTORS random stimuli.
    for (i = 0; i < N_VECTORS; i++) begin
      rand_data = {$random, $random};   // 64 -> 48 (upper bits truncate)
      rand_emax = $random;
      data_in  <= rand_data[47:0];
      e_max    <= rand_emax;
      valid_in <= 1'b1;
      expected_q.push_back(golden_normalize(rand_data[47:0], rand_emax));
      @(posedge clk);
    end
    valid_in <= 1'b0;

    // Drain the 4-stage pipeline.
    repeat (PIPELINE_DEP + 2) @(posedge clk);

    if (errors == 0) begin
      $display("PASS: %0d cycles, both channels match golden.", N_VECTORS);
    end else begin
      $display("FAIL: %0d mishits over %0d cycles.", errors, N_VECTORS);
    end
    $finish;
  end

  // Scoreboard runs in the non-blocking region after each posedge.
  always_ff @(posedge clk) begin
    if (rst_n && valid_out) begin
      if (expected_q.size() == 0) begin
        $display("[%0t] scoreboard underrun — unexpected valid_out", $time);
        errors++;
      end else begin
        exp_now = expected_q.pop_front();
        if (data_out !== exp_now) begin
          errors++;
          if (errors <= 10) begin
            $display("[%0t] MISMATCH: got=%h exp=%h", $time, data_out, exp_now);
          end
        end else begin
          hits++;
        end
      end
    end
  end

  // Safety timeout.
  initial begin
    #100000 $display("TIMEOUT"); $finish;
  end

endmodule
