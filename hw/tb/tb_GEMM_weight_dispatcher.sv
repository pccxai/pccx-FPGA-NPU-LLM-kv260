`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_GEMM_weight_dispatcher
// Phase : pccx v002, Phase A (W4A8 dual-channel weight staging)
//
// Purpose
// -------
//   Drives 128 random upper/lower INT4 weight frames through the
//   dispatcher and confirms:
//
//     * `weight_valid` is `fifo_upper_valid & fifo_lower_valid`, delayed
//       by exactly one cycle.
//     * Every lane of `weight_upper` / `weight_lower` mirrors its input,
//       again at one cycle latency.
//
//   Prints the canonical `PASS: <N> cycles, ...` line the pccx-lab xsim
//   bridge picks up.
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module tb_GEMM_weight_dispatcher;

  localparam int WEIGHT_SIZE = `INT4_WIDTH;
  localparam int WEIGHT_CNT  = `HP_PORT_SINGLE_WIDTH / `INT4_WIDTH;
  localparam int N_FRAMES    = 128;

  // ===| Clock + reset |=========================================================
  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;  // 250 MHz

  // ===| DUT IO |================================================================
  logic [WEIGHT_SIZE-1:0] fifo_upper [0:WEIGHT_CNT-1];
  logic [WEIGHT_SIZE-1:0] fifo_lower [0:WEIGHT_CNT-1];
  logic fifo_upper_valid, fifo_lower_valid;
  logic fifo_upper_ready, fifo_lower_ready;

  logic [WEIGHT_SIZE-1:0] weight_upper [0:WEIGHT_CNT-1];
  logic [WEIGHT_SIZE-1:0] weight_lower [0:WEIGHT_CNT-1];
  logic                   weight_valid;

  GEMM_weight_dispatcher #(
    .weight_size (WEIGHT_SIZE),
    .weight_cnt  (WEIGHT_CNT)
  ) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .fifo_upper      (fifo_upper),
    .fifo_upper_valid(fifo_upper_valid),
    .fifo_upper_ready(fifo_upper_ready),
    .fifo_lower      (fifo_lower),
    .fifo_lower_valid(fifo_lower_valid),
    .fifo_lower_ready(fifo_lower_ready),
    .weight_upper    (weight_upper),
    .weight_lower    (weight_lower),
    .weight_valid    (weight_valid)
  );

  // ===| Scoreboard |============================================================
  // Shadow pipeline: one-cycle-delayed copy of the drivers so we can
  // compare against the DUT's output after the single flop stage.
  logic [WEIGHT_SIZE-1:0] sb_upper     [0:WEIGHT_CNT-1];
  logic [WEIGHT_SIZE-1:0] sb_lower     [0:WEIGHT_CNT-1];
  logic                   sb_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sb_valid <= 1'b0;
      for (int i = 0; i < WEIGHT_CNT; i++) begin
        sb_upper[i] <= '0;
        sb_lower[i] <= '0;
      end
    end else begin
      sb_valid <= fifo_upper_valid & fifo_lower_valid;
      for (int i = 0; i < WEIGHT_CNT; i++) begin
        sb_upper[i] <= fifo_upper[i];
        sb_lower[i] <= fifo_lower[i];
      end
    end
  end

  // ===| Stimulus |==============================================================
  int errors = 0;
  int i;

  initial begin
    rst_n            = 1'b0;
    fifo_upper_valid = 1'b0;
    fifo_lower_valid = 1'b0;
    for (int k = 0; k < WEIGHT_CNT; k++) begin
      fifo_upper[k] = '0;
      fifo_lower[k] = '0;
    end

    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    // Frame generator: randomise data + valid patterns so the scoreboard
    // covers AND-of-valids edge cases (only-upper / only-lower / both /
    // neither).
    for (i = 0; i < N_FRAMES; i++) begin
      for (int k = 0; k < WEIGHT_CNT; k++) begin
        fifo_upper[k] = $random;
        fifo_lower[k] = $random;
      end
      fifo_upper_valid = $random & 1;
      fifo_lower_valid = $random & 1;
      @(posedge clk);
      #1; // let the flop propagate before sampling

      if (weight_valid !== sb_valid) begin
        errors++;
        if (errors <= 10) begin
          $display("[%0t] frame %0d weight_valid mismatch: got=%b exp=%b",
                   $time, i, weight_valid, sb_valid);
        end
      end
      for (int k = 0; k < WEIGHT_CNT; k++) begin
        if (weight_upper[k] !== sb_upper[k]) begin
          errors++;
          if (errors <= 10) begin
            $display("[%0t] frame %0d upper[%0d] mismatch: got=%h exp=%h",
                     $time, i, k, weight_upper[k], sb_upper[k]);
          end
        end
        if (weight_lower[k] !== sb_lower[k]) begin
          errors++;
          if (errors <= 10) begin
            $display("[%0t] frame %0d lower[%0d] mismatch: got=%h exp=%h",
                     $time, i, k, weight_lower[k], sb_lower[k]);
          end
        end
      end
    end

    if (errors == 0) begin
      $display("PASS: %0d cycles, both channels match golden.", N_FRAMES);
    end else begin
      $display("FAIL: %0d mismatches over %0d cycles.", errors, N_FRAMES);
    end
    $finish;
  end

  initial begin
    #100000 $display("TIMEOUT"); $finish;
  end

endmodule
