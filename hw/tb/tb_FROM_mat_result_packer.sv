`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_FROM_mat_result_packer
// Phase : pccx v002 — MAT_CORE -> MEM egress packing
//
// Purpose
// -------
//   Drives the 32-lane staggered result interface with a known pattern
//   (row_res[i] = 16'hA000 | i) and verifies the packer emits exactly
//   four 128-bit words, each reconstructing eight consecutive 16-bit
//   inputs in the documented little-end-first ordering:
//
//     packed_data @ beat k = {row_res[k*8+7], ..., row_res[k*8]}
//
//   packed_ready follows a deterministic pseudo-random stall pattern so the
//   TB checks both packing order and ready/valid hold behavior.
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module tb_FROM_mat_result_packer;

  localparam int ARRAY_SIZE = 32;
  localparam int N_BEATS    = 4;   // 32 × 16 / 128

  // ===| Clock + reset |=========================================================
  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;

  // ===| DUT IO |================================================================
  logic [`BF16_WIDTH-1:0] row_res      [0:ARRAY_SIZE-1];
  logic                   row_res_valid[0:ARRAY_SIZE-1];
  logic [`AXI_STREAM_WIDTH-1:0] packed_data;
  logic                   packed_valid;
  logic                   packed_ready;
  logic                   o_busy;

  FROM_gemm_result_packer #(
    .ARRAY_SIZE (ARRAY_SIZE)
  ) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .row_res      (row_res),
    .row_res_valid(row_res_valid),
    .packed_data  (packed_data),
    .packed_valid (packed_valid),
    .packed_ready (packed_ready),
    .o_busy       (o_busy)
  );

  // ===| Scoreboard |============================================================
  logic [127:0] expected_beats [0:N_BEATS-1];
  int           beats_seen = 0;
  int           errors     = 0;
  int           stall_cycles = 0;
  int           ready_step = 0;
  logic         drive_ready;

  // Reference packing — must match the DUT's bit-ordering.
  initial begin : build_expected
    for (int k = 0; k < N_BEATS; k++) begin
      logic [127:0] word;
      word = '0;
      for (int j = 0; j < 8; j++) begin
        // Pattern that survives the packing: high nibble 0xA plus lane id.
        word[j*16 +: 16] = 16'hA000 | (k*8 + j);
      end
      expected_beats[k] = word;
    end
  end

  // ===| Stimulus |==============================================================
  int i;
  initial begin
    rst_n        = 1'b0;
    packed_ready = 1'b0;
    drive_ready  = 1'b0;
    for (int k = 0; k < ARRAY_SIZE; k++) begin
      row_res[k]       = '0;
      row_res_valid[k] = 1'b0;
    end

    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    // Broadcast all 32 results in a single cycle.
    @(posedge clk);
    for (int k = 0; k < ARRAY_SIZE; k++) begin
      row_res[k]       = 16'hA000 | k;
      row_res_valid[k] = 1'b1;
    end
    @(posedge clk);
    // Results latched — drop the valids so the FSM reads out the buffer.
    for (int k = 0; k < ARRAY_SIZE; k++) row_res_valid[k] = 1'b0;
    drive_ready = 1'b1;

    // Collect N_BEATS beats with a generous drain window.
    i = 0;
    while (beats_seen < N_BEATS && i < 2000) begin
      @(posedge clk);
      i++;
    end

    drive_ready = 1'b0;
    packed_ready = 1'b0;

    if (beats_seen != N_BEATS) begin
      $display("FAIL: %0d mismatches over %0d cycles.",
               (N_BEATS - beats_seen) + errors, i);
    end else if (stall_cycles == 0) begin
      $display("FAIL: ready/valid stall path was not exercised.");
    end else if (errors == 0) begin
      $display("PASS: %0d beats, packer ready-stall order matches golden.", N_BEATS);
    end else begin
      $display("FAIL: %0d mismatches over %0d cycles.", errors, N_BEATS);
    end
    $finish;
  end

  always @(negedge clk) begin
    if (!rst_n || !drive_ready) begin
      packed_ready = 1'b0;
      ready_step   = 0;
    end else begin
      // Repeatable stall pattern with both single- and multi-cycle bubbles.
      packed_ready = ((ready_step % 5) != 1) && ((ready_step % 7) != 3);
      ready_step++;
    end
  end

  // Capture every valid/ready beat and ensure data stays stable while stalled.
  logic [`AXI_STREAM_WIDTH-1:0] stalled_data = '0;
  logic                         was_stalled  = 1'b0;

  always_ff @(posedge clk) begin
    if (rst_n && packed_valid && packed_ready) begin
      if (beats_seen < N_BEATS) begin
        if (packed_data !== expected_beats[beats_seen]) begin
          errors++;
          $display("[%0t] beat %0d mismatch:\n got=%h\n exp=%h",
                   $time, beats_seen, packed_data, expected_beats[beats_seen]);
        end
        beats_seen++;
      end
      was_stalled <= 1'b0;
    end else if (rst_n && packed_valid && !packed_ready) begin
      stall_cycles++;
      if (was_stalled && packed_data !== stalled_data) begin
        errors++;
        $display("[%0t] stalled data changed: got=%h prev=%h",
                 $time, packed_data, stalled_data);
      end
      stalled_data <= packed_data;
      was_stalled  <= 1'b1;
    end else if (rst_n && !packed_valid) begin
      was_stalled <= 1'b0;
    end
  end

  initial begin
    #100000 $display("TIMEOUT"); $finish;
  end

endmodule
