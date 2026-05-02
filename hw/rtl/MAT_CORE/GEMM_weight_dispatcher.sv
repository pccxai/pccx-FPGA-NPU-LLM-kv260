`timescale 1ns / 1ps

// ===| Module: GEMM_weight_dispatcher — dual-lane INT4 weight register stage |==
// Purpose      : Final pipeline flop for the dual-lane (upper/lower) INT4
//                weight bus before it enters the systolic array. Decouples
//                the HP CDC FIFOs from the 32×32 PE grid timing.
// Spec ref     : pccx v002 §2.2.2 (weight stationary fan-out).
// Phase        : pccx v002 (W4A8, 1 DSP = 2 MAC).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// Width        : weight_size (= INT4 = 4) × weight_cnt (= 32).
// Latency      : 1 register stage (input → weight_upper/lower / weight_valid).
// Throughput   : 1 dual-INT4 vector per cycle while both lanes valid.
// Handshake    : fifo_upper_ready / fifo_lower_ready tied to 1'b1 — module
//                is push-only and never stalls. Misaligned valids starve
//                the array of pairs (weight_valid is fifo_upper_valid &
//                fifo_lower_valid).
// Reset state  : weight_upper/lower zeroed; weight_valid = 0.
// Counters     : none.
// Migration    : v001 had a single 32 × INT4 stream; v002 needs two
//                streams (upper + lower) because each DSP MAC processes
//                two weights per cycle (GEMM_dsp_packer pairs them).
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module GEMM_weight_dispatcher #(
  parameter int weight_size = `INT4_WIDTH,                         // 4
  parameter int weight_cnt  = `HP_PORT_SINGLE_WIDTH / `INT4_WIDTH  // 32 = 128/4
) (
  input  logic clk,
  input  logic rst_n,

  // ===| Upper-channel input (e.g. HP0 lane, already unpacked) |================
  input  logic [weight_size-1:0] fifo_upper     [0:weight_cnt-1],
  input  logic                   fifo_upper_valid,
  output logic                   fifo_upper_ready,

  // ===| Lower-channel input (e.g. HP1 lane, already unpacked) |================
  input  logic [weight_size-1:0] fifo_lower     [0:weight_cnt-1],
  input  logic                   fifo_lower_valid,
  output logic                   fifo_lower_ready,

  // ===| Registered outputs to the systolic array |=============================
  output logic [weight_size-1:0] weight_upper [0:weight_cnt-1],
  output logic [weight_size-1:0] weight_lower [0:weight_cnt-1],
  output logic                   weight_valid
);

  // ===| Flow control — always accept while the pipeline is not stalled |=======
  assign fifo_upper_ready = 1'b1;
  assign fifo_lower_ready = 1'b1;

  // ===| Pipeline register stage |==============================================
  //   Fires only when both channels deliver valid data in the same cycle,
  //   which is how the upstream scheduler is supposed to pair them for W4A8
  //   dual-MAC. A misalignment starves the array of valid pairs, so the
  //   valid is an AND, not an OR.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      weight_valid <= 1'b0;
      for (int i = 0; i < weight_cnt; i++) begin
        weight_upper[i] <= '0;
        weight_lower[i] <= '0;
      end
    end else begin
      weight_valid <= fifo_upper_valid & fifo_lower_valid;
      for (int i = 0; i < weight_cnt; i++) begin
        weight_upper[i] <= fifo_upper[i];
        weight_lower[i] <= fifo_lower[i];
      end
    end
  end

endmodule
