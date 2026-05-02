`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

// ===| Module: GEMM_systolic_top — 32×32 dual-lane W4A8 systolic engine |========
// Purpose      : Prefill-phase matrix multiply for transformer decode/prefill.
//                Hosts the weight-stationary 32×32 PE array plus its weight
//                unpacker, staggered fmap delay lines, and e_max pipe.
// Spec ref     : pccx v002 §2.2 (Matrix Core), §3.3 (GEMM uop).
// Target       : KV260 @ 400 MHz core; cascade break at row 16 per spec.
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low; i_clear synchronous soft-clear.
// Topology     : weight_dispatcher → systolic_array → V_ACC_out (raw_res_sum).
//                emax_pipe[0..ARRAY_SIZE_H-1][0..TOTAL_LATENCY-1] keeps
//                e_max aligned with raw_res_sum_valid for column normalisers.
// Latency      : SYSTOLIC_TOTAL_LATENCY cycles (input fmap → V_ACC_out).
// Throughput   : 1 PE-row per clock once the array is filled; 1 DSP = 2 MAC
//                via INT4 packing on the A-port (see GEMM_dsp_packer).
// Data widths  : INT4 weight (4b), INT8 activation (8b on B-port — current
//                staggered_fmap_INT8 truncation is a v001→v002 migration
//                placeholder, see TODO inside).
// Handshake    : Weight side — IN_weight_*_valid + IN_weight_*_ready (dual-lane);
//                fmap side — push-only (IN_fmap_broadcast / valid).
// Reset state  : raw_res_sum* = 0, emax_pipe = 0.
// Errors       : none.
// Counters     : none.
// Assertions   : (Stage C) IN_weight_upper_valid and IN_weight_lower_valid
//                must rise in the same cycle for dual-MAC packing;
//                staggered_inst_valid one-hot per row at issue.
// Migration TODO (v002 §2.2): replace BF16 mantissa truncation with INT8
//                fmap arriving directly from PREPROCESS.
// ===============================================================================

module GEMM_systolic_top #(
    parameter weight_lane_cnt      = `DEVICE_HP_PORT_CNT,
    parameter weight_width_per_lane = `HP_SINGLE_WIDTH,
    parameter weight_size          = `INT4_WIDTH,

    // 32 = 128 bit / int4 (4-bit)
    parameter weight_cnt           = `HP_SINGLE_WIDTH / `INT4_WIDTH,

    parameter array_horizontal     = `ARRAY_SIZE_H,
    parameter array_vertical       = `ARRAY_SIZE_V,

    // v001 fmap was BF16 mantissa on DSP A-port. v002 will replace this
    // with INT8 on B-port; the staggered delay still carries the v001
    // width until the PREPROCESS stage is ported.
    parameter dsp_A_port           = `DEVICE_DSP_A_WIDTH,
    parameter IN_fmap_brodcast     = `FIXED_MANT_WIDTH
)(
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // Control & Inst
    input logic global_weight_valid,
    input logic [2:0] global_inst,
    input logic global_inst_valid,

    // Feature Map Broadcast (from SRAM Cache)
    input logic [IN_fmap_brodcast-1:0] IN_fmap_broadcast      [0:`ARRAY_SIZE_H-1],
    input logic                        IN_fmap_broadcast_valid,

    // e_max (from Cache for Normalization alignment)
    input logic [`BF16_EXP_WIDTH-1:0]  IN_cached_emax_out[0:`ARRAY_SIZE_H-1],

    // ===| Weight input lanes |===================================================
    //   HP0 -> upper INT4 channel, HP1 -> lower INT4 channel. Both lanes must
    //   present valid data in the same cycle for the W4A8 dual-MAC pipeline.
    //   Arrays are already unpacked to 32 × INT4 upstream (128 bit / 4 bit = 32).
    input  logic [`INT4_WIDTH-1:0] IN_weight_upper      [0:(`HP_SINGLE_WIDTH/`INT4_WIDTH)-1],
    input  logic                   IN_weight_upper_valid,
    output logic                   IN_weight_upper_ready,
    input  logic [`INT4_WIDTH-1:0] IN_weight_lower      [0:(`HP_SINGLE_WIDTH/`INT4_WIDTH)-1],
    input  logic                   IN_weight_lower_valid,
    output logic                   IN_weight_lower_ready,

    // Output Results (Raw)
    output logic [`DSP48E2_POUT_SIZE-1:0] raw_res_sum      [0:`ARRAY_SIZE_H-1],
    output logic                          raw_res_sum_valid[0:`ARRAY_SIZE_H-1],

    // Delayed e_max for Normalizers
    output logic [`BF16_EXP_WIDTH-1:0] delayed_emax_32[0:`ARRAY_SIZE_H-1]
);

  // ===| Weight Dispatcher (dual-lane pipeline FF) |============================
  logic [weight_size-1:0] weight_upper [0:weight_cnt-1];
  logic [weight_size-1:0] weight_lower [0:weight_cnt-1];
  logic                   weights_ready_for_array;

  GEMM_weight_dispatcher #(
    .weight_size(weight_size),
    .weight_cnt (weight_cnt)
  ) u_weight_unpacker (
      .clk  (clk),
      .rst_n(rst_n),

      .fifo_upper      (IN_weight_upper),
      .fifo_upper_valid(IN_weight_upper_valid),
      .fifo_upper_ready(IN_weight_upper_ready),

      .fifo_lower      (IN_weight_lower),
      .fifo_lower_valid(IN_weight_lower_valid),
      .fifo_lower_ready(IN_weight_lower_ready),

      .weight_upper(weight_upper),
      .weight_lower(weight_lower),
      .weight_valid(weights_ready_for_array)
  );

  // ===| Staggered Delay Line for FMap & Instructions |=======
  logic [dsp_A_port-1:0] staggered_fmap      [0:`ARRAY_SIZE_H-1];
  logic                  staggered_fmap_valid[0:`ARRAY_SIZE_H-1];
  logic [           2:0] staggered_inst      [0:`ARRAY_SIZE_H-1];
  logic                  staggered_inst_valid[0:`ARRAY_SIZE_H-1];

  GEMM_fmap_staggered_dispatch #(
      .fmap_width(IN_fmap_brodcast),
      .array_size(array_vertical),
      .fmap_out_width(dsp_A_port)
  ) u_delay_line (
      .clk(clk),
      .rst_n(rst_n),
      .fmap_in(IN_fmap_broadcast),
      .fmap_valid(IN_fmap_broadcast_valid),
      .global_inst(global_inst),
      .global_inst_valid(global_inst_valid),
      .row_data(staggered_fmap),
      .row_valid(staggered_fmap_valid),
      .row_inst(staggered_inst),
      .row_inst_valid(staggered_inst_valid)
  );

  // ===| Systolic Array Core (The Engine) |=======
  logic [`DSP48E2_POUT_SIZE-1:0] raw_res_seq[0:`ARRAY_SIZE_H-1];

  // TODO(pccx v002 §2.2 follow-up):
  //   * Fmap must arrive as INT8 from the PREPROCESS stage. Today the
  //     staggered_fmap is 27-bit BF16 mantissa (v001 carryover); we
  //     truncate to its low 8 bits as a placeholder.
  logic [7:0] staggered_fmap_INT8 [0:`ARRAY_SIZE_H-1];
  genvar col_idx;
  generate
    for (col_idx = 0; col_idx < `ARRAY_SIZE_H; col_idx++) begin : act_trunc
      assign staggered_fmap_INT8[col_idx] = staggered_fmap[col_idx][7:0];
    end
  endgenerate

  GEMM_systolic_array #(
      .array_horizontal(`ARRAY_SIZE_H),
      .array_vertical  (`ARRAY_SIZE_V),
      .INT4_BITS       (`INT4_WIDTH),
      .INT8_BITS       (8),
      .B_PORT_W        (`DEVICE_DSP_B_WIDTH)
  ) u_compute_core (
      .clk(clk),
      .rst_n(rst_n),
      .i_clear(i_clear),
      .i_weight_valid(global_weight_valid),

      // Horizontal: distinct upper / lower INT4 weight streams.
      .H_in_upper(weight_upper),
      .H_in_lower(weight_lower),

      // Vertical: Feature Map (INT8 truncation placeholder) + Instructions.
      .V_in         (staggered_fmap_INT8),
      .in_valid     (staggered_fmap_valid),
      .inst_in      (staggered_inst),
      .inst_valid_in(staggered_inst_valid),

      .V_out      (raw_res_seq),
      .V_ACC_out  (raw_res_sum),
      .V_ACC_valid(raw_res_sum_valid)
  );

  // ===| e_max Delay Pipe for Normalization alignment |=======
  localparam TOTAL_LATENCY = `SYSTOLIC_TOTAL_LATENCY;
  logic [`BF16_EXP_WIDTH-1:0] emax_pipe[0:`ARRAY_SIZE_H-1][0:TOTAL_LATENCY-1];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
        for (int d = 0; d < TOTAL_LATENCY; d++) begin
          emax_pipe[c][d] <= 0;
        end
      end
    end else begin
      for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
        emax_pipe[c][0] <= IN_cached_emax_out[c];
        for (int d = 1; d < TOTAL_LATENCY; d++) begin
          emax_pipe[c][d] <= emax_pipe[c][d-1];
        end
      end
    end
  end

  always_comb begin
    for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
      delayed_emax_32[c] = emax_pipe[c][TOTAL_LATENCY-1];
    end
  end

endmodule
