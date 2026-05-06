`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;
import perf_counter_pkg::*;

// ===| Module: mem_u_operation_queue — scheduler↔L2 decoupling FIFOs |==========
// Purpose      : Decouple Global_Scheduler issuing rate from L2 cache
//                controller throughput. Provides two independent FIFO
//                channels (ACP and NPU) so neither blocks the other.
// Spec ref     : pccx v002 §5.4 (op queue), §4.2 (uop semantics).
// Clock        : clk_core @ 400 MHz.
// Reset        : rst_n_core active-low.
// Topology     : 2 × xpm_fifo_sync, depth FifoDepth, width UopWidth=35,
//                BRAM-backed, first-word fall-through, prog_full threshold
//                ProgFullThresh
//                (grace window = FifoDepth - ProgFullThresh entries before
//                back-pressuring the upstream scheduler).
// Uop layout   : acp_uop_t / npu_uop_t are 35-bit packed structs:
//                  {write_en[0], base_addr[16:0], end_addr[16:0]} = 1+17+17 = 35.
// Latency      : FWFT read contract keeps OUT_*_cmd_valid aligned with dout.
// Throughput   : 1 push + 1 pop per channel per cycle (independent channels).
// Handshake    : OUT_*_cmd_valid asserts when (~busy && ~empty); with FWFT the
//                command data is already valid when the consumer samples it.
// Backpressure : OUT_*_cmd_fifo_full asserts at PROG_FULL_THRESH; upstream
//                stops issuing.
// Reset state  : Both FIFOs cleared.
// Counters     : Per-channel handshake_counter_t (acp_perf / npu_perf) wired
//                to perf_counter_pkg::handshake_counter_t. Gated by
//                EnablePerfCounters parameter (default 0 — opt-in). When 0
//                the counter logic constant-props to zero in synthesis.
//                Visibility: TB / waveform via hierarchical reference. No
//                top-level port changes.
//                  in_count    : accepted push (IN_*_rdy && !*_fifo_full).
//                  out_count   : accepted pop  (~IN_*_is_busy && !*_fifo_empty).
//                  stall_cycles: dropped push (IN_*_rdy && *_fifo_full) — i.e.
//                                upstream ignored back-pressure.
//                  busy_cycles : cycles with at least one entry in flight
//                                (~*_fifo_empty).
// Assertions   : (Stage C) sim-only `ifndef SYNTHESIS block at the bottom
//                asserts no silent push-drop on either channel.
// ===============================================================================

module mem_u_operation_queue #(
    parameter logic EnablePerfCounters = perf_counter_pkg::PerfCountersEnableDefault
) (
    input logic clk_core,
    input logic rst_n_core,

    // ===| ACP channel |=========================================================
    input  logic     IN_acp_rdy,
    input  acp_uop_t IN_acp_cmd,
    output acp_uop_t OUT_acp_cmd,
    output logic     OUT_acp_cmd_valid,
    output logic     OUT_acp_cmd_fifo_full,
    input  logic     IN_acp_is_busy,

    // ===| NPU internal channel |================================================
    input  logic     IN_npu_rdy,
    input  npu_uop_t IN_npu_cmd,
    output npu_uop_t OUT_npu_cmd,
    output logic     OUT_npu_cmd_valid,
    output logic     OUT_npu_cmd_fifo_full,
    input  logic     IN_npu_is_busy
);

  localparam int UopWidth = 35;  // 1 + 17 + 17 = write_en + base_addr + end_addr

  // ===| Queue Sizing |===========================================================
  // FifoDepth     : entries per channel (BRAM-backed xpm_fifo_sync).
  // ProgFullThresh: prog_full asserts at this depth and is exposed as
  //                 OUT_*_cmd_fifo_full so the scheduler back-pressures the
  //                 issuer with (FifoDepth - ProgFullThresh) entries of grace.
  localparam int FifoDepth      = 128;
  localparam int ProgFullThresh = 100;

  logic acp_fifo_empty;
  logic acp_fifo_full;
  logic npu_fifo_empty;
  logic npu_fifo_full;

  assign OUT_acp_cmd_fifo_full = acp_fifo_full;
  assign OUT_npu_cmd_fifo_full = npu_fifo_full;

  always_comb begin
    OUT_acp_cmd_valid = ~IN_acp_is_busy & ~acp_fifo_empty;
    OUT_npu_cmd_valid = ~IN_npu_is_busy & ~npu_fifo_empty;
  end

  // ===| ACP FIFO |==============================================================
  xpm_fifo_sync #(
      .FIFO_WRITE_DEPTH  (FifoDepth),
      .WRITE_DATA_WIDTH  (UopWidth),
      .READ_DATA_WIDTH   (UopWidth),
      .FIFO_MEMORY_TYPE  ("block"),
      .READ_MODE         ("fwft"),
      .FULL_RESET_VALUE  (0),
      .PROG_FULL_THRESH  (ProgFullThresh)
  ) u_acp_uop_fifo (
      .sleep    (1'b0),
      .rst      (~rst_n_core),
      .wr_clk   (clk_core),
      .wr_en    (IN_acp_rdy & ~acp_fifo_full),
      .din      (IN_acp_cmd),
      .prog_full(acp_fifo_full),
      .rd_en    (~IN_acp_is_busy & ~acp_fifo_empty),
      .dout     (OUT_acp_cmd),
      .empty    (acp_fifo_empty)
  );

  // ===| NPU FIFO |==============================================================
  xpm_fifo_sync #(
      .FIFO_WRITE_DEPTH  (FifoDepth),
      .WRITE_DATA_WIDTH  (UopWidth),
      .READ_DATA_WIDTH   (UopWidth),
      .FIFO_MEMORY_TYPE  ("block"),
      .READ_MODE         ("fwft"),
      .FULL_RESET_VALUE  (0),
      .PROG_FULL_THRESH  (ProgFullThresh)
  ) u_npu_uop_fifo (
      .sleep    (1'b0),
      .rst      (~rst_n_core),
      .wr_clk   (clk_core),
      .wr_en    (IN_npu_rdy & ~npu_fifo_full),
      .din      (IN_npu_cmd),
      .prog_full(npu_fifo_full),
      .rd_en    (~IN_npu_is_busy & ~npu_fifo_empty),
      .dout     (OUT_npu_cmd),
      .empty    (npu_fifo_empty)
  );

  // ===| Performance counters (opt-in via EnablePerfCounters) |==================
  // Internal-only — no port changes. TB / waveform inspection via
  // hierarchical reference (e.g. `dut.acp_perf.in_count`). When
  // EnablePerfCounters = 0 the always_ff block degenerates to constant 0
  // and synthesis prunes the flops.
  //
  // TODO(i_clear): The mem_dispatcher / mem_u_operation_queue subtree does
  // not currently propagate `i_clear`. These counter flops therefore reset
  // on `rst_n_core` only; a soft-clear pulse would not zero them. A follow-
  // up PR must propagate `i_clear` through the mem subtree before this
  // counter pattern is copied to additional boundary modules. See
  // docs/internal/counter_mvp_notes.md §8 for the gating note.
  handshake_counter_t acp_perf;
  handshake_counter_t npu_perf;

  // Per-cycle event terms. Naming mirrors the .wr_en / .rd_en expressions
  // above so a reader can match a counter to the corresponding FIFO action.
  logic acp_push_fire;
  logic acp_pop_fire;
  logic acp_push_drop;
  logic npu_push_fire;
  logic npu_pop_fire;
  logic npu_push_drop;

  assign acp_push_fire = IN_acp_rdy & ~acp_fifo_full;
  assign acp_pop_fire  = ~IN_acp_is_busy & ~acp_fifo_empty;
  assign acp_push_drop = IN_acp_rdy &  acp_fifo_full;
  assign npu_push_fire = IN_npu_rdy & ~npu_fifo_full;
  assign npu_pop_fire  = ~IN_npu_is_busy & ~npu_fifo_empty;
  assign npu_push_drop = IN_npu_rdy &  npu_fifo_full;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_perf <= '0;
      npu_perf <= '0;
    end else if (EnablePerfCounters) begin
      // ACP channel
      if (acp_push_fire) acp_perf.in_count     <= sat_inc_handshake(acp_perf.in_count);
      if (acp_pop_fire)  acp_perf.out_count    <= sat_inc_handshake(acp_perf.out_count);
      if (acp_push_drop) acp_perf.stall_cycles <= sat_inc_cycle(acp_perf.stall_cycles);
      if (~acp_fifo_empty) acp_perf.busy_cycles <= sat_inc_cycle(acp_perf.busy_cycles);

      // NPU channel
      if (npu_push_fire) npu_perf.in_count     <= sat_inc_handshake(npu_perf.in_count);
      if (npu_pop_fire)  npu_perf.out_count    <= sat_inc_handshake(npu_perf.out_count);
      if (npu_push_drop) npu_perf.stall_cycles <= sat_inc_cycle(npu_perf.stall_cycles);
      if (~npu_fifo_empty) npu_perf.busy_cycles <= sat_inc_cycle(npu_perf.busy_cycles);
    end
  end

  // ===| Assertions (sim-only) |================================================
  // Tier 1 SVA from docs/internal/sva_assertion_candidates.md §2.1: surface
  // the silently-dropped push that the FIFO `wr_en` gating would otherwise
  // hide. Severity is $warning so the simulation does not fail in pathological
  // pre-bring-up tests, but a steady-state production run with these firing
  // means upstream is ignoring OUT_*_cmd_fifo_full back-pressure.
  `ifndef SYNTHESIS
    property p_acp_no_silent_drop;
      @(posedge clk_core) disable iff (!rst_n_core)
        !(IN_acp_rdy && acp_fifo_full);
    endproperty
    a_acp_no_silent_drop : assert property (p_acp_no_silent_drop)
      else $warning("mem_u_operation_queue ACP: push dropped (IN_acp_rdy && fifo_full)");

    property p_npu_no_silent_drop;
      @(posedge clk_core) disable iff (!rst_n_core)
        !(IN_npu_rdy && npu_fifo_full);
    endproperty
    a_npu_no_silent_drop : assert property (p_npu_no_silent_drop)
      else $warning("mem_u_operation_queue NPU: push dropped (IN_npu_rdy && fifo_full)");

    // Counter monotonicity (sat_inc_* cannot wrap). Cheap to add; catches
    // a future regression where the saturating helper gets replaced with a
    // wrapping increment.
    property p_acp_in_monotonic;
      @(posedge clk_core) disable iff (!rst_n_core)
        EnablePerfCounters |-> (acp_perf.in_count >= $past(acp_perf.in_count));
    endproperty
    a_acp_in_monotonic : assert property (p_acp_in_monotonic)
      else $error("mem_u_operation_queue: acp_perf.in_count regressed");

    property p_npu_in_monotonic;
      @(posedge clk_core) disable iff (!rst_n_core)
        EnablePerfCounters |-> (npu_perf.in_count >= $past(npu_perf.in_count));
    endproperty
    a_npu_in_monotonic : assert property (p_npu_in_monotonic)
      else $error("mem_u_operation_queue: npu_perf.in_count regressed");
  `endif

endmodule
