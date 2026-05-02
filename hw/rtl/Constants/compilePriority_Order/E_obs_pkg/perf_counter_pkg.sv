`timescale 1ns / 1ps

// ===| Package: perf_counter_pkg — observability counter vocabulary |===========
// Purpose      : Architecture vocabulary for opt-in performance / occupancy
//                counters across pccx v002 boundary modules. Exposes the
//                default-disable parameter, standard counter widths, and a
//                handshake-counter struct so every module that opts in uses
//                the same shape and the same enable knob.
// Spec ref     : Stage D observability plan (KELLER_REFACTOR_NOTES.md §5).
// Compilation  : E — depends on B_device_pkg / C_type_pkg only via int. Place
//                in filelist.f after vec_core_pkg.sv. Import is required only
//                in modules that actually instantiate counters; this package
//                has no compile-time effect on modules that do not import it.
// Naming        : localparam uses PascalCase per parameter-name-style.
//
// Provides
//   Enable knob        : PerfCountersEnableDefault (default 1'b0 — opt-in).
//   Counter widths     : PerfCntCycleWidth, PerfCntHandshakeWidth,
//                        PerfCntDepthWidth, PerfCntDropWidth.
//   Handshake struct   : handshake_counter_t (in_count / out_count /
//                        stall_cycles / busy_cycles).
//   Saturation helper  : sat_inc(...) — saturates instead of wrapping.
//
// Stage C contract (this PR)
//   This package defines the observability vocabulary AND now includes
//   the first parameter-gated internal counter MVP in
//   mem_u_operation_queue (per-channel handshake_counter_t behind
//   EnablePerfCounters). Counters remain opt-in and default-disabled;
//   further Stage D wiring will land as separate isolated commits per
//   the Stage C decisions memo.
//
// Stability
//   - Widths chosen to fit the worst-case at 400 MHz over a few seconds of
//     simulation: 32-bit cycle counts saturate at ~10.7 s of clk activity
//     before wrap, which exceeds every TB run length in hw/sim/.
//   - Width changes are NOT compatible with already-wired counters; bump
//     PerfCounterPkgVersion when tightening or widening any field.
// ===============================================================================

`ifndef PERF_COUNTER_PKG_SV
`define PERF_COUNTER_PKG_SV

package perf_counter_pkg;

  // ===| Package version (for future width/struct migrations) |==================
  localparam int PerfCounterPkgVersion = 1;

  // ===| Default enable knob |===================================================
  // Counters are OPT-IN. Modules that accept counter wiring must declare a
  // parameter such as
  //     parameter logic EnablePerfCounters = perf_counter_pkg::PerfCountersEnableDefault
  // and gate every counter assignment behind that parameter so the synthesis
  // tools can constant-prop the counter logic away when the knob is 0.
  localparam logic PerfCountersEnableDefault = 1'b0;

  // ===| Standard counter widths |===============================================
  // Cycle counters (busy / stall / phase length): 32 bits.
  // At 400 MHz, 2^32 cycles ≈ 10.74 s — comfortably above any TB run length.
  localparam int PerfCntCycleWidth     = 32;

  // Handshake counters (input / output transactions, dropped pushes):
  // 32 bits matches cycle width so a counter pair fits inside two AXI-Lite
  // 32-bit MMIO words when we eventually expose them through AXIL_STAT_OUT.
  localparam int PerfCntHandshakeWidth = 32;

  // Drop / error counters: same 32-bit field. Kept named separately so a
  // future Stage D pass can narrow them to e.g. 16 bits without disturbing
  // the cycle-counter group.
  localparam int PerfCntDropWidth      = 32;

  // FIFO / queue maximum-occupancy counters. 16 bits is enough for any
  // FIFO depth instantiated in this design (deepest is mem_CVO_stream_bridge
  // at 2048 entries → 12 bits of occupancy).
  localparam int PerfCntDepthWidth     = 16;

  // ===| Handshake counter group struct |========================================
  // Shape used by every push/pop interface that opts in. Modules expose this
  // as a single output struct so the parent (or an aggregator) can wire it
  // straight into a future MMIO debug bank without per-counter glue.
  //
  // Field semantics
  //   in_count      : count of accepted input handshakes (valid && ready).
  //   out_count     : count of accepted output handshakes (valid && ready).
  //   stall_cycles  : count of cycles where input had valid=1 but ready=0
  //                   (or, for push-only sinks, where the module was busy
  //                   and could not accept the next beat).
  //   busy_cycles   : count of cycles the module was active in any pipeline
  //                   stage. Useful for "duty cycle = busy / total" metrics.
  typedef struct packed {
    logic [PerfCntHandshakeWidth-1:0] in_count;
    logic [PerfCntHandshakeWidth-1:0] out_count;
    logic [PerfCntCycleWidth-1:0]     stall_cycles;
    logic [PerfCntCycleWidth-1:0]     busy_cycles;
  } handshake_counter_t;

  // ===| Saturating-increment helper |===========================================
  // Returns prev + 1 unless prev is already all-ones, in which case it pins
  // the counter at the maximum value. Avoids the surprising wrap-to-0 that
  // would otherwise mask sustained activity in long simulation runs.
  function automatic logic [PerfCntCycleWidth-1:0]
      sat_inc_cycle(input logic [PerfCntCycleWidth-1:0] prev);
    sat_inc_cycle = (&prev) ? prev : prev + 1'b1;
  endfunction

  function automatic logic [PerfCntHandshakeWidth-1:0]
      sat_inc_handshake(input logic [PerfCntHandshakeWidth-1:0] prev);
    sat_inc_handshake = (&prev) ? prev : prev + 1'b1;
  endfunction

  function automatic logic [PerfCntDepthWidth-1:0]
      sat_inc_depth(input logic [PerfCntDepthWidth-1:0] prev);
    sat_inc_depth = (&prev) ? prev : prev + 1'b1;
  endfunction

endpackage

`endif  // PERF_COUNTER_PKG_SV
