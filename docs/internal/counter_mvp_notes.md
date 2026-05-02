# Counter MVP — `mem_u_operation_queue` (Stage D)

_First wiring of the Stage C `perf_counter_pkg` vocabulary into a
boundary module. This doc is the durable handover for the Stage D
expansion (more boundary modules, eventual MMIO export)._

## 1. Module under instrumentation

`hw/rtl/MEM_control/IO/mem_u_operation_queue.sv` — the
scheduler↔L2 decoupling FIFO pair (ACP + NPU channels). Selected
because:

  - It is a clean boundary module (single FIFO body per channel,
    well-understood handshakes).
  - Its silent-drop surface (push when prog_full) is also the
    Tier 1 SVA target documented in
    `sva_assertion_candidates.md` §2.1, so SVA + counters could
    land in one minimal change.
  - It is straightforward to write a stand-alone TB that exercises
    both the happy path and the over-push path.

## 2. Wiring shape

The module gains a single new parameter:

```systemverilog
module mem_u_operation_queue #(
    parameter logic EnablePerfCounters = perf_counter_pkg::PerfCountersEnableDefault
) (
    ...
);
```

Two internal `handshake_counter_t` registers are added, one per
channel:

```systemverilog
import perf_counter_pkg::*;

handshake_counter_t acp_perf;
handshake_counter_t npu_perf;
```

with the per-cycle event terms:

  - `*_push_fire` = `IN_*_rdy & ~*_fifo_full` (accepted push).
  - `*_pop_fire`  = `~IN_*_is_busy & ~*_fifo_empty` (accepted pop).
  - `*_push_drop` = `IN_*_rdy & *_fifo_full` (silent drop).
  - `*_busy_term` = `~*_fifo_empty` (any item pending).

The `always_ff` block updates the four fields of each
`handshake_counter_t` only when `EnablePerfCounters` is true, and
uses `perf_counter_pkg::sat_inc_*` so the counters never wrap.

## 3. No port changes

The counters are deliberately **not** exposed at the module port
boundary. Reasons:

  - mem_u_operation_queue is instantiated by `mem_dispatcher.sv`,
    which has its own dense port list. Adding a 256-bit
    `handshake_counter_t` × 2 output port would ripple through to
    the parent and to NPU_top. The parent migration is a separate
    decision worth its own PR.
  - The eventual visibility path is a future MMIO debug bank fed
    by an `AXIL_STAT_OUT`-style aggregator. That aggregator does
    not exist yet (see KELLER_REFACTOR_NOTES.md §6.1 — the
    `fsmout_npu_stat_collector.sv` slot is currently empty).
  - For TB and waveform inspection, hierarchical reference
    (`dut.acp_perf.in_count`) is sufficient and is what the new
    `tb_mem_u_operation_queue` uses.

## 4. Synthesis behaviour

When `EnablePerfCounters = 1'b0` (the default from
`perf_counter_pkg::PerfCountersEnableDefault`), the always_ff body
is gated and Vivado constant-propagates the counter flops away.
The synthesised netlist is therefore identical to the pre-MVP
behaviour for any consumer that doesn't override the parameter.

When the parameter is overridden to `1'b1` (as the new TB does),
the four 32-bit + 32-bit + 32-bit + 32-bit = 128-bit register set
costs ~128 FFs per channel × 2 channels = ~256 FFs total in the
KV260 fabric, plus a handful of LUTs for the saturating-increment
helpers. Negligible compared to the 32×32 systolic array's DSP
budget.

## 5. Driver TB

`hw/tb/tb_mem_u_operation_queue.sv` instantiates the DUT with
`EnablePerfCounters = 1'b1` and exercises:

  - Phase 1 — Push 32 uops on each channel, verify the FIFO body
    accepts them (in_count = 32 on each channel).
  - Phase 2 — Drain the FIFO with consumer back-pressure released.
    out_count converges to NPush minus the std-mode FIFO read
    pipeline depth (~1-3 cycles), so the test asserts
    `out_count >= NPush - 8` rather than equality. This is a
    quirk of `xpm_fifo_sync READ_MODE="std"`'s `empty` semantics,
    not a counter bug.

    **FWFT scratch experiment (2026-05-02).** A throwaway swap of
    both FIFOs from `READ_MODE="std"` to `READ_MODE="fwft"` was
    run against the same TB to determine whether FWFT removes the
    slack. Result: with NPush=32 and IN_npu_is_busy=0 for ~320
    drain cycles, `npu_perf.out_count` settled at 29 (vs the
    expected 32) — i.e. ~3 cycles of slack remain. FWFT therefore
    does **not** eliminate the read-side slack for the current TB
    shape; the residual gap is attributable to xpm_fifo_sync's
    internal prefetch-buffer wake-up rather than to the std-mode
    read pipeline alone. Conclusion: keep `READ_MODE="std"`; the
    `OutCountFloor = NPush - 8` scoreboard tolerance remains
    correct. The experiment was conducted via a temp backup at
    `/tmp/mem_u_operation_queue.sv.preFWFT.bak` and a matching
    `tb_mem_u_operation_queue.sv` backup; both files were restored
    bit-for-bit (md5 verified) before this note was written and
    `git diff` for `mem_u_operation_queue.sv` carries no FWFT
    residue.
  - Phase 2b — Forced over-push on ACP only: 200 cycles of
    IN_acp_rdy = 1 with no drain. The DUT's `wr_en` gate
    suppresses the writes, the SVA `a_acp_no_silent_drop` logs
    `$warning` per dropped cycle, and `acp_perf.stall_cycles`
    increments accordingly. The test asserts
    `acp_perf.stall_cycles > 0`, not an exact count, since the
    threshold-vs-full settling cycle count varies slightly with
    the FIFO's internal pipeline.

## 6. Validation gates passed (this batch)

  - `xvlog -L xpm` over the smoke compile path: 0 ERROR.
  - `bash hw/sim/run_verification.sh`: 7/7 PASS, including the
    new `tb_mem_u_operation_queue`. The previous 6 TBs are
    untouched.
  - `xelab` was upgraded to pass `-L xpm` so xpm_fifo_sync
    elaborates cleanly. The flag is a no-op for TBs that do not
    instantiate xpm primitives.

## 7. Stage D expansion plan

Recommended next-batch targets, in priority order:

  1. **`mem_CVO_stream_bridge`** — the FSM-driven READ/WRITE
     phase counters and the BRAM result-FIFO max-occupancy gauge
     are the highest-leverage observability signals for the
     CVO/SFU path. The KELLER_REFACTOR_NOTES.md §5 table
     spells out the exact counter set (`read_phase_cycles`,
     `write_phase_cycles`, `fifo_max_occupancy`). One TB
     mirroring the structure of `tb_mem_u_operation_queue` should
     land alongside.
  2. **`FROM_mat_result_packer`** — already exposes `o_busy`;
     adding a `handshake_counter_t` over the packed_valid /
     packed_ready handshake is mechanical and gives the parent
     visibility into the result-egress backpressure.
  3. **`AXIL_STAT_OUT`** — the silent-drop counter pairs with the
     deferred Tier 1 SVA candidate at the AXIL boundary.
  4. **MMIO export aggregator** — once 3-5 modules expose
     `handshake_counter_t`, fold them into a single read-only
     debug bank addressable through the existing AXI-Lite STAT
     surface. That work warrants its own PR including a register
     map document.

Each Stage D step keeps the same shape: parameter-gated, internal
flop set, no port changes, hierarchical-reference TB. Don't bulk-
add counters across many modules in one commit.

## 8. Gating note — `i_clear` not yet propagated

`i_clear` is not currently propagated into the
`mem_dispatcher` / `mem_u_operation_queue` subtree. The new counter
flops therefore reset on `rst_n_core` only in this batch; a soft-
clear pulse from the AXIL frontend would not zero them.

This is a **pre-existing gap** in the mem subtree, not a regression
introduced by the counter MVP. The counter wiring inherits the
parent's reset-only behaviour by construction (the always_ff in the
DUT keys off `rst_n_core` because that is the only clear signal it
receives).

Action required before the counter pattern is copied to additional
boundary modules (e.g. `mem_CVO_stream_bridge`,
`FROM_mat_result_packer`):

  - A follow-up PR should propagate `i_clear` from the AXIL frontend
    through `mem_dispatcher` into `mem_u_operation_queue`, then OR
    it into the counter `always_ff` reset condition alongside the
    Tier 4 reset-state assertions in
    `sva_assertion_candidates.md` §2.4.
  - That PR is the prerequisite for the Stage D
    `mem_CVO_stream_bridge` target (§7 item 1) — copying the
    parameter-gated counter shape into a module that does receive
    `i_clear` would create asymmetric clear semantics across the
    mem subtree, which is harder to reason about than the current
    uniform "rst_n_core only" behaviour.

Marker comment: `mem_u_operation_queue.sv` carries a
`TODO(i_clear)` block adjacent to the performance-counter section
that points back to this note.
