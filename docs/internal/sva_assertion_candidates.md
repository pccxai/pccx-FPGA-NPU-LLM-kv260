# SVA Assertion Candidate Map — pccx v002 RTL

_Stage C deliverable, autonomous RTL cleanup pass._

This document enumerates the SystemVerilog Assertion targets surveyed
during the Stage C pass. **No assertions are inserted by this pass**;
this map exists so the next batch can reach for a pre-vetted, ranked
list rather than re-discover them.

## 1. Status — Tier 1 SVA + driver TB landed (this batch)

The Tier 1 candidate `mem_u_operation_queue` push-while-full property
plus a driver TB (`tb_mem_u_operation_queue`) shipped together in
this batch. The flow established here — `\`ifndef SYNTHESIS`-guarded
property/assertion pair at the bottom of the module, plus one TB
that intentionally exercises the warning path — is the template
later batches reuse for Tier 2/3/4 candidates.

Specifically landed:

  - `mem_u_operation_queue` ACP / NPU silent-drop properties:
    `a_acp_no_silent_drop` / `a_npu_no_silent_drop` (severity
    `$warning`).
  - `mem_u_operation_queue` counter monotonicity properties:
    `a_acp_in_monotonic` / `a_npu_in_monotonic` (severity
    `$error` — the saturating counter helper guarantees this).
  - Driver TB `tb_mem_u_operation_queue.sv` exercises both the
    legitimate path (counters increment, no warnings) and the
    over-push path (acp_perf.stall_cycles > 0, $warning fires).
    Now part of the 7/7 PASS run.

The remaining Tier 2/3/4 candidates below stay deferred for the same
reason as before: each one needs a focused TB that drives the
property fault path, otherwise the SVA only buys lint-time syntax
coverage. Land one Tier 2 candidate per future batch alongside its
driver TB.

### 1.1 Caveats on what the Tier 1 batch actually covers

The Tier 1 batch landed in this PR is intentionally narrow. Three
gaps are explicit and stay open for follow-up batches:

  - **Push-side only.** `a_acp_no_silent_drop` and
    `a_npu_no_silent_drop` cover the silent-drop **push** surface
    (`IN_*_rdy && fifo_full`). They do **not** observe the
    pop-side: a consumer that misreads `OUT_*_cmd` because the
    `valid` line and the `dout` payload are out of phase would not
    fire either property.
  - **Monotonicity is a regression guard, not positive coverage.**
    `a_acp_in_monotonic` / `a_npu_in_monotonic` only check that
    `in_count` never decreases. They are cheap insurance against a
    future refactor that swaps `sat_inc_handshake` for a wrapping
    increment; they say **nothing** about whether the count is
    correct, whether pushes were lost, or whether the counter
    actually advanced when expected.
  - **Pop-side stale-data / `READ_MODE` timing is a separate
    follow-up.** xpm_fifo_sync's `READ_MODE="std"` lets `empty`
    deassert one BRAM cycle before `dout` is valid. The current
    batch handles this in the **scoreboard** (the
    `OutCountFloor = NPush - 8` tolerance in
    `tb_mem_u_operation_queue`) and in the
    `counter_mvp_notes.md` §5 FWFT experiment record. A property-
    based check (e.g. `OUT_*_cmd_valid |-> $stable(OUT_*_cmd)`
    until ack) belongs with the §2.3 Tier 3 streaming-handshake
    work, not with this Tier 1 silent-drop pair.

## 2. Recommended assertion targets

Targets are ranked by (a) ease of reasoning, (b) recovery value when
the property fails, and (c) availability of an existing TB that could
be extended to drive the fault path. "Stage" = which downstream batch
is the right home.

### 2.1 Tier 1 — silent-drop / silent-discard surfaces

These modules silently swallow handshakes when their FIFO is full.
The contract documentation flags the surface; an SVA makes it
observable in simulation.

> **Status (this batch):** `mem_u_operation_queue` Tier 1 SVA +
> `tb_mem_u_operation_queue` driver TB shipped together (see §1).
> `AXIL_STAT_OUT` remains the next candidate.

#### `AXIL_STAT_OUT` — push dropped on `fifo_full`

- **Where**: `hw/rtl/NPU_Controller/NPU_frontend/AXIL_STAT_OUT.sv`,
  inside the FIFO push block (`if (IN_valid && !fifo_full) ...`).
- **Property**: At every cycle outside reset/clear,
  `!(IN_valid && fifo_full)`.
- **Severity**: `$warning` — the upstream is permitted to over-issue
  in pathological scenarios, but most workloads should not.
- **Sketch**:
  ```systemverilog
  `ifndef SYNTHESIS
    property p_axil_stat_out_no_silent_drop;
      @(posedge clk) disable iff (!rst_n || IN_clear)
        !(IN_valid && fifo_full);
    endproperty
    a_axil_stat_out_no_silent_drop : assert property
        (p_axil_stat_out_no_silent_drop)
        else $warning("AXIL_STAT_OUT: status push dropped (fifo_full at IN_valid)");
  `endif
  ```
- **TB readiness**: No TB compiles this module yet. Add
  `tb_AXIL_STAT_OUT` first, then land the SVA in the same commit.

#### `mem_u_operation_queue` — push dropped on `fifo_full` (ACP / NPU)

- **Where**: `hw/rtl/MEM_control/IO/mem_u_operation_queue.sv`, both
  channels (push gated by `IN_*_rdy & ~*_fifo_full`).
- **Property**: `!(IN_acp_rdy && acp_fifo_full)` and the same for NPU.
- **Severity**: `$warning`.
- **TB readiness**: No TB. Same recipe — write a queue TB first.

### 2.2 Tier 2 — issue-rate one-hot guarantees

The decoder and scheduler are documented as issuing at most one op
class per cycle. SVA pins this contract so a future decoder rewrite
can not silently regress.

#### `ctrl_npu_decoder` — at most one `OUT_*_op_x64_valid` per cycle

- **Where**: `hw/rtl/NPU_Controller/NPU_Control_Unit/ctrl_npu_decoder.sv`.
- **Property**: `$onehot0({OUT_gemv_op_x64_valid, OUT_gemm_op_x64_valid,
  OUT_memcpy_op_x64_valid, OUT_memset_op_x64_valid, OUT_cvo_op_x64_valid})`.
- **Severity**: `$error` — a violation here means the dispatcher
  receives malformed traffic.
- **TB readiness**: `tb_ctrl_npu_decoder` already exists. The SVA can
  ride on top of the existing baseline; verify all 6 cycles still PASS.

#### `Global_Scheduler` — `OUT_sram_rd_start` is a one-cycle pulse

- **Where**: `hw/rtl/NPU_Controller/Global_Scheduler.sv`.
- **Property**:
  `OUT_sram_rd_start |=> !OUT_sram_rd_start` (concurrent assertion
  guarded against reset).
- **Severity**: `$error` — multi-cycle pulse breaks the one-shot
  contract assumed by downstream consumers.
- **TB readiness**: No TB. Defer until scheduler TB lands.

### 2.3 Tier 3 — handshake / FIFO invariants on streaming paths

These are the largest-value class once a TB drives them, but they
require interface-level cooperation (often a property bound onto
`axis_if`, not the consumer module).

#### `axis_if` payload-stable-while-stalled

- **Where**: `hw/rtl/NPU_Controller/npu_interfaces.svh` (or wherever
  `axis_if` is declared).
- **Property**: `valid && !ready |=> $stable(payload) && valid`.
- **Severity**: `$error` — every protocol-conformant master must
  obey this; a violation indicates a master bug.
- **TB readiness**: Should be added inside the interface so every
  consumer in every TB inherits it. Highest leverage of this list.

#### `mem_CVO_stream_bridge` — write counter never exceeds total

- **Where**: `hw/rtl/MEM_control/top/mem_CVO_stream_bridge.sv`.
- **Property**: `wr_word_cnt <= total_words` and result FIFO never
  overflows when `OUT_cvo_result_ready` is gated.
- **Severity**: `$error`.
- **TB readiness**: No TB.

#### `GEMM_systolic_top` — dual-lane weight valids rise together

- **Where**: `hw/rtl/MAT_CORE/GEMM_systolic_top.sv`.
- **Property**:
  `IN_weight_upper_valid <-> IN_weight_lower_valid` per cycle
  (W4A8 dual-MAC precondition — `GEMM_dsp_packer` requires the pair).
- **Severity**: `$error` — a desync produces wrong arithmetic.
- **TB readiness**: No TB at the top level (component-level
  TBs exist for `GEMM_dsp_packer` and `GEMM_weight_dispatcher`).

### 2.4 Tier 4 — reset-state assertions

Cheap to add, low information density per assertion, but a fast way
to detect partial-reset bugs after RTL surgery. Best added in batches
once the project is comfortable with the SVA workflow.

- Per boundary module: assert `OUT_*_valid == 0` and reset-time
  registers all zero one cycle after `rst_n` deasserts (or after
  `IN_clear` asserts).
- Recommended landing order: `npu_controller_top`, `mem_dispatcher`,
  `Global_Scheduler`, `GEMV_top`, `GEMM_systolic_top`, `CVO_top`.

## 3. Implementation pattern (when the time comes)

Use a uniform block at the **bottom** of each module so contributors
can find them at a glance:

```systemverilog
// ===| Assertions (sim-only) |================================================
`ifndef SYNTHESIS
  // Each assertion gets its own named property + named assertion. Keep
  // properties short; one invariant per assertion. Use $error for
  // contract violations, $warning for soft preconditions the workload
  // might transiently violate.
  property p_<name>;
    @(posedge clk) disable iff (!rst_n || IN_clear)
      <expression>;
  endproperty
  a_<name> : assert property (p_<name>)
      else $<sev>("<module>: <human description>");
`endif
```

The `\`ifndef SYNTHESIS` guard keeps Vivado synthesis output identical;
xsim picks the assertions up automatically.

## 4. Workflow for landing the first SVA

The next batch that lands SVA should follow this sequence so the
toolchain story is established cleanly:

1. Pick **one** Tier 1 candidate (recommendation: `AXIL_STAT_OUT`).
2. Write `tb_AXIL_STAT_OUT` driving the over-push case so the SVA
   provably fires in negative tests and stays silent in positive.
3. Add the SVA block to the module.
4. Re-run `bash hw/sim/run_verification.sh`; verify
   - existing 6 TBs still PASS
   - new tb_AXIL_STAT_OUT PASS for the legitimate path
   - new tb_AXIL_STAT_OUT WARNs (not FAILs) for the over-push path
5. Confirm `xvlog -f filelist.f` reports 0 ERROR / 0 WARNING.
6. Land both files (TB + SVA) in one commit so the trail is
   self-contained.

Repeat for subsequent tiers; do not bulk-land assertions across many
modules in a single commit.
