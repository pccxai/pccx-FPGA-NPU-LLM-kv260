# Stage C Completion Notes — Autonomous RTL Cleanup Pass

_Branch: `stage-c-autonomous-rtl-cleanup`. Date: 2026-05-02._

This document records what landed in the Stage C autonomous batch, the
validation evidence for each chunk, and what was deliberately deferred.
It is the close-out artefact for the batch — a future reviewer who
sees this branch should be able to read this single file and trust
the rest.

## 1. Scope (as agreed before the batch)

Per the Stage C decisions memo:

- **In scope**: package vocabulary, opt-in counter scaffold, dead-module
  inventory, GLOBAL_CONST migration plan, SVA candidate map, contract
  header standardisation.
- **Out of scope** (hard limits):
  - No FPGA RTL functional changes.
  - No top-level interface changes.
  - No KV260 inference claims.
  - No timing-closed claims.
  - No release / tag creation.

Everything below sits inside the in-scope list.

## 2. Commits landed (chronological)

Subsystem header standardisation:

  1. `docs(rtl): add Library + barrel_shifter module contract headers`
  2. `docs(rtl): add NPU_Controller module contract headers`
  3. `docs(rtl): add MEM_control module contract headers`
  4. `docs(rtl): add MAT_CORE module contract headers`
  5. `docs(rtl): add VEC_CORE module contract headers`
  6. `docs(rtl): add CVO_CORE module contract headers`
  7. `docs(rtl): add PREPROCESS contract headers (fmap_cache, preprocess_fmap)`
  8. `docs(rtl): add NPU_top module contract header`

Stage C deliverables:

  9. `feat(rtl): add perf_counter_pkg observability vocabulary scaffold`
  10. `docs: add dead module inventory for hw/rtl`
  11. `chore(rtl): remove dead RTL files (zero active SystemVerilog)`
  12. `docs: add SVA assertion candidate map`
  13. `docs: add GLOBAL_CONST.svh migration plan`
  14. `chore(rtl): drop unused TRUE / FALSE aliases from GLOBAL_CONST.svh`
  15. `docs: add Stage C completion notes` (this commit)

Each commit is reviewable on its own; nothing is squashed.

## 3. What changed by category

### 3.1 Counter scaffold

- New file `hw/rtl/Constants/compilePriority_Order/E_obs_pkg/perf_counter_pkg.sv`
  declares the opt-in vocabulary: `PerfCountersEnableDefault = 1'b0`,
  `PerfCntCycleWidth = 32`, `PerfCntHandshakeWidth = 32`,
  `PerfCntDropWidth = 32`, `PerfCntDepthWidth = 16`, plus the
  `handshake_counter_t` struct and three saturating-increment helpers.
- Wired into `hw/vivado/filelist.f` between vec_core_pkg.sv and the
  Library section as compile-priority section E.
- **Vocabulary-only.** No module imports the package yet, so the
  synthesis output is unchanged. A later Stage D MVP can wire one or
  two boundary modules to the struct behind `PerfCountersEnable`; that
  wiring will be its own isolated commit.

### 3.2 Inventory / dead-module cleanup

- `docs/internal/dead_module_inventory.md` enumerates every `.sv` file
  under `hw/rtl/` and classifies it as active, authored-but-unwired,
  safe-deletion candidate, or deferred-decision candidate.
- Four files removed by an isolated commit:
  - `NPU_Controller/NPU_Control_Unit/ctrl_npu_dispatcher.sv` — 176
    lines, every line `//` commented; the dispatch role moved to
    `Global_Scheduler`.
  - `NPU_Controller/NPU_frontend/ctrl_npu_interface.sv` — 0 bytes.
  - `NPU_Controller/NPU_fsm_out_Logic/fsmout_npu_stat_collector.sv` —
    0 bytes.
  - `NPU_Controller/NPU_fsm_out_Logic/fsmout_npu_stat_encoder.sv` — 0
    bytes.
- `hw/vivado/filelist.f` was trimmed of the two now-deleted entries
  that it referenced (`ctrl_npu_dispatcher.sv`,
  `ctrl_npu_interface.sv`).
- The empty `NPU_fsm_out_Logic/` directory was removed; if a future
  stat-collector lands, the new file can recreate its parent.

### 3.3 SVA / assertion readiness

- `docs/internal/sva_assertion_candidates.md` enumerates the 8
  highest-value SVA targets (silent-drop, one-hot issue, handshake
  invariants, reset-state) with property sketches and tier rankings.
- **No SVA was added in this batch.** Reason: of the 6 currently
  passing TBs, none compiles the modules where Tier-1 SVAs would
  land, so an assertion added today only buys lint-time coverage.
  The recommended workflow (TB + SVA in the same commit, starting
  with `tb_AXIL_STAT_OUT`) is documented in the candidate map.

### 3.4 Package cleanup

- `perf_counter_pkg` (above) is the only new package.
- Compile-priority section E was added to `filelist.f` with a comment
  block explaining its dependency surface.
- No mass rename of internal signals (Stage C decisions memo: review
  noise).

### 3.5 GLOBAL_CONST migration path

- `docs/internal/global_const_migration_plan.md` records the consumer
  survey (3 HP_PORT_* consumers, 5 DSP48E2_* / PREG_SIZE consumers,
  ~33 cold includers) and the 5-phase staged migration order.
- Phase 1 (drop TRUE / FALSE — 0 consumers) shipped in an isolated
  commit. The deprecation comment block in `GLOBAL_CONST.svh` was
  updated to record migration progress.
- Phases 2–5 stay for later batches per the Stage C decisions memo.

### 3.6 Documentation / decision logs

- Three new internal docs under `docs/internal/`
  (dead-module inventory, SVA candidate map, GLOBAL_CONST migration
  plan), plus this completion-notes file.

### 3.7 Subsystem contract headers

- 8 commits standardised the Keller-style contract block (Purpose /
  Spec ref / Clock / Reset / Latency / Throughput / Handshake / Reset
  state / Counters / Errors / Notes / Protected) across ~38 RTL
  files.
- Files in CLAUDE.md §6.2 protected list got the header block but
  no internal-compute changes (CVO CORDIC / SFU, GEMV reduction /
  reduction_branch / accumulate, BF16 fixed-point preprocess).

## 4. Validation evidence

### 4.1 xvlog lint

```bash
$ cd hw/build/stage_c_lint && xvlog -sv \
    -i hw/rtl/Constants/compilePriority_Order/A_const_svh \
    -i hw/rtl/NPU_Controller \
    -i hw/rtl/MEM_control/IO \
    -i hw/rtl/MAT_CORE \
    $(sed -n 's|^rtl/|hw/rtl/|p' hw/vivado/filelist.f)
$ echo $?
0
$ grep -cE "(ERROR|CRITICAL|WARNING)" xvlog.log
0
```

Run after every meaningful change. 0 ERROR / 0 WARNING throughout.

### 4.2 xsim baseline

`bash hw/sim/run_verification.sh` before the batch:

| TB                                   | Result              |
|---|---|
| tb_GEMM_dsp_packer_sign_recovery    | PASS — 1024 cycles  |
| tb_GEMM_weight_dispatcher           | PASS — 128 cycles   |
| tb_mat_result_normalizer            | PASS — 256 cycles   |
| tb_FROM_mat_result_packer           | PASS — 4 cycles     |
| tb_barrel_shifter_BF16              | PASS — 512 cycles   |
| tb_ctrl_npu_decoder                 | PASS — 6 cycles     |

Re-run at the end of the batch must show the same 6 PASS — see
`/tmp/sim_post_stage_c.log` snapshot referenced in the final report.

### 4.3 git diff --check

Run before every commit; whitespace and conflict-marker check clean
throughout.

## 5. Deferred items (not in this batch)

- **Counter MVP wiring** — `perf_counter_pkg` is vocabulary-only. A
  Stage D pass should pick one or two boundary modules
  (recommendation: `mem_dispatcher`, `mem_u_operation_queue`) and
  wire `handshake_counter_t` behind `parameter logic
  EnablePerfCounters = perf_counter_pkg::PerfCountersEnableDefault`.
  Capture a baseline synthesis report before / after to bound the
  area cost.
- **SVA insertion** — see `docs/internal/sva_assertion_candidates.md`
  §4 for the recommended landing order. First SVA + driver TB ships
  together.
- **GLOBAL_CONST Phases 2–5** — three HP_PORT_* migrations, five
  DSP48E2_* / PREG_SIZE migrations, cold-includer strip, file delete.
- **Shape RAM consolidation** (`fmap_array_shape` +
  `weight_array_shape` → parameterised `shape_const_ram`) — separate
  architecture issue per Stage C decisions memo item 5.
- **File / module name mismatches** flagged by verible
  (`GEMM_fmap_staggered_delay.sv` declares
  `GEMM_fmap_staggered_dispatch`, `mat_result_normalizer.sv` declares
  `gemm_result_normalizer`, etc.). Cosmetic; affects navigation only.
- **`bf16_int8_quant_pkg.sv` wire-up** — package stays
  authored-but-unwired per Stage C decisions memo item 6 until the
  W4A8 PREPROCESS path is ready to import it.
- **`preprocess_bf16_fixed_pipeline.sv` working-tree changes** — the
  user has whitespace + contract-header alignment in flight on this
  protected file; not auto-staged. The contract header itself will
  ride along with the user's W4A8 quantizer commit when that ships.
- **`KELLER_REFACTOR_NOTES.md` and `jim_keller_design_philosophy_*.md`**
  — left untracked at the user's location (`hw/rtl/` and repo root
  respectively). The portion of the Keller notes that documents
  dead-module candidates is reproduced in
  `docs/internal/dead_module_inventory.md` so it reaches the public
  repo via the inventory commit. Decide later whether the full notes
  belong under `docs/internal/` (recommended location:
  `docs/internal/keller_refactor_notes_2026_05_02.md`) or stay
  per-clone scratch.

## 6. Risks

- The 8 contract-header commits touched ~38 files including protected
  ones. Every diff is header-only, but a thorough review on the
  protected MAT_CORE / VEC_CORE / CVO_CORE / PREPROCESS files is
  cheap and worth doing once.
- `perf_counter_pkg` widths (32 / 32 / 16) are sized for current
  expected sim length and FIFO depth. If a future MVP wires a counter
  on a much hotter path, the cycle width may need to grow — bump
  `PerfCounterPkgVersion` and adjust.
- The dead-file removal commit touched `filelist.f`. Vivado TCL
  wrappers that hard-coded the old paths (`ctrl_npu_dispatcher.sv`,
  `ctrl_npu_interface.sv`) will fail at re-run; verify the build
  scripts under `hw/vivado/` reference only the trimmed list.
- `GLOBAL_CONST.svh` Phase 1 only removes 4 lines, but the broader
  migration touches protected DSP files in Phase 3. Plan that batch
  with explicit user review.

## 7. Public-claim posture (unchanged)

This batch did **not** touch:

- Release tags (no v0.1.1-alpha or anything similar).
- README claims about FPGA stable / timing-closed / KV260 inference.
- The `docs/RELEASE_EVIDENCE_CHECKLIST.md` content.

The internal docs added under `docs/internal/` describe engineering
intent; they do not make performance, timing, or readiness claims.
