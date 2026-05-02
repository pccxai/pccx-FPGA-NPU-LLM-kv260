# Dead Module Inventory — `hw/rtl/`

_Stage C deliverable, autonomous RTL cleanup pass._

This document is the engineering inventory of every `.sv` file under
`hw/rtl/` as of the Stage C autonomous cleanup batch. It classifies
each file as **active**, **authored-but-unwired**, **safe-deletion
candidate**, or **deferred-decision candidate**, and records the
evidence used for each call so a later reviewer can trust the
classification without re-running the analysis.

The companion commit immediately following this document deletes the
files in the **safe-deletion** group; the others stay where they are.

## 1. Methodology

For each `.sv` file under `hw/rtl/`:

1. Count active (non-commented) lines via `wc -l` after stripping
   `//`-prefixed lines.
2. Check for a `module` / `package` / `interface` declaration.
3. Search the rest of `hw/rtl/` for instantiations of the declared
   identifier (`grep -rn "<name>" hw/rtl/`).
4. Check whether the file is listed in `hw/vivado/filelist.f`.
5. Classify based on the four-way grid:

   | Has body | In filelist | Instantiated elsewhere | Class |
   |---|---|---|---|
   | Yes | Yes | Yes | active |
   | Yes | Yes | No (top-level only) | active (top) |
   | Yes | No | No | authored-but-unwired |
   | No / all-comment | Yes | No | safe-deletion candidate |
   | No / all-comment | No | No | safe-deletion candidate |

Top-level wrapper `NPU_top.sv` is treated as a synthesis root and
counted as active even though no other RTL instantiates it (Vivado
does, externally).

## 2. Full inventory

### 2.1 Active modules / packages / interfaces

These are wired into the build, instantiated at least once (or are the
synthesis top), and carry behavioural code. **No action needed.**

**Constants / packages** (`Constants/compilePriority_Order/`)
- `B_device_pkg/device_pkg.sv` — type-width policy.
- `C_type_pkg/dtype_pkg.sv` — BF16 / INT4 / INT8 / FP32 widths.
- `C_type_pkg/mem_pkg.sv` — L2 / HP / cache geometries.
- `D_pipeline_pkg/vec_core_pkg.sv` — GEMV lane configuration.
- `E_obs_pkg/perf_counter_pkg.sv` — observability counter vocabulary
  (Stage C addition; vocabulary-only, no consumers yet).

**Library**
- `Library/Algorithms/BF16_math.sv` (declares `bf16_math_pkg`).
- `Library/Algorithms/Algorithms.sv` (declares `algorithms_pkg`).
- `Library/Algorithms/QUEUE/IF_queue.sv` (declares `if_queue` interface).
- `Library/Algorithms/QUEUE/QUEUE.sv` (declares `QUEUE` module).

**ISA**
- `NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv` — 64-bit
  VLIW packed encoding.

**MAT_CORE** (GEMM systolic path, all wired)
- `GEMM_dsp_packer.sv`, `GEMM_sign_recovery.sv`, `GEMM_dsp_unit.sv`,
  `GEMM_dsp_unit_last_ROW.sv`, `GEMM_accumulator.sv`,
  `GEMM_fmap_staggered_delay.sv`, `GEMM_weight_dispatcher.sv`,
  `GEMM_systolic_array.sv`, `GEMM_systolic_top.sv`,
  `FROM_mat_result_packer.sv`, `mat_result_normalizer.sv`.

**VEC_CORE** (GEMV path, all wired)
- `GEMV_accumulate.sv`, `GEMV_generate_lut.sv`, `GEMV_reduction.sv`,
  `GEMV_reduction_branch.sv`, `GEMV_top.sv`.

**CVO_CORE** (SFU + CORDIC, all wired)
- `CVO_cordic_unit.sv`, `CVO_sfu_unit.sv`, `CVO_top.sv`.

**PREPROCESS** (BF16 / fixed-point pipeline, wired)
- `preprocess_bf16_fixed_pipeline.sv`, `fmap_cache.sv`,
  `preprocess_fmap.sv`.

**MEM_control**
- `memory/Constant_Memory/fmap_array_shape.sv`,
  `memory/Constant_Memory/weight_array_shape.sv`,
  `memory/mem_BUFFER.sv`, `memory/mem_GLOBAL_cache.sv`,
  `IO/mem_u_operation_queue.sv`, `top/mem_HP_buffer.sv`,
  `top/mem_CVO_stream_bridge.sv`, `top/mem_L2_cache_fmap.sv`,
  `top/mem_dispatcher.sv`.

**NPU_Controller**
- `NPU_Control_Unit/ctrl_npu_decoder.sv`,
  `NPU_frontend/AXIL_CMD_IN.sv`,
  `NPU_frontend/AXIL_STAT_OUT.sv`,
  `NPU_frontend/ctrl_npu_frontend.sv`, `Global_Scheduler.sv`,
  `npu_controller_top.sv`.

**Misc / top**
- `barrel_shifter_BF16.sv` (root-level utility).
- `NPU_top.sv` (synthesis top).

### 2.2 Authored-but-unwired (intentional, do not delete)

These files exist on disk and contain real code, but are **not in
`filelist.f`** and are not instantiated. They are scaffolding for
in-flight work. Do not delete.

| File | Reason kept | Owner / next step |
|---|---|---|
| `Library/Typedef/bf16_int8_quant_pkg.sv` | W4A8 BF16 → INT8 BFP vocabulary; will be wired when `preprocess_bf16_int8_32_pipeline` lands. | User; gating hold (see W4A8 gates below). Do not stage in this batch. |
| `PREPROCESS/preprocess_bf16_int8_32_pipeline.sv` | New W4A8 path under construction. | User; gating hold (see W4A8 gates below). Do not auto-stage. |
| `MEM_control/memory/Constant_Memory/shape_const_ram.sv` | Parameterised replacement for the duplicate `fmap_array_shape` + `weight_array_shape` pair (Stage C decisions item 5; KELLER §6.3.1). Authored to fix the consolidation shape; not yet wired so existing `mem_dispatcher.sv` still instantiates the legacy modules. | **Separate focused PR** — a self-contained shape-RAM consolidation commit per the "Migration path" comment block in the new file's header. Do NOT bundle with this batch. |

> **W4A8 gating hold.** The two W4A8 files above
> (`bf16_int8_quant_pkg.sv`, `preprocess_bf16_int8_32_pipeline.sv`)
> stay untracked until **all five** of the following gates clear:
>
> 1. Package-import refactor in
>    `preprocess_bf16_fixed_pipeline.sv` so the existing pipeline
>    no longer hard-codes its own width macros and can be diffed
>    against the new path on equal footing.
> 2. Golden Python quantizer in `llm-lite/` that emits the bit-
>    exact reference INT8 BFP stream the RTL must reproduce.
> 3. Bit-exact RTL TB driving
>    `preprocess_bf16_int8_32_pipeline` against the Python golden
>    output, with PASS verdict on the same `run_verification.sh`
>    harness.
> 4. `hw/vivado/filelist.f` integration in compile order (after
>    `dtype_pkg.sv`, before `preprocess_fmap.sv`).
> 5. `preprocess_fmap` swap to consume the new path on the W4A8
>    workload selector.
>
> Any partial subset of the five would land an authored-but-unwired
> compile dependency; the user has been explicit that these belong
> in their own focused PR sequence.

### 2.3 Safe-deletion candidates (deleted by the companion commit)

All four files below have **zero active SystemVerilog content** and
**no instantiations anywhere in `hw/rtl/`**. Deleting them does not
change synthesis output, simulation behaviour, or the public RTL
surface.

| File | Active LOC | In filelist | References | Evidence |
|---|---|---|---|---|
| `NPU_Controller/NPU_Control_Unit/ctrl_npu_dispatcher.sv` | 0 (176 lines, all `//`-prefixed) | yes | none | `head` shows every line begins with `//`; the dispatch role moved to `Global_Scheduler.sv`. |
| `NPU_Controller/NPU_frontend/ctrl_npu_interface.sv` | 0 (file is 0 bytes) | yes | none | `wc -l` returns 0. |
| `NPU_Controller/NPU_fsm_out_Logic/fsmout_npu_stat_collector.sv` | 0 (file is 0 bytes) | no | none | `wc -l` returns 0. The status word `mmio_npu_stat[31:0]` is currently composed inline in `NPU_top.sv` (lines 397-400), so a future stat-collector module would replace that ad-hoc assignment — but the empty placeholder file does not help that work happen. |
| `NPU_Controller/NPU_fsm_out_Logic/fsmout_npu_stat_encoder.sv` | 0 (file is 0 bytes) | no | hard-tied (`AXIL_STAT_OUT.IN_enc_stat = '0`, `IN_enc_valid = 1'b0` in `npu_controller_top.sv`) | The encoder slot exists at the AXIL side but is permanently disabled. The empty file is unrelated to that wiring decision and provides no value. |

`filelist.f` lines for the first two are removed in the same commit so
the build does not reference paths that no longer exist.

The empty `NPU_fsm_out_Logic/` directory is also removed; if a future
stat-collector lands, the new file can recreate its parent directory.

### 2.4 Deferred-decision candidates (NOT deleted in this batch)

None at this scope. The shape-RAM consolidation
(`fmap_array_shape` + `weight_array_shape` → one parameterised
`shape_const_ram`) is tracked separately as an architecture issue per
Stage C decisions memo item 5; both shape RAMs remain active here.

## 3. Recommended follow-up (out of Stage C scope)

These are noted for the next refactor pass, not actioned here:

- **Shape RAM consolidation** (Stage C decision item 5) — produces a
  single parameterised `shape_const_ram`. Architecture issue.
- **File / module name mismatches** flagged by verible
  (`GEMM_fmap_staggered_delay.sv` declares
  `GEMM_fmap_staggered_dispatch`, `mat_result_normalizer.sv` declares
  `gemm_result_normalizer`, etc.). Cosmetic; affects navigation only.
- **`mmio_npu_stat` aggregation** — implementing a real
  `fsmout_npu_stat_collector` would replace the inline assignment in
  `NPU_top.sv` and give the AXIL_STAT_OUT slot a driver. Not started
  here because that is a feature, not cleanup.

## 4. Validation gates passed

- `xvlog` over `hw/vivado/filelist.f` after the dead-file deletion +
  filelist trim must report 0 ERROR / 0 WARNING.
- The deletion commit changes nothing observable in any TB output.
- All currently-tracked TBs still pass `bash hw/sim/run_verification.sh`.
