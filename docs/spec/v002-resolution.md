# v002 Datapath Resolution

## Scope

This note resolves three v002.1 architecture choices before arithmetic RTL changes: activation quantization, K-split drain cadence, and DSP accounting. It defines defaults and parameter handles for follow-on RTL, testbench, software, and synthesis work; it does not change datapath behavior, timing constraints, utilization evidence, or the 20 tok/s target, which remains v002.2 work after KV260 boots a built bitstream and runs pccx end-to-end.

## Three decisions

### 1. Activation quantization policy

Options considered: `e_max`/BFP power-of-two scale | true symmetric INT8 scale | driver-computed scale via Constant Cache.

Trade-offs found:

- The current preprocess pipeline already finds a 32-element `global_emax`, aligns BF16 mantissas, and emits `16 x 27-bit` fixed-point values (`hw/rtl/PREPROCESS/preprocess_bf16_fixed_pipeline.sv:8`, `hw/rtl/PREPROCESS/preprocess_bf16_fixed_pipeline.sv:12`, `hw/rtl/PREPROCESS/preprocess_bf16_fixed_pipeline.sv:25`, `hw/rtl/PREPROCESS/preprocess_bf16_fixed_pipeline.sv:77`).
- The GEMM path already expects signed INT8 activation on the DSP B-port, but `GEMM_systolic_top` still truncates the 27-bit fixed-point stream to low 8 bits as a placeholder (`hw/rtl/MAT_CORE/GEMM_systolic_top.sv:20`, `hw/rtl/MAT_CORE/GEMM_systolic_top.sv:138`, `hw/rtl/MAT_CORE/GEMM_systolic_top.sv:146`, `hw/rtl/MAT_CORE/GEMM_systolic_top.sv:166`).
- The ISA exposes `MEMSET` and 6-bit pointers for shape/size-like constants (`README.md:211`, `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv:115`, `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv:134`), but the live dispatcher only writes fmap and weight shape banks today (`hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv:204`, `hw/rtl/MEM_control/top/mem_dispatcher.sv:75`, `hw/rtl/MEM_control/top/mem_dispatcher.sv:122`).
- TODO records the same choice and notes that true symmetric INT8 uses range better, but costs more hardware if max-abs, reciprocal, multiply, and rounding live in RTL (`TODO.md:116`, `TODO.md:135`, `TODO.md:149`, `TODO.md:155`).

Recommendation: choose `e_max`/BFP power-of-two scale for the v002.1 default. It reuses the existing 32-element exponent scan, gives the BF16-to-INT8 quantizer a small and deterministic first implementation, and leaves true symmetric or driver-computed scale as reviewed modes once the software scale table and post-process restore path are specified. That fits v002.1 because the near-term acceptance is board bring-up plus pccx end-to-end execution, while quantization quality and the 20 tok/s target belong to v002.2 tuning.

Parameter name: `ACT_SCALE_POLICY`, with enum type `act_scale_policy_e`.

### 2. K-split / drain limit

Options considered: `1024` | `4096` | parameterized value.

Trade-offs found:

- README states a guard-band limited drain every `1024` cycles for K above that value (`README.md:163`, `README.md:165`).
- TODO records the mismatch against the v002 formula for a `4096`-cycle limit and explicitly says the value must not remain hardcoded (`TODO.md:38`, `TODO.md:40`, `TODO.md:42`).
- The current W4A8 packer and sign recovery comments are internally aligned around `UPPER_SHIFT = 21`, 21-bit per-channel fields, and `1024` accumulations (`hw/rtl/MAT_CORE/GEMM_dsp_packer.sv:22`, `hw/rtl/MAT_CORE/GEMM_dsp_packer.sv:23`, `hw/rtl/MAT_CORE/GEMM_dsp_packer.sv:46`, `hw/rtl/MAT_CORE/GEMM_sign_recovery.sv:31`, `hw/rtl/MAT_CORE/GEMM_sign_recovery.sv:39`).
- The scheduler and top-level wiring do not yet carry a real resolved K count: GEMM/GEMV uops pass `size_ptr_addr`, and `NPU_top` currently forms `gemv_num_recur` directly from that 6-bit pointer (`hw/rtl/NPU_Controller/Global_Scheduler.sv:184`, `hw/rtl/NPU_Controller/Global_Scheduler.sv:197`, `hw/rtl/NPU_top.sv:312`, `hw/rtl/NPU_top.sv:317`).

Recommendation: choose a parameterized value, with v002.1 default `1024`. The default matches the current W4A8 field budget and existing smoke coverage while allowing `4096` to become a reviewed setting after the packer, sign-recovery, scheduler, and TB math all derive from the same parameter. This avoids making throughput-driven K-drain changes part of the v002.1 bring-up path.

Parameter name: `K_DRAIN_LIMIT`.

### 3. DSP accounting baseline

Options considered: `GEMM 1024 + GEMV 64 + alpha` | current RTL-specific higher count.

Trade-offs found:

- TODO frames the mismatch directly: GEMM `32 x 32 = 1024` plus GEMV `16 DSP/core x 4 = 64` gives a `1088 + alpha` architectural baseline, while the current RTL may synthesize higher once extras are counted (`TODO.md:49`, `TODO.md:51`, `TODO.md:595`, `TODO.md:596`).
- The GEMM array physically instantiates a `32 x 32` W4A8 PE grid and uses an additional final-row accumulator strip (`hw/rtl/MAT_CORE/GEMM_systolic_array.sv:3`, `hw/rtl/MAT_CORE/GEMM_systolic_array.sv:11`, `hw/rtl/MAT_CORE/GEMM_systolic_array.sv:227`, `hw/rtl/MAT_CORE/GEMM_systolic_array.sv:230`).
- Each GEMV lane reduction instantiates 16 DSP48E2 slices for the first 32-to-16 reduction stage, and `device_pkg` sets four vector lanes (`hw/rtl/VEC_CORE/GEMV_reduction.sv:14`, `hw/rtl/VEC_CORE/GEMV_reduction.sv:82`, `hw/rtl/VEC_CORE/GEMV_reduction.sv:87`, `hw/rtl/Constants/compilePriority_Order/B_device_pkg/device_pkg.sv:26`).
- Existing package style uses PascalCase localparams for numeric defaults and `_e` suffixes for enums (`hw/rtl/Constants/compilePriority_Order/C_type_pkg/dtype_pkg.sv:8`, `hw/rtl/Constants/compilePriority_Order/C_type_pkg/dtype_pkg.sv:33`, `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv:92`).

Recommendation: choose `GEMM 1024 + GEMV 64 + alpha`. The architecture baseline should remain the count for the intended compute cores; extra DSPs from final accumulators, CVO/SFU, post-process, debug, or inference side effects should be reported as `alpha` by synthesis. That gives issue #42 a stable denominator without hiding current implementation cost, and it keeps v002.1 focused on a reliable first hardware baseline rather than prematurely optimizing every inferred DSP.

Parameter name: `DSP_BASELINE_GEMM`, `DSP_BASELINE_GEMV`, and `DSP_BASELINE_ALPHA`.

## Parameter interface table

| Parameter | Type | Default | Allowed values | Owner module |
| --- | --- | --- | --- | --- |
| `ACT_SCALE_POLICY` | `act_scale_policy_e` | `ACT_SCALE_EMAX_BFP` | `ACT_SCALE_EMAX_BFP`, `ACT_SCALE_SYM_INT8`, `ACT_SCALE_CONST_CACHE` | `NPU_top` -> preprocess path |
| `K_DRAIN_LIMIT` | `int unsigned` | `1024` | `1024`, `4096`, or reviewed positive integer with matching packer, recovery, scheduler, and TB bounds | `NPU_top` -> scheduler/GEMM/GEMV control |
| `DSP_BASELINE_GEMM` | `int unsigned` | `1024` | Derived from GEMM PE grid geometry | synthesis/resource reporting |
| `DSP_BASELINE_GEMV` | `int unsigned` | `64` | Derived from GEMV lane count and stage-1 DSP reduction geometry | synthesis/resource reporting |
| `DSP_BASELINE_ALPHA` | `int unsigned` | `0` | Reported implementation extras above GEMM and GEMV compute baseline | synthesis/resource reporting |

## Follow-up issues to file

- [P1] Reference this note from #32 and #33 before changing the BF16-to-INT8 path; default `ACT_SCALE_POLICY` to `ACT_SCALE_EMAX_BFP`, and keep `ACT_SCALE_SYM_INT8` plus `ACT_SCALE_CONST_CACHE` as non-default reviewed modes.
- [P1] File or update a scheduler/control issue so `K_DRAIN_LIMIT` drives GEMM drain/flush cadence, GEMV recurrence windows, and K-split tests; #34 and #35 should use the same parameter for 1024 and 4096 cases.
- [P1] Reference this note from #42 so utilization reports show `DSP_BASELINE_GEMM + DSP_BASELINE_GEMV + DSP_BASELINE_ALPHA`, with alpha broken out by GEMM accumulator, GEMV, CVO/SFU, post-process, and debug buckets.
- [P2] Reference this note from #30 when documenting scale table layout, `MEMSET` encoding for activation/weight scale metadata, and the post-process restore path.

## Open questions for human review

- [CONFIRM] Is `ACT_SCALE_EMAX_BFP` acceptable as the v002.1 default even if true symmetric INT8 gives better range use later?
- [CONFIRM] Should `ACT_SCALE_CONST_CACHE` reuse the existing GEMM/GEMV `shape_ptr_addr`/`size_ptr_addr` fields, or does it require a new pointer convention?
- [CONFIRM] Should activation restore use the existing `flags.w_scale` path or a separate activation-scale convention?
- [CONFIRM] Should the INT8 saturation range be `[-127, 127]` or `[-128, 127]` for #32 and #33?
- [CONFIRM] Is `K_DRAIN_LIMIT = 1024` the only enabled v002.1 setting until the 4096 field-width proof and TB coverage are reviewed?
- [CONFIRM] Should `DSP_BASELINE_ALPHA` include the GEMM final-row accumulator strip in reports, or should that strip be grouped under GEMM implementation overhead?
