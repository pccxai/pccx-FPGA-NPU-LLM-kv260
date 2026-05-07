# Open Issue Milestone Tracker

Snapshot source: `gh issue list --state open --limit 200`, captured
2026-05-07.

This tracker maps every currently open issue into the active planning
milestones:

- [M01 - v002 correctness & bring-up](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/milestone/1)
- [M02 - software, model pipeline & hardware closure](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/milestone/2)

Rows where the live GitHub milestone is not M01 or M02 are included in the
nearest planning bucket and called out in the metadata cleanup section.

## Summary

| Tracker milestone | Open issues | Scope |
| --- | ---: | --- |
| M01 | 14 | v002 spec correctness, RTL bring-up, top-level integration, simulation, and first synthesis evidence |
| M02 | 11 | bare-metal software, model/golden pipeline, memory validation, implementation closure, release evidence, and repo contribution docs |

## M01 - v002 correctness & bring-up

| Issue | Live GitHub milestone | Primary area |
| --- | --- | --- |
| [#16](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/16) - `[P0][SPEC]` Resolve v002 datapath/spec mismatches before RTL changes | M01 | spec |
| [#17](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/17) - `[P1][STAT]` Connect engine completion to mmio_npu_stat | M01 | npu-top / dispatcher |
| [#32](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/32) - `[P0][PREPROCESS]` Implement BF16 to INT8 activation quantizer path | M01 | preprocess |
| [#33](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/33) - `[P0][TB]` Add preprocess BF16 to INT8 golden-model testbench | M01 | preprocess / verification |
| [#34](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/34) - `[P0][GEMM]` Verify W4A8 DSP dual-MAC behavior | M01 | gemm |
| [#35](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/35) - `[P0][GEMM]` Validate DSP packer and sign recovery | M01 | gemm |
| [#36](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/36) - `[P0][GEMV]` Remove GEMV lane activation hardcode and connect lane mask | M01 | gemv / npu-top |
| [#37](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/37) - `[P0][NPU_TOP]` Add minimal NPU_top end-to-end integration TB | M01 | npu-top / verification |
| [#38](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/38) - `[P1][DISPATCHER]` Redesign ctrl_npu_dispatcher for decoupled uop FIFOs | M01 | dispatcher |
| [#39](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/39) - `[P1][NPU_TOP]` Connect result packer ready/valid and STORE writeback path | M01 | npu-top |
| [#40](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/40) - `[P1][GEMV]` Verify GEMV reduction pipeline and SFU/L2 output mux | M01 | gemv |
| [#41](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/41) - `[P1][CVO]` Add CVO/SFU function and softmax-path testbench | M01 | cvo |
| [#42](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/42) - `[P1][SYNTH]` Run Vivado synth and record timing/resource baseline | M01 | synthesis |
| [#43](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/43) - `[EPIC]` v002 correctness, bring-up, and hardware closure tracking | M01 | spec / admin |

## M02 - software, model pipeline & hardware closure

| Issue | Live GitHub milestone | Primary area |
| --- | --- | --- |
| [#23](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/23) - `[P2][SW]` Build bare-metal app, HAL timer, and Gemma weight pipeline | M02 | software |
| [#24](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/24) - `[P2][GOLDEN]` Add llm-lite golden model and bit-similarity verification | M02 | software / golden-model |
| [#25](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/25) - `[P2][MEMORY]` Verify URAM L2 mapping and memory arbitration | M02 | memory / synthesis |
| [#26](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/26) - `[P2][MEMCPY]` Add mem_dispatcher testbench | M02 | memory / dispatcher |
| [#27](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/27) - `[P2][SCHED]` Add Global_Scheduler hazard and chaining testbench | M02 | scheduler / dispatcher |
| [#28](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/28) - `[P2][GEMM]` Re-enable tb_GEMM_fmap_staggered_delay after INT8 transition | M02 | gemm / preprocess |
| [#29](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/29) - `[P2][IMPL]` Run post-implementation timing closure and bitstream smoke plan | M02 | implementation / synthesis |
| [#30](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/30) - `[P2][DOCS]` Document v002 scale, quantization, and artifact policy | M02 | spec / software |
| [#31](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/31) - `[P2][ADMIN]` Prepare TRC application and HuggingFace Gemma access | M02 | admin |
| [#51](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/51) - docs: add repo-level CONTRIBUTING.md scaffold | v0.1.0-alpha | contributor docs |
| [#58](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/58) - `[P1][EVIDENCE]` Collect and organize v0.2.0 release evidence | Unassigned | release evidence |

## Metadata Cleanup

The tracker bucket above is the planning source for M01/M02 handoff. These
open issues need live GitHub milestone cleanup if the repository issue board
should match this document exactly:

| Issue | Current live milestone | Tracker bucket | Suggested action |
| --- | --- | --- | --- |
| [#51](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/51) | v0.1.0-alpha | M02 | Retarget to M02 or close/defer after the contributor scaffold lands. |
| [#58](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/58) | Unassigned | M02 | Assign to M02 so release evidence work is tracked with hardware closure. |
