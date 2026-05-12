# M01 v002 Correctness & Bring-Up Close Criteria

Snapshot source: `gh issue list --state open --limit 100`, captured
2026-05-07.

This document defines what the M01 milestone requires before a maintainer
can close it. It is not a completion statement: every M01 issue cited
below was open at the snapshot time, and the existing KV260 evidence in
this repository is blocked-board evidence, not board-success evidence.

Milestone: [M01 - v002 correctness & bring-up](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/milestone/1)

## Close Requirements

M01 can be closed only after all of the following are true on the target
base branch or an explicitly named release-candidate SHA:

1. All open M01 P0/P1 issues listed in this document are either closed
   by maintainers or explicitly deferred out of M01 with a linked
   rationale.
2. The v002 datapath/spec decisions are recorded and reflected in RTL,
   parameters, tests, and docs without unresolved contradictions.
3. The deterministic xsim suite passes for the SHA under review, including
   the M01 correctness testbenches.
4. A Vivado synthesis baseline exists for the SHA under review, with
   timing/resource status recorded without claiming post-implementation
   timing closure.
5. KV260 bring-up status is backed by a `PASS_KV260_NPU` evidence
   directory, or by a precise `BLOCKED_*` evidence directory plus an
   open follow-up outside M01. Blocked or fallback runs do not count as
   board inference success.
6. Release/readme wording states the remaining timing, bitstream,
   runtime, and model-artifact limitations plainly.

## Open Issue Gates

| Gate | Open issue evidence | What must be true before M01 close |
| --- | --- | --- |
| Parent tracking | [#43](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/43) - `[EPIC] v002 correctness, bring-up, and hardware closure tracking` | The epic checklist is updated with the final M01 evidence links and no M01 blocker is still open without a defer rationale. |
| Spec decisions | [#16](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/16) - `[P0][SPEC] Resolve v002 datapath/spec mismatches before RTL changes` | Activation quantization, K-split/drain limit, and DSP accounting are decided, documented, and referenced by follow-up RTL/TB work. |
| Preprocess path | [#32](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/32), [#33](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/33) | BF16-to-INT8 activation quantizer behavior is implemented or explicitly deferred, and a golden-model testbench covers the accepted policy. |
| GEMM correctness | [#34](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/34), [#35](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/35) | W4A8 dual-MAC packing and sign recovery are validated with reviewable xsim/golden evidence. |
| GEMV correctness | [#36](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/36), [#40](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/40) | Lane masks, reduction behavior, and SFU/L2 output mux behavior are wired and verified, or the remaining gap is moved out of M01 with rationale. |
| Top-level integration | [#17](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/17), [#37](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/37), [#39](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/39) | Engine completion status, result-packer ready/valid, STORE writeback, and a minimal `NPU_top` end-to-end TB are landed and green. |
| Dispatcher | [#38](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/38) - `[P1][DISPATCHER] Redesign ctrl_npu_dispatcher for decoupled uop FIFOs` | Dispatch behavior no longer depends on unresolved coupling that can invalidate M01 integration evidence. |
| CVO/SFU | [#41](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/41) - `[P1][CVO] Add CVO/SFU function and softmax-path testbench` | The accepted softmax/CVO path has function-level verification or an explicit non-M01 defer record. |
| Synthesis baseline | [#42](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/42) - `[P1][SYNTH] Run Vivado synth and record timing/resource baseline` | Vivado version, command, utilization, DRC, clocks, and timing summary are recorded for the SHA under review. |

## Evidence References

Use these evidence contracts when reviewing whether the gates above are
satisfied:

| Evidence source | Required use for M01 |
| --- | --- |
| [xsim verification workflow](../SIMULATION.md) | Record the exact `bash hw/sim/run_verification.sh` command, Vivado version, per-TB PASS summary, and any retained `.pccx` traces. xsim evidence is source-level verification only. |
| [release evidence checklist](../RELEASE_EVIDENCE_CHECKLIST.md) | Use sections 1-5 as the M01 review checklist for implementation, simulation, synthesis, timing status, and KV260 bring-up status. |
| [Vivado timing evidence checklist](../TIMING_EVIDENCE.md) | Record post-synthesis evidence and avoid timing-closure wording unless a post-implementation timing summary supports it. M01 does not inherit an unstated timing-closed bitstream claim. |
| [KV260 bring-up checklist](../KV260_BRINGUP.md) | Treat only `PASS_KV260_NPU` with captured logs, bitstream hash, commit SHA, and runtime output as board NPU smoke evidence. |
| [Gemma 3N handoff boundary](../GEMMA3N_HANDOFF.md) | Use readiness states and avoid Gemma 3N board-execution, throughput, or production-readiness claims without tied evidence. |
| [`20260502T054224Z-5c69049cf7ba`](../evidence/kv260-gemma3n-e4b/20260502T054224Z-5c69049cf7ba/summary.txt) | Existing evidence records `result_status=BLOCKED_BOARD`, `board_reachable=no`, `bitstream_found=no`, and `model_found=no`. |
| [`20260502T142200Z-b1b0dee-blocked-board`](../evidence/kv260-gemma3n-e4b/20260502T142200Z-b1b0dee-blocked-board/summary.txt) | Existing evidence records `result_status=BLOCKED_BOARD`, `board_reachable=no`, `bitstream_found=no`, and `model_found=no`. |

## Non-Closure Wording

Until every gate above is satisfied, use wording such as:

- "M01 close criteria are documented."
- "M01 blockers remain open."
- "KV260 bring-up evidence is blocked or pending."
- "post-synthesis timing evidence exists" only when tied to a cited
  report.

Avoid wording that says or implies:

- M01 is complete.
- KV260 board inference is proven by a blocked or fallback run.
- Gemma 3N E4B runs on the NPU without `PASS_KV260_NPU` evidence.
- timing is closed without a supporting post-implementation timing
  summary.
