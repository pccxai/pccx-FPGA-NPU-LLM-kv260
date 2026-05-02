# pccx v002 Runnable Candidate Evidence Summary

Date: 2026-05-02
Branch: `closure/v002-runnable-candidate`

## Scope

This is a runnable-candidate evidence record, not a release claim.
It records what was verified locally and what remains blocked by missing
hardware, runtime, or completed Vivado reports.

## PR Stack Status

- The mem-dispatcher shape lookup test and shape RAM consolidation are
  already included in `origin/main`.
- This candidate branch adds:
  - `scripts/v002/run-local-candidate.sh`
  - `PCCX_VIVADO_JOBS` support for memory-constrained Vivado synth runs
  - current KV260 blocker evidence for the candidate SHA

## Local Runnable Candidate

Candidate SHA under test: `b1b0dee023c4aeb161415fa3fda652dcf3d1c99a`

Command:

```bash
bash scripts/v002/run-local-candidate.sh
```

Evidence:

```text
hw/build/v002-local-candidate/20260502T141937Z-b1b0dee/summary.txt
```

Results:

- `bash_syntax`: PASS
- `xsim_list`: PASS
- `tb_shape_const_ram`: PASS
- `tb_mem_dispatcher_shape_lookup`: PASS
- `tb_GEMM_fmap_staggered_delay`: PASS
- full xsim regression: PASS
- Vivado `xvlog` filelist compile: PASS
- `npu_core_wrapper` elaboration with `xpm` + `unisims_ver`: PASS

## Vivado Synth / Timing

Vivado version observed: 2025.2.

Attempts:

- `./vivado/build.sh synth`
  - Started: 2026-05-02 21:43 KST.
  - Result: failed after 31m46s; Vivado parent process was killed during
    synthesis under local memory pressure.
- `PCCX_VIVADO_JOBS=1 ./vivado/build.sh synth`
  - Started: 2026-05-02 22:16 KST.
  - Result: stopped after the 60-minute local cutoff with no generated
    `hw/build/reports/*_post_synth.rpt` reports.
  - Last run log stage: `Start Cross Boundary and Area Optimization`.

Observed blocker signals:

- `Synth 8-11357` warned that `emax_cache_mem_reg` and `emax_pipe_reg`
  3D RAM / struct RAM patterns may create large register implementations.
- `pccx_timing.xdc` produced critical warnings for unmatched false-path
  and multicycle objects at lines 34, 41, 43, 53, and 56.

Timing status:

- No fresh completed post-synth timing report was produced in this run.
- No post-implementation timing report exists for this candidate.
- Timing closure is not claimed.

## KV260 / Gemma 3N E4B

Command:

```bash
PCCX_RUN_ID=20260502T142200Z-b1b0dee-blocked-board \
  bash scripts/kv260/run_gemma3n_e4b_smoke.sh
```

Evidence:

```text
docs/evidence/kv260-gemma3n-e4b/20260502T142200Z-b1b0dee-blocked-board/
```

Result:

- `result_status=BLOCKED_BOARD`
- `board_reachable=no`
- `model_found=no`
- `bitstream_found=no`
- `bitstream_loaded=no`
- `token_count=unknown`
- `tok_per_sec=unknown`

Blocker:

- Required board/runtime environment variables were not present:
  `PCCX_KV260_HOST`, `PCCX_KV260_USER`, `PCCX_MODEL_DIR`,
  `PCCX_BITSTREAM_PATH`, `PCCX_RUN_PROMPT`, `PCCX_RUN_TOKENS`,
  `PCCX_BOARD_RUNTIME_DIR`.

A maintainer-local Gemma host artifact directory was inspected read-only;
the local absolute path is intentionally omitted from tracked evidence.
The host has INT4 safetensor shards and large mmap/cache artifacts, but no
KV260 staging or model execution was performed.

## Performance Boundary

- A 20 tok/s target remains a target only.
- No token throughput was measured on KV260.
- No Gemma 3N E4B board runtime evidence was produced.

## Non-Claims

- No production readiness claim.
- No stable-release claim.
- No timing-closure claim.
- No KV260 inference claim.
- No Gemma 3N E4B on-KV260 runtime claim.
- No achieved-throughput claim.
