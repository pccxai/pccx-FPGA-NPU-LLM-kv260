# v002.1 Bitstream Attempt 1 Evidence

Branch: `evidence/v002.1-bitstream-attempt-1`

Base commit: `18d4631f54721684ef6747bc37cf8538653a7a9e`

Vivado: `vivado v2025.2 (64-bit)`

## Result

`SYNTH_RESULT=FAIL`

The full-top BD synth flow from PR #82 failed during BD preparation before
`synth_design` completed. No implementation, routing, or bitstream step was
run after this failure.

The failing command was:

```bash
vivado -mode batch \
  -log build/logs/v002_1_full_top_synth.log \
  -journal build/logs/v002_1_full_top_synth.jou \
  -source "$PCCX_TOP_BD_TCL" \
  -source vivado/scripts/record_synth_baseline.tcl \
  -tclargs synth
```

PR #82's `system_bd.tcl` and PR #88's `record_synth_baseline.tcl` were used as
transient local scaffolding only. They are not committed in this reports-only
evidence branch.

## Captured Files

- `logs/v002_1_clean.log`: output from `bash hw/vivado/build.sh clean`
- `logs/v002_1_full_top_synth.log`: Vivado full-top synth/BD failure log
- `logs/v002_1_full_top_synth.jou`: Vivado journal for the failed run
- `synth_status.txt`: synth verdict and root-cause summary
- `synth_baseline_status.txt`: PR #88 hook status
- `bitstream_status.txt`: bitstream artifact status
- `top_level_bitstream_status.txt`: top-level bitstream flow status
- `vivado_version.txt`: Vivado version used

## Failure Excerpt

```text
ERROR: [filemgmt 56-591] Given include File '/tmp/kv260-bitstream-wt/hw/rtl/Constants/compilePriority_Order/A_const_svh/npu_arch.svh' needs to be addded to the project in order to use it as an RTL module.
ERROR: [Common 17-39] 'create_bd_cell' failed due to earlier errors.

    while executing
"create_bd_cell -type module -reference $NPU_BD_MODULE_REF npu_0"
    (procedure "pccx_prepare_bd" line 24)
```

## Claim Guard

`SYNTH_FAILED_NO_TIMING_REPORT_NO_CLOSURE_CLAIM`

No timing-closure claim is made. No post-implementation timing report exists
for this attempt.
