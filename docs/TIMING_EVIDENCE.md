# Vivado Timing Evidence Checklist

This checklist describes how to reproduce Vivado timing evidence for the
pccx v002 FPGA path. It does not claim timing closure. A timing report is
evidence; closure requires the relevant post-implementation timing summary
to show that the constraints are met.

## Inputs

- Source list: `hw/vivado/filelist.f`
- Project script: `hw/vivado/create_project.tcl`
- Synthesis script: `hw/vivado/synth.tcl`
- Implementation script: `hw/vivado/impl.tcl`
- Timing constraints: `hw/constraints/pccx_timing.xdc`
- Target part: `xck26-sfvc784-2LV-c`

## Commands

From `hw/`:

```bash
./vivado/build.sh project
./vivado/build.sh synth
./vivado/build.sh impl
```

`project` and `synth` are the normal early evidence path. `impl` is a
long board-flow job and should only be launched when synthesis is clean
and the wrapper/block-design boundary is ready.

## Expected Report Locations

Generated files land under `hw/build/` and are ignored by git unless a
future curated-evidence policy says otherwise.

Post-synthesis:

```text
hw/build/reports/utilization_post_synth.rpt
hw/build/reports/clocks_post_synth.rpt
hw/build/reports/clock_interaction_post_synth.rpt
hw/build/reports/timing_summary_post_synth.rpt
hw/build/reports/drc_post_synth.rpt
```

Post-implementation:

```text
hw/build/reports/utilization_post_impl.rpt
hw/build/reports/clock_interaction_post_impl.rpt
hw/build/reports/timing_summary_post_impl.rpt
hw/build/reports/drc_post_impl.rpt
hw/build/reports/power_post_impl.rpt
hw/build/pccx_v002_kv260.bit
```

Do not commit bitstreams or generated Vivado build directories.

## Evidence To Record In Reviews

- Git commit or PR SHA under test.
- Vivado version.
- Exact command used.
- Whether the run was post-synthesis or post-implementation.
- Target constraints: `core_clk` 2.500 ns and `axi_clk` 4.000 ns.
- Timing summary verdict line.
- WNS, TNS, failing endpoint count, and worst clock domain.
- Utilization summary.
- DRC critical warnings or violations.
- Clock interaction report status for `core_clk` and `axi_clk`.

## Current Status Wording

Use wording such as:

- "post-synthesis timing evidence exists"
- "post-implementation timing evidence is missing"
- "timing constraints are not met in the current report"

Do not write that timing is closed until the post-implementation timing
summary supports that claim.
