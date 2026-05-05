# Vivado build for pccx v002 — KV260

This directory drives the RTL → bitstream flow for the pccx v002 NPU.
Target device: **Kria KV260 SOM, `xck26-sfvc784-2LV-c` (ZU5EV)**.

## Files

| Path | Role |
|------|------|
| `filelist.f`            | Ordered SystemVerilog source list, read by both `xvlog -f` and the Vivado TCL. |
| `create_project.tcl`    | Build the Vivado project (part, sources, includes, XDC). |
| `synth.tcl`             | Out-of-context synthesis of `NPU_top`. Reports into `build/reports/`. |
| `impl.tcl`              | OOC opt/place/route evidence. The bitstream mode is blocked for OOC modules. |
| `top_level_status.tcl`  | Full KV260 top-level / BD bitstream-flow prerequisite gate. |
| `build.sh`              | Wrapper: `./build.sh {project\|synth\|impl\|bitstream\|top-status\|top-bitstream\|clean}`. |
| `npu_core_wrapper.sv`   | Plain-signal shim around `NPU_top`'s SV-interface ports, used by the BD / IP packager. |

## Quick start

```bash
cd hw

# Create the project (fast)
./vivado/build.sh project

# Out-of-context synthesis (minutes to tens of minutes, depending on machine)
./vivado/build.sh synth

# Routed OOC implementation evidence (long job)
./vivado/build.sh impl

# Full KV260 top-level / BD bitstream-flow prerequisite status (lightweight)
./vivado/build.sh top-status

# Wipe all generated state
./vivado/build.sh clean
```

## What is and isn't covered

| Stage | Status |
|-------|--------|
| RTL compile (`xvlog`) — 52 files | ✅ clean |
| `xelab` on `GEMV_top`, `CVO_top`, `GEMM_systolic_top` | ✅ clean |
| `xelab` on `NPU_top` standalone | ⚪ expected fail — interface ports need a wrapper; see `npu_core_wrapper.sv` |
| `create_project.tcl` | ✅ runs green |
| `synth.tcl` (out-of-context) | ✅ current PR #79 evidence: runs to reports with post-synth timing clean |
| `impl.tcl` (out-of-context route) | 🟡 current PR #79 evidence: route completes, post-impl setup still violates by a small margin |
| OOC `bitstream` mode | 🔴 blocked by Vivado HDOOC-3; OOC route evidence is not a KV260 bitstream |
| Full top-level / Zynq PS integration | 🔴 not written yet (see §Next steps and `top-status`) |
| Device-tree overlay | 🔴 not written yet |
| Driver smoke test on board | 🔴 not attempted yet |

## Timing evidence

Generated Vivado evidence lands under `hw/build/reports/` and is ignored
by git unless a future curated-evidence policy says otherwise. Record the
summary values in PRs or release notes instead of committing full build
directories.

Minimum report set:

| Stage | Reports |
|-------|---------|
| Post-synthesis | `utilization_post_synth.rpt`, `clocks_post_synth.rpt`, `clock_interaction_post_synth.rpt`, `timing_summary_post_synth.rpt`, `drc_post_synth.rpt` |
| Post-implementation | `utilization_post_impl.rpt`, `clock_interaction_post_impl.rpt`, `timing_summary_post_impl.rpt`, `drc_post_impl.rpt`, `power_post_impl.rpt` |

See [`../../docs/TIMING_EVIDENCE.md`](../../docs/TIMING_EVIDENCE.md) for
the review checklist and wording rules. A generated timing report is
evidence, not a timing-closure claim.

## Next steps to reach a running board

1. **Synth clean** — resolve whatever `synth.tcl` turns up in
   `build/reports/drc_post_synth.rpt` and the Vivado log. Common
   first-pass issues to expect:
   - unresolved `XPM_*` references needing `-L xpm` in synth
   - untied output ports on placeholder modules
   - multi-driver or ambiguous interface modport warnings
2. **Block design** — create `vivado/system_bd.tcl` or set
   `PCCX_TOP_BD_TCL` to a board-design script that drops:
   - Zynq UltraScale+ MPSoC IP (KV260 preset)
   - `npu_core_wrapper` as a packaged IP
   - AXI Interconnect / Smart Connect between PS HP/HPC/HPM and the NPU
   - Clock Wizard for the 400 MHz core clock
3. **Top-level status gate** — run `./vivado/build.sh top-status` to
   record whether the full top-level prerequisites exist. This writes
   `build/reports/top_level_bitstream_status.txt`.
4. **`write_bitstream`** — only run after full top-level implementation is
   valid and post-implementation timing is clean; otherwise this is not board
   evidence.
5. **Device-tree overlay** — see `sw/dtbo/` (to be created) for the
   Ubuntu 22.04 FPGA Manager flow. Files: `pccx_npu.dtsi`,
   `shell.json`, `Makefile`.
6. **KV260 deploy**:
   ```bash
   sudo xmutil unloadapp
   sudo cp build/pccx_v002_kv260.bit.bin \
           sw/dtbo/pccx_npu.dtbo \
           sw/dtbo/shell.json \
           /lib/firmware/xilinx/pccx_npu/
   sudo xmutil loadapp pccx_npu
   dmesg | tail -30     # look for successful overlay + no AXI timeouts
   ```
7. **Driver smoke** — `sw/driver/` is still skeleton per the repo's
   implementation-status table; bring that up in parallel with the
   board-side work.

## Phase A placeholders still in effect

These do not block synth but make the board-side behaviour non-functional.
Flagged in `tools/phase0/phase_A_audit.md` on the docs repo:

- `GEMM_systolic_top.sv` truncates the 27-bit BF16 mantissa to 8 bits
  as a placeholder for the real PREPROCESS → INT8 path.
- No drain-every-1024 counter yet; the packer's 21-bit per-channel
  accumulator can overflow on long GEMM tiles.
- Weight streamer / DMA upstream of `GEMM_weight_dispatcher` does not
  yet emit two INT4 lanes per row pair — the plumbing exists, the
  source content still has to be organized.
