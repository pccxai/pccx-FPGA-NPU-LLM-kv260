# KV260 Gemma 3N E4B — Hardware Handoff Notes

_Branch: `rtl/kv260-gemma3n-e4b-vivado`. Date: 2026-05-02._

This file is the hardware-side handoff for the parallel runtime worker
working in `pccx-FPGA-NPU-LLM-kv260-bringup` on
`bringup/kv260-gemma3n-e4b-runtime`. It tells the runtime worker what
the hardware path can and cannot do tonight, what command to run to
rebuild the bitstream when the BD lands, and which assumptions the
runtime side must respect.

## 1. Current hardware status (snapshot)

| Stage | Status | Evidence |
|-------|--------|----------|
| RTL compile (`xvlog -f filelist.f`, 52 files) | clean — 0 error, 0 warning | `hw/build/xvlog_filelist/xvlog_with_wrapper.log` |
| TB regression (7 testbenches) | 7/7 PASS | `hw/sim/work/<tb>/xsim.log` |
| `npu_core_wrapper` xelab (param fix) | clean past wrapper, hits pre-existing GEMV/`glbl` warnings | `hw/build/xvlog_filelist/xelab_wrapper2.log` |
| Vivado `create_project.tcl` (KV260 part) | success | `hw/build/vivado_project.log` |
| Vivado `synth.tcl` (OOC, NPU_top) | **aborted intermediate run** — see §1.b below |
| Vivado `impl.tcl` (place / route / bitstream) | not attempted — gated on BD + clean synth |
| Block design (`system_bd.tcl`) | not yet authored |
| Device-tree overlay (`sw/dtbo/`) | not yet authored |
| Bitstream output (`hw/build/pccx_v002_kv260.bit`) | not yet produced |
| KV260 smoke run | not possible tonight — see §5 blockers |

### 1.b OOC synth was an intermediate validation run only

OOC synth was attempted as an intermediate wrapper/top validation
run. It did not complete on this 12 GB host due to swap pressure.
This is **not** the final bitstream flow; final synthesis and
implementation should be rerun on a higher-memory host after BD /
DTBO packaging is in place.

Concretely:

- The run was launched with `./vivado/build.sh synth` against
  `NPU_top` in OOC mode. It progressed through RTL elaboration,
  FSM inference, DSP48E2 transformation (1120 instances), into the
  Cross Boundary and Area Optimization phase, and then stalled with
  the Vivado-internal "Thrashing Detected" notice as the host (12 GB
  RAM + 31 GB swap) ran out of resident memory.
- The run was deliberately stopped — not killed by OOM. The final
  log tail is preserved in
  `hw/build/synth_aborted_2026_05_02/runme.log` (1442 lines) so a
  later host can compare what was reachable.
- The wrapper / interface fixes documented in §4 do not depend on
  this run for validation — they were verified independently by
  xvlog, xelab on the wrapper, and the TB regression suite.
- No timing report, no DRC report, no utilization report, no
  bitstream was produced. The runtime worker must not interpret this
  as silicon-readiness.

When the BD/DTBO path lands and the build is moved to a higher-
memory host, the same `./vivado/build.sh synth` (or `impl`) command
re-runs the standard flow.

## 2. Top module and build commands

The synthesis target is `NPU_top` (interface-port form, used under
out-of-context synth). The packaging target for BD insertion is
`npu_core_wrapper` (plain-signal shim around `NPU_top`).

```bash
cd hw

# 1. Create the project (fast)
./vivado/build.sh project

# 2. Out-of-context synthesis of NPU_top (~10-30 min on this host)
./vivado/build.sh synth

# 3. Full implementation + write_bitstream (~1 h; requires BD first)
./vivado/build.sh impl
```

Vivado install used: `/tools/Xilinx/2025.2/Vivado/bin/vivado`.
Target part: `xck26-sfvc784-2LV-c` (Kria KV260 SOM, ZU5EV).
Target board file: `xilinx.com:kv260_som:part0:1.4` (detected, applied).

After successful `impl`, the bitstream is copied to:

```
hw/build/pccx_v002_kv260.bit
```

The KV260 firmware-load flow expects `.bit.bin` (bif-converted),
`.dtbo`, and `shell.json` under `/lib/firmware/xilinx/pccx_npu/` — see
`hw/vivado/README.md §Next steps to reach a running board` for the
exact sequence.

## 3. Runtime-visible interface contract

This is the contract the runtime worker must follow to drive the NPU
once a bitstream is loaded. All values come from the actual RTL, not
guesses.

### 3.1 Clocks and resets

| Net          | Direction | Period   | Notes |
|--------------|-----------|----------|-------|
| `clk_core`   | input     | 2.500 ns | 400 MHz, compute domain (DSP48E2 array, GEMV, CVO). |
| `clk_axi`    | input     | 4.000 ns | 250 MHz, AXI / control plane. |
| `rst_n_core` | input     | sync     | active-low; release synchronously to `clk_core`. |
| `rst_axi_n`  | input     | sync     | active-low; release synchronously to `clk_axi`. |
| `i_clear`    | input     | sync     | active-high soft-clear, gated by reset; one pulse clears latched state across MAT/VEC/CVO/PREPROCESS/MEM. |

The two clock domains are declared `set_clock_groups -asynchronous` in
`hw/constraints/pccx_timing.xdc`. Every PS↔NPU path must terminate at
one of the existing CDC FIFOs (`mem_HP_buffer`, `mem_dispatcher`,
ACP path) — do not add ad-hoc CDC.

### 3.2 AXI-Lite control plane (`S_AXIL_CTRL`)

| Property | Value |
|----------|-------|
| Address width (`ADDR_W`) | **12 bits** (4 KB MMIO window) |
| Data width   (`DATA_W`)  | **64 bits** |
| Defined in  | `hw/rtl/NPU_Controller/npu_interfaces.svh` (`axil_if`) |
| Consumed by | `hw/rtl/NPU_Controller/NPU_frontend/AXIL_CMD_IN.sv` (12-bit `s_awaddr`, 64-bit `s_wdata`) and `AXIL_STAT_OUT.sv` |

The Zynq PS HPM port is 32-bit AXI-Lite. The Block Design must place a
**Smart Connect (or AXI Interconnect) between the PS HPM port and the
IP `S_AXIL_CTRL`** so the 32→64 width step is handled by the BD, not
RTL. This is the standard Vivado pattern; do not change RTL widths to
work around this.

`ISA_WIDTH = 64` (see `rtl/Constants/compilePriority_Order/A_const_svh/npu_arch.svh`)
— a single AXI-Lite write of one 64-bit VLIW instruction issues one
NPU op. The frontend does not require a doubled-write protocol once
the Smart Connect is in place.

### 3.3 Status word (`AXIL_STAT_OUT`, bits aggregated by `NPU_top`)

```
mmio_npu_stat[31:0]:
  bit  0 : BUSY  = fifo_full | cvo_busy | cvo_disp_busy
  bit  1 : DONE  = CVO operation complete (one-cycle pulse)
  bits 31:2 reserved (driven 0)
```

Source: `hw/rtl/NPU_top.sv:393-400`.

### 3.4 AXI-Stream ports

| Port | Direction | Width | Routed via | Backpressure |
|------|-----------|-------|------------|--------------|
| `S_AXI_HP0_WEIGHT` | slave  | 128 b | mem_HP_buffer (CDC, axi→core) | tready |
| `S_AXI_HP1_WEIGHT` | slave  | 128 b | mem_HP_buffer | tready |
| `S_AXI_HP2_WEIGHT` | slave  | 128 b | mem_HP_buffer | tready |
| `S_AXI_HP3_WEIGHT` | slave  | 128 b | mem_HP_buffer | tready |
| `S_AXIS_ACP_FMAP`  | slave  | 128 b | mem_dispatcher / preprocess_fmap | tready |
| `M_AXIS_ACP_RESULT`| master | 128 b | mem_dispatcher | tready (PS-side) |

HP0 / HP1 carry packed INT4 weights for the 32×32 systolic array
(dual-lane W4A8 — HP0 is upper INT4 channel, HP1 is lower). HP2 / HP3
feed the GEMV core. ACP carries fmap input and result output.

### 3.5 ISA opcodes (one active per cycle at the frontend)

```
OP_GEMM   : ACP_FMAP → preprocess_fmap → systolic → normalizer → packer → ACP_RESULT
OP_GEMV   : ACP_FMAP → preprocess_fmap → GEMV_top  (HP2/HP3 weights)
OP_MEMCPY : ACP DDR4 ↔ L2 (mem_dispatcher via ACP)
OP_MEMSET : Shape constant RAM write (mem_dispatcher)
OP_CVO    : L2 → CVO_top → L2 (mem_dispatcher ↔ CVO stream bridge)
```

Source: `hw/rtl/NPU_top.sv:29-34`.

## 4. What was fixed in this branch

`npu_core_wrapper.sv` (the BD packaging shim) had a parameter-name and
parameter-value mismatch against `axil_if` in `npu_interfaces.svh`,
which was caught by `xelab` static elaboration:

```
ERROR: [VRFC 10-3480] interface 'axil_if' does not have a parameter
named 'ADDR_WIDTH'  (line 80)
```

The interface declares parameters `ADDR_W` / `DATA_W`, defaults
`12 / 64`. The wrapper was overriding `ADDR_WIDTH` / `DATA_WIDTH` (no
such names) with values `32 / 32` (silently inactive override → would
fall back to defaults) and was instantiating the interface with an
empty port list against the `(input clk, input rst_n)` declaration.

Fix (single hunk, blocker-only):

- Rename overrides: `ADDR_WIDTH`→`ADDR_W`, `DATA_WIDTH`→`DATA_W`.
- Align defaults to `axil_if`: `AXIL_ADDR_W = 12`, `AXIL_DATA_W = 64`.
- Pass the AXI clock / reset to the interface instance (`axil_inst (.clk(clk_axi), .rst_n(rst_axi_n))`).
- Add a header comment explaining the BD Smart Connect 32→64 step so
  the wrapper does not look “wrong” to a reviewer used to 32-bit AXIL.

This unblocks the wrapper for IP packaging in the BD step.

## 5. Remaining hardware blockers (in order)

These prevent a true KV260 board run tonight. They are the multi-hour
items the BD owner needs to land before the runtime worker can do a
genuine KV260 smoke test on the actual NPU.

1. **`hw/vivado/system_bd.tcl` missing.** Needs:
   - Zynq UltraScale+ MPSoC IP (KV260 preset, HPM 32-bit AXI-Lite,
     four HP weight ports, ACP fmap/result paths).
   - `npu_core_wrapper` packaged as IP and instantiated.
   - AXI Smart Connect between PS HPM and IP `S_AXIL_CTRL` (32→64).
   - Clock Wizard producing `clk_core` (400 MHz) and `clk_axi`
     (250 MHz) from the PS reference clock.
   - Processor System Reset blocks for both domains.
2. **`./vivado/build.sh impl`** — runs implementation + write_bitstream
   only after `system_bd.tcl` lands and `synth_1` is clean. Hour-scale.
3. **`sw/dtbo/`** (does not yet exist) — needs `pccx_npu.dtsi`,
   `shell.json`, `Makefile`. The KV260 firmware-load flow is
   `xmutil unloadapp / loadapp pccx_npu`.
4. **Driver bring-up** — `sw/driver/` is skeleton-only per the
   implementation status table; does not block bitstream creation but
   is required for the runtime worker’s smoke test.

## 6. Exact handoff for the runtime worker (tonight)

Until item (1) and (2) above land, the runtime worker should treat the
hardware path as **not yet runnable on a real board**. Concretely:

- Do not assume `/lib/firmware/xilinx/pccx_npu/` is populated.
- Do not assume `xmutil loadapp pccx_npu` will succeed; it will not
  find the overlay.
- The MMIO contract above (§3) is stable and can be wired into the
  user-space driver and ISA encoder ahead of bitstream availability.
- Soft-loopback / null-bitstream stand-ins must be clearly labelled as
  such in any logs that get committed.

When the bitstream is ready (after BD + impl land in this branch):

```bash
# from the rtl worktree, after `./vivado/build.sh impl`:
ls hw/build/pccx_v002_kv260.bit              # must exist
# bif → .bit.bin conversion happens in the BD owner's commit
# install:
sudo cp hw/build/pccx_v002_kv260.bit.bin \
        sw/dtbo/pccx_npu.dtbo \
        sw/dtbo/shell.json \
        /lib/firmware/xilinx/pccx_npu/
sudo xmutil unloadapp
sudo xmutil loadapp pccx_npu
dmesg | tail -30
```

The MMIO base address and HP/ACP DMA bindings come from the BD-
generated address map; the runtime smoke test should consume those at
load time, not hard-code them.

## 7. Phase A placeholders still in effect

Documented in `hw/vivado/README.md §Phase A placeholders` and in the
docs repo Phase A audit. They do not block synth but do limit what a
board run can prove:

- `GEMM_systolic_top.sv` truncates the 27-bit BF16 mantissa to 8 bits
  (placeholder for the real PREPROCESS → INT8 path).
- No drain-every-1024 counter yet; the packer’s 21-bit per-channel
  accumulator can overflow on long GEMM tiles.
- Weight streamer / DMA upstream of `GEMM_weight_dispatcher` does not
  yet emit two INT4 lanes per row pair — the plumbing exists, the
  source content still has to be organised.

These are the “does the bitstream actually run a Gemma 3N E4B layer
end-to-end?” items, separate from “does the bitstream build at all?”.

## 8. RTL gaps surfaced during this synth pass (not Phase A placeholders)

These are real RTL bugs (or near-bugs) that the first OOC synth pass on
this branch surfaced. They are out of scope for the unblock task but
must be fixed before the runtime worker trusts the GEMV path on
silicon. They sit in a different category from §7 — those are
*deferred design choices*, these are *implementation gaps the build
just told us about*.

1. **`GEMV_generate_lut.sv:47` — out-of-bounds 2D array write.**
   `OUT_fmap_LUT[idx][w]` is declared `[0:param.weight_width-1]`
   (`weight_width = 4`) but the inner loop runs `for (w = 0; w < 16;
   w++)`. Writes to `w = 4..15` are out of bounds. xvlog warns
   (`VRFC 10-3705 select index 4 into OUT_fmap_LUT is out of bounds`);
   synth proceeds. Likely the array dimension is wrong — the comment
   above the loop says “LUT entry order: descending (w = 0 .. 15 maps
   to weight = -8 .. 7)”, so the array should be `[0:15]` (or
   `[0:(2**weight_width)-1]`).
2. **`GEMV_generate_lut.sv:37` — `OUT_fmap_ready` has no driver.**
   Synth log: `Synth 8-3848 Net OUT_fmap_ready in module GEMV_generate_lut
   does not have driver.` The output is declared but never assigned.
3. **`emax_pipe_reg` — 16384-register 3D RAM warning.**
   Synth log: `Synth 8-11357 Potential Runtime issue for 3D-RAM or RAM
   from Record/Structs for RAM emax_pipe_reg with 16384 registers.`
   This is a synthesis runtime / quality-of-result warning, not a
   correctness blocker, but it points at a register-array structure
   that should probably be moved to a BRAM-friendly layout before
   impl.

The runtime worker can ignore these for the AXIL/MMIO and GEMM/CVO
paths (they are GEMV-internal) but must not advertise the GEMV path as
silicon-ready until at least items (1) and (2) are fixed.

A deeper xelab pass on `npu_core_wrapper` (with `unisims_ver` /
`unimacro_ver` / `secureip` / `glbl` linked in) surfaces a few more
integration-level findings that are also pre-existing — they do not
fire on any of the unit testbenches because TBs do not stitch the
full top together:

4. **`mem_dispatcher.sv:77` — `fmap_shape_read_address` multi-driver.**
   `ERROR: [VRFC 10-3818] variable 'fmap_shape_read_address' is driven
   by invalid combination of procedural drivers`. This is a genuine
   RTL bug (one variable assigned in two always blocks).
5. **`ctrl_npu_frontend.sv:95` — port width mismatch.** `WARNING:
   [VRFC 10-9543] actual bit length 65 differs from formal bit length
   64 for port 'IN_data'`. Likely an off-by-one on a sign-extended
   inst path; not load-bearing but dirty.
6. **`preprocess_fmap.sv:82` — port width mismatch.** `WARNING: [VRFC
   10-9543] actual bit length 128 differs from formal bit length 256
   for port 's_axis_tdata'`. The ACP fmap path is 128-bit at the wrapper
   port; whatever is declaring 256-bit is stale.
7. **`ctrl_npu_frontend.sv:62` — `s_awprot` not connected.** Tied off
   silently; harmless on Zynq HPM but should be wired.

These all sit upstream / sideways of the wrapper change in this PR
and are not introduced by it. They are listed here so the BD / impl
owner is not surprised when the next clean-host synth surfaces them.

## 9. Risk notes

- The OOC synthesis run is the first end-to-end synth pass on this
  branch since the wrapper / Stage C / Phase 3 step 1 changes; expect
  warning-level findings (driver inference, unused signals) that must
  be checked against `hw/build/reports/timing_summary_post_synth.rpt`
  and `drc_post_synth.rpt` before promoting to impl.
- The wrapper change is signal-level only — no behavioural delta on
  `NPU_top`. The TB suite is the regression evidence (7/7 PASS).
- Do not claim KV260 inference success or timing closure unless the
  board run logs prove it. Until §5 items land, the public alpha
  scope statement (“FPGA bring-up in progress, no production claims”)
  is the only honest description.
