# KV260 XDC Constraints Policy

This document records the pccx v002 KV260 constraint boundary for pin
mapping, clock constraints, and false-path usage. It is a policy and
review checklist; it does not claim timing closure. Timing closure still
requires a post-implementation timing summary showing the checked
constraints are met.

## Files And Ownership

| Path | Owner | Scope |
|------|-------|-------|
| `hw/constraints/pccx_timing.xdc` | pccx NPU core | OOC timing constraints for `NPU_top`: clocks, CDC intent, reset synchronizer paths, and reviewed multicycle paths. |
| `hw/vivado/create_project.tcl` | Vivado project flow | Adds every `*.xdc` under `hw/constraints/` to `constrs_1`. |
| Future BD-level XDC | KV260 block-design owner | Any board-facing `PACKAGE_PIN`, `IOSTANDARD`, or physical port constraints introduced by the block design. |

Do not add KV260 board pin placement to `pccx_timing.xdc`. That file is
for the PL-side NPU core and must remain reusable for out-of-context
synthesis.

## Pin Map Policy

The current `NPU_top` is a PS-connected PL IP block. It does not expose
board-facing GPIO, PMOD, HDMI, MIPI, Ethernet, or other package pins.
Physical KV260 pin ownership is therefore outside the NPU core XDC:

- PS MIO, DDR, board clocks, and fixed board peripherals are owned by
  the Zynq UltraScale+ MPSoC configuration and the KV260 board preset
  (`xilinx.com:kv260_som:part0:1.4`) when that board file is installed.
- PL fabric connections between the Zynq PS, SmartConnect / AXI
  interconnect, clocking resources, resets, and `npu_core_wrapper` are
  owned by the block design.
- `hw/constraints/pccx_timing.xdc` must not contain `PACKAGE_PIN`,
  `IOSTANDARD`, or LOC constraints for board pins.

Logical NPU wrapper ports map to the KV260 block design as follows:

| Wrapper port or interface | Width / type | Intended BD source or sink |
|---------------------------|--------------|----------------------------|
| `clk_axi` | Clock | 250 MHz AXI/control-plane clock from PS FCLK or BD clocking resource. |
| `rst_axi_n` | Active-low reset | AXI-domain reset synchronized in the BD or reset controller. |
| `clk_core` | Clock | 400 MHz NPU compute clock from Clock Wizard / MMCM or equivalent BD clocking resource. |
| `rst_n_core` | Active-low reset | Core-domain reset synchronized in the BD or reset controller. |
| `i_clear` | 1-bit soft clear | Control-plane register output or BD-owned control signal. |
| `S_AXIL_CTRL` | AXI4-Lite slave, 12-bit address, 64-bit data | Zynq PS HPM through SmartConnect / width conversion as needed. |
| `S_AXI_HP0_WEIGHT` | AXIS slave, 128-bit data | GEMM weight stream corresponding to PS HP0-side DMA or stream bridge. |
| `S_AXI_HP1_WEIGHT` | AXIS slave, 128-bit data | GEMM weight stream corresponding to PS HP1-side DMA or stream bridge. |
| `S_AXI_HP2_WEIGHT` | AXIS slave, 128-bit data | GEMV weight stream corresponding to PS HP2-side DMA or stream bridge. |
| `S_AXI_HP3_WEIGHT` | AXIS slave, 128-bit data | GEMV weight stream corresponding to PS HP3-side DMA or stream bridge. |
| `S_AXIS_ACP_FMAP` | AXIS slave, 128-bit data | Coherent feature-map input path from the ACP/HPC-side BD bridge. |
| `M_AXIS_ACP_RESULT` | AXIS master, 128-bit data | Coherent result output path to the ACP/HPC-side BD bridge. |

If a future change adds external PL IO, put its physical constraints in a
separate board-level XDC and document the board revision, connector, pin,
voltage standard, and source schematic. Keep that board XDC separate from
the OOC timing policy.

## Clock Constraints

`hw/constraints/pccx_timing.xdc` defines the two NPU clock domains used
by `NPU_top`:

| Clock | Port | Period | Frequency | Scope |
|-------|------|--------|-----------|-------|
| `axi_clk` | `clk_axi` | 4.000 ns | 250 MHz | AXI-Lite control path, AXI-side FIFO ports, and stream ingress / egress attached to the AXI domain. |
| `core_clk` | `clk_core` | 2.500 ns | 400 MHz | DSP48E2 compute path, GEMV lanes, CVO SFU, schedulers, and core-side FIFO ports. |

The XDC also declares `axi_clk` and `core_clk` as asynchronous clock
groups. A crossing between these domains must be implemented through a
CDC FIFO, reset synchronizer, or another explicitly reviewed CDC
structure. Do not rely on the asynchronous group to excuse an ordinary
single-cycle data path.

The BD must generate and connect the actual clock nets that feed
`clk_axi` and `clk_core`. The core XDC only constrains the port-level
timing intent for analysis.

## False-Path Policy

False paths are allowed only for paths that are structurally
asynchronous and reviewed as CDC or reset behavior. They must not be used
to hide normal failing timing paths.

Current allowed false-path classes:

| XDC pattern | Reason |
|-------------|--------|
| `set_false_path -to .../u_reset_sync*/sync_reg_reg[0]` | The first flop of a reset synchronizer receives an asynchronous reset-side input by construction. |
| `set_false_path -from .../wr_pntr_gray_reg* -to .../wr_pntr_gray_sync_reg*` | XPM async FIFO write-pointer gray-code crossing into the opposite clock domain. |
| `set_false_path -from .../rd_pntr_gray_reg* -to .../rd_pntr_gray_sync_reg*` | XPM async FIFO read-pointer gray-code crossing into the opposite clock domain. |

Any new false path must include:

- A short comment in the XDC explaining the CDC or reset structure.
- A bounded object pattern that matches only the intended cells or pins.
- Review evidence from `report_clock_interaction`, `report_cdc`, or the
  relevant Vivado timing report.
- A note in the PR explaining why a max-delay, generated clock, or
  ordinary timing fix is not the right constraint.

The GEMM accumulator drain multicycle path in `pccx_timing.xdc` is not a
false path. It must remain documented as a protocol-timed path where the
controller stalls new MAC work during flush.

## Review Checklist

Before merging an XDC change:

- Confirm no board-level pin placement was added to
  `hw/constraints/pccx_timing.xdc`.
- Confirm `core_clk` remains 2.500 ns and `axi_clk` remains 4.000 ns, or
  document the architectural reason for a target change.
- Confirm every async clock crossing has an implemented CDC structure.
- Confirm every false path is narrow, commented, and backed by report
  evidence.
- Confirm PR wording avoids timing-closure claims unless a real
  post-implementation timing summary supports them.

