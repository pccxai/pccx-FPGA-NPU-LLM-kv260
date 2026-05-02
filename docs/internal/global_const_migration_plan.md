# `GLOBAL_CONST.svh` Migration Plan

_Stage C deliverable, autonomous RTL cleanup pass._

`hw/rtl/Constants/compilePriority_Order/A_const_svh/GLOBAL_CONST.svh`
self-identifies as `DEPRECATED — use npu_arch.svh + kv260_device.svh
instead`. This document records the consumer survey, the destination
mapping for each surviving alias, and the staged migration order so a
future batch can retire the file without surprise.

A small first slice (TRUE / FALSE alias removal — 0 consumers) lands
in this Stage C pass; the rest stays for later batches per the
"file-by-file in later passes" wording of the Stage C decisions memo.

## 1. Consumer survey (current state)

`grep -rln "GLOBAL_CONST" hw/rtl/` shows the file is `\`include`-ed by
~38 RTL files plus `KELLER_REFACTOR_NOTES.md`. Most consumers include
it for legacy convention; only a small number actually reference one
of the legacy aliases. Splitting consumers by which alias they use:

> **Companion shim**: `hw/rtl/Constants/compilePriority_Order/A_const_svh/DEVICE_INFO.svh`
> is a second deprecated header in the same directory, declaring two
> legacy aliases (`DEVICE_HP_SINGLE_LANE_MAX_IN_BIT`, `DEVICE_HP_CNT`)
> over `kv260_device.svh`. As of the Stage C audit, both
> `\`include "DEVICE_INFO.svh"` and the two aliases have **zero active
> consumers** anywhere in `hw/rtl/`. The file is removed outright in
> the Stage C batch — see the companion deletion commit.

### 1.1 `\`TRUE` / `\`FALSE`

```
$ grep -lE '\`(TRUE|FALSE)\b' hw/rtl/ -r
hw/rtl/KELLER_REFACTOR_NOTES.md            (documentation only)
```

**Active consumers: 0.** These aliases predate the user's switch to
explicit `1'b1` / `1'b0` literals and are no longer reached by any
SystemVerilog source. Safe to delete from `GLOBAL_CONST.svh` now.

### 1.2 `\`HP_PORT_MAX_WIDTH` / `\`HP_PORT_SINGLE_WIDTH` / `\`HP_PORT_CNT`

```
$ grep -lE '\`HP_PORT_(MAX_WIDTH|SINGLE_WIDTH|CNT)\b' hw/rtl/ -r
hw/rtl/MAT_CORE/GEMM_systolic_top.sv
hw/rtl/MAT_CORE/GEMM_weight_dispatcher.sv
hw/rtl/NPU_top.sv
```

**Active consumers: 3.** All three are MAT_CORE / top-level files.
Each occurrence aliases through to a single `\`define` already present
in `npu_arch.svh` / `kv260_device.svh`:

| Legacy alias              | Authoritative source                  | Replacement              |
|---|---|---|
| `\`HP_PORT_MAX_WIDTH`     | `npu_arch.svh`                        | `\`HP_TOTAL_WIDTH`       |
| `\`HP_PORT_SINGLE_WIDTH`  | `npu_arch.svh`                        | `\`HP_SINGLE_WIDTH`      |
| `\`HP_PORT_CNT`           | `kv260_device.svh`                    | `\`DEVICE_HP_PORT_CNT`   |

### 1.3 `\`DSP48E2_*` / `\`PREG_SIZE`

```
$ grep -lE '\`(DSP48E2_|PREG_SIZE)' hw/rtl/ -r
hw/rtl/MAT_CORE/GEMM_dsp_unit.sv
hw/rtl/MAT_CORE/GEMM_dsp_unit_last_ROW.sv
hw/rtl/MAT_CORE/GEMM_systolic_array.sv
hw/rtl/MAT_CORE/GEMM_systolic_top.sv
hw/rtl/NPU_top.sv
```

**Active consumers: 5.** All MAT_CORE / top, mostly the protected DSP
files. Mapping:

| Legacy alias               | Authoritative source       | Replacement                |
|---|---|---|
| `\`DSP48E2_POUT_SIZE`      | `npu_arch.svh`             | `\`DSP_P_OUT_WIDTH`        |
| `\`DSP48E2_A_WIDTH`        | `kv260_device.svh`         | `\`DEVICE_DSP_A_WIDTH`     |
| `\`DSP48E2_B_WIDTH`        | `kv260_device.svh`         | `\`DEVICE_DSP_B_WIDTH`     |
| `\`PREG_SIZE`              | `npu_arch.svh`             | `\`DSP_P_OUT_WIDTH`        |

`\`PREG_SIZE` is identical to `\`DSP_P_OUT_WIDTH` already; the alias
is a textual leftover.

### 1.4 Cold includers

The remaining ~33 files `\`include "GLOBAL_CONST.svh"` but never
reference one of the legacy aliases. They probably include it for
historical reasons (the file used to be the umbrella header for
device + arch constants before those split out). Once §1.2 and §1.3
are migrated, the cold includers can have their `\`include` line
dropped mechanically — they will continue to compile because the same
`NUMBERS.svh` / `npu_arch.svh` / `kv260_device.svh` content reaches
them through other includes anyway.

## 2. Staged migration order

Each phase below is meant to be **its own commit**, with `xvlog`
clean and (where applicable) `bash hw/sim/run_verification.sh`
re-passing before the next phase starts.

### Phase 1 — drop `TRUE` / `FALSE` aliases (this Stage C batch)

Zero consumers. Remove the four lines from `GLOBAL_CONST.svh`. Lint
must remain clean. **Actioned in the next commit alongside this
document.**

### Phase 2 — migrate `\`HP_PORT_*` consumers (next batch)

Replace the three files in §1.2 sequentially:

  - `hw/rtl/MAT_CORE/GEMM_systolic_top.sv`
  - `hw/rtl/MAT_CORE/GEMM_weight_dispatcher.sv`
  - `hw/rtl/NPU_top.sv`

Mechanical text substitution per the table in §1.2. Re-run
`xvlog` and the existing TBs after each file (`tb_GEMM_weight_dispatcher`
covers one of the three directly; the others rely on lint).

Once all three migrate, delete the three `\`define HP_PORT_*` lines
from `GLOBAL_CONST.svh`.

### Phase 3 — migrate `\`DSP48E2_*` / `\`PREG_SIZE` consumers (next batch)

Same pattern for the five files in §1.3. Three of them are protected
(`GEMM_dsp_unit.sv`, `GEMM_dsp_unit_last_ROW.sv`,
`GEMM_systolic_array.sv` per CLAUDE.md §6.2 are partially or fully
protected) — touching the `\`define` reference inside a port
declaration is mechanical and does not change compute, but the
diff still warrants explicit review-with-the-user.

Re-run `tb_GEMM_dsp_packer_sign_recovery` after each file.

Once all five migrate, delete the four `\`define DSP48E2_* /
PREG_SIZE` lines from `GLOBAL_CONST.svh`.

### Phase 4 — strip cold `\`include "GLOBAL_CONST.svh"` lines

Mechanical pass over the ~33 files in §1.4: drop the include line.
Each file's other includes (`NUMBERS.svh`, `npu_arch.svh`,
`kv260_device.svh`, package imports) cover the symbols it actually
needs.

Recommend committing in subsystem-sized chunks (one commit per
top-level directory) so a regression bisects to a small surface.

### Phase 5 — delete `GLOBAL_CONST.svh`

Once §1.1, §1.2, §1.3, §1.4 are all done, `grep -r "GLOBAL_CONST"
hw/rtl/` returns zero hits. Remove the file and its include-path
entry (none today, but verify). Lint must remain clean.

## 3. Validation gates per phase

- `xvlog -f hw/vivado/filelist.f` reports 0 ERROR / 0 WARNING.
- `bash hw/sim/run_verification.sh` reports the same 6 PASS as the
  Stage C baseline.
- Synthesis output equivalent (utilization / timing reports unchanged
  modulo retiming noise) — strongly recommended before deleting the
  file in Phase 5, since the legacy `\`define` symbols may show up in
  third-party IP wrapper code that this audit did not cover.

## 4. Risk notes

- The protected files in §1.3 (`GEMM_dsp_unit*.sv`, `GEMM_systolic_array.sv`)
  use the legacy alias inside actual port declarations and DSP-attribute
  expressions. The replacement is textual but should be reviewed
  with the user even though it does not touch compute math.
- `KELLER_REFACTOR_NOTES.md` references the legacy aliases as
  documentation. Update or remove the file when its references are
  no longer applicable.
- `GLOBAL_CONST.svh` is included via `-i .../A_const_svh` on the xvlog
  command line, not via the filelist itself. The include-path entry
  in `hw/vivado/filelist.f` and any TCL build wrapper must stay until
  Phase 5 lands.
