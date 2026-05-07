# xsim Verification Workflow

This repo uses a source-level xsim smoke suite for RTL bring-up evidence.
It is verification evidence only; it is not KV260 board evidence, timing
closure evidence, or a performance claim.

## Prerequisites

- Vivado xsim tools (`xvlog`, `xelab`, `xsim`) available on `PATH`.
- A sibling `pccx-lab` checkout, or `PCCX_LAB_DIR` set to that checkout.
  The runner builds `from_xsim_log` when the binary is missing.

## Runner Commands

From the repository root:

```bash
bash hw/sim/run_verification.sh
```

The same script can also be run from `hw/sim/`:

```bash
bash run_verification.sh
```

Selection flags:

| Flag | Behavior |
| --- | --- |
| `--full` | Run every testbench in the deterministic `TB_LIST`. This is the default when no selector is supplied. |
| `--quick` | Run the stable local smoke subset in `QUICK_TB_LIST`. Use this for fast pre-review checks. |
| `--tb <name>` | Run one known testbench. The name must exist in `TB_DEPS`. |
| `--list` | Print known testbench names in full-suite order and exit. |
| `-h`, `--help` | Print usage and exit. |

Common examples:

```bash
bash hw/sim/run_verification.sh --full
bash hw/sim/run_verification.sh --quick
bash hw/sim/run_verification.sh --tb tb_v002_runtime_smoke_program
bash hw/sim/run_verification.sh --list
```

Use one selector per command in normal evidence logs. If multiple
selectors are passed, the script processes them left to right and the
last selector that assigns a testbench set wins.

The runner executes testbenches in deterministic order. A testbench passes
only when its `xsim.log` contains a `PASS:` verdict. A missing verdict or
an explicit `FAIL:` verdict makes the runner exit nonzero.

Environment variables:

| Variable | Behavior |
| --- | --- |
| `PCCX_LAB_DIR` | Path to a sibling or external `pccx-lab` checkout. Defaults to `../pccx-lab` relative to this repo. |
| `XILINX_VIVADO` | Vivado install root. When set and `glbl.v` exists under it, the runner compiles `glbl` for XPM/CDC models. |

## Active Testbenches

| Testbench | Coverage focus |
| --- | --- |
| `tb_shape_const_ram` | `shape_const_ram` reset, write, read, hold, overwrite contract |
| `tb_mem_dispatcher_shape_lookup` | `mem_dispatcher` shape-constant lookup and LOAD pointer routing |
| `tb_GEMM_dsp_packer_sign_recovery` | W4A8 packer and sign recovery lane behavior |
| `tb_mat_result_normalizer` | GEMM result normalization to BF16 path |
| `tb_GEMM_weight_dispatcher` | GEMM weight dispatcher lane-valid handling |
| `tb_FROM_mat_result_packer` | GEMM result packer FSM |
| `tb_barrel_shifter_BF16` | BF16 to fixed-point barrel shifter |
| `tb_ctrl_npu_decoder` | ISA opcode decode to one-hot control |
| `tb_mem_u_operation_queue` | Memory operation queue smoke |
| `tb_GEMM_fmap_staggered_delay` | GEMM fmap/data-valid/instruction stagger alignment |
| `tb_v002_runtime_smoke_program` | Generated v002 runtime `.memh` program through decoder and Global Scheduler |

`tb_v002_runtime_smoke_program` is driven by
`tools/v002/generate_smoke_program.py`. The runner generates
`program.json` and `v002_runtime_smoke.memh` inside the testbench work
directory before xsim starts. This is a runtime handoff smoke only; it is
not model inference, board execution, or throughput evidence.

## Adding a Testbench

The runner is intentionally explicit: every testbench must declare its
RTL dependencies, trace lane, and suite membership in
`hw/sim/run_verification.sh`.

1. Add the self-checking testbench at `hw/tb/tb_<name>.sv`.
   The module name must match the filename without `.sv` because the
   runner invokes `xelab <tb_name>`.

2. Emit exactly one final verdict that starts with `PASS:` or `FAIL:`.
   The runner scans `xsim.log` for the first matching verdict line.
   Missing verdicts are treated as failures.

3. Add the compile dependency map entry. Paths are relative to `hw/rtl/`
   and should include packages, interfaces, and DUT modules in compile
   order:

   ```bash
   [tb_new_module]="NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv SUB_DIR/new_module.sv"
   ```

4. Add a trace core ID in `TB_CORE`. Pick an unused integer. Keeping IDs
   contiguous makes composed pccx-lab timelines easier to read:

   ```bash
   [tb_new_module]=11
   ```

5. Append the testbench name to `TB_LIST` in the order the full suite
   should run. Add it to `QUICK_TB_LIST` only when the test is stable,
   fast, and suitable for local smoke checks.

6. If the testbench needs generated inputs, write them under
   `hw/sim/work/<tb_name>/` from inside `run_tb()` before `xvlog`.
   Do not write generated stimuli into `hw/tb/` or commit simulator
   output.

7. Verify the runner metadata and the new test:

   ```bash
   bash hw/sim/run_verification.sh --list
   bash hw/sim/run_verification.sh --tb tb_new_module
   ```

8. For review evidence, include the exact command, Vivado version,
   runner summary, and the new `hw/sim/work/<tb_name>/` path if the
   test fails.

## Generated Evidence

Each testbench writes under:

```text
hw/sim/work/<tb_name>/
```

Typical files:

```text
xvlog.log
xelab.log
xsim.log
from_xsim_log.log
<tb_name>.pccx
```

The runner also prints every generated `.pccx` trace path so the trace can
be opened in pccx-lab.

## Git Policy

Generated simulator outputs stay untracked. `.gitignore` excludes the
normal generated locations and file types, including:

```text
hw/sim/work/
hw/build/
*.log
*.pb
*.wdb
*.jou
xsim.dir/
```

Do not commit generated logs, waveforms, build directories, bitstreams, or
model artifacts unless the repo adopts an explicit curated-evidence policy
for a specific file.

## Evidence Checklist

For a review or release handoff, record:

- Git commit or PR SHA under test.
- Vivado version.
- Exact command used.
- Full runner summary.
- Any non-PASS verdict and the corresponding `hw/sim/work/<tb_name>/` path.
- Whether `.pccx` traces were retained locally for pccx-lab inspection.
