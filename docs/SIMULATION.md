# xsim Verification Workflow

This repo uses a source-level xsim smoke suite for RTL bring-up evidence.
It is verification evidence only; it is not KV260 board evidence, timing
closure evidence, or a performance claim.

## Prerequisites

- Vivado xsim tools (`xvlog`, `xelab`, `xsim`) available on `PATH`.
- A sibling `pccx-lab` checkout, or `PCCX_LAB_DIR` set to that checkout.
  The runner builds `from_xsim_log` when the binary is missing.

## Run Command

From the repository root:

```bash
bash scripts/v002/use_submodule_sources.sh
```

The wrapper forwards arguments to the v002 runner inside the
submodule, so quick smoke and per-testbench modes work the same way:

```bash
bash scripts/v002/use_submodule_sources.sh --quick
bash scripts/v002/use_submodule_sources.sh --tb tb_v002_runtime_smoke_program
```

The runner executes testbenches in deterministic order. A testbench passes
only when its `xsim.log` contains a `PASS:` verdict. A missing verdict or
an explicit `FAIL:` verdict makes the runner exit nonzero.

## Active Testbenches

| Testbench | Coverage focus |
| --- | --- |
| `tb_shape_const_ram` | `shape_const_ram` reset, write, read, hold, overwrite contract |
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

## Generated Evidence

Each testbench writes under the submodule's sim work directory:

```text
third_party/pccx-v002/LLM/sim/work/<tb_name>/
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
- Any non-PASS verdict and the corresponding `third_party/pccx-v002/LLM/sim/work/<tb_name>/` path.
- Whether `.pccx` traces were retained locally for pccx-lab inspection.
