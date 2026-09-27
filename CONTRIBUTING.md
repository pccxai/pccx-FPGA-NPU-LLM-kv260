# Contributing to pccx-FPGA-NPU-LLM-kv260

Thank you for your interest in contributing to the PCCX KV260 project.

This file covers repo-specific guidance. For the full org-wide guidelines
(branch naming, commit conventions, PR workflow), see the
**[pccxai org-wide CONTRIBUTING.md](https://github.com/pccxai/.github/blob/main/CONTRIBUTING.md)**.

## Where to start

- Browse issues labelled [`good first issue`](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/labels/good%20first%20issue).
- Open a [discussion](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/discussions) before starting non-trivial work.
- Search existing issues and PRs before filing a duplicate.

## Architecture spec

The canonical design rationale, ISA, memory map, and model-mapping notes
live on the **[pccx documentation site](https://docs.pccx.ai/en/docs/v002/index.html)**.
This repo holds the KV260 integration RTL and board bring-up — read the
spec first, then come back here for implementation details.

## How to run xsim verification

The repo uses a source-level xsim smoke suite for RTL bring-up evidence.

**Prerequisites:**

- Vivado xsim tools (`xvlog`, `xelab`, `xsim`) on `PATH`.
- A sibling legacy `pccx-lab` checkout, or `PCCX_LAB_DIR` set to that checkout.
  That repository is no longer an active public PCCX entry point. The current
  runner still requires its trace bridge; a clean contributor checkout must
  resolve this dependency before xsim can be advertised as self-contained.
  Track that onboarding gap in [#152](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/152).

**Run the full suite** from the repo root:

```bash
bash scripts/v002/use_submodule_sources.sh
```

**Quick smoke mode** (fewer testbenches):

```bash
bash scripts/v002/use_submodule_sources.sh --quick
```

**Single testbench:**

```bash
bash scripts/v002/use_submodule_sources.sh --tb tb_v002_runtime_smoke_program
```

See [docs/SIMULATION.md](docs/SIMULATION.md) for run log paths and
[pccx-lab's verification-workflow doc](https://docs.pccx.ai/en/lab/verification-workflow.html)
for the `from_xsim_log` converter.

## Pull requests

- Target `main`. Keep the diff focused on one concern.
- Describe motivation, approach, and any verification steps in the PR body.
- Link the related issue or discussion.
- Ensure CI (`repo-validate`) is green before requesting review.

## Code style

- Match the file's existing style; do not reformat unrelated code.
- Add tests or verification notes when behavior changes.
- Keep documentation in sync with behaviour changes.

## Engineering discipline

Architecture, release decisions, and hardware-sensitive verification are
maintainer-owned. Do not make production-ready, timing-closure, or
throughput claims without linked evidence.

## Reporting security issues

Do not file a public issue for security findings. See [SECURITY.md](SECURITY.md).
