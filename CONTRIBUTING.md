# Contributing

Start with the organization-wide guide: <https://github.com/pccxai/.github/blob/main/CONTRIBUTING.md>.
This file only adds the FPGA/KV260-specific review rules for this repo.

The canonical architecture and ISA documentation lives in the pccx docs site:
<https://pccxai.github.io/pccx/en/docs/v002/index.html>.

## Filing Issues

Use the issue form that matches the work. The org community-health templates
tracked in `pccxai/.github` PR #9 define Bug, Feature, and Evidence forms:
<https://github.com/pccxai/.github/pull/9>.

- Bug: include the affected commit or branch, minimal reproduction steps,
  observed result, expected result, environment, and logs.
- Feature: state the problem, smallest useful proposal, explicit boundaries,
  references, and acceptance checks.
- Evidence: link the claim, release gate, or review decision, then attach logs,
  reports, commits, board records, or say what evidence is still missing.

For starter work, search issues with the `agent-safe` label:
<https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/labels/agent-safe>.
Good starter issues are normally narrow docs, tests, scripts, wiring, or
scaffolding tasks with explicit validation steps.

## Making Pull Requests

Use a short topic branch, for example `docs/<slug>`, `tb/<slug>`,
`feat/<slug>`, or `fix/<slug>`. Keep each PR tied to one issue or one small
reviewable change. Do not mix generated simulator output, Vivado build
artifacts, or unrelated cleanup into the branch.

Before opening the PR:

- Rebase or merge from the intended base branch deliberately; do not hide base
  drift with an unexplained diff.
- Run the relevant local checks and paste the exact commands and verdicts.
- Run `git diff --check`.
- Run `bash scripts/v002/claim-scan.sh` when touching public docs, release
  notes, README text, evidence summaries, or PR-facing wording.
- Do not use `--no-verify` for commits or pushes.
- Do not force-push.

PR wording must be evidence-gated. Say what the patch changes and what checks
passed. Do not claim timing closure, generated bitstreams, KV260 board
execution, Gemma 3N E4B hardware execution, measured throughput, production
readiness, release readiness, or stable interfaces unless the PR includes the
specific evidence that proves it.

## Running xsim Regression

The local xsim entry point is `hw/sim/run_verification.sh`:

```bash
bash hw/sim/run_verification.sh
```

```bash
bash hw/sim/run_verification.sh --quick
bash hw/sim/run_verification.sh --tb tb_v002_runtime_smoke_program
bash hw/sim/run_verification.sh --list
```

The runner expects Vivado xsim tools on `PATH`. It writes per-testbench logs
and `.pccx` traces under `hw/sim/work/<tb_name>/`; those files are generated
evidence and should stay untracked. A testbench passes only when its
`xsim.log` contains a `PASS:` verdict.

## Coordinating With PR #79

PR #79 is the active v002 synth/timing evidence branch:
<https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/79>.

If your work changes RTL, timing evidence, Vivado scripts, driver handoff
state, or wording about the current implementation status, check whether it
should target or be rebased against PR #79's active branch before opening a
main-based PR. Docs-only starter tasks that do not alter those claims can
target `main`, but should still avoid contradicting PR #79's evidence state.
