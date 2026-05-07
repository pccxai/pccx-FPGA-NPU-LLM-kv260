# Repository Documentation Index

This page is a navigation aid for repo-local documentation. The
architecture and ISA reference remains the pccx documentation site:
<https://pccxai.github.io/pccx/en/docs/v002/index.html>.

## v002.1 Ramp Docs

| Area | Document | Source PR |
| --- | --- | --- |
| Spec decisions | [v002 resolution notes](spec/v002-resolution.md) | [#80](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/80) |
| Build | [KV260 bitstream build runbook](runbooks/v002.1-bitstream-build.md) | [#82](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/82) |
| Evidence | [v002.1 evidence inventory](evidence/v002.1-evidence-inventory.md) | [#95](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/95) |
| Timing / smoke | [Post-implementation timing and bitstream smoke plan](runbooks/v002.1-post-impl-timing-and-bitstream-smoke.md) | [#96](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/96) |
| Release | [Release docs README](releases/README.md) | [#97](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/97) |
| Release | [v002.1-rc.0 notes draft](releases/v002.1-rc.0-notes-DRAFT.md) | [#97](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/97) |
| Contributor workflow | [Contributing guide](../CONTRIBUTING.md) | [#98](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/98) |
| Repo entry point | [README v002.1 ramp links](../README.md) | [#99](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/99) |
| Deploy | [KV260 bitstream deploy procedure](runbooks/v002.1-bitstream-deploy.md) | [#100](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/100) |
| Release history | [Changelog](../CHANGELOG.md) | [#102](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/102) |
| Timing | [Full-top timing improvement playbook](spec/v002.1-timing-improvement-playbook.md) | [#105](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/pull/105) |

## Existing Repo-Local Docs

| Area | Document |
| --- | --- |
| KV260 bring-up | [KV260 bring-up evidence checklist](KV260_BRINGUP.md) |
| Model handoff | [Gemma 3N E4B target-model handoff boundary](GEMMA3N_HANDOFF.md) |
| Verification | [xsim evidence workflow](SIMULATION.md) |
| Timing evidence | [Vivado timing evidence checklist](TIMING_EVIDENCE.md) |
| W4A8 validation | [W4A8 golden-vector gate](W4A8_GOLDEN_VECTOR_GATE.md) |
| Release readiness | [Release evidence checklist](RELEASE_EVIDENCE_CHECKLIST.md) |
| Published prerelease | [v0.1.0-alpha release notes](releases/v0.1.0-alpha.md) |

## Notes

- Source PR links identify where each v002.1 ramp document entered
  review.
- Plans, drafts, inventories, and runbooks are indexed by their stated
  purpose; this page does not claim bitstream, timing, or board-smoke
  completion.
