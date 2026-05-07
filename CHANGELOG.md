# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## v002.1 ramp PR inventory — 2026-05-07

Evidence-gated inventory of draft PRs #80-#100 queried for the v002.1
ramp on 2026-05-07. This is not a release note and does not claim
merge, tag, timing closure, bitstream availability, or KV260 board
bring-up.

### Spec

- #80 `docs(spec): resolve v002 quantization, K-drain, and DSP baseline`

### TBs

- #81 `tb(gemm): verify W4A8 DSP dual-MAC and sign recovery`
- #83 `tb(npu_top): minimal end-to-end integration tb`
- #84 `tb(preprocess): bf16 to int8 e_max golden`
- #85 `tb(gemm): validate dsp packer and sign recovery`
- #89 `tb(gemv): verify reduction pipeline and sfu/l2 output mux`
- #91 `tb(gemm): re-enable fmap staggered-delay after int8 transition`
- #92 `tb(sched): global_scheduler hazard and chaining`
- #93 `tb(memcpy): add mem_dispatcher comprehensive testbench`
- #94 `tb(memory): verify uram l2 mapping and arbitration`

### RTL fixes

- #86 `feat(stat): connect engine completion to mmio_npu_stat`
- #87 `feat(gemv): connect lane mask and remove activation hardcode`
- #90 `feat(npu_top): connect result packer ready/valid and STORE writeback`

### Infra

- #82 `build(vivado): add KV260 bitstream runbook and BD scaffold`
- #88 `infra(synth): add baseline recording scripts and runbook`

### Docs

- #95 `docs(evidence): v002.1 evidence inventory`
- #96 `docs(runbook): post-impl timing closure and bitstream smoke plan`
- #97 `docs(release): v002.1-rc.0 notes draft`
- #98 `docs(contrib): add CONTRIBUTING.md`
- #99 `docs(readme): link v002.1 ramp artifacts`
- #100 `docs(runbook): v002.1 bitstream deploy procedure`

## v0.1.0-alpha — 2026-05-01

First public RTL preview of the pccx v002 NPU implementation targeting
Xilinx Kria KV260 (Zynq UltraScale+ ZU5EV) under the `pccxai`
organization. Mirrors the release notes at
[`docs/releases/v0.1.0-alpha.md`](docs/releases/v0.1.0-alpha.md).

### Highlights

- First public RTL preview of the pccx v002 NPU implementation
  targeting Xilinx Kria KV260 (Zynq UltraScale+ ZU5EV) under the
  `pccxai` organization.
- SystemVerilog testbenches for `tb_ctrl_npu_decoder` and
  `tb_barrel_shifter_BF16` (BF16 to 27-bit fixed-point) wired into the
  verification flow.
- Multi-driver issue on `OUT_fetch_PC_ready` repaired in the decoder
  testbench path.
- Sail ISA model — types, regs, decode, and execute increments 1 to 3
  (per-opcode MAC, DMA, and SFU effects, and per-opcode operand
  latching). CI installs `z3` and runs the `Sail typecheck` workflow on
  every PR.
- `repo-validate` required status check on `main`; covers URL hygiene,
  brand-name guard, and license / citation sanity.
- Verification workflow section in the README explains how to run the
  testbench suite locally and what each `tb` covers.
- Phantom `llm-lite` submodule reference removed; the clone footprint
  is now clean.
- Project renamed from `uXC` to `pccx`; legacy strings purged from
  tracked sources.

### Known limitations

- This is a reference implementation, not a timing-closed production
  bitstream. No place-and-route closure or KV260 board bring-up
  artifacts are included.
- The Sail ISA model covers decode and the first three execute
  increments only; later opcodes are still being modelled.
- Some testbenches (notably `tb_GEMM_fmap_staggered_delay`, the
  shift-chain timing model) are parked WIP and are not part of CI.
- No bit-accurate co-simulation harness with `pccx-lab` yet — that
  bridge lands in v0.2.0.
- Wiki is enabled on this repository but currently empty; the intended
  documentation surface is the `pccxai/pccx` site, not the wiki.

### Validation

- `repo-validate` and `Sail typecheck` required checks green on
  `main`.
- Stage 1, 2, and 3 ruleset active; direct push to `main` blocked.
- README and `CITATION.cff` reference `pccxai/pccx` as the canonical
  architecture repo.
- No standalone-vendor brand-token leaks in tracked sources.

[Unreleased]: https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/compare/HEAD...HEAD
