# FPGA Release Evidence Checklist

This checklist defines the minimum evidence required before tagging any
production release of `pccx-FPGA-NPU-LLM-kv260`.  Complete source-only
snapshots (e.g. `v0.1.0-alpha`) do not require all items; hardware-gated
items are marked **[HW]**.

See also: [`RELEASING.md`](../RELEASING.md) for the tagging process,
and [`CHANGELOG.md`](../CHANGELOG.md) for the version under preparation.
For the explicit future pccx v002.1 KV260 release gate, see
[`docs/releases/v002.1-acceptance-criteria.md`](releases/v002.1-acceptance-criteria.md).

---

## 1. Implementation completion

- [ ] All P0/P1 GitHub issues resolved or explicitly deferred with rationale
- [ ] Known TODOs listed in release notes (copy from open issues)
- [ ] No uncommitted user work (`git status` clean on tagged SHA)
- [ ] Version and commit hash recorded in `CHANGELOG.md` `[Unreleased]` cut
- [ ] `CITATION.cff` metadata consistent with tag

---

## 2. Simulation evidence

For each testbench run include the following.  Current TB inventory:
`tb_barrel_shifter_BF16`, `tb_ctrl_npu_decoder`, `tb_FROM_mat_result_packer`,
`tb_GEMM_fmap_staggered_delay`,
`tb_GEMM_dsp_packer_sign_recovery`, `tb_GEMM_weight_dispatcher`,
`tb_mat_result_normalizer`, `tb_mem_u_operation_queue`,
`tb_shape_const_ram`.

- [ ] Command used to run: `cd hw/sim && bash run_verification.sh`
- [ ] Vivado / xsim version recorded (currently: Vivado v2025.2)
- [ ] Per-TB pass/fail summary from the deterministic runner
- [ ] Known warnings or elaboration errors documented
- [ ] `.pccx` trace artifacts retained in `hw/sim/work/<tb>/`
- [ ] Waveform snapshots (`.wdb`) retained for failed TBs
- [ ] E2E NPU_top integration TB added and green (tracked: [#37](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/37))
- [ ] BF16-to-INT8 preprocess golden-model TB added (tracked: [#33](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/33))

---

## 3. Synthesis evidence

Existing post-synthesis reports: `hw/build/reports/` (tool: Vivado v2025.2,
date: 2026-04-20, design: NPU_top, device: xck26-sfvc784-2LV-c).

- [x] Synthesis run completed and summary recorded (utilization, timing,
      DRC, clocks)
- [x] Vivado version recorded in report headers
- [ ] Target clock and achieved slack documented in release notes
  - Current post-synth: `core_clk` WNS = **−9.792 ns** (NOT MET, 4194 failing
    endpoints); `axi_clk` WNS = +2.253 ns (MET).  Post-synthesis slack is
    pessimistic; post-implementation timing is the authoritative figure.
  - Post-synthesis utilization snapshot: 5 611 LUTs, 8 458 FFs, 80 RAMB36,
    56 URAM, 4 DSP (GEMM systolic array not yet fully instantiated).
- [ ] DRC critical violations resolved or documented
- [ ] Constraints file (`hw/constraints/`) referenced in release notes
- [ ] Clock interaction report reviewed for `core_clk` / `axi_clk`

---

## 4. Post-implementation and timing closure evidence

**[HW]** These items require a full Vivado implementation run (tracked:
[#29](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/29),
[#42](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/42)).

- [ ] Implementation run completed (route_design)
- [ ] Post-route timing summary: WNS / TNS / failing endpoints
- [ ] Timing constraints met for both `core_clk` (target 400 MHz) and
      `axi_clk` (target 250 MHz)
- [ ] Utilization post-implementation recorded
- [ ] Clock interaction post-implementation recorded
- [ ] Critical path summary noted in release notes
- [ ] Bitstream generated (`.bit` not committed to git — attach to GitHub
      Release or record SHA + generation command)

---

## 5. KV260 bring-up evidence

**[HW]** None of these items are satisfied as of v0.1.0-alpha.  Do not
claim bring-up success until all mandatory items are checked.

- [ ] Board revision and serial number recorded
- [ ] Vivado / Vitis version used for bitstream generation
- [ ] Bitstream path and SHA256 recorded
- [ ] Boot sequence steps documented (JTAG / SD / network boot)
- [ ] UART or JTAG console output captured at load time
- [ ] AXI-Lite register read/write test passing (loopback or status poll)
- [ ] Observed behavior documented (with log excerpt)
- [ ] Failure modes and recovery steps noted if bring-up was not clean

---

## 6. Runtime / inference evidence

**[HW]** Only to be collected after KV260 bring-up is confirmed.  Do not
include estimated or simulated numbers as hardware-measured results.

- [ ] Model configuration used (model name, weight source, quantization config)
- [ ] Input sample documented (token IDs or prompt excerpt)
- [ ] Output sample documented (decoded tokens)
- [ ] Measurement method for latency/throughput (cycle counter, wall clock,
      UART timestamp)
- [ ] Measured tok/s figure with methodology note
- [ ] Power / thermal notes if available (board PSU measurement or on-chip
      sensor)
- [ ] Known limitations listed (sequence length cap, unsupported ops, etc.)

---

## 7. Reproducibility

- [ ] Exact `run_verification.sh` command and expected runtime noted
- [ ] Required Vivado version and license type documented
- [ ] Required hardware (KV260 board revision) listed
- [ ] Known non-reproducible parts identified (e.g. Vivado seed, licensed
      model weights)
- [ ] Steps for contributor to reproduce simulation without hardware noted
      in README or a `docs/SIMULATION.md` page

---

## 8. Release gate summary

A tag **MUST NOT** be created unless:

1. All non-`[HW]` items above are checked.
2. `[HW]` items are either checked **or** explicitly deferred with a
   written rationale in the release notes.
3. `repo-validate` and `Sail typecheck` CI are green on the tagged SHA.
4. `RELEASING.md` pre-flight checklist is complete.
5. Release notes state clearly: timing closure status, bitstream
   availability, and bring-up status.

**No production release before full implementation + evidence collection.**
`v0.2.0` is the earliest realistic target; see EPIC
[#43](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/43).

## 9. v002.1 KV260 acceptance criteria

The future pccx v002.1 KV260 release gate requires all of the following
from the same candidate commit, bitstream, board, and Gemma 3N E4B
runtime fixture:

- [ ] Bitstream generated, with Vivado version, command, post-implementation
      timing summary, basename, and SHA256 recorded.
- [ ] KV260 board run captured, with board identifier, programming log,
      runtime command, sanitized environment, stdout, stderr, console, and
      summary files retained.
- [ ] Gemma 3N E4B token observed from the KV260 NPU runtime path, with
      input fixture, output token excerpt, model fixture identity, commit
      SHA, bitstream SHA256, and board run evidence tied together.

Blocked, fallback, simulation-only, synthesis-only, or host-only evidence
does not satisfy the v002.1 gate. This checklist defines the required
criteria; it does not mark them complete.
