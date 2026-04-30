# Releasing pccx-FPGA-NPU-LLM-kv260

This repository ships **source snapshots** of the bare-metal Kria KV260
implementation of the pccx v002 NPU architecture.  It does NOT ship
bitstreams, compiled hardware artefacts, or timing-closed Vivado
projects.  Production hardware deliverables are out of scope until a
separately-versioned bringup line opens.

## Versioning

- Pre-1.0 (`0.x.y`): pre-bringup; minor bumps may carry breaking RTL
  or driver changes.
- Tag format: `vX.Y.Z[-alpha|-beta|-rc]`.
- The first tag is planned as `v0.1.0-alpha` — see `CHANGELOG.md`.

## What is in a release

A release is a **source archive** at the tagged commit:

- `hw/rtl/` — SystemVerilog modules and interface headers.
- `hw/sim/` — testbench harnesses, run scripts, expected traces.
- `sw/driver/` — bare-metal C driver and HAL.
- `formal/sail/` — Sail ISA model and typecheck setup.

A release does **not** include:

- compiled bitstreams (`*.bit`, `*.dcp`),
- Vivado / Vitis project state (`*.xpr`, `.Xil/`, `.cache/`,
  `*.runs/`),
- model checkpoints or licensed weights.

## Pre-flight checks

Before tagging:

1. The Sail typecheck CI (`.github/workflows/sail-check.yml`) is
   green on `main` for the tagged SHA.
2. Local `bash hw/sim/run_verification.sh` completes without `xsim`
   compile errors on the maintainer's machine.
3. `xvlog` lint is clean for any modules added since the previous
   tag (the authoritative lint — Verilator on hosted runners is not
   sufficient because of the project's compile order and `.svh`
   include search paths).
4. `CHANGELOG.md` `[Unreleased]` block is cut to `## [X.Y.Z] - YYYY-MM-DD`.
5. `CITATION.cff` is consistent with the tag's authors / metadata.

## Tagging

```bash
git tag -a vX.Y.Z -m "pccx-kv260-rtl vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z
```

## Drafting on GitHub

```bash
gh release create vX.Y.Z --draft --prerelease \
   --title "pccx-kv260-rtl vX.Y.Z — early source snapshot" \
   --notes-file release_notes/vX.Y.Z.md
```

The release notes MUST state:

- Whether timing closure has been demonstrated for the tagged commit
  (today: not yet).
- Which Vivado / Vitis version was used for any synthesis number
  cited in the notes.
- Whether a bitstream is attached (today: no).
- The Sail typecheck SHA the tag passed.

## After-release

- Open a fresh `## [Unreleased]` block in `CHANGELOG.md`.
- Open issues for any "Skeleton" or "Not started" rows in the README
  Implementation Status table.
- If the tag changes the public driver API, mirror it in the matching
  `pccx/docs/v002/Drivers/api.html` page and bump the "Last verified
  against" admonition there.
