# Open-Source Components Disclosure - pccx v002 KV260

This disclosure covers the v002 source snapshot in this repository:
SystemVerilog RTL, bare-metal driver source, formal Sail model,
verification scripts, CI workflows, and KV260 bring-up helpers.

Vivado-generated IP, Xilinx primitive libraries, XPM/UNISIM simulation
libraries, bitstreams, Vivado/Vitis project state, board firmware
packages, and model weights are intentionally out of scope. They are not
distributed by this repository.

## Bundled Source

| Component | Use in v002 | Location | License |
| --- | --- | --- | --- |
| pccx-FPGA-NPU-LLM-kv260 | KV260 RTL, driver, formal model, tests, docs, and scripts in this source snapshot | this repository | Apache-2.0, see [`LICENSE`](../LICENSE) |

No third-party source tree is vendored into this repository. There is no
`.gitmodules` file, package lockfile, Python requirements file, or Cargo
workspace in this source snapshot.

## Referenced Open-Source Projects

These projects are used as source references or local companion tools but
are not copied into this repository.

| Component | Use in v002 | Distribution state | License |
| --- | --- | --- | --- |
| pccx | Canonical v002 architecture, ISA, driver, and Gemma 3N mapping specification referenced by this RTL | External documentation/source repository | Apache-2.0 |
| pccx-lab | Optional trace visualization and `from_xsim_log` converter used by `hw/sim/run_verification.sh` | External companion repository; built locally when available | Apache-2.0 |

## Tooling Used by Verification and CI

These tools are referenced by scripts or GitHub Actions. They are
execution dependencies only; their source code is not bundled here.

| Component | Use in v002 | Where referenced | License |
| --- | --- | --- | --- |
| Sail | Typechecks the formal pccx v002 ISA model under `formal/sail/` | `formal/sail/Makefile`, `.github/workflows/sail-check.yml` | BSD-2-Clause for the Sail implementation |
| Z3 | SMT solver used by Sail's typechecker | `.github/workflows/sail-check.yml` | MIT |
| GitHub Actions checkout | CI source checkout action | `.github/workflows/*.yml` | MIT |
| GitHub Actions cache | CI opam cache action for Sail installs | `.github/workflows/sail-check.yml` | MIT |
| Python 3 standard library | Smoke-program generation, evidence parsing, and JSON validation helpers | `tools/v002/*.py`, scripts, CI | PSF-2.0 |
| PyYAML | YAML syntax validation for GitHub workflow files | `.github/workflows/validate.yml` | MIT |
| Bash | Shell script runtime for verification, release, and board-smoke helpers | `hw/sim/*.sh`, `hw/vivado/*.sh`, `scripts/**/*.sh` | GPL-3.0-or-later |
| Git | Repository metadata, evidence commit IDs, and artifact scans | scripts and CI workflows | GPL-2.0-only |
| GNU Make | Sail model check/doc convenience targets | `formal/sail/Makefile` | GPL-3.0-or-later |
| opam / OCaml toolchain | Installs Sail 0.20.1 in CI | `.github/workflows/sail-check.yml` | LGPL-3.0-only with linking exception for opam; LGPL-2.1-only with OCaml linking exception for OCaml |
| zlib | Sail build dependency installed in CI | `.github/workflows/sail-check.yml` | zlib |
| GMP | Sail build dependency installed in CI | `.github/workflows/sail-check.yml` | LGPL-3.0-or-later / GPL-2.0-or-later dual license |
| GNU M4 | Sail build dependency installed in CI | `.github/workflows/sail-check.yml` | GPL-3.0-or-later |
| pkgconf / pkg-config | Sail build dependency discovery in CI | `.github/workflows/sail-check.yml` | ISC |
| OpenSSH client | Optional KV260 board-smoke transport | `scripts/kv260/run_gemma3n_e4b_smoke.sh` | BSD-style OpenSSH license |
| rsync | Optional host-to-board model staging helper | `scripts/kv260/run_gemma3n_e4b_smoke.sh` | GPL-3.0-or-later |

## Explicitly Not Included

| Item | Disclosure status |
| --- | --- |
| AMD/Xilinx Vivado, Vitis, xsim, xvlog, xelab, XPM, UNISIM, `glbl.v`, generated IP blocks, and board files | Required or optional proprietary FPGA tooling; not open-source components and not distributed here |
| KV260 bitstreams, DCPs, XSA/XCLBIN artifacts, Vivado project state, `.Xil/`, and build reports | Generated artifacts; excluded by `.gitignore` and release policy |
| Gemma 3N E4B model weights, tokenizer files, GGUF/safetensors files, NumPy packed weight blobs, and datasets | External assets; not committed, not redistributed, and not represented as open-source components |
| pccx-lab build outputs, Cargo registry cache, opam switch cache, and system package cache | Local build/runtime state only; not part of the source snapshot |

## Maintenance Notes

Update this file whenever the repository adds a vendored dependency,
package manifest, new CI action, new board-side helper dependency, or a
new external companion project that is required to reproduce v002
verification evidence.
