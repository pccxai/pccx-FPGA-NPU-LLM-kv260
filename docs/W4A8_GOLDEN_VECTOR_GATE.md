# W4A8 / BF16-INT8 Golden-Vector Gate

W4A8 and BF16-to-INT8 pipeline work must be gated by reproducible
golden vectors before arithmetic RTL behavior changes. This note is a
planning gate only; it does not modify the current preprocess, GEMM, or
GEMV RTL.

## Protected Scope

Treat these areas as arithmetic-sensitive:

- `hw/rtl/PREPROCESS/preprocess_bf16_fixed_pipeline.sv`
- future BF16-to-INT8 preprocess pipeline files
- `hw/rtl/MAT_CORE/GEMM_dsp_unit*.sv`
- `hw/rtl/MAT_CORE/GEMM_dsp_packer.sv`
- `hw/rtl/MAT_CORE/GEMM_sign_recovery.sv`
- GEMV reduction and accumulation modules

Do not mix W4A8 arithmetic changes into shape RAM cleanup, documentation,
or board bring-up PRs.

## Minimum Golden-Vector Set

Before changing arithmetic behavior, prepare fixtures for:

| Area | Required fixture |
| --- | --- |
| BF16 decode | sign, exponent, mantissa edge cases |
| Emax alignment | max exponent ties, negative values, zero lanes |
| BF16 to INT8 quantization | clamp, round, saturation, sign handling |
| INT4 weight packing | upper/lower lane sign extension and pairing |
| DSP dual-MAC accumulation | borrow/carry recovery across guard bits |
| Drain cadence | accumulation windows up to the 1024-cycle limit |
| Result normalization | positive/negative large sums and near-zero sums |

Each fixture should include:

- Source generator commit or script name.
- Input tensor shape and lane order.
- Exact packed bytes or words consumed by RTL.
- Expected per-cycle RTL-visible values.
- Expected final outputs.
- Hash of the fixture file.

## Acceptance Criteria

A W4A8 arithmetic PR should include:

- A focused RTL change with no unrelated docs or bring-up edits.
- A testbench that consumes golden vectors or embeds a clearly explained
  deterministic golden model.
- PASS/FAIL verdict lines compatible with `hw/sim/run_verification.sh`.
- `xvlog` compile evidence for the touched path.
- A note explaining whether the fixture is public-safe or must remain
  local because it was derived from licensed model weights.

## Public-Safe Policy

Do not commit model weights, generated weight blobs, private cache paths,
or board dumps. If a fixture is derived from licensed model data, commit
only the generator contract and expected shape, then attach the private
fixture through the maintainer-approved evidence channel.

Use "planned W4A8 validation gate" or "golden-vector target" until the
fixtures and xsim results exist.
