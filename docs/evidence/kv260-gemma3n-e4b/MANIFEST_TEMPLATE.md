# Gemma 3N E4B External Manifest Template

Use `configs/v002/gemma3n_e4b_manifest.example.json` as the source-only
template for board smoke inputs. The manifest is a pointer to external assets;
it must not copy model weights, converted blobs, bitstreams, or board dumps into
this repository.

Required fields:

- `model.external_model_dir`: board-side or host-side Gemma 3N E4B directory.
- `runtime.bitstream`: local or board-side bitstream path for the intended run.
- `runtime.runtime_bin`: board-side runtime binary or launcher path.
- `runtime.target_board`: `KV260`.

Optional fields may list tokenizer files, packed/scale arrays, GGUF files, or
runtime notes. Paths should be environment-specific and kept out of committed
run evidence when they reveal private host layout.

Connection to board smoke:

- `scripts/kv260/run_gemma3n_e4b_smoke.sh --dry-run` validates the handoff
  shape without contacting the board.
- A real run writes evidence under `docs/evidence/kv260-gemma3n-e4b/<run_id>/`.
- Missing board, model, bitstream, or runtime inputs produce blocked evidence
  rather than a success claim.

Not yet verified by this template:

- KV260 NPU execution.
- Gemma 3N E4B hardware execution.
- Measured throughput.
- Timing closure.
