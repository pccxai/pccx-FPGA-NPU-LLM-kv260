# Gemma 3N E4B Target-Model Handoff Boundary

Gemma 3N E4B on KV260 is a target path for pccx v002. This repository
must not imply that the model already runs on the board until KV260 NPU
smoke evidence exists for the bitstream, runtime, model fixture, and
commit under test.

## Repository Boundary

This FPGA repository owns:

- RTL and Vivado build inputs.
- Bare-metal driver skeletons and board-side runtime hooks.
- KV260 bring-up evidence structure.
- Public-safe model handoff contracts.

This FPGA repository does not own:

- Model weights.
- Generated weight blobs.
- Private staging paths.
- CPU benchmark corpora or research artifacts from separate repos.
- User-facing launcher state beyond documented readiness contracts.

## Manifest Contract

Runtime or launcher tooling may pass a manifest-like object into the
board smoke flow. Public docs and PRs should use placeholders, not local
absolute paths.

Example shape:

```json
{
  "model_id": "google/gemma-3n-e4b",
  "target": "kv260",
  "precision": "w4a8-target",
  "files": [
    {
      "name": "<model-file-name>",
      "sha256": "<sha256-if-shareable>",
      "bytes": "<size-if-shareable>"
    }
  ],
  "runtime": {
    "bitstream": "<bitstream-basename>",
    "bitstream_sha256": "<sha256>",
    "runner": "scripts/kv260/run_gemma3n_e4b_smoke.sh"
  }
}
```

Do not commit a manifest containing private host paths, model cache
directories, credentials, tokens, or proprietary/generated model blobs.

## Readiness Contract

Use readiness states instead of success claims:

| Field | Meaning |
| --- | --- |
| `board_reachable` | KV260 accepted non-interactive SSH |
| `bitstream_found` | Bitstream path resolved locally or on board |
| `bitstream_loaded` | PL programming was attempted and reported success |
| `model_found` | Model directory exists on the board, or was staged intentionally |
| `runtime_path` | `FPGA_NPU`, `PS_FALLBACK`, or `blocked` |
| `result_status` | `PASS_KV260_NPU`, `PASS_KV260_FALLBACK`, or `BLOCKED_*` |

`PASS_KV260_FALLBACK` is useful runtime readiness evidence, but it is
not KV260 NPU inference evidence. Treat `BLOCKED_*` outputs as blockers
with logs, not failures to hide.

## Public Wording

Use:

- "Gemma 3N E4B target path"
- "runtime readiness checks"
- "KV260 bring-up evidence"
- "planned model handoff"

Avoid:

- wording that presents Gemma 3N E4B board execution as complete
- wording that presents KV260 inference as proven without logs
- throughput wording without tied evidence
- production-readiness wording

Measured performance belongs only in evidence tied to a specific commit,
bitstream hash, model fixture, board run, and measurement method.
