# KV260 Bring-Up Evidence Checklist

This checklist describes the evidence path for KV260 board smoke runs.
It is a bring-up scaffold, not proof that the NPU bitstream or Gemma 3N
E4B runtime works on hardware.

## Current Runner

The repo includes:

```bash
scripts/kv260/run_gemma3n_e4b_smoke.sh
```

Use `--dry-run` to validate the handoff shape and write blocked evidence
without contacting the board:

```bash
scripts/kv260/run_gemma3n_e4b_smoke.sh --dry-run
```

Provide a generated runtime handoff artifact to validate the MMIO handoff
contract:

```bash
scripts/kv260/run_gemma3n_e4b_smoke.sh --dry-run --handoff <runtime_smoke/handoff/handoff.json>
# or
PCCX_RUNTIME_HANDOFF_JSON=runtime_smoke/handoff/handoff.json \
scripts/kv260/run_gemma3n_e4b_smoke.sh --dry-run
```

The script is explicit by default: without the required environment it
exits with a `BLOCKED_*` status and writes a blocker summary under
`docs/evidence/kv260-gemma3n-e4b/<run_id>/`.

Do not run it against a board unless the board, bitstream, model path,
runtime directory, and prompt fixture are intentionally configured.

## Required Configuration

| Variable | Purpose |
| --- | --- |
| `PCCX_KV260_HOST` | KV260 hostname or IP for SSH |
| `PCCX_KV260_USER` | SSH user with key-based login |
| `PCCX_MODEL_DIR` | Board-side model directory, or host directory when staging is enabled |
| `PCCX_BITSTREAM_PATH` | Local or board-side bitstream path |
| `PCCX_RUN_PROMPT` | Smoke input text |
| `PCCX_RUN_TOKENS` | Small positive max-new-token count |
| `PCCX_BOARD_RUNTIME_DIR` | Writable directory on the KV260 for runner and evidence files |

Optional controls:

| Variable | Purpose |
| --- | --- |
| `PCCX_STAGE_MODEL_TO_BOARD=1` | Copy a host model directory to the board runtime area |
| `PCCX_SKIP_BITSTREAM_LOAD=1` | Skip programming when the intended image is already loaded |
| `PCCX_REMOTE_RUN_CMD` | Override the board-side NPU runtime command |
| `PCCX_RUN_ID` | Force a stable evidence directory name |
| `PCCX_RUNTIME_HANDOFF_JSON` | Path to a runtime handoff artifact to validate in dry-run |

Recognized aliases for runtime handoff manifests:

| Alias | Maps to |
| --- | --- |
| `PCCX_KV260_BOARD_ADDR` | `PCCX_KV260_HOST` |
| `PCCX_GEMMA3N_E4B_MODEL_DIR` | `PCCX_MODEL_DIR` |
| `PCCX_KV260_BITSTREAM` | `PCCX_BITSTREAM_PATH` |

## Evidence Flow

1. Confirm board reachability over non-interactive SSH.
2. Create a board-side runtime/evidence directory.
3. Confirm the model directory exists on the board, or stage it only when
   `PCCX_STAGE_MODEL_TO_BOARD=1` is set.
4. Confirm the bitstream exists locally or on the board.
5. Record the bitstream basename and SHA256.
6. Program the PL with `fpgautil` or `xbutil`, unless
   `PCCX_SKIP_BITSTREAM_LOAD=1` is set.
7. Stage the remote smoke runner.
8. Run the NPU runtime candidate or record a precise blocker.
9. Fetch sanitized stdout, stderr, generated output, and `result.env`.
10. Write `summary.txt` and, when blocked, `blocker.txt`.

## Result Status

| Status | Meaning |
| --- | --- |
| `PASS_KV260_NPU` | Board-side NPU runtime reported a pass |
| `PASS_KV260_FALLBACK` | Board-side software fallback reported a pass; this is not NPU inference evidence |
| `BLOCKED_BOARD` | SSH or board setup is missing or unreachable |
| `BLOCKED_MODEL` | Model path or staging is missing |
| `BLOCKED_BITSTREAM` | Bitstream path, copy, or programming failed |
| `BLOCKED_RUNTIME` | Runtime wrapper, output, or metrics are missing |
| `BLOCKED_RTL` | The runtime identified an RTL/Vivado readiness blocker |
| `READY_FOR_BOARD_INPUTS` | Dry-run validated a handoff artifact; board inputs can be supplied in non-dry-run mode |
| `BLOCKED_BOARD_INPUTS` | Dry-run missing/invalid handoff shape or missing board-input-ready artifacts for this lane |

Only `PASS_KV260_NPU` with the captured logs, bitstream hash, commit SHA,
and runtime output should be treated as KV260 NPU smoke evidence.

## What To Record

- Git commit SHA and branch.
- Vivado/Vitis version used to create the bitstream.
- Bitstream basename and SHA256.
- Board identifier, if available.
- Exact environment variable set, with private hostnames and paths
  redacted in public notes.
- Prompt fixture and requested token count.
- `summary.txt`, `blocker.txt` when present, and sanitized runtime logs.
- Whether the runtime path was `FPGA_NPU`, `PS_FALLBACK`, or `blocked`.

## Public Wording

Use "KV260 bring-up logs", "runtime readiness checks", or "target
path" until a real `PASS_KV260_NPU` evidence directory exists for the
current bitstream and commit. Do not claim board inference, Gemma 3N E4B
runtime success, timing closure, or measured throughput from a blocked
or fallback run.
