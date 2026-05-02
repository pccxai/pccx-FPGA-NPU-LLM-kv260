#!/usr/bin/env python3
"""Generate deterministic v002 driver handoff artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTROL_ADDR_INSTRUCTION = "0x000"
CONTROL_ADDR_KICK = "0x008"
KICK_WORD = 0x8000_0000_0000_0000
INSN_HEX_RE = re.compile(r"^[0-9a-fA-F]{1,16}$")
REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_MODES = ("dry-run", "board", "future")
VALID_TRANSPORTS = ("mmio", "axi-lite", "axi", "future")
ENCODER = [
    "OP_GEMM",
    "OP_GEMV",
    "OP_MEMCPY",
    "OP_MEMSET",
    "OP_CVO",
]


def git_value(args: list[str], default: str) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", *args],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            or default
        )
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def require_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    return False


def parse_hex_word(raw: str) -> int:
    value = raw.strip().lower()
    if value.startswith("0x"):
        value = value[2:]
    if not INSN_HEX_RE.fullmatch(value):
        raise SystemExit(f"invalid hex word: {raw!r}")
    return int(value, 16)


def normalize_word(raw: str) -> str:
    return f"0x{parse_hex_word(raw):016x}"


def read_memh_lines(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing memh file: {path}")
    lines: list[str] = []
    for raw in path.read_text(encoding="ascii", errors="replace").splitlines():
        token = raw.strip()
        if not token:
            continue
        if token.startswith("#") or token.startswith("//"):
            continue
        token = token.split("#", 1)[0].split("//", 1)[0].strip()
        if not token:
            continue
        lines.append(normalize_word(token))
    if not lines:
        raise SystemExit(f"memh file {path} had no instruction words")
    return lines


def validate_program_payload(payload: dict[str, Any], program_path: Path, memh_words: list[str]) -> tuple[int, list[str]]:
    required = [
        "schema_version",
        "kind",
        "instructions",
        "claim_boundary",
        "target",
    ]
    missing = [name for name in required if name not in payload]
    if missing:
        raise SystemExit(f"program missing required fields: {', '.join(missing)}")

    kind = payload.get("kind")
    if kind != "pccx_v002_runtime_smoke_program":
        raise SystemExit(f"unsupported program kind: {kind}")

    instructions = payload.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise SystemExit("program instructions must be a non-empty list")

    if payload.get("schema_version") != 1:
        raise SystemExit(f"unsupported schema_version: {payload.get('schema_version')}")

    word_hexes: list[str] = []
    for idx, item in enumerate(instructions):
        if not isinstance(item, dict):
            raise SystemExit(f"instruction {idx} is not an object")
        if "word_hex" not in item:
            raise SystemExit(f"instruction {idx} missing word_hex")
        word_hexes.append(normalize_word(item["word_hex"]))

    if len(word_hexes) != len(memh_words):
        raise SystemExit(
            "program and memh instruction count mismatch: "
            f"program={len(word_hexes)} memh={len(memh_words)}"
        )

    for idx, (program_word, memh_word) in enumerate(zip(word_hexes, memh_words), start=1):
        if program_word != memh_word:
            raise SystemExit(
                f"instruction {idx} mismatch: program={program_word} memh={memh_word}"
            )

    return len(word_hexes), word_hexes


def expected_register_writes(commands: list[str], run_id: str) -> list[dict[str, Any]]:
    writes: list[dict[str, Any]] = []
    for index, word in enumerate(commands):
        writes.append(
            {
                "index": index,
                "registerAddress": CONTROL_ADDR_INSTRUCTION,
                "registerName": "S_AXIL_CMD_IN_INSTRUCTION",
                "wstrb": "0xff",
                "value": word,
                "transport": "mmio",
                "description": f"program instruction #{index} for handoff run {run_id}",
            }
        )
    writes.append(
        {
            "index": len(commands),
            "registerAddress": CONTROL_ADDR_KICK,
            "registerName": "S_AXIL_CMD_IN_KICK",
            "wstrb": "0xff",
            "value": normalize_word(hex(KICK_WORD)),
            "transport": "mmio",
            "description": f"kick execution for handoff run {run_id}",
        }
    )
    return writes


def extract_claim_flags(payload: dict[str, Any]) -> dict[str, bool]:
    claim_boundary = payload.get("claim_boundary", {})
    return {
        "kv260_inference_success": require_bool(
            claim_boundary.get("kv260_inference_success"), "kv260_inference_success"
        ),
        "gemma3n_e4b_hardware_execution": require_bool(
            claim_boundary.get("gemma3n_e4b_hardware_execution"),
            "gemma3n_e4b_hardware_execution",
        ),
        "timing_closed": require_bool(
            claim_boundary.get("timing_closed"),
            "timing_closed",
        ),
        "tokens_per_second_achieved": claim_boundary.get("tokens_per_second_achieved")
        is not None,
    }


def validate_handoff_payload(payload: dict[str, Any], mode: str) -> None:
    required = [
        "schemaVersion",
        "runId",
        "sourceProgramJson",
        "sourceMemh",
        "mode",
        "transport",
        "registerWrites",
        "commandWords",
        "target",
        "expectedDecoderPath",
        "expectedSchedulerPath",
        "boardInputsRequired",
        "safetyFlags",
        "unsupportedClaims",
    ]
    missing = [name for name in required if name not in payload]
    if missing:
        raise SystemExit(f"handoff missing required fields: {', '.join(missing)}")

    if payload.get("schemaVersion") != 1:
        raise SystemExit(f"unsupported handoff schemaVersion: {payload.get('schemaVersion')}")

    if payload.get("mode") not in VALID_MODES:
        raise SystemExit(f"unsupported handoff mode: {payload.get('mode')}")
    if mode == "dry-run" and payload.get("mode") not in ("dry-run", "board", "future"):
        raise SystemExit(f"handoff mode mismatch: {payload.get('mode')}")
    if not isinstance(payload.get("registerWrites"), list):
        raise SystemExit("handoff registerWrites must be a list")
    if not isinstance(payload.get("commandWords"), list):
        raise SystemExit("handoff commandWords must be a list")


def stable_payload(
    program: dict[str, Any],
    memh_words: list[str],
    program_path: Path,
    memh_path: Path,
    args: argparse.Namespace,
    instruction_count: int,
) -> dict[str, Any]:
    run_id = args.run_id
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short = git_value(["rev-parse", "--short=12", "HEAD"], "unknown")
        run_id = f"{ts}-{short}"

    program_path_abs = str(program_path.resolve())
    memh_path_abs = str(memh_path.resolve())

    claim_flags = extract_claim_flags(program)
    unsupported = [name for name, value in claim_flags.items() if value]
    evidence_dir = args.out.parent if args.out is not None else None

    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "runId": run_id,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gitCommit": {
            "short": git_value(["rev-parse", "--short=12", "HEAD"], "unknown"),
            "long": git_value(["rev-parse", "HEAD"], "unknown"),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], "unknown"),
        },
        "sourceProgramJson": program_path_abs,
        "sourceMemh": memh_path_abs,
        "mode": args.mode,
        "transport": args.transport,
        "target": {
            "board": "KV260",
            "architecture": "pccx_v002",
            "isaWidth": 64,
            "programBoundary": program.get("target", {}).get(
                "program_boundary", "AXI-Lite command FIFO / ctrl_npu_decoder"
            ),
        },
        "expectedDecoderPath": "hw/rtl/NPU_Controller/NPU_Control_Unit/ctrl_npu_decoder.sv",
        "expectedSchedulerPath": "hw/rtl/NPU_Controller/Global_Scheduler.sv",
        "registerWrites": expected_register_writes(memh_words, run_id),
        "commandWords": memh_words,
        "programSummary": {
            "kind": program.get("kind"),
            "preset": program.get("preset"),
            "instructionCount": instruction_count,
            "encoderCandidates": ENCODER,
            "sourceProgramSchema": program.get("schema_version"),
            "sourceClaimBoundary": program.get("claim_boundary", {}),
        },
        "safetyFlags": {
            "noBoardCredentialsRequired": True,
            "noBoardContactInDryRun": True,
            "noModelWeightsRequiredForDryRun": True,
            "noBitstreamRequiredForDryRun": True,
            "supportsBoardInputs": True,
            "requiresBoardInputsForBoardMode": args.mode != "dry-run",
        },
        "unsupportedClaims": claim_flags,
        "boardInputsRequired": (
            [] if args.mode == "dry-run"
            else [
                "PCCX_KV260_HOST",
                "PCCX_KV260_USER",
                "PCCX_MODEL_DIR",
                "PCCX_BITSTREAM_PATH",
                "PCCX_BOARD_RUNTIME_DIR",
                "PCCX_RUN_PROMPT",
                "PCCX_RUN_TOKENS",
            ]
        ),
        "evidencePaths": {
            "runtimeSmokeProgram": program_path_abs,
            "runtimeSmokeMemh": memh_path_abs,
            "handoff": str(args.out.resolve()) if args.out else "",
        },
        "hashes": {
            "programJson": hashlib.sha256(
                program_path.read_bytes()
            ).hexdigest(),
            "programMemh": hashlib.sha256(
                "\n".join(memh_words).encode("ascii")
            ).hexdigest(),
        },
        "validation": {
            "unsupportedClaims": unsupported,
            "commandCount": len(memh_words),
            "supported": len(unsupported) == 0,
        },
    }

    if evidence_dir is not None:
        payload["evidencePaths"]["smokeRunLogRoot"] = str(evidence_dir)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic v002 runtime smoke handoff plans for "
            "driver/MMIO dry-run and board mode."
        )
    )
    parser.add_argument(
        "--program-json",
        type=Path,
        help="generated v002 runtime program JSON",
    )
    parser.add_argument(
        "--memh",
        type=Path,
        help="generated v002 runtime .memh file (64-bit words)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output handoff artifact JSON path (stdout if omitted)",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="dry-run",
        help="handoff mode (dry-run|board|future)",
    )
    parser.add_argument(
        "--transport",
        choices=VALID_TRANSPORTS,
        default="axi-lite",
        help="command transport name for evidence",
    )
    parser.add_argument(
        "--run-id",
        help="stable run identifier (defaults to timestamp+git short SHA)",
    )
    parser.add_argument(
        "--validate-handoff",
        type=Path,
        help="validate an existing handoff artifact and exit",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.validate_handoff is not None:
        payload = read_json(args.validate_handoff)
        validate_handoff_payload(payload, args.mode)
        unsupported = [
            name for name, active in payload.get("unsupportedClaims", {}).items() if active
        ]
        board_inputs = payload.get("boardInputsRequired") or []
        transport = payload.get("transport")
        if transport not in VALID_TRANSPORTS:
            raise SystemExit(f"unsupported transport: {transport}")
        result = {
            "result": "valid",
            "runId": payload.get("runId"),
            "mode": payload.get("mode"),
            "transport": transport,
            "unsupportedClaims": unsupported,
            "boardInputsRequired": board_inputs,
            "commandCount": len(payload.get("commandWords", []) or []),
        }
        if args.out is not None:
            write_json(args.out, result)
        else:
            print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    if args.program_json is None or args.memh is None:
        raise SystemExit("--program-json and --memh are required unless --validate-handoff is used")

    program_path = args.program_json
    memh_path = args.memh
    if not program_path.exists():
        raise SystemExit(f"missing program JSON: {program_path}")
    if not memh_path.exists():
        raise SystemExit(f"missing memh: {memh_path}")

    program = read_json(program_path)
    memh_words = read_memh_lines(memh_path)
    instruction_count, command_words = validate_program_payload(program, program_path, memh_words)
    payload = stable_payload(
        program,
        command_words,
        program_path,
        memh_path,
        args,
        instruction_count,
    )

    if args.out:
        write_json(args.out, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
