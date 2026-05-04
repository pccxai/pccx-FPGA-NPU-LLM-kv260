#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
# Collect bounded full top-level / BD bitstream-flow readiness evidence.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HW_DIR="$ROOT_DIR/hw"
REPORT_DIR="$HW_DIR/build/reports"

RUN_ID="${PCCX_RUN_ID:-}"
DRY_RUN=0
MODE="status"
ORIGINAL_ARGS=("$@")

usage() {
    cat <<'USAGE'
usage: scripts/v002/run-top-level-evidence.sh [--dry-run] [--run-id <id>] [--mode status|bitstream]

Collects explicit evidence for the full KV260 top-level / block-design flow.
The status mode is lightweight and does not run implementation or write a
bitstream. The bitstream mode is a gate: it fails fast until the full top-level
BD flow exists.
USAGE
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --run-id)
            if (($# < 2)); then
                echo "error: --run-id requires a value" >&2
                exit 2
            fi
            RUN_ID="$2"
            shift 2
            ;;
        --mode)
            if (($# < 2)); then
                echo "error: --mode requires status or bitstream" >&2
                exit 2
            fi
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    status|bitstream) ;;
    *)
        echo "error: --mode must be status or bitstream" >&2
        exit 2
        ;;
esac

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT_SHORT="$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || printf 'nogit')"
COMMIT_FULL="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')"
BRANCH="$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf 'unknown')"
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$STAMP-$COMMIT_SHORT"
fi
RUN_ID="${RUN_ID//\//_}"
EVIDENCE_DIR="$HW_DIR/build/v002-top-level-evidence/$RUN_ID"
SUMMARY="$EVIDENCE_DIR/summary.txt"
mkdir -p "$EVIDENCE_DIR"

STATUS_FILE="$REPORT_DIR/top_level_bitstream_status.txt"
LOG_FILE="$EVIDENCE_DIR/top_level_status.log"
TOP_STATUS="not_run"

if (( DRY_RUN )); then
    TOP_STATUS="dry_run_not_executed"
elif bash "$HW_DIR/vivado/build.sh" top-status >"$LOG_FILE" 2>&1; then
    TOP_STATUS="pass"
else
    TOP_STATUS="failed"
fi

field_from_status_file() {
    local report="$1"
    local field="$2"
    if [[ ! -f "$report" ]]; then
        printf 'unavailable'
        return
    fi
    awk -F= -v key="$field" '$1 == key {print substr($0, length(key) + 2); found=1; exit} END {if (!found) print "unavailable"}' "$report"
}

if [[ -f "$STATUS_FILE" ]]; then
    cp "$STATUS_FILE" "$EVIDENCE_DIR/top_level_bitstream_status.txt"
fi

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'branch=%s\n' "$BRANCH"
    printf 'git_commit=%s\n' "$COMMIT_FULL"
    printf 'dry_run=%s\n' "$DRY_RUN"
    printf 'mode=%s\n' "$MODE"
    printf 'command_line=%q ' "$0" "${ORIGINAL_ARGS[@]}"
    printf '\n'
    printf 'top_level_status_command=%s\n' "$TOP_STATUS"
    printf 'full_top_level_flow=%s\n' "$(field_from_status_file "$STATUS_FILE" 'full_top_level_flow')"
    printf 'bitstream_status=%s\n' "$(field_from_status_file "$STATUS_FILE" 'bitstream_status')"
    printf 'board_part_status=%s\n' "$(field_from_status_file "$STATUS_FILE" 'board_part_status')"
    printf 'wrapper_status=%s\n' "$(field_from_status_file "$STATUS_FILE" 'wrapper_status')"
    printf 'bd_status=%s\n' "$(field_from_status_file "$STATUS_FILE" 'bd_status')"
    printf 'bd_script=%s\n' "$(field_from_status_file "$STATUS_FILE" 'bd_script')"
    printf 'blocker=%s\n' "$(field_from_status_file "$STATUS_FILE" 'blocker')"
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'log_file=%s\n' "$LOG_FILE"
    printf 'KV260_board_status=NOT_RUN\n'
    printf 'Gemma_3N_E4B_runtime_status=NO_HARDWARE_EVIDENCE\n'
    printf 'measured_throughput_status=MEASUREMENT_NOT_AVAILABLE\n'
} >"$SUMMARY"

printf 'summary=%s\n' "$SUMMARY"
printf 'full_top_level_flow=%s\n' "$(field_from_status_file "$STATUS_FILE" 'full_top_level_flow')"
printf 'bitstream_status=%s\n' "$(field_from_status_file "$STATUS_FILE" 'bitstream_status')"

if [[ "$MODE" == "bitstream" ]]; then
    exit 3
fi
