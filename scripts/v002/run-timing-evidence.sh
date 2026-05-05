#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
# Collect bounded v002 timing evidence without claiming closure.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HW_DIR="$ROOT_DIR/hw"
REPORT_DIR="$HW_DIR/build/reports"

DRY_RUN=0
RUN_SYNTH=0
REQUIRE_IMPL_CLEAN=0
RUN_ID="${PCCX_RUN_ID:-}"
ORIGINAL_ARGS=("$@")

usage() {
    cat <<'USAGE'
usage: scripts/v002/run-timing-evidence.sh [--dry-run] [--run-synth] [--require-impl-clean] [--run-id <id>]

Options:
  --dry-run             collect environment/report status only
  --run-synth           run hw/vivado/build.sh synth before collecting status
  --require-impl-clean  exit nonzero unless post-impl timing is present and clean
  --run-id <id>         use a deterministic evidence directory suffix
  -h, --help            print this help
USAGE
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --run-synth)
            RUN_SYNTH=1
            shift
            ;;
        --require-impl-clean)
            REQUIRE_IMPL_CLEAN=1
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

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT_SHORT="$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || printf 'nogit')"
COMMIT_FULL="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')"
BRANCH="$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf 'unknown')"
GIT_STATUS_SHORT="$(git -C "$ROOT_DIR" status --short 2>/dev/null || true)"
WORKTREE_DIRTY=0
if [[ -n "$GIT_STATUS_SHORT" ]]; then
    WORKTREE_DIRTY=1
fi
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$STAMP-$COMMIT_SHORT"
fi
RUN_ID="${RUN_ID//\//_}"
EVIDENCE_DIR="$HW_DIR/build/v002-timing-evidence/$RUN_ID"
SUMMARY="$EVIDENCE_DIR/summary.txt"
mkdir -p "$EVIDENCE_DIR"

status_from_report() {
    local report="$1"
    if [[ ! -f "$report" ]]; then
        printf 'TIMING_NOT_RUN'
        return
    fi
    if grep -qi 'Timing constraints are met' "$report"; then
        printf 'TIMING_REPORT_PRESENT_CLOSED'
    elif grep -qi 'Timing constraints are not met' "$report"; then
        printf 'TIMING_REPORT_PRESENT_NOT_CLOSED'
    else
        printf 'TIMING_REPORT_PRESENT_NOT_CLOSED'
    fi
}

metric_from_report() {
    local report="$1"
    local metric="$2"
    if [[ ! -f "$report" ]]; then
        printf 'unavailable'
        return
    fi
    python3 - "$report" "$metric" <<'PY'
import re
import sys

path, metric = sys.argv[1], sys.argv[2]
metric_index = {
    "WNS(ns)": 0,
    "TNS(ns)": 1,
    "WHS(ns)": 4,
    "THS(ns)": 5,
}.get(metric)
lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()
for idx, line in enumerate(lines):
    if metric_index is None or metric not in line:
        continue
    if not all(name in line for name in ("WNS(ns)", "TNS(ns)", "WHS(ns)", "THS(ns)")):
        continue
    for candidate in lines[idx + 1 : idx + 8]:
        values = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", candidate)
        if len(values) > metric_index:
            print(values[metric_index])
            raise SystemExit(0)
print("unavailable")
PY
}

capture_report_tail() {
    local report="$1"
    local name="$2"
    if [[ -f "$report" ]]; then
        tail -n 160 "$report" >"$EVIDENCE_DIR/$name.tail.txt" || true
    fi
}

SYNTH_LOG="$EVIDENCE_DIR/synth.log"
if (( RUN_SYNTH && ! DRY_RUN )); then
    if ! PCCX_VIVADO_JOBS="${PCCX_VIVADO_JOBS:-1}" bash "$HW_DIR/vivado/build.sh" synth \
        >"$SYNTH_LOG" 2>&1; then
        SYNTH_STATUS="failed"
    else
        SYNTH_STATUS="pass"
    fi
elif (( RUN_SYNTH && DRY_RUN )); then
    SYNTH_STATUS="dry_run_not_executed"
else
    SYNTH_STATUS="not_requested"
fi

TIMING_REPORT="$REPORT_DIR/timing_summary_post_synth.rpt"
UTIL_REPORT="$REPORT_DIR/utilization_post_synth.rpt"
IMPL_TIMING_REPORT="$REPORT_DIR/timing_summary_post_impl.rpt"
IMPL_UTIL_REPORT="$REPORT_DIR/utilization_post_impl.rpt"
BITSTREAM_STATUS_FILE="$REPORT_DIR/bitstream_status.txt"
TOP_LEVEL_STATUS_FILE="$REPORT_DIR/top_level_bitstream_status.txt"
TIMING_STATUS="$(status_from_report "$TIMING_REPORT")"
IMPL_TIMING_STATUS="$(status_from_report "$IMPL_TIMING_REPORT")"
if [[ "$SYNTH_STATUS" == "failed" && ! -f "$TIMING_REPORT" ]]; then
    TIMING_STATUS="TIMING_ATTEMPTED_NO_REPORT"
fi
if [[ -f "$IMPL_TIMING_REPORT" ]]; then
    if [[ "$IMPL_TIMING_STATUS" == "TIMING_REPORT_PRESENT_CLOSED" ]]; then
        OVERALL_TIMING_STATUS="POST_IMPL_TIMING_CLOSED"
    else
        OVERALL_TIMING_STATUS="POST_IMPL_TIMING_NOT_CLOSED"
    fi
elif [[ "$TIMING_STATUS" == "TIMING_REPORT_PRESENT_CLOSED" ]]; then
    OVERALL_TIMING_STATUS="POST_SYNTH_TIMING_CLOSED_IMPL_NOT_RUN"
elif [[ "$TIMING_STATUS" == "TIMING_NOT_RUN" ]]; then
    OVERALL_TIMING_STATUS="TIMING_NOT_RUN"
else
    OVERALL_TIMING_STATUS="POST_SYNTH_TIMING_NOT_CLOSED"
fi

capture_report_tail "$TIMING_REPORT" timing_summary_post_synth
capture_report_tail "$UTIL_REPORT" utilization_post_synth
capture_report_tail "$IMPL_TIMING_REPORT" timing_summary_post_impl
capture_report_tail "$IMPL_UTIL_REPORT" utilization_post_impl
capture_report_tail "$BITSTREAM_STATUS_FILE" bitstream_status
capture_report_tail "$TOP_LEVEL_STATUS_FILE" top_level_bitstream_status

field_from_status_file() {
    local report="$1"
    local field="$2"
    if [[ ! -f "$report" ]]; then
        printf 'unavailable'
        return
    fi
    awk -F= -v key="$field" '$1 == key {print substr($0, length(key) + 2); found=1; exit} END {if (!found) print "unavailable"}' "$report"
}

vivado_version() {
    local vivado_bin
    vivado_bin="$(command -v vivado 2>/dev/null || true)"
    if [[ -z "$vivado_bin" ]]; then
        printf 'MISSING'
        return
    fi
    "$vivado_bin" -version 2>/dev/null | awk 'NR == 1 {print; found=1} END {if (!found) exit 1}' || printf 'MISSING'
}

VIVADO_VERSION="$(vivado_version)"
BITSTREAM_STATUS="$(field_from_status_file "$BITSTREAM_STATUS_FILE" 'bitstream_status')"
IMPLEMENTATION_SCOPE="$(field_from_status_file "$BITSTREAM_STATUS_FILE" 'implementation_scope')"
if [[ "$IMPLEMENTATION_SCOPE" == "unavailable" ]]; then
    case "$BITSTREAM_STATUS" in
        BITSTREAM_NOT_REQUESTED|BITSTREAM_BLOCKED_OOC)
            IMPLEMENTATION_SCOPE="OOC_ROUTE_ONLY"
            ;;
    esac
fi
FULL_TOP_LEVEL_FLOW="$(field_from_status_file "$TOP_LEVEL_STATUS_FILE" 'full_top_level_flow')"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'branch=%s\n' "$BRANCH"
    printf 'git_commit=%s\n' "$COMMIT_FULL"
    printf 'worktree_dirty=%s\n' "$WORKTREE_DIRTY"
    printf 'dry_run=%s\n' "$DRY_RUN"
    printf 'run_synth=%s\n' "$RUN_SYNTH"
    printf 'require_impl_clean=%s\n' "$REQUIRE_IMPL_CLEAN"
    printf 'command_line=%q ' "$0" "${ORIGINAL_ARGS[@]}"
    printf '\n'
    printf 'synth_status=%s\n' "$SYNTH_STATUS"
    printf 'overall_timing_status=%s\n' "$OVERALL_TIMING_STATUS"
    printf 'timing_status=%s\n' "$TIMING_STATUS"
    printf 'timing_report=%s\n' "$TIMING_REPORT"
    printf 'timing_wns_ns=%s\n' "$(metric_from_report "$TIMING_REPORT" 'WNS(ns)')"
    printf 'timing_tns_ns=%s\n' "$(metric_from_report "$TIMING_REPORT" 'TNS(ns)')"
    printf 'timing_whs_ns=%s\n' "$(metric_from_report "$TIMING_REPORT" 'WHS(ns)')"
    printf 'timing_ths_ns=%s\n' "$(metric_from_report "$TIMING_REPORT" 'THS(ns)')"
    printf 'impl_timing_status=%s\n' "$IMPL_TIMING_STATUS"
    printf 'impl_timing_report=%s\n' "$IMPL_TIMING_REPORT"
    printf 'impl_timing_wns_ns=%s\n' "$(metric_from_report "$IMPL_TIMING_REPORT" 'WNS(ns)')"
    printf 'impl_timing_tns_ns=%s\n' "$(metric_from_report "$IMPL_TIMING_REPORT" 'TNS(ns)')"
    printf 'impl_timing_whs_ns=%s\n' "$(metric_from_report "$IMPL_TIMING_REPORT" 'WHS(ns)')"
    printf 'impl_timing_ths_ns=%s\n' "$(metric_from_report "$IMPL_TIMING_REPORT" 'THS(ns)')"
    printf 'utilization_report=%s\n' "$UTIL_REPORT"
    printf 'impl_utilization_report=%s\n' "$IMPL_UTIL_REPORT"
    printf 'bitstream_status_file=%s\n' "$BITSTREAM_STATUS_FILE"
    printf 'bitstream_status=%s\n' "$BITSTREAM_STATUS"
    printf 'implementation_scope=%s\n' "$IMPLEMENTATION_SCOPE"
    printf 'full_top_level_status_file=%s\n' "$TOP_LEVEL_STATUS_FILE"
    printf 'full_top_level_flow=%s\n' "$FULL_TOP_LEVEL_FLOW"
    printf 'vivado=%s\n' "$(command -v vivado 2>/dev/null || printf MISSING)"
    printf 'vivado_version=%s\n' "$VIVADO_VERSION"
    printf 'xvlog=%s\n' "$(command -v xvlog 2>/dev/null || printf MISSING)"
    printf 'part=xck26-sfvc784-2LV-c\n'
    printf 'constraints=%s\n' "$HW_DIR/constraints/pccx_timing.xdc"
    printf 'timing_closure_claim=no\n'
} >"$SUMMARY"

printf 'summary=%s\n' "$SUMMARY"
printf 'overall_timing_status=%s\n' "$OVERALL_TIMING_STATUS"
printf 'timing_status=%s\n' "$TIMING_STATUS"
printf 'impl_timing_status=%s\n' "$IMPL_TIMING_STATUS"

if (( REQUIRE_IMPL_CLEAN )); then
    case "$IMPL_TIMING_STATUS" in
        TIMING_REPORT_PRESENT_CLOSED)
            exit 0
            ;;
        *)
            exit 2
            ;;
    esac
else
    case "$TIMING_STATUS" in
        TIMING_REPORT_PRESENT_CLOSED)
            exit 0
            ;;
        *)
            exit 2
            ;;
    esac
fi
