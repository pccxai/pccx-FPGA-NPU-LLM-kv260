#!/usr/bin/env bash
# Collect bounded v002 timing evidence without claiming closure.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HW_DIR="$ROOT_DIR/hw"
REPORT_DIR="$HW_DIR/build/reports"

DRY_RUN=0
RUN_SYNTH=0
RUN_ID="${PCCX_RUN_ID:-}"
ORIGINAL_ARGS=("$@")

usage() {
    cat <<'USAGE'
usage: scripts/v002/run-timing-evidence.sh [--dry-run] [--run-synth] [--run-id <id>]

Options:
  --dry-run       collect environment/report status only
  --run-synth     run hw/vivado/build.sh synth before collecting status
  --run-id <id>   use a deterministic evidence directory suffix
  -h, --help      print this help
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
TIMING_STATUS="$(status_from_report "$TIMING_REPORT")"
IMPL_TIMING_STATUS="$(status_from_report "$IMPL_TIMING_REPORT")"
if [[ "$SYNTH_STATUS" == "failed" && ! -f "$TIMING_REPORT" ]]; then
    TIMING_STATUS="TIMING_ATTEMPTED_NO_REPORT"
fi

capture_report_tail "$TIMING_REPORT" timing_summary_post_synth
capture_report_tail "$UTIL_REPORT" utilization_post_synth
capture_report_tail "$IMPL_TIMING_REPORT" timing_summary_post_impl
capture_report_tail "$IMPL_UTIL_REPORT" utilization_post_impl

VIVADO_VERSION="$(vivado -version 2>/dev/null | head -n 1 || printf MISSING)"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'branch=%s\n' "$BRANCH"
    printf 'git_commit=%s\n' "$COMMIT_FULL"
    printf 'dry_run=%s\n' "$DRY_RUN"
    printf 'run_synth=%s\n' "$RUN_SYNTH"
    printf 'command_line=%q ' "$0" "${ORIGINAL_ARGS[@]}"
    printf '\n'
    printf 'synth_status=%s\n' "$SYNTH_STATUS"
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
    printf 'vivado=%s\n' "$(command -v vivado 2>/dev/null || printf MISSING)"
    printf 'vivado_version=%s\n' "$VIVADO_VERSION"
    printf 'xvlog=%s\n' "$(command -v xvlog 2>/dev/null || printf MISSING)"
    printf 'part=xck26-sfvc784-2LV-c\n'
    printf 'constraints=%s\n' "$HW_DIR/constraints/pccx_timing.xdc"
    printf 'timing_closure_claim=no\n'
} >"$SUMMARY"

printf 'summary=%s\n' "$SUMMARY"
printf 'timing_status=%s\n' "$TIMING_STATUS"

case "$TIMING_STATUS" in
    TIMING_REPORT_PRESENT_CLOSED)
        exit 0
        ;;
    *)
        exit 2
        ;;
esac
