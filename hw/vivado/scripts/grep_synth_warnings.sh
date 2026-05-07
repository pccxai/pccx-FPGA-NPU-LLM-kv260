#!/usr/bin/env bash
# Print Vivado critical warnings from a synthesis log.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'USAGE'
usage: hw/vivado/scripts/grep_synth_warnings.sh [synth.log|-]

Print CRITICAL WARNING records from a Vivado synthesis log to stdout.

With no argument, the script probes common local build outputs:
  ./synth.log
  hw/build/synth.log
  hw/build/vivado_synth.log
  hw/build/pccx_v002_kv260/pccx_v002_kv260.runs/synth_1/runme.log
  newest hw/build/v002-timing-evidence/*/synth.log

Use "-" to read from stdin.
USAGE
}

latest_timing_evidence_log() {
    [[ -d "$HW_DIR/build/v002-timing-evidence" ]] || return 0

    find "$HW_DIR/build/v002-timing-evidence" -type f -name synth.log \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print}'
}

default_log() {
    local candidate
    local candidates=(
        "$PWD/synth.log"
        "$HW_DIR/build/synth.log"
        "$HW_DIR/build/vivado_synth.log"
        "$HW_DIR/build/pccx_v002_kv260/pccx_v002_kv260.runs/synth_1/runme.log"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    candidate="$(latest_timing_evidence_log)"
    if [[ -n "$candidate" && -f "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    return 1
}

extract_critical_warnings() {
    awk '
        /^CRITICAL WARNING:/ {
            print
            in_record = 1
            next
        }
        /^[[:space:]]+/ && in_record {
            print
            next
        }
        /^[A-Z][A-Z ]*:/ {
            in_record = 0
        }
    ' "$@"
}

if (($# > 1)); then
    usage >&2
    exit 2
fi

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    -)
        extract_critical_warnings /dev/stdin
        ;;
    "")
        log_file="$(default_log)" || {
            echo "error: no synth log found; pass a path or use '-' for stdin" >&2
            exit 1
        }
        extract_critical_warnings "$log_file"
        ;;
    *)
        log_file="$1"
        if [[ ! -f "$log_file" ]]; then
            echo "error: synth log not found: $log_file" >&2
            exit 1
        fi
        extract_critical_warnings "$log_file"
        ;;
esac
