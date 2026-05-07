#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
PCCX_V002_RTL_ROOT="$ROOT_DIR/third_party/pccx-v002"
LOG="$ROOT_DIR/build/sim_v002_submodule.log"

mkdir -p "$ROOT_DIR/build"

fail_with_log() {
    local message="$1"
    printf '%s\n' "$message" >"$LOG"
    printf '%s\n' "$message" >&2
    exit 2
}

if [[ ! -d "$PCCX_V002_RTL_ROOT" ]]; then
    fail_with_log "pccx-v002 submodule not found at $PCCX_V002_RTL_ROOT"
fi

find_smoke_program_tool() {
    local candidate
    for candidate in \
        "$ROOT_DIR/tools/v002/generate_smoke_program.py" \
        "$ROOT_DIR/scripts/v002/generate_smoke_program.py" \
        "$ROOT_DIR/sw/v002/generate_smoke_program.py"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    find "$ROOT_DIR/sw" "$ROOT_DIR/tools" "$ROOT_DIR/scripts" \
        -type f -name 'generate_smoke_program.py' -print 2>/dev/null \
        | sort \
        | head -n 1
}

PCCX_SMOKE_PROGRAM_TOOL="$(find_smoke_program_tool)"
if [[ -z "$PCCX_SMOKE_PROGRAM_TOOL" ]]; then
    fail_with_log "PCCX_SMOKE_PROGRAM_TOOL not found: searched sw/, tools/, scripts/ for generate_smoke_program.py"
fi

export PCCX_V002_RTL_ROOT
export PCCX_SMOKE_PROGRAM_TOOL
if [[ -z "${PCCX_LAB_DIR:-}" && -d "$ROOT_DIR/../pccx-lab" ]]; then
    export PCCX_LAB_DIR="$ROOT_DIR/../pccx-lab"
fi

set +e
(
    cd "$ROOT_DIR"
    printf 'PCCX_V002_RTL_ROOT=%s\n' "$PCCX_V002_RTL_ROOT"
    printf 'PCCX_SMOKE_PROGRAM_TOOL=%s\n' "$PCCX_SMOKE_PROGRAM_TOOL"
    if [[ -n "${PCCX_LAB_DIR:-}" ]]; then
        printf 'PCCX_LAB_DIR=%s\n' "$PCCX_LAB_DIR"
    fi
    printf 'runner=third_party/pccx-v002/LLM/sim/run_verification.sh\n'
    printf '\n'
    PCCX_V002_RTL_ROOT="$PCCX_V002_RTL_ROOT" \
    PCCX_SMOKE_PROGRAM_TOOL="$PCCX_SMOKE_PROGRAM_TOOL" \
        bash third_party/pccx-v002/LLM/sim/run_verification.sh "$@"
) >"$LOG" 2>&1
status=$?
set -e

if (( status == 0 )); then
    printf 'PASS: submodule simulation complete, log: %s\n' "$LOG"
else
    printf 'FAIL: submodule simulation failed (%d), log: %s\n' "$status" "$LOG" >&2
fi
exit "$status"
