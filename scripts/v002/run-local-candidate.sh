#!/usr/bin/env bash
# Run the strongest local v002 candidate checks that do not require a board.
#
# Logs are written under hw/build/ so generated evidence stays out of git.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HW_DIR="$ROOT_DIR/hw"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT="$(git -C "$ROOT_DIR" rev-parse --short HEAD)"
LOG_ROOT="${PCCX_V002_EVIDENCE_DIR:-$HW_DIR/build/v002-local-candidate/$STAMP-$COMMIT}"
SUMMARY="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

log_summary() {
    printf '%s\n' "$*" | tee -a "$SUMMARY"
}

run_logged() {
    local name="$1"
    shift
    local log="$LOG_ROOT/$name.log"

    log_summary "==> $name"
    log_summary "command: $*"
    if "$@" >"$log" 2>&1; then
        log_summary "result: PASS"
    else
        local status=$?
        log_summary "result: FAIL ($status), see $log"
        return "$status"
    fi
}

run_shell() {
    local name="$1"
    shift
    local log="$LOG_ROOT/$name.log"

    log_summary "==> $name"
    log_summary "command: $*"
    if (cd "$ROOT_DIR" && bash -lc "$*") >"$log" 2>&1; then
        log_summary "result: PASS"
    else
        local status=$?
        log_summary "result: FAIL ($status), see $log"
        return "$status"
    fi
}

make_abs_filelist() {
    local out="$1"
    while IFS= read -r line; do
        case "$line" in
            rtl/*)    printf '%s/%s\n' "$HW_DIR" "$line" ;;
            vivado/*) printf '%s/%s\n' "$HW_DIR" "$line" ;;
            *)        printf '%s\n' "$line" ;;
        esac
    done <"$HW_DIR/vivado/filelist.f" >"$out"
}

run_optional_vivado_compile() {
    if ! command -v xvlog >/dev/null 2>&1; then
        log_summary "==> vivado_filelist_compile"
        log_summary "result: SKIP, xvlog not found"
        return 0
    fi

    local work="$LOG_ROOT/vivado_filelist_work"
    local abs_filelist="$LOG_ROOT/filelist.abs.f"
    mkdir -p "$work"
    make_abs_filelist "$abs_filelist"

    run_shell vivado_filelist_compile \
        "cd '$work' && xvlog -sv \
          -i '$HW_DIR/rtl' \
          -i '$HW_DIR/rtl/Constants/compilePriority_Order/A_const_svh' \
          -i '$HW_DIR/rtl/MAT_CORE' \
          -i '$HW_DIR/rtl/MEM_control/IO' \
          -i '$HW_DIR/rtl/NPU_Controller' \
          -i '$HW_DIR/rtl/NPU_Controller/NPU_Control_Unit' \
          -i '$HW_DIR/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE' \
          -i '$HW_DIR/rtl/VEC_CORE' \
          -f '$abs_filelist'"
}

run_optional_wrapper_elab() {
    if ! command -v xvlog >/dev/null 2>&1 || ! command -v xelab >/dev/null 2>&1; then
        log_summary "==> npu_core_wrapper_elab"
        log_summary "result: SKIP, xvlog/xelab not found"
        return 0
    fi

    local vivado_root="${XILINX_VIVADO:-}"
    local glbl_v="${vivado_root:+$vivado_root/ids_lite/ISE/verilog/src/glbl.v}"
    if [[ -z "$glbl_v" || ! -f "$glbl_v" ]]; then
        log_summary "==> npu_core_wrapper_elab"
        log_summary "result: SKIP, XILINX_VIVADO glbl.v not found"
        return 0
    fi

    local work="$LOG_ROOT/top_elab_work"
    local abs_filelist="$LOG_ROOT/filelist.top.abs.f"
    mkdir -p "$work"
    make_abs_filelist "$abs_filelist"

    run_shell npu_core_wrapper_elab \
        "cd '$work' && xvlog -sv \
          -i '$HW_DIR/rtl' \
          -i '$HW_DIR/rtl/Constants/compilePriority_Order/A_const_svh' \
          -i '$HW_DIR/rtl/MAT_CORE' \
          -i '$HW_DIR/rtl/MEM_control/IO' \
          -i '$HW_DIR/rtl/NPU_Controller' \
          -i '$HW_DIR/rtl/NPU_Controller/NPU_Control_Unit' \
          -i '$HW_DIR/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE' \
          -i '$HW_DIR/rtl/VEC_CORE' \
          -f '$abs_filelist' '$glbl_v' && \
          xelab -L xpm -L unisims_ver -debug typical npu_core_wrapper glbl -s npu_core_wrapper_snap"
}

: >"$SUMMARY"
log_summary "pccx v002 local candidate"
log_summary "utc: $STAMP"
log_summary "branch: $(git -C "$ROOT_DIR" branch --show-current)"
log_summary "commit: $(git -C "$ROOT_DIR" rev-parse HEAD)"
log_summary "logs: $LOG_ROOT"
log_summary ""

run_logged bash_syntax bash -n \
    "$ROOT_DIR/hw/sim/run_verification.sh" \
    "$ROOT_DIR/scripts/kv260/run_gemma3n_e4b_smoke.sh" \
    "$ROOT_DIR/scripts/v002/run-local-candidate.sh"

run_logged xsim_list "$ROOT_DIR/hw/sim/run_verification.sh" --list
run_logged xsim_shape_const_ram "$ROOT_DIR/hw/sim/run_verification.sh" --tb tb_shape_const_ram
run_logged xsim_mem_dispatcher_shape_lookup "$ROOT_DIR/hw/sim/run_verification.sh" --tb tb_mem_dispatcher_shape_lookup
run_logged xsim_fmap_staggered_delay "$ROOT_DIR/hw/sim/run_verification.sh" --tb tb_GEMM_fmap_staggered_delay
run_logged xsim_full_regression "$ROOT_DIR/hw/sim/run_verification.sh"

run_optional_vivado_compile
run_optional_wrapper_elab

log_summary ""
log_summary "LOCAL-RUNNABLE-CANDIDATE checks completed."
