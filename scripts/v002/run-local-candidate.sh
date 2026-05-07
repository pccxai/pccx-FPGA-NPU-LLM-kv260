#!/usr/bin/env bash
# Run local v002 candidate checks that do not require a board.
#
# Logs are written under hw/build/ so run evidence stays out of git.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HW_DIR="$ROOT_DIR/hw"
V002_ROOT="$ROOT_DIR/third_party/pccx-v002"

MODE="full"
LIST_ONLY=0
NO_VIVADO=0
DO_VIVADO_COMPILE="auto"
DO_TOP_ELAB="auto"
DO_RUNTIME_DRY_RUN="auto"
RUN_ID="${PCCX_RUN_ID:-}"

usage() {
    cat <<'USAGE'
usage: scripts/v002/run-local-candidate.sh [options]

Options:
  --list                    print planned checks and exit
  --quick                   run syntax, runtime dry-run, and smoke xsim subset
  --full                    run the full local validation set (default)
  --no-vivado               skip Vivado compile and top elaboration checks
  --with-vivado-compile     include Vivado filelist compile when tools exist
  --with-top-elab           include npu_core_wrapper elaboration when tools exist
  --with-runtime-dry-run    include runtime smoke program dry-run
  --run-id <id>             use a deterministic evidence directory suffix
  -h, --help                print this help
USAGE
}

print_step_list() {
    cat <<'STEPS'
available checks:
  bash_syntax
  tool_summary
  runtime_smoke_generate
  runtime_smoke_json
  xsim_list
  xsim_shape_const_ram
  xsim_mem_dispatcher_shape_lookup
  xsim_v002_runtime_smoke_program
  xsim_fmap_staggered_delay
  xsim_full_regression
  vivado_filelist_compile
  npu_core_wrapper_elab

quick mode:
  bash_syntax
  runtime_smoke_generate
  runtime_smoke_json
  xsim_list
  xsim_shape_const_ram
  xsim_mem_dispatcher_shape_lookup
  xsim_v002_runtime_smoke_program

full mode:
  quick mode plus xsim_fmap_staggered_delay, xsim_full_regression,
  and Vivado compile/elaboration unless --no-vivado is used or tools are absent.
STEPS
}

while (($#)); do
    case "$1" in
        --list)
            LIST_ONLY=1
            shift
            ;;
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --no-vivado)
            NO_VIVADO=1
            shift
            ;;
        --with-vivado-compile)
            DO_VIVADO_COMPILE=1
            shift
            ;;
        --with-top-elab)
            DO_TOP_ELAB=1
            shift
            ;;
        --with-runtime-dry-run)
            DO_RUNTIME_DRY_RUN=1
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

if (( LIST_ONLY )); then
    print_step_list
    exit 0
fi

if [[ "$DO_RUNTIME_DRY_RUN" == "auto" ]]; then
    DO_RUNTIME_DRY_RUN=1
fi
if [[ "$DO_VIVADO_COMPILE" == "auto" ]]; then
    [[ "$MODE" == "full" ]] && DO_VIVADO_COMPILE=1 || DO_VIVADO_COMPILE=0
fi
if [[ "$DO_TOP_ELAB" == "auto" ]]; then
    [[ "$MODE" == "full" ]] && DO_TOP_ELAB=1 || DO_TOP_ELAB=0
fi
if (( NO_VIVADO )); then
    DO_VIVADO_COMPILE=0
    DO_TOP_ELAB=0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT="$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || printf 'nogit')"
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$STAMP-$COMMIT"
fi
RUN_ID="${RUN_ID//\//_}"
LOG_ROOT="${PCCX_V002_EVIDENCE_DIR:-$HW_DIR/build/v002-local-candidate/$RUN_ID}"
SUMMARY="$LOG_ROOT/summary.txt"
FAILED_COMMANDS="$LOG_ROOT/failed-commands.txt"

mkdir -p "$LOG_ROOT"
: >"$SUMMARY"
: >"$FAILED_COMMANDS"

PASSED_STEPS=()
FAILED_STEPS=()

quote_command() {
    local quoted=""
    printf -v quoted '%q ' "$@"
    printf '%s' "${quoted% }"
}

log_summary() {
    printf '%s\n' "$*" | tee -a "$SUMMARY"
}

record_failure() {
    local name="$1"
    local status="$2"
    local command="$3"
    local log="$4"
    local tail_log="$5"
    {
        printf 'name=%s\n' "$name"
        printf 'status=%s\n' "$status"
        printf 'command=%s\n' "$command"
        printf 'log=%s\n' "$log"
        printf 'tail=%s\n' "$tail_log"
        printf '\n'
    } >>"$FAILED_COMMANDS"
}

run_logged() {
    local name="$1"
    shift
    local log="$LOG_ROOT/$name.log"
    local tail_log="$LOG_ROOT/$name.tail.txt"
    local command
    command="$(quote_command "$@")"

    log_summary "==> $name"
    log_summary "command: $command"
    if "$@" >"$log" 2>&1; then
        log_summary "result: PASS"
        return 0
    fi

    local status=$?
    tail -n 80 "$log" >"$tail_log" 2>/dev/null || true
    log_summary "result: FAIL ($status), see $log"
    log_summary "tail: $tail_log"
    record_failure "$name" "$status" "$command" "$log" "$tail_log"
    return "$status"
}

run_shell() {
    local name="$1"
    shift
    local command="$*"
    local log="$LOG_ROOT/$name.log"
    local tail_log="$LOG_ROOT/$name.tail.txt"

    log_summary "==> $name"
    log_summary "command: $command"
    if (cd "$ROOT_DIR" && bash -lc "$command") >"$log" 2>&1; then
        log_summary "result: PASS"
        return 0
    fi

    local status=$?
    tail -n 80 "$log" >"$tail_log" 2>/dev/null || true
    log_summary "result: FAIL ($status), see $log"
    log_summary "tail: $tail_log"
    record_failure "$name" "$status" "$command" "$log" "$tail_log"
    return "$status"
}

run_step() {
    local step="$1"
    shift
    if "$@"; then
        PASSED_STEPS+=("$step")
    else
        FAILED_STEPS+=("$step")
    fi
}

tool_summary() {
    local tool path
    log_summary "==> tool_summary"
    for tool in bash git python3 xvlog xelab xsim vivado gh; do
        if path="$(command -v "$tool" 2>/dev/null)"; then
            log_summary "tool.$tool=$path"
        else
            log_summary "tool.$tool=MISSING"
        fi
    done
    log_summary "result: PASS"
}

make_abs_filelist() {
    local out="$1"
    local filelist="$HW_DIR/vivado/filelist.v002.f"
    : >"$out"
    while IFS= read -r line; do
        case "$line" in
            ""|\#*) continue ;;
            "-f "*)
                local nested="${line#-f }"
                case "$nested" in
                    ../third_party/pccx-v002/LLM/scripts/filelist.f)
                        while IFS= read -r subline; do
                            case "$subline" in
                                ""|\#*) continue ;;
                                *) printf '%s/%s\n' "$V002_ROOT" "$subline" ;;
                            esac
                        done <"$V002_ROOT/LLM/scripts/filelist.f"
                        ;;
                    *)
                        printf '%s\n' "$line"
                        ;;
                esac
                ;;
            rtl/*)    printf '%s/%s\n' "$HW_DIR" "$line" ;;
            vivado/*) printf '%s/%s\n' "$HW_DIR" "$line" ;;
            *)        printf '%s\n' "$line" ;;
        esac
    done <"$filelist"
}

run_runtime_dry_run() {
    if (( ! DO_RUNTIME_DRY_RUN )); then
        log_summary "==> runtime_smoke_generate"
        log_summary "result: SKIP, runtime dry-run disabled"
        return 0
    fi

    local out_dir="$LOG_ROOT/runtime_smoke"
    mkdir -p "$out_dir"

    run_logged runtime_smoke_generate \
        python3 "$ROOT_DIR/tools/v002/generate_smoke_program.py" \
            --preset tiny-shape-lookup \
            --out "$out_dir/program.json" \
            --memh "$out_dir/program.memh" || return $?

    run_logged runtime_smoke_json \
        python3 -m json.tool "$out_dir/program.json" || return $?

    log_summary "runtime_smoke_program=$out_dir/program.json"
    log_summary "runtime_smoke_memh=$out_dir/program.memh"
}

run_optional_vivado_compile() {
    if (( ! DO_VIVADO_COMPILE )); then
        log_summary "==> vivado_filelist_compile"
        log_summary "result: SKIP, disabled"
        return 0
    fi
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
          -i '$V002_ROOT/common/rtl/packages/legacy' \
          -i '$V002_ROOT/common/rtl/packages' \
          -i '$V002_ROOT/common/rtl/interfaces' \
          -i '$V002_ROOT/LLM/rtl/packages/isa' \
          -i '$V002_ROOT/LLM/rtl/packages/controller' \
          -i '$V002_ROOT/LLM/rtl/core/mat' \
          -i '$V002_ROOT/LLM/rtl/core/vec' \
          -i '$V002_ROOT/LLM/rtl/interfaces' \
          -f '$abs_filelist'"
}

run_optional_wrapper_elab() {
    if (( ! DO_TOP_ELAB )); then
        log_summary "==> npu_core_wrapper_elab"
        log_summary "result: SKIP, disabled"
        return 0
    fi
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
          -i '$V002_ROOT/common/rtl/packages/legacy' \
          -i '$V002_ROOT/common/rtl/packages' \
          -i '$V002_ROOT/common/rtl/interfaces' \
          -i '$V002_ROOT/LLM/rtl/packages/isa' \
          -i '$V002_ROOT/LLM/rtl/packages/controller' \
          -i '$V002_ROOT/LLM/rtl/core/mat' \
          -i '$V002_ROOT/LLM/rtl/core/vec' \
          -i '$V002_ROOT/LLM/rtl/interfaces' \
          -f '$abs_filelist' '$glbl_v' && \
          xelab -L xpm -L unisims_ver -debug typical npu_core_wrapper glbl -s npu_core_wrapper_snap"
}

write_header() {
    log_summary "pccx v002 local candidate"
    log_summary "utc: $STAMP"
    log_summary "run_id: $RUN_ID"
    log_summary "mode: $MODE"
    log_summary "branch: $(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf unknown)"
    log_summary "commit: $(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
    log_summary "commit_short: $COMMIT"
    if [[ -n "$(git -C "$ROOT_DIR" status --short 2>/dev/null)" ]]; then
        log_summary "worktree: dirty"
        git -C "$ROOT_DIR" status --short >"$LOG_ROOT/git_status_short.txt" 2>/dev/null || true
        log_summary "git_status_short: $LOG_ROOT/git_status_short.txt"
    else
        log_summary "worktree: clean"
    fi
    log_summary "logs: $LOG_ROOT"
    log_summary "vivado_compile_enabled: $DO_VIVADO_COMPILE"
    log_summary "top_elab_enabled: $DO_TOP_ELAB"
    log_summary "runtime_dry_run_enabled: $DO_RUNTIME_DRY_RUN"
    log_summary ""
}

write_footer() {
    local status="PASS"
    if ((${#FAILED_STEPS[@]})); then
        status="FAIL"
    fi

    log_summary ""
    log_summary "exit_status: $status"
    log_summary "passed_steps: ${#PASSED_STEPS[@]}"
    log_summary "failed_steps: ${#FAILED_STEPS[@]}"
    if ((${#FAILED_STEPS[@]})); then
        log_summary "failed_step_names: ${FAILED_STEPS[*]}"
        log_summary "failed_commands: $FAILED_COMMANDS"
    fi
    log_summary ""
    log_summary "non_claim_summary:"
    log_summary "  not_hardware_execution: true unless KV260 evidence says otherwise"
    log_summary "  kv260_inference_success_claim: no"
    log_summary "  gemma3n_e4b_hardware_execution_claim: no"
    log_summary "  tokens_per_second_achieved_claim: no"
    log_summary "  timing_closure_claim: no"
    log_summary ""
    log_summary "LOCAL-RUNNABLE-CANDIDATE checks completed."
}

syntax_files=(
    "$ROOT_DIR/scripts/v002/use_submodule_sources.sh"
    "$ROOT_DIR/scripts/kv260/run_gemma3n_e4b_smoke.sh"
    "$ROOT_DIR/scripts/v002/run-local-candidate.sh"
)
[[ -f "$ROOT_DIR/scripts/v002/run-timing-evidence.sh" ]] && \
    syntax_files+=("$ROOT_DIR/scripts/v002/run-timing-evidence.sh")
[[ -f "$ROOT_DIR/scripts/v002/run-throughput-report.sh" ]] && \
    syntax_files+=("$ROOT_DIR/scripts/v002/run-throughput-report.sh")
[[ -f "$ROOT_DIR/scripts/v002/claim-scan.sh" ]] && \
    syntax_files+=("$ROOT_DIR/scripts/v002/claim-scan.sh")
[[ -f "$ROOT_DIR/scripts/v002/artifact-safety-check.sh" ]] && \
    syntax_files+=("$ROOT_DIR/scripts/v002/artifact-safety-check.sh")

write_header
tool_summary

run_step bash_syntax run_logged bash_syntax bash -n "${syntax_files[@]}"
run_step runtime_dry_run run_runtime_dry_run
run_step xsim_list run_logged xsim_list "$ROOT_DIR/scripts/v002/use_submodule_sources.sh" --list
run_step xsim_shape_const_ram run_logged xsim_shape_const_ram "$ROOT_DIR/scripts/v002/use_submodule_sources.sh" --tb tb_shape_const_ram
run_step xsim_mem_dispatcher_shape_lookup run_logged xsim_mem_dispatcher_shape_lookup "$ROOT_DIR/scripts/v002/use_submodule_sources.sh" --tb tb_mem_dispatcher_shape_lookup
run_step xsim_v002_runtime_smoke_program run_logged xsim_v002_runtime_smoke_program "$ROOT_DIR/scripts/v002/use_submodule_sources.sh" --tb tb_v002_runtime_smoke_program

if [[ "$MODE" == "full" ]]; then
    run_step xsim_fmap_staggered_delay run_logged xsim_fmap_staggered_delay "$ROOT_DIR/scripts/v002/use_submodule_sources.sh" --tb tb_GEMM_fmap_staggered_delay
    run_step xsim_full_regression run_logged xsim_full_regression "$ROOT_DIR/scripts/v002/use_submodule_sources.sh" --full
fi

run_step vivado_filelist_compile run_optional_vivado_compile
run_step npu_core_wrapper_elab run_optional_wrapper_elab

write_footer

if ((${#FAILED_STEPS[@]})); then
    exit 1
fi
exit 0
