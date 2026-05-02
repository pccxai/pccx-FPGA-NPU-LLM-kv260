#!/usr/bin/env bash
# Unified verification runner for pccx-FPGA.
#
# Runs every known testbench in deterministic order and reports a one-line
# summary plus the path to the generated .pccx traces so pccx-lab can
# visualise them.
#
# Intentionally idempotent: each tb writes to its own work dir under
# hw/sim/work/<tb_name>/ so concurrent / repeated runs don't stomp.

set -euo pipefail

HW_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SIM_DIR="$HW_DIR/sim"
WORK_ROOT="$SIM_DIR/work"
PCCX_LAB_DIR="${PCCX_LAB_DIR:-$HW_DIR/../../pccx-lab}"
PCCX_CLI_BIN="$PCCX_LAB_DIR/target/release/from_xsim_log"

mkdir -p "$WORK_ROOT"

# Build the pccx-lab bridge binary if stale / missing.
if [[ ! -x "$PCCX_CLI_BIN" ]]; then
    echo "==> Building from_xsim_log (one-time setup)"
    (cd "$PCCX_LAB_DIR" && cargo build --release --bin from_xsim_log)
fi

# ─── Per-testbench RTL dependency map ───────────────────────────────────────
# Each entry lists the .sv modules xvlog must pick up, relative to hw/rtl/.
declare -A TB_DEPS=(
    [tb_shape_const_ram]="NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv MEM_control/memory/Constant_Memory/shape_const_ram.sv"
    [tb_GEMM_dsp_packer_sign_recovery]="MAT_CORE/GEMM_dsp_packer.sv MAT_CORE/GEMM_sign_recovery.sv"
    [tb_mat_result_normalizer]="Constants/compilePriority_Order/C_type_pkg/dtype_pkg.sv MAT_CORE/mat_result_normalizer.sv"
    [tb_GEMM_weight_dispatcher]="MAT_CORE/GEMM_weight_dispatcher.sv"
    [tb_FROM_mat_result_packer]="MAT_CORE/FROM_mat_result_packer.sv"
    [tb_barrel_shifter_BF16]="barrel_shifter_BF16.sv"
    [tb_ctrl_npu_decoder]="NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv NPU_Controller/NPU_Control_Unit/ctrl_npu_decoder.sv"
    [tb_mem_u_operation_queue]="Constants/compilePriority_Order/E_obs_pkg/perf_counter_pkg.sv NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv MEM_control/IO/mem_u_operation_queue.sv"
    [tb_GEMM_fmap_staggered_delay]="MAT_CORE/GEMM_fmap_staggered_delay.sv"
)

# Core-id assigned to a tb's emitted pccx trace. Kept contiguous so the UI
# Timeline shows one lane per testbench when the traces are composed.
declare -A TB_CORE=(
    [tb_shape_const_ram]=0
    [tb_GEMM_dsp_packer_sign_recovery]=1
    [tb_mat_result_normalizer]=2
    [tb_GEMM_weight_dispatcher]=3
    [tb_FROM_mat_result_packer]=4
    [tb_barrel_shifter_BF16]=5
    [tb_ctrl_npu_decoder]=6
    [tb_mem_u_operation_queue]=7
    [tb_GEMM_fmap_staggered_delay]=8
)

TB_LIST=(
    tb_shape_const_ram
    tb_GEMM_dsp_packer_sign_recovery
    tb_mat_result_normalizer
    tb_GEMM_weight_dispatcher
    tb_FROM_mat_result_packer
    tb_barrel_shifter_BF16
    tb_ctrl_npu_decoder
    tb_mem_u_operation_queue
    tb_GEMM_fmap_staggered_delay
)

run_tb() {
    local tb="$1"
    local work="$WORK_ROOT/$tb"
    mkdir -p "$work"

    local -a rtl_args=()
    for rel in ${TB_DEPS[$tb]}; do
        rtl_args+=("$HW_DIR/rtl/$rel")
    done

    local tool_status=0
    (
        set -euo pipefail
        cd "$work"
        echo "  [xvlog]"
        xvlog -sv \
            -i "$HW_DIR/rtl/Constants/compilePriority_Order/A_const_svh" \
            -i "$HW_DIR/rtl/MAT_CORE" \
            "${rtl_args[@]}" \
            "$HW_DIR/tb/$tb.sv" \
            >xvlog.log 2>&1

        echo "  [xelab]"
        # -L xpm: needed for TBs that compile xpm_fifo_sync (e.g. the
        # mem_u_operation_queue queue smoke). No-op for TBs that don't.
        xelab -L xpm -debug typical "$tb" -s snap >xelab.log 2>&1

        echo "  [xsim]"
        xsim snap -R >xsim.log 2>&1

        "$PCCX_CLI_BIN" \
            --log    "$work/xsim.log" \
            --output "$work/$tb.pccx" \
            --testbench "$tb" \
            --core-id "${TB_CORE[$tb]:-0}" \
            >"$work/from_xsim_log.log" 2>&1
    ) || tool_status=$?

    if (( tool_status != 0 )); then
        printf '%-50s  TOOL FAIL: see %s\n' "$tb" "$work"
        return 1
    fi

    local verdict
    verdict="$(awk '/^(PASS|FAIL):/ { print; exit }' "$work/xsim.log")"
    if [[ -z "$verdict" ]]; then
        verdict="FAIL: no PASS/FAIL verdict in $work/xsim.log"
    fi
    printf '%-50s  %s\n' "$tb" "$verdict"

    [[ "$verdict" == PASS:* ]]
}

# ─── Main ───────────────────────────────────────────────────────────────────
echo "==> Running pccx-FPGA testbench suite"
echo ""
printf '%-50s  %s\n' "TESTBENCH" "RESULT"
printf '%-50s  %s\n' "---------" "------"
overall_status=0
for tb in "${TB_LIST[@]}"; do
    if ! run_tb "$tb"; then
        overall_status=1
    fi
done

echo ""
echo "==> Generated .pccx traces:"
find "$WORK_ROOT" -name '*.pccx' -print

echo ""
echo "==> Synthesis status (from existing hw/build/reports):"
if [[ -f "$HW_DIR/build/reports/timing_summary_post_synth.rpt" ]]; then
    grep -E 'Timing constraints are (not )?met' \
        "$HW_DIR/build/reports/timing_summary_post_synth.rpt" \
        | head -1 \
        || echo "  (no verdict line in timing report)"
else
    echo "  No synth report — run hw/vivado/synth.tcl first."
fi

exit "$overall_status"
