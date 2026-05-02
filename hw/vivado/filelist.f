# pccx v002 RTL compile filelist
#
# Ordering matters: packages and interfaces before modules that import them.
# Invoke with:  xvlog -sv -f filelist.f  (xvlog expects paths relative to $PWD)
# Use this list from any tool via -sourceset / tcl add_files too.

# ===| A: pure `define headers (no compile, included via `include) |===========
# Present here for reference; they are picked up via -i <dir> on the command
# line, not compiled. The Vivado TCL wrapper translates these to
# set_property include_dirs on the fileset.
#
# rtl/Constants/compilePriority_Order/A_const_svh/*.svh

# ===| B: device_pkg (depends on A) |==========================================
rtl/Constants/compilePriority_Order/B_device_pkg/device_pkg.sv

# ===| C: dtype / mem packages (depend on B) |=================================
rtl/Constants/compilePriority_Order/C_type_pkg/dtype_pkg.sv
rtl/Constants/compilePriority_Order/C_type_pkg/mem_pkg.sv

# ===| D: vector-core configuration package (depends on B+C) |=================
rtl/Constants/compilePriority_Order/D_pipeline_pkg/vec_core_pkg.sv

# ===| E: observability counter package (depends on int only) |================
# Vocabulary-only at Stage C: PerfCountersEnableDefault, counter widths, and
# the handshake_counter_t struct. No module imports it yet; opt-in counter
# wiring will arrive as a separate Stage D MVP commit.
rtl/Constants/compilePriority_Order/E_obs_pkg/perf_counter_pkg.sv

# ===| Library packages and interfaces |=======================================
rtl/Library/Algorithms/BF16_math.sv
rtl/Library/Algorithms/Algorithms.sv
rtl/Library/Algorithms/QUEUE/IF_queue.sv

# ===| ISA package (depends on types) |========================================
rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv
rtl/MEM_control/memory/Constant_Memory/shape_const_ram.sv

# ===| MAT_CORE |==============================================================
rtl/MAT_CORE/GEMM_dsp_packer.sv
rtl/MAT_CORE/GEMM_sign_recovery.sv
rtl/MAT_CORE/GEMM_dsp_unit.sv
rtl/MAT_CORE/GEMM_dsp_unit_last_ROW.sv
rtl/MAT_CORE/GEMM_accumulator.sv
rtl/MAT_CORE/GEMM_fmap_staggered_delay.sv
rtl/MAT_CORE/GEMM_weight_dispatcher.sv
rtl/MAT_CORE/GEMM_systolic_array.sv
rtl/MAT_CORE/GEMM_systolic_top.sv
rtl/MAT_CORE/FROM_mat_result_packer.sv
rtl/MAT_CORE/mat_result_normalizer.sv

# ===| VEC_CORE |==============================================================
rtl/VEC_CORE/GEMV_accumulate.sv
rtl/VEC_CORE/GEMV_generate_lut.sv
rtl/VEC_CORE/GEMV_reduction.sv
rtl/VEC_CORE/GEMV_reduction_branch.sv
rtl/VEC_CORE/GEMV_top.sv

# ===| CVO_CORE |==============================================================
rtl/CVO_CORE/CVO_cordic_unit.sv
rtl/CVO_CORE/CVO_sfu_unit.sv
rtl/CVO_CORE/CVO_top.sv

# ===| PREPROCESS |============================================================
rtl/PREPROCESS/preprocess_bf16_fixed_pipeline.sv
rtl/PREPROCESS/fmap_cache.sv
rtl/PREPROCESS/preprocess_fmap.sv

# ===| MEM_control |===========================================================
rtl/MEM_control/memory/Constant_Memory/fmap_array_shape.sv
rtl/MEM_control/memory/Constant_Memory/weight_array_shape.sv
rtl/MEM_control/memory/mem_BUFFER.sv
rtl/MEM_control/memory/mem_GLOBAL_cache.sv
rtl/MEM_control/IO/mem_u_operation_queue.sv
rtl/MEM_control/top/mem_HP_buffer.sv
rtl/MEM_control/top/mem_CVO_stream_bridge.sv
rtl/MEM_control/top/mem_L2_cache_fmap.sv
rtl/MEM_control/top/mem_dispatcher.sv

# ===| NPU_Controller |========================================================
rtl/NPU_Controller/NPU_Control_Unit/ctrl_npu_decoder.sv
rtl/NPU_Controller/NPU_frontend/AXIL_CMD_IN.sv
rtl/NPU_Controller/NPU_frontend/AXIL_STAT_OUT.sv
rtl/NPU_Controller/NPU_frontend/ctrl_npu_frontend.sv
rtl/NPU_Controller/Global_Scheduler.sv
rtl/NPU_Controller/npu_controller_top.sv

# ===| Library.QUEUE (after algorithms_pkg) |===================================
rtl/Library/Algorithms/QUEUE/QUEUE.sv

# ===| Misc utility |==========================================================
rtl/barrel_shifter_BF16.sv

# ===| Top level (last) |=====================================================
rtl/NPU_top.sv
