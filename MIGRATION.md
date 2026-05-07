# v002 IP-core extraction migration map

This map records reusable IP-core files copied from this KV260 integration repository into `pccx-v002`. The duplicate in-tree copies were removed after this repository was redirected to consume `pccx-v002` through `third_party/pccx-v002`.

Source: `pccx-v002/SOURCE_MANIFEST.md`

| old path | new owner repo | new path | reason |
| --- | --- | --- | --- |
| `hw/rtl/Constants/compilePriority_Order/A_const_svh/GLOBAL_CONST.svh` | `pccx-v002` | `common/rtl/packages/legacy/GLOBAL_CONST.svh` | include updated for device profile rename |
| `hw/rtl/Constants/compilePriority_Order/A_const_svh/NUMBERS.svh` | `pccx-v002` | `common/rtl/packages/NUMBERS.svh` | none |
| `hw/rtl/Constants/compilePriority_Order/A_const_svh/npu_arch.svh` | `pccx-v002` | `common/rtl/packages/npu_arch.svh` | board reference removed from copied comment |
| `hw/rtl/Constants/compilePriority_Order/B_device_pkg/device_pkg.sv` | `pccx-v002` | `common/rtl/packages/device_pkg.sv` | none |
| `hw/rtl/Constants/compilePriority_Order/C_type_pkg/dtype_pkg.sv` | `pccx-v002` | `common/rtl/packages/dtype_pkg.sv` | none |
| `hw/rtl/Constants/compilePriority_Order/C_type_pkg/mem_pkg.sv` | `pccx-v002` | `common/rtl/packages/mem_pkg.sv` | include updated for device profile rename |
| `hw/rtl/Constants/compilePriority_Order/D_pipeline_pkg/vec_core_pkg.sv` | `pccx-v002` | `common/rtl/packages/vec_core_pkg.sv` | target-specific comment removed |
| `hw/rtl/Constants/compilePriority_Order/E_obs_pkg/perf_counter_pkg.sv` | `pccx-v002` | `common/rtl/packages/perf_counter_pkg.sv` | none |
| `hw/rtl/Library/Algorithms/Algorithms.sv` | `pccx-v002` | `common/rtl/packages/Algorithms.sv` | none |
| `hw/rtl/Library/Algorithms/BF16_math.sv` | `pccx-v002` | `common/rtl/packages/BF16_math.sv` | none |
| `hw/rtl/Library/Algorithms/QUEUE/IF_queue.sv` | `pccx-v002` | `common/rtl/interfaces/IF_queue.sv` | none |
| `hw/rtl/Library/Algorithms/QUEUE/QUEUE.sv` | `pccx-v002` | `common/rtl/wrappers/QUEUE.sv` | none |
| `hw/rtl/NPU_Controller/npu_interfaces.svh` | `pccx-v002` | `common/rtl/interfaces/npu_interfaces.svh` | none |
| `hw/rtl/barrel_shifter_BF16.sv` | `pccx-v002` | `common/rtl/wrappers/barrel_shifter_BF16.sv` | none |
| `hw/rtl/Constants/compilePriority_Order/A_const_svh/kv260_device.svh` | `pccx-v002` | `common/rtl/packages/device_profile.svh` | board-named device profile generalized |
| `hw/rtl/CVO_CORE/CVO_cordic_unit.sv` | `pccx-v002` | `LLM/rtl/core/cvo/CVO_cordic_unit.sv` | none |
| `hw/rtl/CVO_CORE/CVO_sfu_unit.sv` | `pccx-v002` | `LLM/rtl/core/cvo/CVO_sfu_unit.sv` | none |
| `hw/rtl/CVO_CORE/CVO_top.sv` | `pccx-v002` | `LLM/rtl/core/cvo/CVO_top.sv` | none |
| `hw/rtl/MAT_CORE/FROM_mat_result_packer.sv` | `pccx-v002` | `LLM/rtl/core/mat/FROM_mat_result_packer.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_Array.svh` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_Array.svh` | none |
| `hw/rtl/MAT_CORE/GEMM_accumulator.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_accumulator.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_dsp_packer.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_dsp_packer.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_dsp_unit.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_dsp_unit.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_dsp_unit_last_ROW.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_dsp_unit_last_ROW.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_fmap_staggered_delay.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_fmap_staggered_delay.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_sign_recovery.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_sign_recovery.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_systolic_array.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_systolic_array.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_systolic_top.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_systolic_top.sv` | none |
| `hw/rtl/MAT_CORE/GEMM_weight_dispatcher.sv` | `pccx-v002` | `LLM/rtl/core/mat/GEMM_weight_dispatcher.sv` | none |
| `hw/rtl/MAT_CORE/mat_result_normalizer.sv` | `pccx-v002` | `LLM/rtl/core/mat/mat_result_normalizer.sv` | none |
| `hw/rtl/MEM_control/IO/mem_IO.svh` | `pccx-v002` | `LLM/rtl/interfaces/mem_IO.svh` | none |
| `hw/rtl/MEM_control/IO/mem_u_operation_queue.sv` | `pccx-v002` | `LLM/rtl/core/memory/mem_u_operation_queue.sv` | none |
| `hw/rtl/MEM_control/memory/Constant_Memory/shape_const_ram.sv` | `pccx-v002` | `LLM/rtl/core/memory/Constant_Memory/shape_const_ram.sv` | none |
| `hw/rtl/MEM_control/memory/mem_BUFFER.sv` | `pccx-v002` | `LLM/rtl/core/memory/mem_BUFFER.sv` | none |
| `hw/rtl/MEM_control/memory/mem_GLOBAL_cache.sv` | `pccx-v002` | `LLM/rtl/core/memory/mem_GLOBAL_cache.sv` | none |
| `hw/rtl/MEM_control/top/mem_CVO_stream_bridge.sv` | `pccx-v002` | `LLM/rtl/core/memory/mem_CVO_stream_bridge.sv` | none |
| `hw/rtl/MEM_control/top/mem_HP_buffer.sv` | `pccx-v002` | `LLM/rtl/core/memory/mem_HP_buffer.sv` | none |
| `hw/rtl/MEM_control/top/mem_L2_cache_fmap.sv` | `pccx-v002` | `LLM/rtl/core/memory/mem_L2_cache_fmap.sv` | none |
| `hw/rtl/MEM_control/top/mem_dispatcher.sv` | `pccx-v002` | `LLM/rtl/core/memory/mem_dispatcher.sv` | none |
| `hw/rtl/NPU_Controller/Global_Scheduler.sv` | `pccx-v002` | `LLM/rtl/core/controller/Global_Scheduler.sv` | none |
| `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_memctrl.svh` | `pccx-v002` | `LLM/rtl/packages/isa/isa_memctrl.svh` | none |
| `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv` | `pccx-v002` | `LLM/rtl/packages/isa/isa_pkg.sv` | none |
| `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_x32.svh` | `pccx-v002` | `LLM/rtl/packages/isa/isa_x32.svh` | none |
| `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_x64.svh` | `pccx-v002` | `LLM/rtl/packages/isa/isa_x64.svh` | none |
| `hw/rtl/NPU_Controller/NPU_Control_Unit/ctrl_decode_const.svh` | `pccx-v002` | `LLM/rtl/packages/controller/ctrl_decode_const.svh` | none |
| `hw/rtl/NPU_Controller/NPU_Control_Unit/ctrl_npu_decoder.sv` | `pccx-v002` | `LLM/rtl/core/controller/ctrl_npu_decoder.sv` | none |
| `hw/rtl/NPU_Controller/NPU_frontend/AXIL_CMD_IN.sv` | `pccx-v002` | `LLM/rtl/core/controller/AXIL_CMD_IN.sv` | none |
| `hw/rtl/NPU_Controller/NPU_frontend/AXIL_STAT_OUT.sv` | `pccx-v002` | `LLM/rtl/core/controller/AXIL_STAT_OUT.sv` | none |
| `hw/rtl/NPU_Controller/NPU_frontend/ctrl_npu_frontend.sv` | `pccx-v002` | `LLM/rtl/core/controller/ctrl_npu_frontend.sv` | none |
| `hw/rtl/NPU_Controller/npu_controller_top.sv` | `pccx-v002` | `LLM/rtl/core/controller/npu_controller_top.sv` | none |
| `hw/rtl/NPU_top.sv` | `pccx-v002` | `LLM/rtl/top/pccx_npu_top.sv` | package top rename |
| `hw/rtl/PREPROCESS/fmap_cache.sv` | `pccx-v002` | `LLM/rtl/core/preprocess/fmap_cache.sv` | none |
| `hw/rtl/PREPROCESS/preprocess_bf16_fixed_pipeline.sv` | `pccx-v002` | `LLM/rtl/core/preprocess/preprocess_bf16_fixed_pipeline.sv` | none |
| `hw/rtl/PREPROCESS/preprocess_fmap.sv` | `pccx-v002` | `LLM/rtl/core/preprocess/preprocess_fmap.sv` | none |
| `hw/rtl/VEC_CORE/GEMV_Vec_Matrix_MUL.svh` | `pccx-v002` | `LLM/rtl/core/vec/GEMV_Vec_Matrix_MUL.svh` | none |
| `hw/rtl/VEC_CORE/GEMV_accumulate.sv` | `pccx-v002` | `LLM/rtl/core/vec/GEMV_accumulate.sv` | none |
| `hw/rtl/VEC_CORE/GEMV_generate_lut.sv` | `pccx-v002` | `LLM/rtl/core/vec/GEMV_generate_lut.sv` | none |
| `hw/rtl/VEC_CORE/GEMV_reduction.sv` | `pccx-v002` | `LLM/rtl/core/vec/GEMV_reduction.sv` | none |
| `hw/rtl/VEC_CORE/GEMV_reduction_branch.sv` | `pccx-v002` | `LLM/rtl/core/vec/GEMV_reduction_branch.sv` | none |
| `hw/rtl/VEC_CORE/GEMV_top.sv` | `pccx-v002` | `LLM/rtl/core/vec/GEMV_top.sv` | none |
| `hw/sim/run_verification.sh` | `pccx-v002` | `LLM/sim/run_verification.sh` | simulation paths updated for package layout |
| `hw/tb/tb_FROM_mat_result_packer.sv` | `pccx-v002` | `LLM/tb/tb_FROM_mat_result_packer.sv` | none |
| `hw/tb/tb_GEMM_dsp_packer_sign_recovery.sv` | `pccx-v002` | `LLM/tb/tb_GEMM_dsp_packer_sign_recovery.sv` | none |
| `hw/tb/tb_GEMM_fmap_staggered_delay.sv` | `pccx-v002` | `LLM/tb/tb_GEMM_fmap_staggered_delay.sv` | none |
| `hw/tb/tb_GEMM_weight_dispatcher.sv` | `pccx-v002` | `LLM/tb/tb_GEMM_weight_dispatcher.sv` | none |
| `hw/tb/tb_barrel_shifter_BF16.sv` | `pccx-v002` | `LLM/tb/tb_barrel_shifter_BF16.sv` | none |
| `hw/tb/tb_ctrl_npu_decoder.sv` | `pccx-v002` | `LLM/tb/tb_ctrl_npu_decoder.sv` | none |
| `hw/tb/tb_mat_result_normalizer.sv` | `pccx-v002` | `LLM/tb/tb_mat_result_normalizer.sv` | none |
| `hw/tb/tb_mem_dispatcher_shape_lookup.sv` | `pccx-v002` | `LLM/tb/tb_mem_dispatcher_shape_lookup.sv` | none |
| `hw/tb/tb_mem_u_operation_queue.sv` | `pccx-v002` | `LLM/tb/tb_mem_u_operation_queue.sv` | none |
| `hw/tb/tb_shape_const_ram.sv` | `pccx-v002` | `LLM/tb/tb_shape_const_ram.sv` | none |
| `hw/tb/tb_v002_runtime_smoke_program.sv` | `pccx-v002` | `LLM/tb/tb_v002_runtime_smoke_program.sv` | none |
| `formal/sail/Makefile` | `pccx-v002` | `LLM/formal/sail/Makefile` | none |
| `formal/sail/README.md` | `pccx-v002` | `LLM/formal/sail/README.md` | none |
| `formal/sail/pccx.sail_project` | `pccx-v002` | `LLM/formal/sail/pccx.sail_project` | none |
| `formal/sail/src/pccx_decode.sail` | `pccx-v002` | `LLM/formal/sail/src/pccx_decode.sail` | none |
| `formal/sail/src/pccx_execute.sail` | `pccx-v002` | `LLM/formal/sail/src/pccx_execute.sail` | none |
| `formal/sail/src/pccx_regs.sail` | `pccx-v002` | `LLM/formal/sail/src/pccx_regs.sail` | target-model example removed from copied comment |
| `formal/sail/src/pccx_types.sail` | `pccx-v002` | `LLM/formal/sail/src/pccx_types.sail` | none |
| `formal/sail/src/prelude.sail` | `pccx-v002` | `LLM/formal/sail/src/prelude.sail` | none |
| `formal/sail/tests/smoke_decode.sail` | `pccx-v002` | `LLM/formal/sail/tests/smoke_decode.sail` | none |
| `hw/vivado/filelist.f` | `pccx-v002` | `LLM/scripts/filelist.f` | package compile list path with BD shim removed |

## Removed in submodule transition

| directory | commit | reason |
| --- | --- | --- |
| `hw/rtl/Constants/`, `hw/rtl/Library/`, `hw/rtl/barrel_shifter_BF16.sv` | `d018959` | Common RTL now comes from `third_party/pccx-v002/common/rtl/`. |
| `hw/rtl/MAT_CORE/`, `hw/rtl/CVO_CORE/`, `hw/rtl/VEC_CORE/`, `hw/rtl/PREPROCESS/`, `hw/rtl/NPU_Controller/`, `hw/rtl/MEM_control/`, `hw/rtl/NPU_top.sv` | `8d7f8d0` | LLM core RTL now comes from `third_party/pccx-v002/LLM/rtl/`. |
| `hw/tb/`, `hw/sim/run_verification.sh` | `610cdd0` | Testbench and sim runner now come from `third_party/pccx-v002/LLM/`. |
| `formal/sail/` | `4e3ff17` | Formal sources now come from `third_party/pccx-v002/LLM/formal/sail/`. |
| `hw/vivado/filelist.f` | `596116f` | KV260 Vivado flow now enters through `hw/vivado/filelist.v002.f`. |
| `.github/workflows/sail-check.yml` | `30fea1f` | Verification job lives with the IP-core in `pccx-v002`. |
