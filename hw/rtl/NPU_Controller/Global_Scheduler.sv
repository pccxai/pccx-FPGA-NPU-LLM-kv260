`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===| Global Scheduler |========================================================
// Translates decoded VLIW instructions into engine micro-ops.
//
// Single always_ff drives each output to avoid multiple-driver conflicts.
// Priority for OUT_LOAD_uop: GEMM > GEMV > MEMCPY > CVO (one active per cycle).
//
// OUT_STORE_uop  : registered at issue time; mem_dispatcher uses it to initiate
//                  result writeback after the engine signals completion.
// OUT_sram_rd_start : one-cycle pulse when a GEMM or GEMV load is dispatched,
//                     triggering preprocess_fmap to begin broadcasting from cache.
// ===============================================================================

module Global_Scheduler #() (
    input logic clk_core,
    input logic rst_n_core,

    // ===| From ctrl_npu_decoder |===============================================
    input logic IN_GEMV_op_x64_valid,
    input logic IN_GEMM_op_x64_valid,
    input logic IN_memcpy_op_x64_valid,
    input logic IN_memset_op_x64_valid,
    input logic IN_cvo_op_x64_valid,

    input instruction_op_x64_t instruction,

    // ===| Engine micro-ops |====================================================
    output gemm_control_uop_t   OUT_GEMM_uop,
    output GEMV_control_uop_t   OUT_GEMV_uop,
    output memory_control_uop_t OUT_LOAD_uop,
    output memory_control_uop_t OUT_STORE_uop,
    output memory_set_uop_t     OUT_mem_set_uop,
    output cvo_control_uop_t    OUT_CVO_uop,

    // ===| Datapath control |====================================================
    output logic OUT_sram_rd_start   // pulse: start fmap cache broadcast
);

  // ===| Combinational instruction body casts |==================================
  GEMV_op_x64_t   GEMV_op_x64;
  GEMM_op_x64_t   GEMM_op_x64;
  memcpy_op_x64_t memcpy_op_x64;
  memset_op_x64_t memset_op_x64;
  cvo_op_x64_t    cvo_op_x64;

  always_comb begin
    GEMV_op_x64   = GEMV_op_x64_t'(instruction.instruction);
    GEMM_op_x64   = GEMM_op_x64_t'(instruction.instruction);
    memcpy_op_x64 = memcpy_op_x64_t'(instruction.instruction);
    memset_op_x64 = memset_op_x64_t'(instruction.instruction);
    cvo_op_x64    = cvo_op_x64_t'(instruction.instruction);
  end

  // ===| MEMSET uop |============================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_mem_set_uop <= '0;
    end else if (IN_memset_op_x64_valid) begin
      OUT_mem_set_uop <= '{
          dest_cache : dest_cache_e'(memset_op_x64.dest_cache),
          dest_addr  : memset_op_x64.dest_addr,
          a_value    : memset_op_x64.a_value,
          b_value    : memset_op_x64.b_value,
          c_value    : memset_op_x64.c_value
      };
    end
  end

  // ===| MEMCPY route translation ===============================================
  // from_device/to_device (1-bit each) → data_route_e (8-bit enum)
  data_route_e memcpy_route;
  always_comb begin
    if (memcpy_op_x64.from_device == FROM_HOST && memcpy_op_x64.to_device == TO_NPU)
      memcpy_route = from_host_to_L2;
    else
      memcpy_route = from_L2_to_host;
  end

  // ===| LOAD uop — single driver (priority: GEMM > GEMV > MEMCPY > CVO) |======
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_LOAD_uop      <= '0;
      OUT_sram_rd_start <= 1'b0;
    end else begin
      OUT_sram_rd_start <= 1'b0;   // default: no pulse

      if (IN_GEMM_op_x64_valid) begin
        OUT_LOAD_uop <= '{
            data_dest      : from_L2_to_L1_GEMM,
            dest_addr      : '0,
            src_addr       : GEMM_op_x64.src_addr,
            shape_ptr_addr : GEMM_op_x64.shape_ptr_addr,
            async          : SYNC_OP
        };
        OUT_sram_rd_start <= 1'b1;

      end else if (IN_GEMV_op_x64_valid) begin
        OUT_LOAD_uop <= '{
            data_dest      : from_L2_to_L1_GEMV,
            dest_addr      : '0,
            src_addr       : GEMV_op_x64.src_addr,
            shape_ptr_addr : GEMV_op_x64.shape_ptr_addr,
            async          : SYNC_OP
        };
        OUT_sram_rd_start <= 1'b1;

      end else if (IN_memcpy_op_x64_valid) begin
        OUT_LOAD_uop <= '{
            data_dest      : memcpy_route,
            dest_addr      : memcpy_op_x64.dest_addr,
            src_addr       : memcpy_op_x64.src_addr,
            shape_ptr_addr : memcpy_op_x64.shape_ptr_addr,
            async          : memcpy_op_x64.async
        };

      end else if (IN_cvo_op_x64_valid) begin
        OUT_LOAD_uop <= '{
            data_dest      : from_L2_to_CVO,
            dest_addr      : '0,
            src_addr       : cvo_op_x64.src_addr,
            shape_ptr_addr : '0,
            async          : cvo_op_x64.async
        };
      end
    end
  end

  // ===| STORE uop — latched at issue time |=====================================
  // Held until the engine signals completion (external handshake, not shown here).
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_STORE_uop <= '0;
    end else if (IN_GEMM_op_x64_valid) begin
      OUT_STORE_uop <= '{
          data_dest      : from_GEMM_res_to_L2,
          dest_addr      : GEMM_op_x64.dest_reg,
          src_addr       : '0,
          shape_ptr_addr : GEMM_op_x64.shape_ptr_addr,
          async          : SYNC_OP
      };
    end else if (IN_GEMV_op_x64_valid) begin
      OUT_STORE_uop <= '{
          data_dest      : from_GEMV_res_to_L2,
          dest_addr      : GEMV_op_x64.dest_reg,
          src_addr       : '0,
          shape_ptr_addr : GEMV_op_x64.shape_ptr_addr,
          async          : SYNC_OP
      };
    end else if (IN_cvo_op_x64_valid) begin
      OUT_STORE_uop <= '{
          data_dest      : from_CVO_res_to_L2,
          dest_addr      : cvo_op_x64.dst_addr,
          src_addr       : '0,
          shape_ptr_addr : '0,
          async          : cvo_op_x64.async
      };
    end
  end

  // ===| GEMM uop |==============================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_GEMM_uop <= '0;
    end else if (IN_GEMM_op_x64_valid) begin
      OUT_GEMM_uop <= '{
          flags         : GEMM_op_x64.flags,
          size_ptr_addr : GEMM_op_x64.size_ptr_addr,
          parallel_lane : GEMM_op_x64.parallel_lane
      };
    end
  end

  // ===| GEMV uop |==============================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_GEMV_uop <= '0;
    end else if (IN_GEMV_op_x64_valid) begin
      OUT_GEMV_uop <= '{
          flags         : GEMV_op_x64.flags,
          size_ptr_addr : GEMV_op_x64.size_ptr_addr,
          parallel_lane : GEMV_op_x64.parallel_lane
      };
    end
  end

  // ===| CVO uop |===============================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_CVO_uop <= '0;
    end else if (IN_cvo_op_x64_valid) begin
      OUT_CVO_uop <= '{
          cvo_func : cvo_func_e'(cvo_op_x64.cvo_func),
          src_addr : cvo_op_x64.src_addr,
          dst_addr : cvo_op_x64.dst_addr,
          length   : cvo_op_x64.length,
          flags    : cvo_flags_t'(cvo_op_x64.flags),
          async    : cvo_op_x64.async
      };
    end
  end

endmodule
