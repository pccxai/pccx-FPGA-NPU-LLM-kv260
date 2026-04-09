`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module Global_Scheduler #(
) (
    input logic clk_core,
    input logic rst_n_core,

    input logic IN_memcpy_op_x64_valid,
    input memory_op_x64_t memcpy_op_x64,

    input logic IN_vdotm_op_x64_valid,
    input vdotm_op_x64_t vdotm_op_x64,

    input logic IN_mdotm_op_x64_valid,
    input mdotm_op_x64_t mdotm_op_x64,

    output memory_control_uop_t OUT_mem_uop,

    output stlc_control_uop_t OUT_stlc_uop,

    output vdotm_control_uop_t OUT_vdotm_uop
);

  memory_control_uop_t mem_uop;
  stlc_control_uop_t   stlc_uop;
  vdotm_control_uop_t  vdotm_uop;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
    end else begin
      if (IN_memcpy_op_x64_valid) begin
        mem_uop <= '{
            data_dest      : {memcpy_op_x64.from_device, memcpy_op_x64.to_device},
            dest_addr      : memcpy_op_x64.dest_addr,
            src_addr       : memcpy_op_x64.src_addr,
            shape_ptr_addr : memcpy_op_x64.shape_ptr_addr,
            async          : memcpy_op_x64.async
        };
      end
    end
  end


  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
    end else begin
      if (IN_vdotm_op_x64_valid) begin
        mem_uop <= '{
            data_dest      : {memcpy_op_x64.from_device, memcpy_op_x64.to_device},,
            dest_addr      : memcpy_op_x64.dest_addr,
            src_addr       : memcpy_op_x64.src_addr,
            shape_ptr_addr : memcpy_op_x64.shape_ptr_addr,
            async          : memcpy_op_x64.async
        };

        stlc_uop <= '{
            flags_t         : vdotm_op_x64.flags,
            ptr_addr_t      : vdotm_op_x64.size_ptr_addr,
            parallel_lane_t : vdotm_op_x64.parallel_lane
        };
      end
    end
  end

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
    end else begin
      if (IN_mdotm_op_x64_valid) begin
        mem_uop <= '{
            data_dest      : {memcpy_op_x64.from_device, memcpy_op_x64.to_device},,
            dest_addr      : memcpy_op_x64.dest_addr,
            src_addr       : memcpy_op_x64.src_addr,
            shape_ptr_addr : memcpy_op_x64.shape_ptr_addr,
            async          : memcpy_op_x64.async
        };

        stlc_uop <= '{
            flags_t         : mdotm_op_x64.flags,
            ptr_addr_t      : mdotm_op_x64.size_ptr_addr,
            parallel_lane_t : mdotm_op_x64.parallel_lane
        };
      end
    end
  end



endmodule
