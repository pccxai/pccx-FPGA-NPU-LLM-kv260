`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;


module mem_u_operation_queue #(
) (
    input logic clk_core,
    input logic rst_n_core,

    input logic IN_acp_rdy,
    input acp_uop_t IN_acp_cmd,

    output acp_uop_t OUT_acp_cmd,
    output logic OUT_acp_cmd_valid,
    output logic OUT_acp_cmd_fifo_full,

    input logic IN_acp_is_busy,


    input logic IN_npu_rdy,
    input npu_uop_t IN_npu_cmd,

    output npu_uop_t OUT_npu_cmd,
    output logic OUT_npu_cmd_valid,
    output logic OUT_npu_cmd_fifo_full,

    input logic IN_npu_is_busy
);

  logic acp_fifo_empty;
  logic acp_fifo_full;

  logic npu_fifo_empty;
  logic npu_fifo_full;

  assign OUT_acp_cmd_fifo_full = acp_fifo_full;
  assign OUT_npu_cmd_fifo_full = npu_fifo_full;

  always_comb begin
    OUT_acp_cmd_valid = ~IN_acp_is_busy & ~acp_fifo_empty;
    OUT_npu_cmd_valid = ~IN_npu_is_busy & ~npu_fifo_empty;
  end

  // ACP port instruction
  xpm_fifo_sync #(
      .FIFO_DEPTH(128),
      .WRITE_DATA_WIDTH(33),
      .READ_DATA_WIDTH(33),
      .FIFO_MEMORY_TYPE("block"),
      .READ_MODE("std"),

      .FULL_RESET_VALUE(0),
      .PROG_FULL_THRESH(100)
  ) acp_port_uops_mem_x64_t (
      .sleep(1'b0),
      .rst(~rst_n_core),
      .wr_clk(clk_core),

      //write port
      .wr_en(IN_acp_rdy & ~acp_fifo_full),
      .din(IN_acp_cmd),
      .prog_full(acp_fifo_full),

      //read port
      .rd_en(~IN_acp_is_busy & ~acp_fifo_empty),
      .dout (OUT_acp_cmd),
      .empty(acp_fifo_empty)
  );


  // internal port instruction
  xpm_fifo_sync #(
      .FIFO_DEPTH(128),
      .WRITE_DATA_WIDTH(33),
      .READ_DATA_WIDTH(33),
      .FIFO_MEMORY_TYPE("block"),
      .READ_MODE("std"),

      .FULL_RESET_VALUE(0),
      .PROG_FULL_THRESH(100)
  ) internal_port_uops_mem_x64_t (
      .sleep(1'b0),
      .rst  (rst_n_core),
      .clk  (clk_core),

      //write port
      .wr_en(IN_npu_rdy & ~npu_fifo_full),
      .din(IN_npu_cmd),
      .prog_full(npu_fifo_full),

      //read port
      .rd_en(~IN_npu_is_busy & ~npu_fifo_empty),
      .dout (OUT_npu_cmd),
      .empty(npu_fifo_empty)
  );



endmodule
