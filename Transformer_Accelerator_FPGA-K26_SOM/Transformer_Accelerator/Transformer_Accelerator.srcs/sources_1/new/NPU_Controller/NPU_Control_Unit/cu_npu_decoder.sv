`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module cu_npu_decoder (
    input logic clk,
    input logic rst_n,
    input logic [`ISA_WIDTH-1:0] IN_raw_instruction,
    input logic raw_instruction_pop_valid,

    output logic OUT_fetch_PC_ready,


    //output instruction_t OUT_inst,

    output logic OUT_memcpy_op_x64_valid;
    output memcpy_op_x64_t OUT_memcpy_op_x64;

    output logic OUT_vdotm_op_x64_valid;
    output vdotm_op_x64_t  OUT_vdotm_op_x64;

    output logic OUT_mdotm_op_x64_valid;
    output mdotm_op_x64_t  OUT_mdotm_op_x64;
);

  logic [2:0] OUT_valid;
  assign OUT_vdotm_op_x64_valid = OUT_valid[0];
  assign OUT_mdotm_op_x64_valid = OUT_valid[1];
  assign OUT_memcpy_op_x64_valid = OUT_valid[2];

  VLIW_instruction_x64 instruction_VLIW_x64;


  always_ff @(posedge clk) begin
    if(!rst_n) begin
      OUT_valid <= 3'b000;
      OUT_fetch_PC_ready <= `TRUE;
    end else begin
      if(raw_instruction_pop_valid) begin

        OUT_memcpy_VALID <= raw_instruction_pop_valid;

        case (o_inst.opcode)
            OP_VDOTM: begin
              OUT_vdotm_cmd_x64  <= vdotm_op_x64_t'(IN_raw_instruction[3 +:59]);
              OUT_valid <= 3'b001;
            end
            OP_MDOTM: begin
              OUT_mdotm_cmd_x64 <= mdotm_op_x64_t'(IN_raw_instruction[3 +:59]);
              OUT_valid <= 3'b010;
            end
            OP_MEMCPY: begin
              OUT_memcpy_cmd_x64 <= memcpy_op_x64_t'(IN_raw_instruction[3 +:59]);
              OUT_valid <= 3'b100;
            end
            default: o_valid <= 1'b0;  // unknown opcode
        endcase
      end else begin

        OUT_valid <= 3'b000;

      end
    end
  end

endmodule

