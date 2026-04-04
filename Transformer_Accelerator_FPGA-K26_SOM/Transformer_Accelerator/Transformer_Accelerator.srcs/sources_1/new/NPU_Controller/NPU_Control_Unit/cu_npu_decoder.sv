`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module cu_npu_decoder (
    input logic [`ISA_WIDTH-1:0] IN_raw_instruction,  // Instruction_reg
    input logic raw_instruction_pop_valid,

    output logic OUT_fetch_PC_ready,

    output instruction_t OUT_inst,
    output logic         OUT_valid
);

  // 캐스팅 한 줄로 해독 끝
  assign o_inst = instruction_t'(i_raw);

  always_comb begin
    o_valid = 1'b1;
    case (o_inst.opcode)
      OP_VDOTM: begin

      end
      OP_MDOTM: begin
        // o_inst.payload.dotm.dest 이런식으로 접근
      end
      OP_MEMCPY: begin
        // o_inst.payload.memcpy.dim_x 이런식으로
      end
      default: o_valid = 1'b0;  // unknown opcode
    endcase
  end  // opcode별 추가 처리


endmodule

