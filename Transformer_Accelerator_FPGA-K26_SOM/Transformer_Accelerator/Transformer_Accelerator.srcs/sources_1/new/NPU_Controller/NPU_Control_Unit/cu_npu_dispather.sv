`timeOUT_scale 1ns / 1ps
`include "stlc_Array.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module cu_npu_dispatcher (
    input  logic         clk,
    input  logic         rst_n,
    input  instruction_t IN_inst,
    input  logic         IN_valid,
    output logic         o_valid,

    // VdotM / MdotM controls
    output logic [3:0] OUT_activate_top,
    output logic [3:0] OUT_activate_lane,
    output logic       OUT_emax_align,
    output logic       OUT_accm,
    output logic       OUT_scale,

    // memcpy
    output logic [`MAX_MATRIX_WIDTH-1:0] OUT_matrix_shape     [0:`MAX_MATRIX_DIM-1],
    output logic [                  3:0] OUT_destination_queue
);

  /*─────────────────────────────────────────────
  Lane activation bitmask
  bit[0]=lane1, bit[1]=lane2 ...
  ─────────────────────────────────────────────*/
  localparam logic [3:0] LANE_1 = 4'b0001;
  localparam logic [3:0] LANE_2 = 4'b0010;
  localparam logic [3:0] LANE_3 = 4'b0100;
  localparam logic [3:0] LANE_4 = 4'b1000;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      o_valid           <= 1'b0;
      OUT_activate_lane     <= '0;
      OUT_emax_align        <= 1'b0;
      OUT_accm              <= 1'b0;
      OUT_scale             <= 1'b0;
      OUT_destination_queue <= '0;
      for (int i = 0; i < `MAX_MATRIX_DIM; i++) OUT_matrix_shape[i] <= '0;
    end else begin

      o_valid <= 1'b0;  // default : deassert every cycle

      if (IN_valid) begin
        case (IN_inst.opcode)

          OP_VDOTM: begin
            o_valid <= 1'b1;

            if (IN_inst.cmd_chaining) begin
              // TODO: chaining logic
            end
            if (IN_inst.override) begin
              // TODO: override logic
            end

            // lane activation (OR mask, cumulative)
            case (IN_inst.payload.dotm.lane_idq)
              2'b00: OUT_activate_lane <= LANE_1;
              2'b01: OUT_activate_lane <= LANE_1 | LANE_2;
              2'b10: OUT_activate_lane <= LANE_1 | LANE_2 | LANE_3;
              2'b11: OUT_activate_lane <= LANE_1 | LANE_2 | LANE_3 | LANE_4;
              default: begin
                o_valid <= 1'b0;  // unknown → drop + TODO: interrupt
              end
            endcase

            OUT_emax_align <= IN_inst.payload.dotm.align;
            OUT_accm       <= IN_inst.payload.dotm.OUT_accm;
            // activate when added to ISA
            // OUT_scale      <= IN_inst.payload.dotm.OUT_scale;
            OUT_activate_top[`TOP_VDOTM] <= `TRUE;
          end

          OP_MDOTM: begin
            if (IN_inst.override) begin
              if (IN_inst.cmd_chaining) begin
                // TODO
              end else begin
                // TODO
              end
            end else begin
              if (IN_inst.cmd_chaining) begin
                // TODO
              end else begin
                o_valid <= 1'b1;

                OUT_emax_align <= IN_inst.payload.dotm.align;
                OUT_accm       <= IN_inst.payload.dotm.OUT_accm;
                OUT_activate_top[`TOP_VDOTM] <= `TRUE;
              end
            end

          end

          OP_MEMCPY: begin
            if (IN_inst.override) begin
              if (IN_inst.cmd_chaining) begin
                // accumulate matrix shape across chained instructions
                OUT_matrix_shape[IN_inst.payload.memcpy.dim_xyz]
                    <= IN_inst.payload.memcpy.dim_x;  // TODO: dim_N 필드 확정 후 수정
              end else begin
                // chaining end → dispatch memcpy
                o_valid           <= 1'b1;
                OUT_destination_queue <= IN_inst.payload.memcpy.dest_queue;

                case (IN_inst.payload.memcpy.dest_queue[3:2])
                  `MASKING_WEIGHT: begin
                    // TODO: → weight buffer
                  end
                  `MASKING_OUT_scale: begin
                    // TODO: ACP → OUT_scale cache
                  end
                  `MASKING_FMAP: begin
                    // TODO: ACP → find emax & align → cache
                  end
                  default: o_valid <= 1'b0;  // undefined
                endcase
              end
            end else begin
              // non-override memcpy
              // TODO
            end
          end

          default: o_valid <= 1'b0;  // unknown opcode → drop

        endcase
      end
    end
  end

endmodule
