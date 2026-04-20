`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import isa_pkg::*;
import bf16_math_pkg::*;

// ===| CVO Top |=================================================================
// Wraps CVO_sfu_unit (EXP/SQRT/GELU/RECIP/SCALE/REDUCE_SUM) and
// CVO_cordic_unit (SIN/COS) behind a unified streaming interface.
//
// Data flow:
//   Host issues OP_CVO via AXI-Lite → Global_Scheduler produces cvo_control_uop_t
//   → CVO_top latches uop, processes IN_length BF16 elements from L2 stream,
//     writes results back via output stream.
//
// FLAG_SUB_EMAX: subtract IN_e_max from each input before the function.
//   Implements exp(x - e_max) for numerically stable softmax.
// FLAG_ACCM: accumulate output into dst (add OUT_result to prior value).
//   Handled externally by the mem subsystem; CVO_top only signals it via OUT_accm.
//
// FSM states:
//   IDLE    : waiting for valid uop
//   RUNNING : streaming IN_length elements through the chosen unit
//   DONE    : pulse OUT_done for one cycle, return to IDLE
// ===============================================================================

module CVO_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        i_clear,

    // ===| Dispatch from Global_Scheduler |=====================================
    input  cvo_control_uop_t IN_uop,
    input  logic             IN_uop_valid,
    output logic             OUT_uop_ready,

    // ===| BF16 Input Stream (from L2 via mem_dispatcher) |=====================
    input  logic [15:0]  IN_data,
    input  logic         IN_data_valid,
    output logic         OUT_data_ready,

    // ===| BF16 Output Stream (to L2 via mem_dispatcher) |=====================
    output logic [15:0]  OUT_result,
    output logic         OUT_result_valid,
    input  logic         IN_result_ready,

    // ===| e_max for FLAG_SUB_EMAX |============================================
    // Passed in as BF16; CVO subtracts this from each element before the function.
    input  logic [15:0]  IN_e_max,

    // ===| Status |=============================================================
    output logic         OUT_busy,
    output logic         OUT_done,
    output logic         OUT_accm   // mirrors IN_uop.flags.accm to mem subsystem
);

  // ===| FSM |===================================================================
  typedef enum logic [1:0] {
    ST_IDLE    = 2'b00,
    ST_RUNNING = 2'b01,
    ST_DONE    = 2'b10
  } cvo_state_e;

  cvo_state_e state;

  // ===| Latched UOP |===========================================================
  cvo_func_e   uop_func;
  cvo_flags_t  uop_flags;
  logic [15:0] uop_length;
  logic [15:0] elem_count;   // elements processed in current operation

  // ===| BF16 subtract e_max (combinational) |===================================
  // Implements x - e_max in BF16 via bf16_add(x, -e_max).

  logic [15:0] sub_emax_result_wire;

  always_comb begin : comb_sub_emax
    // Negate e_max by flipping sign bit, then add to x
    sub_emax_result_wire = bf16_add(IN_data, {~IN_e_max[15], IN_e_max[14:0]});
  end

  // ===| Input to sub-units (after optional e_max subtraction) |=================
  logic [15:0] data_to_unit_wire;
  logic        data_valid_to_unit_wire;

  always_comb begin
    data_to_unit_wire    = uop_flags.sub_emax ? sub_emax_result_wire : IN_data;
    data_valid_to_unit_wire = (state == ST_RUNNING) && IN_data_valid;
  end

  // ===| Opcode Routing (declared ahead of units that use it as a gating term) |
  logic is_cordic_op_wire;
  always_comb begin
    is_cordic_op_wire = (uop_func == CVO_SIN) || (uop_func == CVO_COS);
  end

  // ===| SFU Instantiation |=====================================================
  logic [15:0] sfu_result;
  logic        sfu_result_valid;
  logic        sfu_ready;

  CVO_sfu_unit u_CVO_sfu_unit (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_clear         (i_clear),

      .IN_func         (uop_func),
      .IN_length       (uop_length),
      .IN_flags        (uop_flags),

      .IN_data         (data_to_unit_wire),
      .IN_valid        (data_valid_to_unit_wire && !is_cordic_op_wire),
      .OUT_data_ready  (sfu_ready),

      .OUT_result      (sfu_result),
      .OUT_result_valid(sfu_result_valid)
  );

  // ===| CORDIC Instantiation |==================================================
  logic [15:0] cordic_sin;
  logic [15:0] cordic_cos;
  logic        cordic_valid;

  CVO_cordic_unit u_CVO_cordic_unit (
      .clk          (clk),
      .rst_n        (rst_n),

      .IN_angle_bf16(data_to_unit_wire),
      .IN_valid     (data_valid_to_unit_wire && is_cordic_op_wire),

      .OUT_sin_bf16 (cordic_sin),
      .OUT_cos_bf16 (cordic_cos),
      .OUT_valid    (cordic_valid)
  );

  // ===| FSM Logic |=============================================================
  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      state      <= ST_IDLE;
      uop_func   <= CVO_EXP;
      uop_flags  <= '0;
      uop_length <= 16'd0;
      elem_count <= 16'd0;
      OUT_done   <= 1'b0;
    end else begin
      OUT_done <= 1'b0;

      case (state)
        // ===| IDLE: wait for dispatch |===
        ST_IDLE: begin
          if (IN_uop_valid) begin
            uop_func   <= IN_uop.cvo_func;
            uop_flags  <= IN_uop.flags;
            uop_length <= IN_uop.length;
            elem_count <= 16'd0;
            state      <= ST_RUNNING;
          end
        end

        // ===| RUNNING: count consumed elements |===
        ST_RUNNING: begin
          if (IN_data_valid && OUT_data_ready) begin
            elem_count <= elem_count + 16'd1;
            if (elem_count == uop_length - 16'd1) begin
              state    <= ST_DONE;
            end
          end
        end

        // ===| DONE: pulse and return |===
        ST_DONE: begin
          OUT_done <= 1'b1;
          state    <= ST_IDLE;
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

  // ===| Output Mux |============================================================
  // CORDIC outputs two results per input; select sin or cos based on function.
  logic [15:0] result_mux_wire;
  logic        result_valid_mux_wire;

  always_comb begin
    if (is_cordic_op_wire) begin
      result_mux_wire       = (uop_func == CVO_SIN) ? cordic_sin : cordic_cos;
      result_valid_mux_wire = cordic_valid;
    end else begin
      result_mux_wire       = sfu_result;
      result_valid_mux_wire = sfu_result_valid;
    end
  end

  // ===| Output Registers |======================================================
  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      OUT_result       <= 16'd0;
      OUT_result_valid <= 1'b0;
    end else begin
      OUT_result       <= result_mux_wire;
      OUT_result_valid <= result_valid_mux_wire && IN_result_ready;
    end
  end

  // ===| Status & Control |======================================================
  assign OUT_busy      = (state != ST_IDLE);
  assign OUT_uop_ready = (state == ST_IDLE);
  assign OUT_data_ready = sfu_ready && (state == ST_RUNNING);
  assign OUT_accm      = uop_flags.accm;

endmodule
