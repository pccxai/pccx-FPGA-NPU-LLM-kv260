// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import isa_pkg::*;
import bf16_math_pkg::*;

// ===| Module: CVO_top — scalar function unit (SFU + CORDIC) wrapper |==========
// Purpose      : Run BF16 element-wise non-linear ops (EXP/SQRT/GELU/RECIP/
//                SCALE/REDUCE_SUM) and trig ops (SIN/COS) behind a unified
//                streaming interface; provides numerically stable softmax via
//                FLAG_SUB_EMAX (x - e_max).
// Spec ref     : pccx v002 §2.4 (CVO core), §3.5 (CVO uop), §5.3 (bridge).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low; i_clear synchronous soft-clear.
// Topology     : Op routing decided by uop_func at IDLE→RUNNING transition:
//                  CVO_SIN/CVO_COS → CVO_cordic_unit
//                  others          → CVO_sfu_unit
// Sub-units    : CVO_sfu_unit (BF16 scalar pipeline), CVO_cordic_unit (rotation
//                CORDIC for sin/cos pair).
// Data flow    : Host issues OP_CVO via AXI-Lite → Global_Scheduler produces
//                cvo_control_uop_t → CVO_top latches uop, processes IN_length
//                BF16 elements from the L2 stream, writes results back via
//                the output stream.
// FLAG_SUB_EMAX: Subtract IN_e_max from each input before the function.
//                Implements exp(x − e_max) for numerically stable softmax.
// FLAG_ACCM    : Accumulate output into dst (add OUT_result to prior value).
//                Handled externally by the mem subsystem; CVO_top only
//                surfaces it via OUT_accm.
// FSM states   : IDLE → RUNNING → DONE → IDLE.
// Latency      : Sub-unit latency (see CVO_sfu_unit / CVO_cordic_unit
//                contracts) + 1 output register cycle.
// Throughput   : 1 result per cycle in steady state for non-CORDIC ops.
// Backpressure : OUT_data_ready ANDs sub-unit ready with (state == RUNNING).
// Reset state  : ST_IDLE, OUT_result = 0, OUT_result_valid = 0, OUT_done = 0.
// Errors       : Unknown uop_func falls back to ST_IDLE on default arm.
// Counters     : none.
// Assertions   : (Stage C) IN_data_valid && OUT_data_ready only during RUNNING;
//                OUT_done is one-cycle pulse only.
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

  // ===| Registered Unit Input Boundary |========================================
  // The L2 bridge deserializes 128-bit words into a 16-bit BF16 stream. Keep that
  // muxing on the bridge side of a flop so the SFU/CORDIC arithmetic paths start
  // from a local CVO register rather than from the L2 deser buffer.

  logic        unit_in_valid;
  logic [15:0] unit_in_data;
  logic [15:0] unit_in_e_max;
  cvo_func_e   unit_in_func;
  cvo_flags_t  unit_in_flags;
  logic [15:0] unit_in_length;
  logic        input_accept_wire;

  assign input_accept_wire = (state == ST_RUNNING) && IN_data_valid && OUT_data_ready;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      unit_in_valid  <= 1'b0;
      unit_in_data   <= 16'd0;
      unit_in_e_max  <= 16'd0;
      unit_in_func   <= CVO_EXP;
      unit_in_flags  <= '0;
      unit_in_length <= 16'd0;
    end else begin
      unit_in_valid <= input_accept_wire;
      if (input_accept_wire) begin
        unit_in_data   <= IN_data;
        unit_in_e_max  <= IN_e_max;
        unit_in_func   <= uop_func;
        unit_in_flags  <= uop_flags;
        unit_in_length <= uop_length;
      end
    end
  end

  // ===| Pipelined BF16 subtract e_max |=========================================
  // Implements x - e_max for FLAG_SUB_EMAX. This is intentionally split across
  // stages because a full BF16 align/add/normalize chain does not meet the
  // 400 MHz core target as one combinational block.

  function automatic logic [15:0] pack_bf16_mag(input logic        out_sign,
                                                input logic [7:0]  emax,
                                                input logic [23:0] mag);
    int          lead;
    logic [7:0]  out_exp;
    logic [6:0]  out_mant;

    if (mag == 24'd0) return 16'd0;

    lead = 23;
    while (lead > 0 && mag[lead] == 1'b0) lead = lead - 1;

    out_exp = emax + 8'(lead - 15);
    if (lead >= 7)
      out_mant = mag[lead-1-:7];
    else
      out_mant = 7'(mag << (7 - lead));

    return {out_sign, out_exp, out_mant};
  endfunction

  bf16_t      sub_s0_a_wire;
  bf16_t      sub_s0_b_wire;
  logic [7:0] sub_s0_emax_wire;

  always_comb begin : comb_sub_emax_classify
    sub_s0_a_wire    = to_bf16(unit_in_data);
    sub_s0_b_wire    = to_bf16({~unit_in_e_max[15], unit_in_e_max[14:0]});
    sub_s0_emax_wire = (sub_s0_a_wire.exp > sub_s0_b_wire.exp) ?
                       sub_s0_a_wire.exp : sub_s0_b_wire.exp;
  end

  logic        sub_s0_valid;
  logic        sub_s0_do_sub;
  logic [15:0] sub_s0_passthrough;
  logic [7:0]  sub_s0_emax;
  bf16_t       sub_s0_a;
  bf16_t       sub_s0_b;
  cvo_func_e   sub_s0_func;
  cvo_flags_t  sub_s0_flags;
  logic [15:0] sub_s0_length;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      sub_s0_valid       <= 1'b0;
      sub_s0_do_sub      <= 1'b0;
      sub_s0_passthrough <= 16'd0;
      sub_s0_emax        <= 8'd0;
      sub_s0_a           <= '0;
      sub_s0_b           <= '0;
      sub_s0_func        <= CVO_EXP;
      sub_s0_flags       <= '0;
      sub_s0_length      <= 16'd0;
    end else begin
      sub_s0_valid <= unit_in_valid;
      if (unit_in_valid) begin
        sub_s0_do_sub      <= unit_in_flags.sub_emax;
        sub_s0_passthrough <= unit_in_data;
        sub_s0_emax        <= sub_s0_emax_wire;
        sub_s0_a           <= sub_s0_a_wire;
        sub_s0_b           <= sub_s0_b_wire;
        sub_s0_func        <= unit_in_func;
        sub_s0_flags       <= unit_in_flags;
        sub_s0_length      <= unit_in_length;
      end
    end
  end

  logic [23:0] sub_aligned_a_wire;
  logic [23:0] sub_aligned_b_wire;

  always_comb begin : comb_sub_emax_align
    sub_aligned_a_wire = align_to_emax(sub_s0_a, sub_s0_emax);
    sub_aligned_b_wire = align_to_emax(sub_s0_b, sub_s0_emax);
  end

  logic        sub_s1_valid;
  logic        sub_s1_do_sub;
  logic [15:0] sub_s1_passthrough;
  logic [7:0]  sub_s1_emax;
  logic [23:0] sub_s1_aligned_a;
  logic [23:0] sub_s1_aligned_b;
  cvo_func_e   sub_s1_func;
  cvo_flags_t  sub_s1_flags;
  logic [15:0] sub_s1_length;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      sub_s1_valid       <= 1'b0;
      sub_s1_do_sub      <= 1'b0;
      sub_s1_passthrough <= 16'd0;
      sub_s1_emax        <= 8'd0;
      sub_s1_aligned_a   <= 24'd0;
      sub_s1_aligned_b   <= 24'd0;
      sub_s1_func        <= CVO_EXP;
      sub_s1_flags       <= '0;
      sub_s1_length      <= 16'd0;
    end else begin
      sub_s1_valid <= sub_s0_valid;
      if (sub_s0_valid) begin
        sub_s1_do_sub      <= sub_s0_do_sub;
        sub_s1_passthrough <= sub_s0_passthrough;
        sub_s1_emax        <= sub_s0_emax;
        sub_s1_aligned_a   <= sub_aligned_a_wire;
        sub_s1_aligned_b   <= sub_aligned_b_wire;
        sub_s1_func        <= sub_s0_func;
        sub_s1_flags       <= sub_s0_flags;
        sub_s1_length      <= sub_s0_length;
      end
    end
  end

  logic              sub_s2_valid;
  logic              sub_s2_do_sub;
  logic [15:0]       sub_s2_passthrough;
  logic [7:0]        sub_s2_emax;
  logic signed [24:0] sub_s2_sum;
  cvo_func_e         sub_s2_func;
  cvo_flags_t        sub_s2_flags;
  logic [15:0]       sub_s2_length;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      sub_s2_valid       <= 1'b0;
      sub_s2_do_sub      <= 1'b0;
      sub_s2_passthrough <= 16'd0;
      sub_s2_emax        <= 8'd0;
      sub_s2_sum         <= 25'sd0;
      sub_s2_func        <= CVO_EXP;
      sub_s2_flags       <= '0;
      sub_s2_length      <= 16'd0;
    end else begin
      sub_s2_valid <= sub_s1_valid;
      if (sub_s1_valid) begin
        sub_s2_do_sub      <= sub_s1_do_sub;
        sub_s2_passthrough <= sub_s1_passthrough;
        sub_s2_emax        <= sub_s1_emax;
        sub_s2_sum         <= $signed({sub_s1_aligned_a[23], sub_s1_aligned_a}) +
                              $signed({sub_s1_aligned_b[23], sub_s1_aligned_b});
        sub_s2_func        <= sub_s1_func;
        sub_s2_flags       <= sub_s1_flags;
        sub_s2_length      <= sub_s1_length;
      end
	    end
	  end

  logic        sub_s3_valid;
  logic        sub_s3_do_sub;
  logic [15:0] sub_s3_passthrough;
  logic [7:0]  sub_s3_emax;
  logic        sub_s3_sign;
  logic [23:0] sub_s3_mag;
  cvo_func_e   sub_s3_func;
  cvo_flags_t  sub_s3_flags;
  logic [15:0] sub_s3_length;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      sub_s3_valid       <= 1'b0;
      sub_s3_do_sub      <= 1'b0;
      sub_s3_passthrough <= 16'd0;
      sub_s3_emax        <= 8'd0;
      sub_s3_sign        <= 1'b0;
      sub_s3_mag         <= 24'd0;
      sub_s3_func        <= CVO_EXP;
      sub_s3_flags       <= '0;
      sub_s3_length      <= 16'd0;
    end else begin
      sub_s3_valid <= sub_s2_valid;
      if (sub_s2_valid) begin
        sub_s3_do_sub      <= sub_s2_do_sub;
        sub_s3_passthrough <= sub_s2_passthrough;
        sub_s3_emax        <= sub_s2_emax;
        sub_s3_sign        <= sub_s2_sum[24];
        sub_s3_mag         <= sub_s2_sum[24] ? (~sub_s2_sum[23:0] + 24'd1) :
                                               sub_s2_sum[23:0];
        sub_s3_func        <= sub_s2_func;
        sub_s3_flags       <= sub_s2_flags;
        sub_s3_length      <= sub_s2_length;
      end
    end
  end

	  // ===| Input to sub-units (after optional e_max subtraction) |=================
	  logic        data_valid_to_unit_wire;
	  logic [15:0] unit_data;
	  logic        unit_valid;
  cvo_func_e   unit_func;
  cvo_flags_t  unit_flags;
  logic [15:0] unit_length;

  always_comb begin
    data_valid_to_unit_wire = unit_valid;
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      unit_valid  <= 1'b0;
      unit_data   <= 16'd0;
      unit_func   <= CVO_EXP;
      unit_flags  <= '0;
      unit_length <= 16'd0;
    end else begin
      unit_valid <= sub_s3_valid;
      if (sub_s3_valid) begin
        unit_data   <= sub_s3_do_sub ? pack_bf16_mag(sub_s3_sign, sub_s3_emax, sub_s3_mag) :
                                       sub_s3_passthrough;
        unit_func   <= sub_s3_func;
        unit_flags  <= sub_s3_flags;
        unit_length <= sub_s3_length;
      end
    end
  end

  // ===| Opcode Routing (declared ahead of units that use it as a gating term) |
  logic is_cordic_op_wire;
  always_comb begin
    is_cordic_op_wire = (unit_func == CVO_SIN) || (unit_func == CVO_COS);
  end

  // ===| SFU Instantiation |=====================================================
  logic [15:0] sfu_result;
  logic        sfu_result_valid;
  logic        sfu_ready;

  CVO_sfu_unit u_CVO_sfu_unit (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_clear         (i_clear),

      .IN_func         (unit_func),
      .IN_length       (unit_length),
      .IN_flags        (unit_flags),

      .IN_data         (unit_data),
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

      .IN_angle_bf16(unit_data),
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
