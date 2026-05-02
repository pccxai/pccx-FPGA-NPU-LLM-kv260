`timescale 1ns / 1ps

// ===| Module: GEMM_systolic_array — 32×32 W4A8 PE grid |========================
// Purpose      : Physical instantiation of the systolic PE grid; owns the
//                cascade break at row 16, the fabric activation delay line,
//                and the final-row accumulator strip.
// Spec ref     : pccx v002 §2.2.1 (PE topology), §2.2.4 (cascade break).
// Phase        : pccx v002 (W4A8, 1 DSP = 2 MAC).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low; i_clear soft-clear (propagated to every PE).
// Geometry     : array_horizontal × array_vertical PE grid (default 32×32).
// Topology
//   Row 0 : GEMM_dsp_unit with IS_TOP_ROW = 1. Activation sourced from
//           V_in[col] fabric input.
//   Row 16: GEMM_dsp_unit with BREAK_CASCADE = 1. Activation re-injected
//           from the fabric delay line, partial sum takes the C-port
//           instead of PCIN.
//   Row 31: GEMM_dsp_unit_last_ROW. Exposes 48-bit P as V_ACC_out[col]
//           (after the GEMM_accumulator strip).
//   Others: GEMM_dsp_unit normal row. Activation via BCIN cascade,
//           partial sum via PCIN cascade.
// Latency      : SYSTOLIC_TOTAL_LATENCY cycles (top of array → V_ACC_out).
// Throughput   : 32 dual-MAC ops per row per cycle once the array is filled.
// Backpressure : None — push-only PE grid; upstream is responsible for
//                pacing fmap and weight valids consistently.
// Reset state  : All cascade wires zeroed via per-PE reset.
// Counters     : none.
// Notes        : The fabric delay line gemm_in_V_fabric carries V_in down
//                the column for the row-16 break re-injection.
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module GEMM_systolic_array #(
  parameter array_horizontal = `ARRAY_SIZE_H,
  parameter array_vertical   = `ARRAY_SIZE_V,
  parameter INT4_BITS        = `INT4_WIDTH,
  parameter INT8_BITS        = 8,
  parameter B_PORT_W         = `DEVICE_DSP_B_WIDTH
) (
  input logic clk,
  input logic rst_n,
  input logic i_clear,

  // ===| Global controls |========================================================
  input logic i_weight_valid,

  // ===| Horizontal weight streams (two INT4 lanes, one pair per row) |==========
  input logic [INT4_BITS-1:0] H_in_upper [0:array_horizontal-1],
  input logic [INT4_BITS-1:0] H_in_lower [0:array_horizontal-1],

  // ===| Vertical INT8 activation (one per column) |=============================
  input logic [INT8_BITS-1:0] V_in       [0:array_vertical-1],

  // ===| Staggered activation valid + VLIW instruction (per column) |============
  input logic                 in_valid      [0:array_vertical-1],
  input logic [2:0]           inst_in       [0:array_horizontal-1],
  input logic                 inst_valid_in [0:array_vertical-1],

  // ===| Outputs (bottom row) |==================================================
  output logic [`DSP48E2_POUT_SIZE-1:0] V_out       [0:array_horizontal-1],
  output logic [`DSP48E2_POUT_SIZE-1:0] V_ACC_out   [0:array_horizontal-1],
  output logic                          V_ACC_valid [0:array_vertical-1]
);

  // ===| Horizontal weight wires (shift-register chain, one per lane) |==========
  logic [INT4_BITS-1:0] gemm_H_upper_wire [0:array_horizontal-1][0:array_vertical];
  logic [INT4_BITS-1:0] gemm_H_lower_wire [0:array_horizontal-1][0:array_vertical];

  // ===| Vertical activation cascade (DSP BCOUT -> BCIN chain) |=================
  logic [B_PORT_W-1:0] gemm_BCIN_wire [0:array_horizontal][0:array_vertical-1];

  // ===| Instruction / valid / partial-sum wires (top->bottom) |=================
  logic [2:0]                         gemm_inst_wire       [0:array_horizontal][0:array_vertical-1];
  logic                               gemm_inst_valid_wire [0:array_horizontal][0:array_vertical-1];
  logic                               gemm_V_valid_wire    [0:array_horizontal][0:array_vertical-1];
  logic [`DSP48E2_POUT_SIZE-1:0]      gemm_V_result_wire   [0:array_horizontal][0:array_vertical-1];

  // ===| Fabric break wires for row 15 -> 16 |===================================
  logic [`DSP48E2_POUT_SIZE-1:0]      gemm_P_fabric_wire   [0:array_horizontal-1][0:array_vertical-1];

  // ===| Fabric activation delay line (used by the break row) |=================
  logic [INT8_BITS-1:0] gemm_in_V_fabric [0:array_horizontal][0:array_vertical-1];

  // ===| Input assignments (top lane / column-0 lane) |==========================
  genvar i;
  generate
    for (i = 0; i < array_vertical; i++) begin : assign_v_inputs
      // Top-row cascades are unused (B_INPUT = DIRECT at row 0).
      assign gemm_BCIN_wire[0][i]        = '0;
      assign gemm_inst_wire[0][i]        = inst_in[i];
      assign gemm_inst_valid_wire[0][i]  = inst_valid_in[i];
      assign gemm_V_valid_wire[0][i]     = in_valid[i];
      assign gemm_V_result_wire[0][i]    = '0;
      assign gemm_in_V_fabric[0][i]      = V_in[i];
    end

    for (i = 0; i < array_horizontal; i++) begin : assign_h_inputs
      assign gemm_H_upper_wire[i][0] = H_in_upper[i];
      assign gemm_H_lower_wire[i][0] = H_in_lower[i];
    end
  endgenerate

  // ===| Fabric delay line for V_in (feeds the break row) |======================
  genvar d_row, d_col;
  generate
    for (d_row = 0; d_row < array_horizontal; d_row++) begin : v_delay_row
      for (d_col = 0; d_col < array_vertical; d_col++) begin : v_delay_col
        always_ff @(posedge clk) begin
          if (gemm_V_valid_wire[d_row][d_col]) begin
            gemm_in_V_fabric[d_row+1][d_col] <= gemm_in_V_fabric[d_row][d_col];
          end
        end
      end
    end
  endgenerate

  // ===| 2D array instantiation |================================================
  genvar row, col;
  generate
    for (row = 0; row < array_horizontal; row++) begin : gemm_row_loop
      for (col = 0; col < array_vertical; col++) begin : gemm_col_loop

        if (row == array_horizontal - 1) begin : last_row
          GEMM_dsp_unit_last_ROW #(
              .IS_TOP_ROW(0)
          ) dsp_unit_last_ROW (
              .clk(clk),
              .rst_n(rst_n),
              .i_clear(i_clear),

              .i_valid        (gemm_V_valid_wire[row][col]),
              .inst_valid_in_V(gemm_inst_valid_wire[row][col]),
              .i_weight_valid (i_weight_valid),
              .o_valid        (gemm_V_valid_wire[row+1][col]),

              .in_H_upper (gemm_H_upper_wire[row][col]),
              .out_H_upper(gemm_H_upper_wire[row][col+1]),
              .in_H_lower (gemm_H_lower_wire[row][col]),
              .out_H_lower(gemm_H_lower_wire[row][col+1]),

              .in_V      (gemm_in_V_fabric[row][col]),
              .BCIN_in   (gemm_BCIN_wire[row][col]),
              .BCOUT_out (gemm_BCIN_wire[row+1][col]),

              .instruction_in_V (gemm_inst_wire[row][col]),
              .instruction_out_V(gemm_inst_wire[row+1][col]),
              .inst_valid_out_V (gemm_inst_valid_wire[row+1][col]),

              .V_result_in (gemm_V_result_wire[row][col]),
              .V_result_out(gemm_V_result_wire[row+1][col]),

              .gemm_unit_results(V_out[col])
          );
        end else if (row == 16) begin : break_row
          GEMM_dsp_unit #(
              .IS_TOP_ROW   (0),
              .BREAK_CASCADE(1)
          ) dsp_unit_break (
              .clk(clk),
              .rst_n(rst_n),
              .i_clear(i_clear),

              .i_valid        (gemm_V_valid_wire[row][col]),
              .inst_valid_in_V(gemm_inst_valid_wire[row][col]),
              .i_weight_valid (i_weight_valid),
              .o_valid        (gemm_V_valid_wire[row+1][col]),

              .in_H_upper (gemm_H_upper_wire[row][col]),
              .out_H_upper(gemm_H_upper_wire[row][col+1]),
              .in_H_lower (gemm_H_lower_wire[row][col]),
              .out_H_lower(gemm_H_lower_wire[row][col+1]),

              // Break row re-injects activation from the fabric delay line
              // (B_INPUT = DIRECT is forced by BREAK_CASCADE inside the unit).
              .in_V      (gemm_in_V_fabric[row][col]),
              .BCIN_in   ('0),
              .BCOUT_out (gemm_BCIN_wire[row+1][col]),

              .instruction_in_V (gemm_inst_wire[row][col]),
              .instruction_out_V(gemm_inst_wire[row+1][col]),
              .inst_valid_out_V (gemm_inst_valid_wire[row+1][col]),

              // Take partial sum from the previous row's fabric P-out.
              .V_result_in (gemm_P_fabric_wire[row-1][col]),
              .V_result_out(gemm_V_result_wire[row+1][col]),
              .P_fabric_out(gemm_P_fabric_wire[row][col])
          );
        end else begin : normal_row
          GEMM_dsp_unit #(
              .IS_TOP_ROW   (row == 0 ? 1 : 0),
              .BREAK_CASCADE(0)
          ) dsp_unit (
              .clk(clk),
              .rst_n(rst_n),
              .i_clear(i_clear),

              .i_valid        (gemm_V_valid_wire[row][col]),
              .inst_valid_in_V(gemm_inst_valid_wire[row][col]),
              .i_weight_valid (i_weight_valid),
              .o_valid        (gemm_V_valid_wire[row+1][col]),

              .in_H_upper (gemm_H_upper_wire[row][col]),
              .out_H_upper(gemm_H_upper_wire[row][col+1]),
              .in_H_lower (gemm_H_lower_wire[row][col]),
              .out_H_lower(gemm_H_lower_wire[row][col+1]),

              // Top row: take INT8 activation from fabric V_in (driven via the
              // delay line's row-0 slot). Other rows: BCIN cascade.
              .in_V      (row == 0 ? gemm_in_V_fabric[row][col] : {INT8_BITS{1'b0}}),
              .BCIN_in   (gemm_BCIN_wire[row][col]),
              .BCOUT_out (gemm_BCIN_wire[row+1][col]),

              .instruction_in_V (gemm_inst_wire[row][col]),
              .instruction_out_V(gemm_inst_wire[row+1][col]),
              .inst_valid_out_V (gemm_inst_valid_wire[row+1][col]),

              .V_result_in (gemm_V_result_wire[row][col]),
              .V_result_out(gemm_V_result_wire[row+1][col]),
              .P_fabric_out(gemm_P_fabric_wire[row][col])
          );
        end
      end
    end

    // ===| Accumulator at the final row |========================================
    for (col = 0; col < array_vertical; col++) begin : gemm_ACC_col_loop
      assign V_ACC_valid[col] = gemm_inst_valid_wire[array_horizontal][col];
      GEMM_accumulator #() gemm_ACC (
          .clk    (clk),
          .rst_n  (rst_n),
          .i_clear(i_clear),
          .i_valid(V_ACC_valid[col]),
          .PCIN   (gemm_V_result_wire[array_horizontal][col]),
          .gemm_ACC_result(V_ACC_out[col])
      );
    end
  endgenerate

endmodule
