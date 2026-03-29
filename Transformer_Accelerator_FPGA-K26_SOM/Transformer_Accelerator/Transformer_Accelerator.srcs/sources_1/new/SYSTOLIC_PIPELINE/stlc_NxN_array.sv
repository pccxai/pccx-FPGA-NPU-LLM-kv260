`timescale 1ns / 1ps
`include "stlc_Array.svh"

module stlc_NxN_array #(
    parameter ARRAY_HORIZONTAL = `ARRAY_SIZE_H,
    parameter ARRAY_VERTICAL   = `ARRAY_SIZE_V
) (
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // =| Global Controls |=
    input logic i_weight_valid,  // Enables horizontal weight shifting

    // =| Delay line input (from FMap Cache and Weight Dispatcher) |=
    input logic [`STLC_MAC_UNIT_IN_H-1:0] H_in[0:`ARRAY_SIZE_H-1],
    input logic [`STLC_MAC_UNIT_IN_V-1:0] V_in[0:`ARRAY_SIZE_V-1],
    input logic in_valid[0:`ARRAY_SIZE_V-1],  // Staggered valid from FMap delay line

    // =| VLIW Instruction Input (Staggered along with V_in) |=
    input logic [2:0] inst_in      [0:`ARRAY_SIZE_V-1],
    input logic       inst_valid_in[0:`ARRAY_SIZE_V-1],

    // =| Outputs |=
    output logic [`DSP_RESULT_SIZE-1:0] V_out      [0:`ARRAY_SIZE_V-1],
    output logic [`DSP_RESULT_SIZE-1:0] V_ACC_out  [0:`ARRAY_SIZE_V-1],
    output logic                        V_ACC_valid[0:`ARRAY_SIZE_V-1]
);

  // ===| Systolic Array Internal Wires |==================================

  // Horizontal logic wires (Weights)
  // Size is [Row][Col], data flows Left to Right.
  // H_in feeds into Col 0.
  logic [`STLC_MAC_UNIT_IN_H - 1 : 0] stlc_H_wire[0 : ARRAY_HORIZONTAL-1][0 : ARRAY_VERTICAL];

  // Vertical logic wires (Feature Map / ACIN)
  // Size is [Row][Col], data flows Top to Bottom.
  logic [29:0] stlc_ACIN_wire[0 : ARRAY_HORIZONTAL][0 : ARRAY_VERTICAL-1];

  // Instruction logic wires (Top to Bottom)
  logic [2:0] stlc_inst_wire[0 : ARRAY_HORIZONTAL][0 : ARRAY_VERTICAL-1];
  logic stlc_inst_valid_wire[0 : ARRAY_HORIZONTAL][0 : ARRAY_VERTICAL-1];

  // Valid signal logic wires (Top to Bottom)
  logic stlc_V_valid_wire[0 : ARRAY_HORIZONTAL][0 : ARRAY_VERTICAL-1];

  // Result shift wires (Top to Bottom)
  logic [`DSP_RESULT_SIZE - 1 : 0] stlc_V_result_wire[0 : ARRAY_HORIZONTAL][0 : ARRAY_VERTICAL-1];

  // Fabric break wires for row 15 -> 16
  logic [47:0] stlc_P_fabric_wire[0 : ARRAY_HORIZONTAL-1][0 : ARRAY_VERTICAL-1];

  // V_in fabric delay line to replace A_fabric_wire
  logic [29:0] stlc_in_V_fabric[0 : ARRAY_HORIZONTAL][0 : ARRAY_VERTICAL-1];

  // ======================================================================

  // ===| Input Assignments |==============================================
  // >>>| TOP INPUT LANE |<<<
  genvar i;
  generate
    for (i = 0; i < ARRAY_VERTICAL; i++) begin : assign_v_inputs
      assign stlc_ACIN_wire[0][i] = 30'd0;  // Top row ACIN is not used (A is used directly)
      assign stlc_inst_wire[0][i] = inst_in[i];
      assign stlc_inst_valid_wire[0][i] = inst_valid_in[i];
      assign stlc_V_valid_wire[0][i] = in_valid[i];
      assign stlc_V_result_wire[0][i] = 48'd0;  // Top row PCIN is 0

      // Initialize the fabric delay line with V_in padded to 30 bits
      assign stlc_in_V_fabric[0][i] = {3'd0, V_in[i]};
    end

    for (i = 0; i < ARRAY_HORIZONTAL; i++) begin : assign_h_inputs
      assign stlc_H_wire[i][0] = H_in[i];
    end
  endgenerate

  // >>>| Normal Lane |<<<
  // Fabric delay line for V_in to reach row 16 correctly
  genvar d_row, d_col;
  generate
    for (d_row = 0; d_row < ARRAY_HORIZONTAL; d_row++) begin : v_delay_row
      for (d_col = 0; d_col < ARRAY_VERTICAL; d_col++) begin : v_delay_col
        always_ff @(posedge clk) begin
          if (stlc_V_valid_wire[d_row][d_col]) begin
            stlc_in_V_fabric[d_row+1][d_col] <= stlc_in_V_fabric[d_row][d_col];
          end
        end
      end
    end
  endgenerate

  // ===| 2D Array Instantiation |=========================================
  genvar row, col;
  generate
    for (row = 0; row < ARRAY_HORIZONTAL; row++) begin : stlc_row_loop
      for (col = 0; col < ARRAY_VERTICAL; col++) begin : stlc_col_loop

        if (row == ARRAY_HORIZONTAL - 1) begin : last_row
          stlc_dsp_unit_last_ROW #(
              .IS_TOP_ROW(0)
          ) dsp_unit_last_ROW (
              .clk(clk),
              .rst_n(rst_n),
              .i_clear(i_clear),

              .i_valid(stlc_V_valid_wire[row][col]),
              .inst_valid_in_V(stlc_inst_valid_wire[row][col]),
              .i_weight_valid(i_weight_valid),
              .o_valid(stlc_V_valid_wire[row+1][col]),

              .in_H (stlc_H_wire[row][col]),
              .out_H(stlc_H_wire[row][col+1]),

              .ACIN_in  (stlc_ACIN_wire[row][col]),
              .ACOUT_out(stlc_ACIN_wire[row+1][col]),

              .instruction_in_V (stlc_inst_wire[row][col]),
              .instruction_out_V(stlc_inst_wire[row+1][col]),
              .inst_valid_out_V (stlc_inst_valid_wire[row+1][col]),

              .V_result_in (stlc_V_result_wire[row][col]),
              .V_result_out(stlc_V_result_wire[row+1][col]),

              .stlc_unit_results(V_out[col])
          );
        end else if (row == 16) begin : break_row
          stlc_dsp_unit #(
              .IS_TOP_ROW(0),
              .BREAK_CASCADE(1)
          ) dsp_unit_break (
              .clk(clk),
              .rst_n(rst_n),
              .i_clear(i_clear),

              .i_valid(stlc_V_valid_wire[row][col]),
              .inst_valid_in_V(stlc_inst_valid_wire[row][col]),
              .i_weight_valid(i_weight_valid),
              .o_valid(stlc_V_valid_wire[row+1][col]),

              .in_H (stlc_H_wire[row][col]),
              .out_H(stlc_H_wire[row][col+1]),

              // Take delayed input from fabric shift register instead of CASCADE
              .in_V(stlc_in_V_fabric[row][col]),
              .ACIN_in(30'd0),
              .ACOUT_out(stlc_ACIN_wire[row+1][col]),

              .instruction_in_V (stlc_inst_wire[row][col]),
              .instruction_out_V(stlc_inst_wire[row+1][col]),
              .inst_valid_out_V (stlc_inst_valid_wire[row+1][col]),

              // Take result from previous row's P fabric out
              .V_result_in (stlc_P_fabric_wire[row-1][col]),
              .V_result_out(stlc_V_result_wire[row+1][col]),
              .P_fabric_out(stlc_P_fabric_wire[row][col])
          );
        end else begin : normal_row
          stlc_dsp_unit #(
              .IS_TOP_ROW(row == 0 ? 1 : 0),
              .BREAK_CASCADE(0)
          ) dsp_unit (
              .clk(clk),
              .rst_n(rst_n),
              .i_clear(i_clear),

              .i_valid(stlc_V_valid_wire[row][col]),
              .inst_valid_in_V(stlc_inst_valid_wire[row][col]),
              .i_weight_valid(i_weight_valid),
              .o_valid(stlc_V_valid_wire[row+1][col]),

              .in_H (stlc_H_wire[row][col]),
              .out_H(stlc_H_wire[row][col+1]),

              .in_V(row == 0 ? {3'd0, V_in[col]} : 30'd0),
              .ACIN_in(stlc_ACIN_wire[row][col]),
              .ACOUT_out(stlc_ACIN_wire[row+1][col]),

              .instruction_in_V (stlc_inst_wire[row][col]),
              .instruction_out_V(stlc_inst_wire[row+1][col]),
              .inst_valid_out_V (stlc_inst_valid_wire[row+1][col]),

              .V_result_in (stlc_V_result_wire[row][col]),
              .V_result_out(stlc_V_result_wire[row+1][col]),
              .P_fabric_out(stlc_P_fabric_wire[row][col])
          );
        end
      end
    end

    // Accumulators for the final row
    for (col = 0; col < ARRAY_VERTICAL; col++) begin : stlc_ACC_col_loop
      assign V_ACC_valid[col] = stlc_inst_valid_wire[ARRAY_HORIZONTAL][col];
      stlc_accumulator #() stlc_ACC (
          .clk(clk),
          .rst_n(rst_n),
          .i_clear(i_clear),
          // Accumulator should trigger when the last row outputs a valid result
          .i_valid(V_ACC_valid[col]),
          // PCIN connects to the V_result_out of the LAST_ROW (which is stored in [ARRAY_HORIZONTAL])
          .PCIN(stlc_V_result_wire[ARRAY_HORIZONTAL][col]),
          .stlc_ACC_result(V_ACC_out[col])
      );
    end
  endgenerate

endmodule
