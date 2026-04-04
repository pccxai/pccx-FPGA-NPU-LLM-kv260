`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"
`include "stlc_Array.svh"

/**
 * Module: stlc_weight_dispatcher
 * Description:
 *   Unpacks 128-bit wide data into 32 individual 4-bit INT4 weights.
 *   Provides registered outputs to maintain 400MHz timing.
 */

module stlc_weight_dispatcher #(
    parameter weight_lane_cnt       = `HP_PORT_CNT,
    parameter weight_width_per_lane = `HP_PORT_SINGLE_WIDTH,
    parameter weight_size           = `INT4,
    parameter weight_cnt            = `HP_WEIGHT_CNT(`HP_PORT_SINGLE_WIDTH, `INT4),
    parameter array_horizontal      = `ARRAY_SIZE_H,
    parameter array_vertical        = `ARRAY_SIZE_V
) (
    input logic clk,
    input logic rst_n,

    // ===| 128-bit * 4 Input from FIFO |===
    input logic [weight_size-1:0] fifo_data [0:weight_cnt-1],
    input logic                   fifo_valid[0:weight_cnt-1],

    output logic fifo_ready[0:weight_cnt-1],

    // ===| 32 x 4-bit Outputs to Systolic Array (V_in) |========
    output logic [weight_size-1:0] weight_out  [0:weight_cnt-1],
    output logic                   weight_valid
);

  // ===| Flow Control: Always ready if not stalled by downstream |=====================
  assign fifo_ready = 1'b1;

  // ===| Unpacking Logic with Pipeline Registers |=====================================
  // ===| This ensures that the massive fan-out (1 to 32) doesn't break timing. |=======
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      weight_valid <= 1'b0;
      for (int i = 0; i < weight_cnt; i++) begin
        weight_out[w_lane_cnt][i] <= '0;
      end
    end else begin
      weight_valid <= fifo_valid;

      // ===| Unpack 128-bit into 32 x 4-bit  |==============================================
      for (int i = 0; i < weight_cnt; i++) begin
        weight_out[i] <= fifo_data[i];
      end
    end
  end
endmodule
