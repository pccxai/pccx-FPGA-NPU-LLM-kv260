`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "GEMM_Array.svh"

/**
 * Module: FROM_gemm_result_packer
 * Description:
 *   Collects 32 staggered 16-bit results and packs them into 128-bit DMA words.
 *   Uses an internal FSM to sequentially scan and pack results from all columns.
 */

module FROM_gemm_result_packer #(
    parameter ARRAY_SIZE = 32
) (
    input logic clk,
    input logic rst_n,

    // =| Input from Normalizers (16-bit BF16) |=
    input logic [`BF16_WIDTH-1:0] row_res      [0:ARRAY_SIZE-1],
    input logic                   row_res_valid[0:ARRAY_SIZE-1],

    // =| Output to FIFO (128-bit) |=
    output logic [`AXI_STREAM_WIDTH-1:0] packed_data,
    output logic                       packed_valid,
    input  logic                       packed_ready,

    // =| Status |=
    output logic o_busy
);

  // ===| Internal Buffer to Hold Results |=======
  // Since systolic array results are staggered, we need to capture them.
  logic [`BF16_WIDTH-1:0] capture_reg[0:ARRAY_SIZE-1];
  logic [ARRAY_SIZE-1:0]  capture_valid;

  // ===| State Machine for Packing (Round-Robin) |=======
  typedef enum logic [1:0] {
    IDLE,
    CHECK_VALID,
    SEND_DATA
  } state_t;
  state_t     state;
  logic [5:0] send_idx;  // 0 to 31, must be declared before the capture FSM
                         // uses it to clear capture_valid bits on SEND_DATA.

  // Busy if any capture_valid bit is set or we are in a non-IDLE state
  assign o_busy = (|capture_valid) || (state != IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      capture_valid <= '0;
      for (int i = 0; i < ARRAY_SIZE; i++) capture_reg[i] <= '0;
    end else begin
      for (int i = 0; i < ARRAY_SIZE; i++) begin
        if (row_res_valid[i]) begin
          capture_reg[i]   <= row_res[i];
          capture_valid[i] <= 1'b1;
        end
      end

      // Clear valid bits once they are consumed (handled by FSM below)
      if (state == SEND_DATA && packed_ready) begin
        for (int i = 0; i < 8; i++) begin
          capture_valid[send_idx+i] <= 1'b0;
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state        <= IDLE;
      send_idx     <= 0;
      packed_valid <= 1'b0;
      packed_data  <= '0;
    end else begin
      case (state)
        IDLE: begin
          packed_valid <= 1'b0;
          if (|capture_valid) begin
            state    <= CHECK_VALID;
            send_idx <= 0;
          end
        end

        CHECK_VALID: begin
          // We need 8 results to form a 128-bit word (16*8=128)
          // In a real systolic array, they might not arrive all at once,
          // but for simplicity, let's wait until we have a chunk of 8.
          if (&capture_valid[send_idx+:8]) begin
            state <= SEND_DATA;
          end
        end

        SEND_DATA: begin
          if (packed_ready) begin
            packed_data <= {
              capture_reg[send_idx+7],
              capture_reg[send_idx+6],
              capture_reg[send_idx+5],
              capture_reg[send_idx+4],
              capture_reg[send_idx+3],
              capture_reg[send_idx+2],
              capture_reg[send_idx+1],
              capture_reg[send_idx+0]
            };
            packed_valid <= 1'b1;

            if (send_idx >= 24) begin
              state    <= IDLE;
              send_idx <= 0;
            end else begin
              send_idx <= send_idx + 8;
              state    <= CHECK_VALID;
            end
          end else begin
            packed_valid <= 1'b1;  // Keep high until ready
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
