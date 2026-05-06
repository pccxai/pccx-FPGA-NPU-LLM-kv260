`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "GEMM_Array.svh"

// ===| Module: FROM_gemm_result_packer — 32×BF16 → 128b AXIS packer |===========
// Purpose      : Capture the staggered 16-bit normalised results from all
//                ARRAY_SIZE columns, then pack 8-element groups into 128-bit
//                DMA words for the result AXIS bus.
// Spec ref     : pccx v002 §2.2.7 (result pack), §5.6 (result writeback).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// FSM          : IDLE → CHECK_VALID → SEND_DATA → (next group | IDLE).
// Latency      : One group (= 8 columns) ready when capture_valid[idx +: 8]
//                all set; pack happens in SEND_DATA when packed_ready high.
// Throughput   : 4 packed 128-bit words per ARRAY_SIZE = 32 normaliser
//                completions (8 BF16 lanes × 16 bit = 128 bit).
// Handshake    : Standard valid/ready on packed_*. packed_valid is held
//                high while waiting for packed_ready.
// Reset state  : capture_valid = 0; capture_reg = 0; state = IDLE.
// Counters     : o_busy reflects (|capture_valid) || (state != IDLE).
// Errors       : none.
// Assertions   : (Stage C) send_idx ≤ ARRAY_SIZE - 8.
// Notes        : Because rows produce results staggered, capture_reg[i]
//                holds row i's BF16 until consumed by the SEND_DATA arm
//                that clears its capture_valid bit.
// ===============================================================================

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

  // ===| Pack Geometry |==========================================================
  // BeatsPerWord  : how many BF16 lanes fit in one AXIS word
  //                 (= AXI_STREAM_WIDTH / BF16_WIDTH = 8 today).
  // LastSendIdx   : the highest send_idx that still leaves a full group inside
  //                 the capture register (= ARRAY_SIZE - BeatsPerWord). Once
  //                 send_idx reaches it, the FSM returns to IDLE.
  localparam int BeatsPerWord = `AXI_STREAM_WIDTH / `BF16_WIDTH;
  localparam int LastSendIdx  = ARRAY_SIZE - BeatsPerWord;

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
        for (int i = 0; i < BeatsPerWord; i++) begin
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
          packed_valid <= 1'b0;
          // Wait until BeatsPerWord adjacent rows have produced a valid result
          // (one 128-bit word = 8 BF16 lanes today).
          if (&capture_valid[send_idx+:BeatsPerWord]) begin
            // Drive data before asserting valid so a downstream stall sees a
            // stable payload until the eventual valid/ready handshake.
            for (int b = 0; b < BeatsPerWord; b++) begin
              packed_data[b*`BF16_WIDTH +: `BF16_WIDTH] <= capture_reg[send_idx+b];
            end
            packed_valid <= 1'b1;
            state        <= SEND_DATA;
          end
        end

        SEND_DATA: begin
          packed_valid <= 1'b1;  // Keep high until ready
          if (packed_ready) begin
            packed_valid <= 1'b0;

            if (send_idx >= LastSendIdx) begin
              state    <= IDLE;
              send_idx <= 0;
            end else begin
              send_idx <= send_idx + BeatsPerWord;
              state    <= CHECK_VALID;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
