// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===| Module: mem_CVO_stream_bridge — L2 ↔ CVO 16b/128b serdes bridge |========
// Purpose      : Convert between 128-bit L2 port-B bursts and 16-bit BF16
//                stream consumed/produced by CVO_top. Owns the per-op FSM
//                that walks src→dst regions.
// Spec ref     : pccx v002 §5.3 (CVO stream bridge), §3.5 (CVO uop).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low.
// Operation flow:
//   Phase 1 — READ : sequential 128-bit bursts from L2[src_addr..src_addr+N_words-1]
//                    → deserialise 8 × 16-bit per word → stream to CVO engine.
//                    CVO results are buffered in an internal 2048-deep XPM FIFO.
//   Phase 2 — WRITE: drain FIFO → serialise 8 × 16-bit → 128-bit bursts
//                    → write to L2[dst_addr..dst_addr+N_words-1].
// L2 addr unit : 128-bit words. base_addr N ↔ bytes [16*N .. 16*N+15].
// L2 read lat  : 4 clocks (URAM READ_LATENCY_B = 4) — tracked via rd_lat_pipe.
// Max vec len  : ResultFifoDepth × 16-bit = 32 KB (1 BRAM36 instance) — the
//                FIFO depth caps the longest CVO op vector.
// Handshake    : OUT_cvo_valid asserts during READ phase only;
//                OUT_cvo_result_ready asserts during READ phase only when
//                the result FIFO is not full.
// L2 port-B mux: Write priority during ST_WRITE; otherwise read addr.
// Reset state  : ST_IDLE, all counters/buffers zeroed, OUT_done = 0.
// Counters     : none.
// Assertions   : (Stage C) wr_word_cnt ≤ total_words; FIFO never overflows
//                (OUT_cvo_result_ready guarded by ~fifo_full).
// ===============================================================================

module mem_CVO_stream_bridge (
    input logic clk,
    input logic rst_n,

    // ===| Dispatch from mem_dispatcher |========================================
    input  cvo_control_uop_t IN_cvo_uop,
    input  logic             IN_cvo_uop_valid,
    output logic             OUT_busy,
    output logic             OUT_done,

    // ===| L2 port B direct interface (128-bit) |================================
    // Single-address mux: write takes priority over read.
    output logic         OUT_l2_we,
    output logic [ 16:0] OUT_l2_addr,
    output logic [127:0] OUT_l2_wdata,
    input  logic [127:0] IN_l2_rdata,   // valid 4 cycles after OUT_l2_addr+~we

    // ===| CVO data stream (to CVO_top.IN_data) |=================================
    output logic [15:0] OUT_cvo_data,
    output logic        OUT_cvo_valid,
    input  logic        IN_cvo_data_ready,

    // ===| CVO result stream (from CVO_top.OUT_result) |==========================
    input  logic [15:0] IN_cvo_result,
    input  logic        IN_cvo_result_valid,
    output logic        OUT_cvo_result_ready
);

  // ===| Bridge Sizing |=========================================================
  // ResultFifoDepth : depth of the on-chip BF16 result FIFO (1 BRAM36 = 2048 ×
  //                   16-bit = 32 KB). Also caps the longest CVO op vector
  //                   that can sit in the FIFO between READ and WRITE phases
  //                   without overrunning the buffer.
  localparam int ResultFifoDepth = 2048;

  // ===| State Machine |=========================================================
  typedef enum logic [1:0] {
    ST_IDLE  = 2'b00,
    ST_READ  = 2'b01,  // reading L2 → CVO (buffering outputs)
    ST_WRITE = 2'b10,  // draining buffer → L2
    ST_DONE  = 2'b11
  } bridge_state_e;

  bridge_state_e state;

  // ===| Latched UOP |===========================================================
  logic [16:0] rd_base;     // L2 word address of src
  logic [16:0] wr_base;     // L2 word address of dst
  logic [15:0] total_elems; // CVO length (elements)
  logic [12:0] total_words; // ceil(total_elems / 8)

  always_comb begin
    total_words = 13'((total_elems + 16'd7) >> 3);
  end

  // ===| Read-side state |=======================================================
  logic [ 12:0] rd_word_cnt;  // words issued so far
  logic [  2:0] rd_elem_idx;  // current element within 128-bit deser buffer
  logic [127:0] rd_deser_buf;  // latched 128-bit L2 word
  logic         rd_buf_valid;  // deser buffer holds valid data
  logic [ 15:0] elems_fed;  // elements delivered to CVO

  // 3-cycle read latency tracking
  logic [  3:0] rd_lat_pipe;  // shift register: [3]=oldest, [0]=newest

  // ===| Write-side state |======================================================
  logic [  2:0] wr_elem_idx;  // accumulation index 0..7
  logic [127:0] wr_ser_buf;  // serialisation buffer
  logic [ 12:0] wr_word_cnt;  // words written so far
  logic [ 15:0] elems_result;  // results drained from FIFO

  // ===| Output FIFO (CVO results → write buffer) |==============================
  // XPM FIFO sync, depth=2048, width=16 bit (max 32 KB = 1 BRAM36)
  logic         fifo_wr_en;
  logic         fifo_rd_en;
  logic [ 15:0] fifo_dout;
  logic         fifo_empty;
  logic         fifo_full;

  assign fifo_wr_en = IN_cvo_result_valid && (state == ST_READ);
  assign OUT_cvo_result_ready = ~fifo_full && (state == ST_READ);

  xpm_fifo_sync #(
      .FIFO_WRITE_DEPTH(ResultFifoDepth),
      .WRITE_DATA_WIDTH(16),
      .READ_DATA_WIDTH (16),
      .FIFO_MEMORY_TYPE("block"),
      .READ_MODE       ("std"),
      .FULL_RESET_VALUE(0)
  ) u_result_fifo (
      .sleep (1'b0),
      .rst   (~rst_n),
      .wr_clk(clk),
      .wr_en (fifo_wr_en),
      .din   (IN_cvo_result),
      .rd_en (fifo_rd_en),
      .dout  (fifo_dout),
      .empty (fifo_empty),
      .full  (fifo_full)
  );

  // ===| Main FSM |==============================================================
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state        <= ST_IDLE;
      rd_base      <= '0;
      wr_base      <= '0;
      total_elems  <= '0;
      rd_word_cnt  <= '0;
      rd_elem_idx  <= '0;
      rd_deser_buf <= '0;
      rd_buf_valid <= 1'b0;
      rd_lat_pipe  <= 4'b0;
      elems_fed    <= '0;
      wr_elem_idx  <= '0;
      wr_ser_buf   <= '0;
      wr_word_cnt  <= '0;
      elems_result <= '0;
      OUT_done     <= 1'b0;
    end else begin
      OUT_done <= 1'b0;

      case (state)
        // ===| IDLE: latch uop, convert element addresses to word addresses |===
        ST_IDLE: begin
          if (IN_cvo_uop_valid) begin
            // src/dst are element (16-bit) addresses; divide by 8 for 128-bit words
            rd_base      <= 17'(IN_cvo_uop.src_addr >> 3);
            wr_base      <= 17'(IN_cvo_uop.dst_addr >> 3);
            total_elems  <= IN_cvo_uop.length;
            rd_word_cnt  <= '0;
            rd_elem_idx  <= '0;
            rd_buf_valid <= 1'b0;
            rd_lat_pipe  <= 4'b0;
            elems_fed    <= '0;
            wr_elem_idx  <= '0;
            wr_ser_buf   <= '0;
            wr_word_cnt  <= '0;
            elems_result <= '0;
            state        <= ST_READ;
          end
        end

        // ===| READ: stream L2 → CVO; capture results in FIFO |================
        ST_READ: begin
          // Advance latency shift register
          rd_lat_pipe <= {rd_lat_pipe[2:0], 1'b0};

          // Issue next L2 read when deser buffer is empty (pre-fetch when 3 left)
          if (!rd_buf_valid && rd_word_cnt < 13'(total_words)) begin
            rd_lat_pipe[0] <= 1'b1;  // mark new read outstanding
            rd_word_cnt    <= rd_word_cnt + 13'd1;
          end

          // Capture L2 data 4 cycles after read issued
          if (rd_lat_pipe[3]) begin
            rd_deser_buf <= IN_l2_rdata;
            rd_buf_valid <= 1'b1;
            rd_elem_idx  <= 3'd0;
          end

          // Feed CVO one element per cycle from deser buffer
          if (rd_buf_valid && IN_cvo_data_ready) begin
            rd_elem_idx <= rd_elem_idx + 3'd1;
            elems_fed   <= elems_fed + 16'd1;
            if (rd_elem_idx == 3'd7 || elems_fed + 16'd1 == total_elems) begin
              rd_buf_valid <= 1'b0;
            end
          end

          // Transition when all elements have been fed and all results captured
          if (elems_fed == total_elems && !fifo_empty) begin
            state <= ST_WRITE;
          end
        end

        // ===| WRITE: drain FIFO → L2 |=========================================
        ST_WRITE: begin
          if (!fifo_empty) begin
            wr_ser_buf   <= {fifo_dout, wr_ser_buf[127:16]};
            wr_elem_idx  <= wr_elem_idx + 3'd1;
            elems_result <= elems_result + 16'd1;
          end

          // When 8 elements accumulated (or last partial word), write to L2
          if (wr_elem_idx == 3'd7 || elems_result == total_elems) begin
            wr_word_cnt <= wr_word_cnt + 13'd1;
            wr_elem_idx <= 3'd0;
          end

          if (elems_result == total_elems && fifo_empty) begin
            state <= ST_DONE;
          end
        end

        // ===| DONE: pulse, return to IDLE |====================================
        ST_DONE: begin
          OUT_done <= 1'b1;
          state    <= ST_IDLE;
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

  // ===| FIFO read enable (draining during WRITE phase) |========================
  assign fifo_rd_en = (state == ST_WRITE) && !fifo_empty;

  // ===| L2 port B output mux ===================================================
  // Priority: write (WRITE phase) > read (READ phase)
  always_comb begin
    OUT_l2_we    = 1'b0;
    OUT_l2_addr  = '0;
    OUT_l2_wdata = '0;

    if (state == ST_WRITE && wr_elem_idx == 3'd0 && wr_word_cnt > 0) begin
      // Write accumulated 128-bit word to dst
      OUT_l2_we    = 1'b1;
      OUT_l2_addr  = 17'(wr_base + (wr_word_cnt - 13'd1));
      OUT_l2_wdata = wr_ser_buf;
    end else if (state == ST_READ && rd_lat_pipe[0]) begin
      // Issue read for next 128-bit word from src
      OUT_l2_we   = 1'b0;
      OUT_l2_addr = 17'(rd_base + (rd_word_cnt - 13'd1));
    end
  end

  // ===| CVO data output ========================================================
  // Mux the correct 16-bit slice from the deser buffer
  always_comb begin
    OUT_cvo_data  = rd_deser_buf[rd_elem_idx*16+:16];
    OUT_cvo_valid = rd_buf_valid && (state == ST_READ);
  end

  // ===| Status |================================================================
  assign OUT_busy = (state != ST_IDLE);

endmodule
