`timescale 1ns / 1ps

// algorithms_pkg is compiled from Library/Algorithms/Algorithms.sv.
`include "GLOBAL_CONST.svh"

// AXIL_CMD_IN
// AXI4-Lite Write path : CPU → NPU
// Stores incoming commands into a FIFO.
// Drains FIFO to upper module when IN_decoder_ready is asserted.

module AXIL_CMD_IN #(
    parameter FIFO_DEPTH = 8  // number of commands to buffer
) (
    input logic clk,
    input logic rst_n,
    input logic IN_clear,

    // AXI4-Lite Write channels (slave)
    // AW
    input logic [11:0] s_awaddr,
    input logic [2:0] s_awprot,
    input logic s_awvalid,
    output logic s_awready,
    // W
    input logic [`ISA_WIDTH-1:0] s_wdata,
    input logic [(`ISA_WIDTH/8)-1:0] s_wstrb,
    input logic s_wvalid,
    output logic s_wready,
    // B
    output logic [1:0] s_bresp,
    output logic s_bvalid,
    input logic s_bready,

    // To upper module (NPU_interface)
    output logic [`ISA_WIDTH-1:0] OUT_data,         // command word
    output logic                  OUT_valid,        // FIFO has data
    input  logic                  IN_decoder_ready  // upper module is ready to consume
);

  /*─────────────────────────────────────────────
  Register Address Map
  ───────────────────────────────────────────────*/
  localparam ADDR_INST = 12'h000;
  localparam ADDR_KICK = 12'h008;

  /*─────────────────────────────────────────────
  AXI4-Lite Write Path
  Latch AW first, write register when W arrives.
  ───────────────────────────────────────────────*/
  logic [          11:0] aw_addr_latch;
  logic                  aw_pending;
  logic                  bvalid_r;
  logic                  fifo_wen;
  logic [`ISA_WIDTH-1:0] fifo_wdata;

  // if queue is full block receive
  assign s_awready = ~aw_pending && ~cmd_q.full;
  assign s_wready  = aw_pending;
  assign s_bresp   = 2'b00;
  assign s_bvalid  = bvalid_r;

  // AW latch
  always_ff @(posedge clk) begin
    if (!rst_n || IN_clear) begin
      aw_addr_latch <= '0;
      aw_pending    <= 1'b0;
    end else begin
      if (s_awvalid && s_awready) begin
        aw_addr_latch <= s_awaddr;
        aw_pending    <= 1'b1;
      end
      if (s_wvalid && s_wready) aw_pending <= 1'b0;
    end
  end

  // W : push into FIFO + B response
  always_ff @(posedge clk) begin
    fifo_wen <= 1'b0;
    bvalid_r <= 1'b0;

    if (!rst_n || IN_clear) begin
      fifo_wdata <= '0;
    end else begin
      if (s_wvalid && s_wready) begin
        case (aw_addr_latch)
          // push instruction word into FIFO
          ADDR_INST: begin
            fifo_wdata <= s_wdata;
            fifo_wen   <= 1'b1;
          end
          // KICK : push a special marker (bit63 = 1 as kick flag)
          ADDR_KICK: begin
            fifo_wdata <= 64'h8000_0000_0000_0000;
            fifo_wen   <= 1'b1;
          end
          default: ;
        endcase
        bvalid_r <= 1'b1;
      end
      if (bvalid_r && s_bready) bvalid_r <= 1'b0;
    end
  end

  /*─────────────────────────────────────────────
  Command Queue  (simple synchronous, FIFO_DEPTH entries)
  Push : fifo_wen
  Pop  : OUT_valid && IN_decoder_ready
  ───────────────────────────────────────────────*/
  import algorithms_pkg::*;

  IF_queue #(
      .DATA_WIDTH(`ISA_WIDTH),
      .DEPTH(FIFO_DEPTH)
  ) cmd_q (
      .clk  (clk),
      .rst_n(rst_n)
  );
  QUEUE u_cmd_q (.q(cmd_q.owner));

  always_ff @(posedge clk) begin
    if (!rst_n || IN_clear) begin
      cmd_q.clear();
    end else begin
      if (fifo_wen) cmd_q.push(fifo_wdata);  // push when AXI write done
      if (OUT_valid && IN_decoder_ready) cmd_q.pop();  // IXED: o_valid -> OUT_valid
    end
  end

  assign OUT_valid = ~cmd_q.empty;
  assign OUT_data  = cmd_q.pop_data;

endmodule
