`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"

// AXIL_STAT_OUT
// AXI4-Lite Read path : NPU → CPU
// Upper module pushes status into FIFO continuously.
// Drains FIFO to CPU when AXI4-Lite read handshake happens.

module AXIL_STAT_OUT #(
    parameter FIFO_DEPTH = 8
) (
    input logic clk,
    input logic rst_n,
    input logic IN_clear,

    // From upper module (NPU_interface → here)
    input logic [`ISA_WIDTH-1:0] IN_data,  // status word to send to CPU
    input logic                  IN_valid, // upper module has data to push

    // AXI4-Lite Read channels (slave)
    // AR
    input  logic [          11:0] s_araddr,
    input  logic                  s_arvalid,
    output logic                  s_arready,
    // R
    output logic [`ISA_WIDTH-1:0] s_rdata,
    output logic [           1:0] s_rresp,
    output logic                  s_rvalid,
    input  logic                  s_rready
);

  /*─────────────────────────────────────────────
  FIFO  (simple synchronous, FIFO_DEPTH entries)
  Push : IN_valid from upper module
  Pop  : AXI4-Lite read handshake with CPU
  ───────────────────────────────────────────────*/
  localparam PTR_W = $clog2(FIFO_DEPTH);

  logic [`ISA_WIDTH-1:0] mem[0:FIFO_DEPTH-1];
  logic [PTR_W:0] wr_ptr, rd_ptr;
  logic fifo_empty, fifo_full;

  assign fifo_empty = (wr_ptr == rd_ptr);
  assign fifo_full  = (wr_ptr[PTR_W] != rd_ptr[PTR_W]) && (wr_ptr[PTR_W-1:0] == rd_ptr[PTR_W-1:0]);

  logic fifo_ren;

  always_ff @(posedge clk) begin
    if (!rst_n || IN_clear) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
    end else begin
      // push : upper module feeds status continuously
      if (IN_valid && !fifo_full) begin
        mem[wr_ptr[PTR_W-1:0]] <= IN_data;
        wr_ptr <= wr_ptr + 1'b1;
      end
      // pop : CPU consumed the data
      if (fifo_ren && !fifo_empty) rd_ptr <= rd_ptr + 1'b1;
    end
  end

  /*─────────────────────────────────────────────
  AXI4-Lite Read Path
  Wait for AR, then pop one entry from FIFO and return it.
  Hold rvalid until CPU acknowledges with rready.
  ───────────────────────────────────────────────*/
  logic [`ISA_WIDTH-1:0] rdata_r;
  logic                  rvalid_r;

  assign s_rdata   = rdata_r;
  assign s_rresp   = 2'b00;
  assign s_rvalid  = rvalid_r;
  assign s_arready = ~rvalid_r && ~fifo_empty;  // ready only when FIFO has data
  assign fifo_ren  = s_arvalid && s_arready;  // pop on AR handshake

  always_ff @(posedge clk) begin
    if (!rst_n || IN_clear) begin
      rdata_r  <= '0;
      rvalid_r <= 1'b0;
    end else begin
      // AR handshake → latch FIFO head and assert rvalid
      if (s_arvalid && s_arready) begin
        rdata_r  <= mem[rd_ptr[PTR_W-1:0]];
        rvalid_r <= 1'b1;
      end
      // R handshake → CPU consumed data, release
      if (rvalid_r && s_rready) rvalid_r <= 1'b0;
    end
  end

endmodule
