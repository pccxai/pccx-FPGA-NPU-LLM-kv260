`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"
`include "npu_interfaces.svh"

module Instruction_decoder (
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // AXI4-Lite Slave : PS talks to NPU
    axil_if.slave S_AXIL_CTRL,

    // From Instruction_encoder : NPU status to report back to PS
    input logic [31:0] i_enc_stat,  // encoded status word from encoder

    // Decoded command output : goes into NPU internals
    output logic [31:0] o_instruction,
    output logic        o_kick          // 1-cycle fire pulse
);

  /*─────────────────────────────────────────────
  Register Address Map
  ───────────────────────────────────────────────*/
  localparam ADDR_INST = 12'h000;  // W : instruction from PS
  localparam ADDR_KICK = 12'h004;  // W : write anything to fire NPU
  localparam ADDR_STAT = 12'h008;  // R : NPU status (from encoder)

  /*─────────────────────────────────────────────
  1. AXI4-Lite Write Path  (PS → NPU)
  AW and W channels are independent.
  Latch AW first, then write register when W arrives.
  ───────────────────────────────────────────────*/
  logic [11:0] aw_addr_latch;
  logic        aw_pending;  // AW received, W not yet
  logic        bvalid_r;

  assign S_AXIL_CTRL.awready = ~aw_pending;
  assign S_AXIL_CTRL.wready  = aw_pending;
  assign S_AXIL_CTRL.bresp   = 2'b00;  // always OKAY
  assign S_AXIL_CTRL.bvalid  = bvalid_r;

  // AW latch
  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      aw_addr_latch <= '0;
      aw_pending    <= 1'b0;
    end else begin
      if (S_AXIL_CTRL.awvalid && S_AXIL_CTRL.awready) begin
        aw_addr_latch <= S_AXIL_CTRL.awaddr;
        aw_pending    <= 1'b1;
      end
      if (S_AXIL_CTRL.wvalid && S_AXIL_CTRL.wready) begin
        aw_pending <= 1'b0;
      end
    end
  end

  // W : write register + B : send response
  always_ff @(posedge clk) begin
    o_kick   <= 1'b0;  // 1-cycle pulse
    bvalid_r <= 1'b0;

    if (!rst_n || i_clear) begin
      o_instruction <= '0;
    end else begin
      if (S_AXIL_CTRL.wvalid && S_AXIL_CTRL.wready) begin
        case (aw_addr_latch)
          ADDR_INST: o_instruction <= S_AXIL_CTRL.wdata;
          ADDR_KICK: o_kick <= 1'b1;
          default:   ;
        endcase
        bvalid_r <= 1'b1;  // notify PS that write is done
      end
      // hold bvalid until PS acknowledges with bready
      if (bvalid_r && S_AXIL_CTRL.bready) bvalid_r <= 1'b0;
    end
  end

  /*─────────────────────────────────────────────
  2. AXI4-Lite Read Path  (NPU → PS)
  Data comes from Instruction_encoder (i_enc_stat).
  Hold rvalid until PS consumes it with rready.
  ───────────────────────────────────────────────*/
  logic [31:0] rdata_r;
  logic        rvalid_r;

  assign S_AXIL_CTRL.rdata   = rdata_r;
  assign S_AXIL_CTRL.rresp   = 2'b00;
  assign S_AXIL_CTRL.rvalid  = rvalid_r;
  assign S_AXIL_CTRL.arready = ~rvalid_r;  // block new reads while responding

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      rdata_r  <= '0;
      rvalid_r <= 1'b0;
    end else begin
      if (S_AXIL_CTRL.arvalid && S_AXIL_CTRL.arready) begin
        rvalid_r <= 1'b1;
        case (S_AXIL_CTRL.araddr)
          ADDR_STAT: rdata_r <= i_enc_stat;  // pass encoder output to PS
          default:   rdata_r <= '0;
        endcase
      end
      if (rvalid_r && S_AXIL_CTRL.rready) rvalid_r <= 1'b0;
    end
  end

endmodule
