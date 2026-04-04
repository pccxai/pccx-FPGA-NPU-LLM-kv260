`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

module ctrl_npu_frontend (
    input logic clk,
    input logic rst_n,
    input logic IN_clear,

    // AXI4-Lite Slave : PS ↔ NPU control plane
    axil_if.slave S_AXIL_CTRL,

    // Control from Brain
    input logic IN_rd_start,

    // Decoded command → Dispatcher / FSM
    output logic [`ISA_WIDTH-1:0] OUT_RAW_instruction,
    output logic                  OUT_kick,

    // Status ← Encoder / FSM
    input logic [`ISA_WIDTH:0] IN_enc_stat,
    input logic                IN_enc_valid

    input logic  IN_fetch_ready;
);

  /*─────────────────────────────────────────────
  Internal wires : AXIL_CMD_IN ↔ upper logic
  ───────────────────────────────────────────────*/
  logic [`ISA_WIDTH-1:0] cmd_data;
  logic                  cmd_valid;
  logic                  decoder_ready;

  assign  IN_fetch_ready =  IN_fetch_ready;  // FSM 붙이면 교체
  assign OUT_RAW_instruction = cmd_data;
  assign OUT_kick            = cmd_valid &  IN_fetch_ready;

  /*─────────────────────────────────────────────
  [1-2] Communication IN : CPU → NPU
  ───────────────────────────────────────────────*/
  AXIL_CMD_IN #(
      .FIFO_DEPTH(8)
  ) u_cmd_in (
      .clk    (clk),
      .rst_n  (rst_n),
      .IN_clear(i_clear),

      // AXI4-Lite Write channels (인터페이스에서 풀어서 연결)
      .s_awaddr (S_AXIL_CTRL.awaddr),
      .s_awvalid(S_AXIL_CTRL.awvalid),
      .s_awready(S_AXIL_CTRL.awready),
      .s_wdata  (S_AXIL_CTRL.wdata),
      .s_wvalid (S_AXIL_CTRL.wvalid),
      .s_wready (S_AXIL_CTRL.wready),
      .s_bresp  (S_AXIL_CTRL.bresp),
      .s_bvalid (S_AXIL_CTRL.bvalid),
      .s_bready (S_AXIL_CTRL.bready),

      .OUT_data(cmd_data),
      .OUT_valid(cmd_valid),
      .IN_decoder_ready( IN_fetch_ready)
  );

  /*─────────────────────────────────────────────
  [1-2] Communication OUT : NPU → CPU
  ───────────────────────────────────────────────*/
  AXIL_STAT_OUT #(
      .FIFO_DEPTH(8)
  ) u_stat_out (
      .clk    (clk),
      .rst_n  (rst_n),
      .IN_clear(i_clear),

      .IN_data (i_enc_stat),
      .IN_valid(i_enc_valid),

      // AXI4-Lite Read channels
      .s_araddr (S_AXIL_CTRL.araddr),
      .s_arvalid(S_AXIL_CTRL.arvalid),
      .s_arready(S_AXIL_CTRL.arready),
      .s_rdata  (S_AXIL_CTRL.rdata),
      .s_rresp  (S_AXIL_CTRL.rresp),
      .s_rvalid (S_AXIL_CTRL.rvalid),
      .s_rready (S_AXIL_CTRL.rready)
  );

endmodule
