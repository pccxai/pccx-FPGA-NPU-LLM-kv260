`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

// ===| Module: ctrl_npu_frontend — AXIL command/status frontend |===============
// Purpose      : Bridge AXI-Lite control plane to internal raw-instruction
//                bus and surface status words back to the host.
// Spec ref     : pccx v002 §4 (control plane), §3 (ISA).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low; IN_clear synchronous soft-clear.
// Submodules   : AXIL_CMD_IN  (FIFO_DEPTH=8) — write-channel command path.
//                AXIL_STAT_OUT (FIFO_DEPTH=8) — read-channel status path.
// Latency      : 1 AXIL handshake → cmd FIFO push → 1 cycle to OUT_RAW_instruction
//                while OUT_kick = (cmd_valid & IN_fetch_ready).
// Throughput   : Up to 1 AXIL transaction/cycle limited by AXIL bresp.
// Handshake    : OUT_kick is single-cycle pulse — downstream decoder must
//                latch on rising edge.
// Reset state  : OUT_RAW_instruction = 0; OUT_kick = 0.
// Errors       : AXIL bresp/rresp surface 2'b00 (OKAY) only — error mapping
//                deferred to the FSM stat encoder.
// Counters     : none.
// ===============================================================================
module ctrl_npu_frontend (
    input logic clk,
    input logic rst_n,
    input logic IN_clear,

    // AXI4-Lite Slave : PS <-> NPU control plane
    axil_if.slave S_AXIL_CTRL,

    // Control from Brain
    //input logic IN_rd_start,

    // Decoded command -> Dispatcher / FSM
    output logic [`ISA_WIDTH-1:0] OUT_RAW_instruction,
    output logic                  OUT_kick,

    // Status <- Encoder / FSM
    input logic [`ISA_WIDTH-1:0] IN_enc_stat,
    input logic                  IN_enc_valid,

    input logic IN_fetch_ready  // FIXED: Removed illegal semicolon
);

  /*─────────────────────────────────────────────
  Internal wires : AXIL_CMD_IN <-> upper logic
  ───────────────────────────────────────────────*/
  logic [`ISA_WIDTH-1:0] cmd_data;
  logic                  cmd_valid;
  // logic               decoder_ready; // (Unused wire commented out)

  // FIXED: Removed 'assign IN_fetch_ready = IN_fetch_ready;'
  // (You cannot continuously assign an input to itself in SystemVerilog)

  assign OUT_RAW_instruction = cmd_data;
  assign OUT_kick            = cmd_valid & IN_fetch_ready;

  /*─────────────────────────────────────────────
  [1-2] Communication IN : CPU -> NPU (Using Write Channels)
  ───────────────────────────────────────────────*/
  AXIL_CMD_IN #(
      .FIFO_DEPTH(8)
  ) u_cmd_in (
      .clk     (clk),
      .rst_n   (rst_n),
      .IN_clear(IN_clear), // FIXED: Typo i_clear -> IN_clear

      // AXI4-Lite Write channels directly routed from the interface
      .s_awaddr (S_AXIL_CTRL.awaddr),
      .s_awprot (S_AXIL_CTRL.awprot),
      .s_awvalid(S_AXIL_CTRL.awvalid),
      .s_awready(S_AXIL_CTRL.awready),
      .s_wdata  (S_AXIL_CTRL.wdata),
      .s_wstrb  (S_AXIL_CTRL.wstrb),
      .s_wvalid (S_AXIL_CTRL.wvalid),
      .s_wready (S_AXIL_CTRL.wready),
      .s_bresp  (S_AXIL_CTRL.bresp),
      .s_bvalid (S_AXIL_CTRL.bvalid),
      .s_bready (S_AXIL_CTRL.bready),

      .OUT_data(cmd_data),
      .OUT_valid(cmd_valid),
      .IN_decoder_ready(IN_fetch_ready)
  );

  /*─────────────────────────────────────────────
  [1-2] Communication OUT : NPU -> CPU (Using Read Channels)
  ───────────────────────────────────────────────*/
  AXIL_STAT_OUT #(
      .FIFO_DEPTH(8)
  ) u_stat_out (
      .clk     (clk),
      .rst_n   (rst_n),
      .IN_clear(IN_clear), // FIXED: Typo i_clear -> IN_clear

      .IN_data (IN_enc_stat),  // FIXED: Typo i_enc_stat -> IN_enc_stat
      .IN_valid(IN_enc_valid), // FIXED: Typo i_enc_valid -> IN_enc_valid

      // AXI4-Lite Read channels directly routed from the interface
      .s_araddr (S_AXIL_CTRL.araddr),
      .s_arvalid(S_AXIL_CTRL.arvalid),
      .s_arready(S_AXIL_CTRL.arready),
      .s_rdata  (S_AXIL_CTRL.rdata),
      .s_rresp  (S_AXIL_CTRL.rresp),
      .s_rvalid (S_AXIL_CTRL.rvalid),
      .s_rready (S_AXIL_CTRL.rready)
  );

endmodule
