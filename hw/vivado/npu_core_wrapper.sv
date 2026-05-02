`timescale 1ns / 1ps

// ===============================================================================
// Module: npu_core_wrapper
// Purpose
// -------
//   Convert `NPU_top`'s SystemVerilog-interface ports into plain AXI4-Lite
//   and AXI4-Stream signal bundles so the core can be packaged as a
//   Vivado IP and dropped into a Block Design alongside the Zynq PS.
//
//   NPU_top itself uses `axil_if` / `axis_if` (see `npu_interfaces.svh`).
//   The BD IP packager accepts plain signal ports much more readily; this
//   wrapper does the one-to-one expansion so the BD / Zynq MPSoC sees
//   normal AXI4 interfaces.
//
//   Kept intentionally thin — no registers, no CDC — just signal wiring.
// ===============================================================================

`include "npu_interfaces.svh"

// AXI-Lite parameter values must match the `axil_if` defaults in
// `npu_interfaces.svh` (ADDR_W=12, DATA_W=64). The Zynq PS HPM is 32-bit
// AXI-Lite; the BD inserts a Smart Connect to convert PS 32→IP 64 width.
module npu_core_wrapper #(
  parameter int AXIL_ADDR_W  = 12,
  parameter int AXIL_DATA_W  = 64,
  parameter int HP_DATA_W    = 128,
  parameter int ACP_DATA_W   = 128
) (
  // ===| Clocks and resets |==================================================
  input  logic clk_core,
  input  logic rst_n_core,
  input  logic clk_axi,
  input  logic rst_axi_n,
  input  logic i_clear,

  // ===| S_AXIL_CTRL (AXI4-Lite slave) |======================================
  input  logic [AXIL_ADDR_W-1:0]  s_axil_awaddr,
  input  logic                    s_axil_awvalid,
  output logic                    s_axil_awready,
  input  logic [AXIL_DATA_W-1:0]  s_axil_wdata,
  input  logic [AXIL_DATA_W/8-1:0] s_axil_wstrb,
  input  logic                    s_axil_wvalid,
  output logic                    s_axil_wready,
  output logic [1:0]              s_axil_bresp,
  output logic                    s_axil_bvalid,
  input  logic                    s_axil_bready,
  input  logic [AXIL_ADDR_W-1:0]  s_axil_araddr,
  input  logic                    s_axil_arvalid,
  output logic                    s_axil_arready,
  output logic [AXIL_DATA_W-1:0]  s_axil_rdata,
  output logic [1:0]              s_axil_rresp,
  output logic                    s_axil_rvalid,
  input  logic                    s_axil_rready,

  // ===| S_AXI_HP0..3_WEIGHT (AXIS slave, 128-bit each) |====================
  input  logic [HP_DATA_W-1:0] s_axis_hp0_tdata,
  input  logic                 s_axis_hp0_tvalid,
  output logic                 s_axis_hp0_tready,
  input  logic [HP_DATA_W-1:0] s_axis_hp1_tdata,
  input  logic                 s_axis_hp1_tvalid,
  output logic                 s_axis_hp1_tready,
  input  logic [HP_DATA_W-1:0] s_axis_hp2_tdata,
  input  logic                 s_axis_hp2_tvalid,
  output logic                 s_axis_hp2_tready,
  input  logic [HP_DATA_W-1:0] s_axis_hp3_tdata,
  input  logic                 s_axis_hp3_tvalid,
  output logic                 s_axis_hp3_tready,

  // ===| ACP FMap (AXIS slave) + Result (AXIS master) |======================
  input  logic [ACP_DATA_W-1:0] s_axis_acp_fmap_tdata,
  input  logic                  s_axis_acp_fmap_tvalid,
  output logic                  s_axis_acp_fmap_tready,

  output logic [ACP_DATA_W-1:0] m_axis_acp_result_tdata,
  output logic                  m_axis_acp_result_tvalid,
  input  logic                  m_axis_acp_result_tready
);

  // ---------------------------------------------------------------------------
  // Interface instances — one per NPU_top port
  // ---------------------------------------------------------------------------
  axil_if #(.ADDR_W(AXIL_ADDR_W), .DATA_W(AXIL_DATA_W)) axil_inst (
    .clk  (clk_axi),
    .rst_n(rst_axi_n)
  );
  axis_if #(.DATA_WIDTH(HP_DATA_W))  hp0_inst ();
  axis_if #(.DATA_WIDTH(HP_DATA_W))  hp1_inst ();
  axis_if #(.DATA_WIDTH(HP_DATA_W))  hp2_inst ();
  axis_if #(.DATA_WIDTH(HP_DATA_W))  hp3_inst ();
  axis_if #(.DATA_WIDTH(ACP_DATA_W)) acp_fmap_inst ();
  axis_if #(.DATA_WIDTH(ACP_DATA_W)) acp_result_inst ();

  // ---------------------------------------------------------------------------
  // AXI-Lite bundle → interface
  // ---------------------------------------------------------------------------
  assign axil_inst.awaddr  = s_axil_awaddr;
  assign axil_inst.awvalid = s_axil_awvalid;
  assign s_axil_awready    = axil_inst.awready;
  assign axil_inst.wdata   = s_axil_wdata;
  assign axil_inst.wstrb   = s_axil_wstrb;
  assign axil_inst.wvalid  = s_axil_wvalid;
  assign s_axil_wready     = axil_inst.wready;
  assign s_axil_bresp      = axil_inst.bresp;
  assign s_axil_bvalid     = axil_inst.bvalid;
  assign axil_inst.bready  = s_axil_bready;
  assign axil_inst.araddr  = s_axil_araddr;
  assign axil_inst.arvalid = s_axil_arvalid;
  assign s_axil_arready    = axil_inst.arready;
  assign s_axil_rdata      = axil_inst.rdata;
  assign s_axil_rresp      = axil_inst.rresp;
  assign s_axil_rvalid     = axil_inst.rvalid;
  assign axil_inst.rready  = s_axil_rready;

  // ---------------------------------------------------------------------------
  // AXIS HP weight bundles (x4)
  // ---------------------------------------------------------------------------
  assign hp0_inst.tdata  = s_axis_hp0_tdata;
  assign hp0_inst.tvalid = s_axis_hp0_tvalid;
  assign s_axis_hp0_tready = hp0_inst.tready;
  assign hp1_inst.tdata  = s_axis_hp1_tdata;
  assign hp1_inst.tvalid = s_axis_hp1_tvalid;
  assign s_axis_hp1_tready = hp1_inst.tready;
  assign hp2_inst.tdata  = s_axis_hp2_tdata;
  assign hp2_inst.tvalid = s_axis_hp2_tvalid;
  assign s_axis_hp2_tready = hp2_inst.tready;
  assign hp3_inst.tdata  = s_axis_hp3_tdata;
  assign hp3_inst.tvalid = s_axis_hp3_tvalid;
  assign s_axis_hp3_tready = hp3_inst.tready;

  // ---------------------------------------------------------------------------
  // ACP FMap (slave) + Result (master)
  // ---------------------------------------------------------------------------
  assign acp_fmap_inst.tdata   = s_axis_acp_fmap_tdata;
  assign acp_fmap_inst.tvalid  = s_axis_acp_fmap_tvalid;
  assign s_axis_acp_fmap_tready = acp_fmap_inst.tready;

  assign m_axis_acp_result_tdata  = acp_result_inst.tdata;
  assign m_axis_acp_result_tvalid = acp_result_inst.tvalid;
  assign acp_result_inst.tready   = m_axis_acp_result_tready;

  // ---------------------------------------------------------------------------
  // Instantiate the core
  // ---------------------------------------------------------------------------
  NPU_top u_npu_top (
    .clk_core          (clk_core),
    .rst_n_core        (rst_n_core),
    .clk_axi           (clk_axi),
    .rst_axi_n         (rst_axi_n),
    .i_clear           (i_clear),
    .S_AXIL_CTRL       (axil_inst.slave),
    .S_AXI_HP0_WEIGHT  (hp0_inst.slave),
    .S_AXI_HP1_WEIGHT  (hp1_inst.slave),
    .S_AXI_HP2_WEIGHT  (hp2_inst.slave),
    .S_AXI_HP3_WEIGHT  (hp3_inst.slave),
    .S_AXIS_ACP_FMAP   (acp_fmap_inst.slave),
    .M_AXIS_ACP_RESULT (acp_result_inst.master)
  );

endmodule
