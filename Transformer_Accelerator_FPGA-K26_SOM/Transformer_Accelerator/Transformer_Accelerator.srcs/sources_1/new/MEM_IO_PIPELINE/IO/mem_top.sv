`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module mem_top (
    input logic clk_core,    // 400MHz
    input logic rst_n_core,
    input logic clk_axi,     // 250MHz
    input logic rst_axi_n,

    // ===| External AXI-Stream (From PS & DDR4) |=================
    axis_if.slave S_AXI_HP0_WEIGHT,
    axis_if.slave S_AXI_HP1_WEIGHT,
    axis_if.slave S_AXI_HP2_WEIGHT,
    axis_if.slave S_AXI_HP3_WEIGHT,

    axis_if.slave  S_AXIS_ACP_FMAP,
    axis_if.master M_AXIS_ACP_RESULT, // TO ps.

    // ===| Weight Pipeline Control (To/From Dispatcher) |=============
    input  logic [`ADDR_WIDTH_L2-1:0] IN_read_addr_hp [0:3],
    input  logic                      IN_read_en_hp   [0:3],
    output logic [             127:0] OUT_read_data_hp[0:3],

    // ===| L2 cache Pipeline Control (To/From Dispatcher)(KV,FMAP) |==
    // ACP (External) Memory Map Control
    // Dispatcher tells where to store incoming FMAP
    input logic        IN_acp_write_en,   // A port mod to write or read
    input logic [16:0] IN_acp_base_addr,
    input logic        IN_acp_rx_start,   // Trigger to accept ACP data
    input logic [16:0] IN_acp_end_addr,

    output logic OUT_acp_is_busy,
    // [127:0] IN_npu_rdata   = connected to acp
    // [127:0] OUT_npu_rdata  = connected to acp

    // NPU (Internal) Compute Access (Port B)
    input logic        IN_npu_write_en,
    input logic [16:0] IN_npu_base_addr,
    input logic        IN_npu_rx_start,
    input logic [16:0] IN_npu_end_addr,

    output logic OUT_npu_is_busy,

    input  logic [127:0] IN_npu_wdata,
    output logic [127:0] OUT_npu_rdata,

    input ptr_addr_t IN_fmap_shape_read_address
);

  // 1. Internal AXIS Interfaces (400MHz Domain)
  axis_if #(.DATA_WIDTH(128)) core_hp_bus[0:3] ();
  axis_if #(.DATA_WIDTH(128)) core_acp_rx_bus ();
  axis_if #(.DATA_WIDTH(128)) core_acp_tx_bus ();

  // 2. mem_BUFFER Instantiation
  mem_BUFFER u_buffer (
      .clk_core(clk_core),
      .rst_n_core(rst_n_core),
      .clk_axi(clk_axi),
      .rst_axi_n(rst_axi_n),

      .S_AXI_HP0_WEIGHT (S_AXI_HP0_WEIGHT),
      .M_CORE_HP0_WEIGHT(core_hp_bus[0]),
      .S_AXI_HP1_WEIGHT (S_AXI_HP1_WEIGHT),
      .M_CORE_HP1_WEIGHT(core_hp_bus[1]),
      .S_AXI_HP2_WEIGHT (S_AXI_HP2_WEIGHT),
      .M_CORE_HP2_WEIGHT(core_hp_bus[2]),
      .S_AXI_HP3_WEIGHT (S_AXI_HP3_WEIGHT),
      .M_CORE_HP3_WEIGHT(core_hp_bus[3]),


      // [RX] Data from DDR4 to NPU
      .S_AXIS_ACP_FMAP(S_AXIS_ACP_FMAP),

      // [TX] Data from NPU to DDR4
      .M_CORE_ACP_RX(core_acp_rx_bus),

      // [RX] Converted to 400MHz Core
      .M_AXIS_ACP_RESULT(M_AXIS_ACP_RESULT),

      // [TX] Coming from 400MHz Core
      .S_CORE_ACP_TX(core_acp_tx_bus)
  );
  logic [16:0] npu_ptr;
  logic        npu_write_en;
  logic        npu_is_busy;
  assign OUT_npu_is_busy = npu_is_busy;
  logic [16:0] npu_end_addr;


  // NPU port : WRITE
  // NPU port : READ
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      npu_ptr <= '0;
      npu_end_addr <= '0;
      npu_is_busy <= 1'b0;
    end else begin
      if (npu_is_busy == 1) begin

        // + 32(fmapsize * 32) = 128
        npu_ptr <= npu_ptr + 128;

        if (npu_ptr + 128 >= npu_end_addr) begin
          npu_is_busy <= 0;
        end

      end else if (IN_npu_rx_start) begin
        npu_ptr <= IN_npu_base_addr;
        npu_is_busy <= 1;
        npu_end_addr <= IN_npu_end_addr;

        if (IN_npu_write_en == `PORT_MOD_E_WRITE) begin
          // MOD write
          npu_write_en <= `PORT_MOD_E_WRITE;

        end else begin
          // MOD read
          npu_write_en <= `PORT_MOD_E_READ;

        end
      end
    end
  end

  logic [16:0] acp_ptr;
  logic        acp_write_en;
  logic        acp_is_busy;
  assign OUT_acp_is_busy = acp_is_busy;
  logic [16:0] acp_end_addr;

  // ===| ACP Read Valid Pipeline |===
  // URAM READ_LATENCY_A = 3
  // read request → after 3-clk tdata valid → tvalid is 1
  logic [ 2:0] acp_rd_valid_pipe;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_rd_valid_pipe <= 3'b000;
    end else begin
      // [0] = READ ?
      // [1] = 1 clk after
      // [2] = 2 clk after → if this is HIGH data -> tdata
      acp_rd_valid_pipe <= {acp_rd_valid_pipe[1:0], (acp_is_busy & ~acp_write_en)};
    end
  end

  // tvalid: is true after URAM 3-clk latency
  assign core_acp_tx_bus.tvalid = acp_rd_valid_pipe[2];
  assign core_acp_tx_bus.tkeep  = '1;
  assign core_acp_tx_bus.tlast  = 1'b0;
  // tdata is connected directly to  mem_L2_cache instance port

  // WRITE MODE: rx bus - tready
  // acp_is_busy & acp_write_en
  assign core_acp_rx_bus.tready = acp_is_busy & acp_write_en;

  // ACP port : WRITE / READ
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_ptr      <= '0;
      acp_end_addr <= '0;
      acp_is_busy  <= 1'b0;
      acp_write_en <= 1'b0;
    end else begin
      if (acp_is_busy) begin

        if (acp_write_en) begin
          // ===| WRITE MOD |===
          if (core_acp_rx_bus.tvalid) begin
            acp_ptr <= acp_ptr + 17'd128;
            if (acp_ptr + 17'd128 >= acp_end_addr) acp_is_busy <= 1'b0;
          end

        end else begin
          // ===| READ MOD |===
          if (core_acp_tx_bus.tready) begin
            acp_ptr <= acp_ptr + 17'd128;
            if (acp_ptr + 17'd128 >= acp_end_addr) acp_is_busy <= 1'b0;
          end
        end

      end else if (IN_acp_rx_start) begin
        acp_ptr      <= IN_acp_base_addr;
        acp_is_busy  <= 1'b1;
        acp_end_addr <= IN_acp_end_addr;
        acp_write_en <= IN_acp_write_en;
      end
    end
  end


  // 5. FMAP & KV Cache (Massive 1.75MB URAM)
  mem_L2_cache #(
      .URAM_DEPTH(114688)
  ) u_fmap_kv_l2 (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),

      // Port A (ACP Side)
      // WRITE when acp_write_en & rx_valid
      .IN_acp_we    (acp_write_en & core_acp_rx_bus.tvalid),
      .IN_acp_addr  (acp_ptr),
      .IN_acp_wdata (core_acp_rx_bus.tdata),
      .OUT_acp_rdata(core_acp_tx_bus.tdata),

      // Port B (NPU Compute Side)
      .IN_npu_we    (npu_write_en),
      .IN_npu_addr  (npu_ptr),
      .IN_npu_wdata (IN_npu_wdata),
      .OUT_npu_rdata(OUT_npu_rdata)
  );

endmodule
