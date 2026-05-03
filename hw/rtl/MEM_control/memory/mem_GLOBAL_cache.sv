`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "GLOBAL_CONST.svh"
`include "mem_IO.svh"

import isa_pkg::*;

// ===| Module: mem_GLOBAL_cache — L2 cache controller (URAM + ACP/NPU FSMs) |==
// Purpose      : Wrap mem_L2_cache_fmap (URAM) with the ACP DMA state machine
//                and the NPU compute access state machine, plus the ACP CDC
//                bridge (mem_BUFFER). Hides URAM port-A/port-B handshaking
//                from upstream. (Keller "deep module" boundary.)
// Spec ref     : pccx v002 §5.1, §5.4 (port arbitration), §5.5 (CDC).
// Clocks       : Dual-clock — clk_core (URAM, NPU FSM) + clk_axi (ACP-side
//                AXIS). CDC isolated to mem_BUFFER.
// Resets       : rst_n_core / rst_axi_n active-low.
//   Port A — ACP DMA : host DDR4 ↔ L2 via AXI-Stream (CDC via mem_BUFFER).
//   Port B — NPU     : compute engines (GEMM / GEMV / CVO) streaming R/W.
// Arbitration  : Port B is driven externally via IN_npu_* signals; the
//                upstream mem_dispatcher mux (final_npu_*) decides which
//                producer (NPU FSM vs CVO bridge) wins port B per cycle.
// Latency      : ACP read = URAM_LATENCY (3) cycles after acp_is_busy & ~we;
//                tracked via acp_rd_valid_pipe.
// Throughput   : 1 ACP transaction + 1 NPU transaction in flight in parallel.
// Address unit : 128-bit words (address 0 = first 128-bit line).
// Reset state  : both FSMs ST_IDLE-equivalent (acp_is_busy = 0, npu_is_busy = 0).
// Counters     : none.
// Assertions   : (Stage C) acp_ptr never exceeds acp_end_addr; the same for
//                npu_ptr / npu_end_addr.
// ===============================================================================

module mem_GLOBAL_cache (
    input logic clk_core,
    input logic rst_n_core,
    input logic clk_axi,
    input logic rst_axi_n,

    // ===| AXI-Stream ACP (external, AXI clock domain) |========================
    axis_if.slave  S_AXIS_ACP_FMAP,    // feature map in  (128-bit, from PS/DDR4)
    axis_if.master M_AXIS_ACP_RESULT,  // result out       (128-bit, to PS/DDR4)
    axis_if.master M_AXIS_NPU_FMAP,    // L2 -> preprocess fmap stream

    // ===| Port A — ACP DMA control |============================================
    input  logic        IN_acp_write_en,   // 1=write (DDR→L2), 0=read (L2→DDR)
    input  logic [16:0] IN_acp_base_addr,
    input  logic        IN_acp_rx_start,   // start ACP transfer
    input  logic [16:0] IN_acp_end_addr,
    output logic        OUT_acp_is_busy,

    // ===| Port B — NPU compute direct access |==================================
    input  logic         IN_npu_write_en,
    input  logic  [16:0] IN_npu_base_addr,
    input  logic         IN_npu_rx_start,
    input  logic  [16:0] IN_npu_end_addr,
    output logic         OUT_npu_is_busy,

    input  logic [127:0] IN_npu_wdata,
    output logic [127:0] OUT_npu_rdata
);

  // ===| ACP CDC FIFO (AXI → Core clock) |=======================================
  axis_if #(.DATA_WIDTH(128)) core_acp_rx_bus ();
  axis_if #(.DATA_WIDTH(128)) core_acp_tx_bus ();

  mem_BUFFER u_acp_cdc (
      .clk_core         (clk_core),
      .rst_n_core       (rst_n_core),
      .clk_axi          (clk_axi),
      .rst_axi_n        (rst_axi_n),
      .S_AXIS_ACP_FMAP  (S_AXIS_ACP_FMAP),
      .M_AXIS_ACP_RESULT(M_AXIS_ACP_RESULT),
      .M_CORE_ACP_RX    (core_acp_rx_bus),
      .S_CORE_ACP_TX    (core_acp_tx_bus)
  );

  // ===| Port A — ACP state machine (core clock domain) |=======================
  logic [16:0] acp_ptr;
  logic        acp_write_en;
  logic        acp_is_busy;
  logic [16:0] acp_end_addr;

  assign OUT_acp_is_busy = acp_is_busy;

  // ACP read pipeline: URAM READ_LATENCY=3
  logic [2:0] acp_rd_valid_pipe;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_rd_valid_pipe <= 3'b000;
    end else begin
      acp_rd_valid_pipe <= {acp_rd_valid_pipe[1:0], (acp_is_busy & ~acp_write_en)};
    end
  end

  assign core_acp_tx_bus.tvalid = acp_rd_valid_pipe[2];
  assign core_acp_tx_bus.tkeep  = '1;
  assign core_acp_tx_bus.tlast  = 1'b0;

  assign core_acp_rx_bus.tready = acp_is_busy & acp_write_en;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_ptr      <= '0;
      acp_end_addr <= '0;
      acp_is_busy  <= 1'b0;
      acp_write_en <= 1'b0;
    end else begin
      if (acp_is_busy) begin
        if (acp_write_en) begin
          if (core_acp_rx_bus.tvalid) begin
            acp_ptr <= acp_ptr + 17'd1;
            if (acp_ptr + 17'd1 >= acp_end_addr) acp_is_busy <= 1'b0;
          end
        end else begin
          if (core_acp_tx_bus.tready) begin
            acp_ptr <= acp_ptr + 17'd1;
            if (acp_ptr + 17'd1 >= acp_end_addr) acp_is_busy <= 1'b0;
          end
        end
      end else if (IN_acp_rx_start) begin
        acp_ptr      <= IN_acp_base_addr;
        acp_end_addr <= IN_acp_end_addr;
        acp_is_busy  <= 1'b1;
        acp_write_en <= IN_acp_write_en;
      end
    end
  end

  // ===| Port B — NPU state machine |============================================
  logic [16:0] npu_ptr;
  logic        npu_write_en;
  logic        npu_is_busy;
  logic [16:0] npu_end_addr;
  logic [2:0]  npu_rd_valid_pipe;
  logic [2:0]  npu_rd_last_pipe;
  logic        npu_read_fire;
  logic        npu_write_fire;

  assign OUT_npu_is_busy = npu_is_busy;
  assign npu_read_fire   = npu_is_busy & ~npu_write_en & M_AXIS_NPU_FMAP.tready;
  assign npu_write_fire  = npu_is_busy &  npu_write_en;

  assign M_AXIS_NPU_FMAP.tdata  = OUT_npu_rdata;
  assign M_AXIS_NPU_FMAP.tvalid = npu_rd_valid_pipe[2];
  assign M_AXIS_NPU_FMAP.tlast  = npu_rd_last_pipe[2];
  assign M_AXIS_NPU_FMAP.tkeep  = '1;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      npu_rd_valid_pipe <= 3'b000;
      npu_rd_last_pipe  <= 3'b000;
    end else begin
      npu_rd_valid_pipe <= {npu_rd_valid_pipe[1:0], npu_read_fire};
      npu_rd_last_pipe  <= {npu_rd_last_pipe[1:0], (npu_read_fire && (npu_ptr + 17'd1 >= npu_end_addr))};
    end
  end

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      npu_ptr      <= '0;
      npu_end_addr <= '0;
      npu_is_busy  <= 1'b0;
      npu_write_en <= 1'b0;
    end else begin
      if (npu_is_busy) begin
        if (npu_write_fire || npu_read_fire) begin
          npu_ptr <= npu_ptr + 17'd1;
          if (npu_ptr + 17'd1 >= npu_end_addr) npu_is_busy <= 1'b0;
        end
      end else if (IN_npu_rx_start) begin
        npu_ptr      <= IN_npu_base_addr;
        npu_end_addr <= IN_npu_end_addr;
        npu_is_busy  <= 1'b1;
        npu_write_en <= IN_npu_write_en;
      end
    end
  end

  // ===| L2 URAM (port B shared between ACP read-out and NPU compute) |=========
  // Port A → ACP DMA (write when host→L2, read when L2→host)
  // Port B → NPU compute (fmap broadcast, CVO streaming)
  mem_L2_cache_fmap #(
      .Depth(114688)
  ) u_l2_uram (
      .clk_core    (clk_core),
      .rst_n_core  (rst_n_core),

      // Port A — ACP
      .IN_acp_we   (acp_write_en & core_acp_rx_bus.tvalid),
      .IN_acp_addr (acp_ptr),
      .IN_acp_wdata(core_acp_rx_bus.tdata),
      .OUT_acp_rdata(core_acp_tx_bus.tdata),

      // Port B — NPU compute
      .IN_npu_we    (npu_write_fire),
      .IN_npu_addr  (npu_ptr),
      .IN_npu_wdata (IN_npu_wdata),
      .OUT_npu_rdata(OUT_npu_rdata)
  );

endmodule
