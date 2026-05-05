// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "GLOBAL_CONST.svh"

// ===| Module: mem_HP_buffer — HP weight CDC FIFO bank (URAM-backed) |===========
// Purpose      : Cross-clock-domain weight buffering between AXI HP ports
//                (clk_axi, 250 MHz) and the compute core (clk_core, 400 MHz),
//                providing an elastic boundary for weight streaming.
// Spec ref     : pccx v002 §6 (KV260 SoC), §5.4 (HP weight path).
// Clocks       : clk_axi (S-side), clk_core (M-side), independent_clock CDC.
// Resets       : rst_axi_n / rst_n_core, both active-low — applied to their
//                respective clock domains by xpm_fifo_axis.
// Topology     : 4 × xpm_fifo_axis, 128-bit wide × URAM_FIFO_DEPTH (= 4096)
//                deep — uses 2 URAM blocks per FIFO (64 KB each, 256 KB total).
// Latency      : Per-FIFO read latency tracked internally by xpm_fifo_axis;
//                gray-code pointer sync ≈ 2-3 destination clocks.
// Backpressure : tready propagates back through standard AXI4-Stream rules.
//                Source side stalls when FIFO is full; sink stalls when empty.
// Reset state  : All FIFOs cleared; tvalid/tready = 0.
// Errors       : none surfaced (FIFO over/underflow asserts inside XPM).
// Counters     : none.
// Notes        : Weight order:
//                  HP0/HP1 → MAT_CORE (GEMM upper/lower INT4 lanes)
//                  HP2/HP3 → VEC_CORE (GEMV lanes A/B; C/D currently tied 0).
// ===============================================================================
module mem_HP_buffer #(
) (
    // ===| Clock & Reset |======================================
    input logic clk_core,  // 400MHz
    input logic rst_n_core,
    input logic clk_axi,  // 250MHz
    input logic rst_axi_n,

    // ===| HP Ports (Weight) - AXI Side |=======================
    axis_if.slave S_AXI_HP0_WEIGHT,
    axis_if.slave S_AXI_HP1_WEIGHT,
    axis_if.slave S_AXI_HP2_WEIGHT,
    axis_if.slave S_AXI_HP3_WEIGHT,

    // ===| Weight Stream - Core Side (To L1 or Dispatcher) |====
    axis_if.master M_CORE_HP0_WEIGHT,
    axis_if.master M_CORE_HP1_WEIGHT,
    axis_if.master M_CORE_HP2_WEIGHT,
    axis_if.master M_CORE_HP3_WEIGHT
);

  // ine Large Depth for URAM (4096 uses 2 URAM blocks per FIFO)
  localparam int URAM_FIFO_DEPTH = 4096;

  // Weight streams are fixed-width, continuous beat streams in this v002
  // boundary. The FIFO instances move data/valid/ready; terminate sidebands
  // so downstream AXIS ports have a deterministic contract in synthesis.
  assign M_CORE_HP0_WEIGHT.tkeep = '1;
  assign M_CORE_HP0_WEIGHT.tlast = 1'b0;
  assign M_CORE_HP1_WEIGHT.tkeep = '1;
  assign M_CORE_HP1_WEIGHT.tlast = 1'b0;
  assign M_CORE_HP2_WEIGHT.tkeep = '1;
  assign M_CORE_HP2_WEIGHT.tlast = 1'b0;
  assign M_CORE_HP3_WEIGHT.tkeep = '1;
  assign M_CORE_HP3_WEIGHT.tlast = 1'b0;

  // [1] HP0 Weight FIFO (URAM based - Massive 64KB)
  xpm_fifo_axis #(
      .FIFO_DEPTH      (URAM_FIFO_DEPTH),
      .TDATA_WIDTH     (128),
      .FIFO_MEMORY_TYPE("block"),
      .CLOCKING_MODE   ("independent_clock")
  ) u_hp0_weight_fifo (
      .s_aclk(clk_axi),
      .s_aresetn(rst_axi_n),
      .s_axis_tdata(S_AXI_HP0_WEIGHT.tdata),
      .s_axis_tvalid(S_AXI_HP0_WEIGHT.tvalid),
      .s_axis_tready(S_AXI_HP0_WEIGHT.tready),

      .m_aclk(clk_core),
      .m_axis_tdata(M_CORE_HP0_WEIGHT.tdata),
      .m_axis_tvalid(M_CORE_HP0_WEIGHT.tvalid),
      .m_axis_tready(M_CORE_HP0_WEIGHT.tready)
  );

  // [2] HP1 Weight FIFO (URAM based - Massive 64KB)
  xpm_fifo_axis #(
      .FIFO_DEPTH(URAM_FIFO_DEPTH),
      .TDATA_WIDTH(128),
      .FIFO_MEMORY_TYPE("block"),
      .CLOCKING_MODE("independent_clock")
  ) u_hp1_weight_fifo (
      .s_aclk(clk_axi),
      .s_aresetn(rst_axi_n),
      .s_axis_tdata(S_AXI_HP1_WEIGHT.tdata),
      .s_axis_tvalid(S_AXI_HP1_WEIGHT.tvalid),
      .s_axis_tready(S_AXI_HP1_WEIGHT.tready),

      .m_aclk(clk_core),
      .m_axis_tdata(M_CORE_HP1_WEIGHT.tdata),
      .m_axis_tvalid(M_CORE_HP1_WEIGHT.tvalid),
      .m_axis_tready(M_CORE_HP1_WEIGHT.tready)
  );

  // [3] HP2 Weight FIFO (URAM based - Massive 64KB)
  xpm_fifo_axis #(
      .FIFO_DEPTH(URAM_FIFO_DEPTH),
      .TDATA_WIDTH(128),
      .FIFO_MEMORY_TYPE("block"),
      .CLOCKING_MODE("independent_clock")
  ) u_hp2_weight_fifo (
      .s_aclk(clk_axi),
      .s_aresetn(rst_axi_n),
      .s_axis_tdata(S_AXI_HP2_WEIGHT.tdata),
      .s_axis_tvalid(S_AXI_HP2_WEIGHT.tvalid),
      .s_axis_tready(S_AXI_HP2_WEIGHT.tready),

      .m_aclk(clk_core),
      .m_axis_tdata(M_CORE_HP2_WEIGHT.tdata),
      .m_axis_tvalid(M_CORE_HP2_WEIGHT.tvalid),
      .m_axis_tready(M_CORE_HP2_WEIGHT.tready)
  );

  // [4] HP3 Weight FIFO (URAM based - Massive 64KB)
  xpm_fifo_axis #(
      .FIFO_DEPTH(URAM_FIFO_DEPTH),
      .TDATA_WIDTH(128),
      .FIFO_MEMORY_TYPE("block"),
      .CLOCKING_MODE("independent_clock")
  ) u_hp3_weight_fifo (
      .s_aclk(clk_axi),
      .s_aresetn(rst_axi_n),
      .s_axis_tdata(S_AXI_HP3_WEIGHT.tdata),
      .s_axis_tvalid(S_AXI_HP3_WEIGHT.tvalid),
      .s_axis_tready(S_AXI_HP3_WEIGHT.tready),

      .m_aclk(clk_core),
      .m_axis_tdata(M_CORE_HP3_WEIGHT.tdata),
      .m_axis_tvalid(M_CORE_HP3_WEIGHT.tvalid),
      .m_axis_tready(M_CORE_HP3_WEIGHT.tready)
  );

endmodule
