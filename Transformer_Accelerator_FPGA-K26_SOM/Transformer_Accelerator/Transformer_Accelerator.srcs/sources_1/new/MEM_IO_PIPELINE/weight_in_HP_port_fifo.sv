`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"


module weight_in_HP_port_fifo #(
    parameter DATA_WIDTH = `HP_PORT_SINGLE_WIDTH
)(
    input logic clk,
    input logic rst_n,

    axis_if.slave IN_HP0,
    axis_if.slave IN_HP1,
    axis_if.slave IN_HP2,
    axis_if.slave IN_HP3,

    output logic [DATA_WIDTH-1:0] weight_fifo_data [0:`AXI_WEIGHT_PORT_CNT-1],
    output logic                  weight_fifo_valid[0:`AXI_WEIGHT_PORT_CNT-1],
    output logic                  weight_fifo_ready[0:`AXI_WEIGHT_PORT_CNT-1]
);
    // ===| HP0 port |==========================
      xpm_fifo_axis #(
          .FIFO_DEPTH(`XPM_FIFO_DEPTH),
          .TDATA_WIDTH(`AXI_DATA_WIDTH),
          .FIFO_MEMORY_TYPE("block"),
          .CLOCKING_MODE("common_clock")
      ) u_w_fifo0 (
          .s_aclk(clk),
          .s_aresetn(rst_n),
          .s_axis_tdata(IN_HP0.tdata),
          .s_axis_tvalid(IN_HP0.tvalid),
          .s_axis_tready(IN_HP0.tready),
          .m_axis_tdata(weight_fifo_data[0]),
          .m_axis_tvalid(weight_fifo_valid[0]),
          .m_axis_tready(weight_fifo_ready[0])
      );

    // ===| HP1 port |==========================
      xpm_fifo_axis #(
          .FIFO_DEPTH(`XPM_FIFO_DEPTH),
          .TDATA_WIDTH(`AXI_DATA_WIDTH),
          .FIFO_MEMORY_TYPE("block"),
          .CLOCKING_MODE("common_clock")
      ) u_w_fifo1 (
          .s_aclk(clk),
          .s_aresetn(rst_n),
          .s_axis_tdata(IN_HP1.tdata),
          .s_axis_tvalid(IN_HP1.tvalid),
          .s_axis_tready(IN_HP1.tready),
          .m_axis_tdata(weight_fifo_data[1]),
          .m_axis_tvalid(weight_fifo_valid[1]),
          .m_axis_tready(weight_fifo_ready[1])
      );

    // ===| HP2 port |==========================
      xpm_fifo_axis #(
          .FIFO_DEPTH(`XPM_FIFO_DEPTH),
          .TDATA_WIDTH(`AXI_DATA_WIDTH),
          .FIFO_MEMORY_TYPE("block"),
          .CLOCKING_MODE("common_clock")
      ) u_w_fifo2 (
          .s_aclk(clk),
          .s_aresetn(rst_n),
          .s_axis_tdata(IN_HP2.tdata),
          .s_axis_tvalid(IN_HP2.tvalid),
          .s_axis_tready(IN_HP2.tready),
          .m_axis_tdata(weight_fifo_data[2]),
          .m_axis_tvalid(weight_fifo_valid[2]),
          .m_axis_tready(weight_fifo_ready[2])
      );

    // ===| HP3 port |==========================
      xpm_fifo_axis #(
          .FIFO_DEPTH(`XPM_FIFO_DEPTH),
          .TDATA_WIDTH(`AXI_DATA_WIDTH),
          .FIFO_MEMORY_TYPE("block"),
          .CLOCKING_MODE("common_clock")
      ) u_w_fifo3 (
          .s_aclk(clk),
          .s_aresetn(rst_n),
          .s_axis_tdata(IN_HP3.tdata),
          .s_axis_tvalid(IN_HP3.tvalid),
          .s_axis_tready(IN_HP3.tready),
          .m_axis_tdata(weight_fifo_data[3]),
          .m_axis_tvalid(weight_fifo_valid[3]),
          .m_axis_tready(weight_fifo_ready[3])
      );