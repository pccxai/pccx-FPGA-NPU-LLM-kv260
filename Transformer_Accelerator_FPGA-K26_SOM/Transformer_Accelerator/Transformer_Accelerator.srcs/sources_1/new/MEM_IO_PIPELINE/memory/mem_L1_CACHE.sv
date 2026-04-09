`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"


module mem_L1_CACHE #(
    parameter DATA_WIDTH = `HP_PORT_SINGLE_WIDTH
) (
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
  // HP0 systolic cache
  // HP0 V dot M cache
  // HP1: to V dot M pipeline
  // HP2: to V dot M pipeline
  // HP3: to V dot M pipeline
  /*
  typedef enum logic [3:0] {
    L2_CACHE_W_A1 = 4'h0,
    L2_CACHE_W_A2 = 4'h1,
    L2_CACHE_W_A3 = 4'h2,
    L2_CACHE_W_A4 = 4'h3,

    L2_CACHE_S_B1 = 4'h4,

    L2_CACHE_F_C1 = 4'h5,
    L2_CACHE_F_C2 = 4'h6,
  } L2_CACHE;
*/
  // ACP0: FMAP Cache
  // ACP1: FMAP Cache

  //CMD

endmodule
