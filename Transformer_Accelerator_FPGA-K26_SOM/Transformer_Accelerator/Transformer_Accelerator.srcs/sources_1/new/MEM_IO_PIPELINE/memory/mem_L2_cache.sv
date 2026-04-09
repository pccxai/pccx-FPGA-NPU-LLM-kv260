`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"

/**
 * Module: mem_FMAP_KV_CACHE
 * Description:
 * Massive Unified L2 Scratchpad for Feature Maps, KV Cache, and any intermediate data.
 * Extracted maximum physical limit of KV260 URAMs (56 blocks reserved for this module).
 * * Capacity: 114,688 x 128-bit = 1.75 MB
 * Memory Map (Software Managed):
 * - e.g., 0x00000 ~ 0x07FFF : Feature Map (Dynamic)
 * - e.g., 0x08000 ~ 0x1BFFF : KV Cache (TurboQuant compressed) & Others
 */
module mem_L2_cache #(
    // MUM DEPTH FOR KV260 (using 56 URAM blocks)
    parameter int URAM_DEPTH = 114688,
    parameter int ADDR_WIDTH = $clog2(URAM_DEPTH)  // 17 bits
) (
    input logic clk_core,
    input logic rst_n_core,

    // ===| PORT A: Dedicated to ACP (External DDR4 Data Transfer) |===
    input  logic                  IN_acp_we,
    input  logic [ADDR_WIDTH-1:0] IN_acp_addr,
    input  logic [         127:0] IN_acp_wdata,
    output logic [         127:0] OUT_acp_rdata,

    // ===| PORT B: Dedicated to NPU Internal Pipeline (Compute) |===
    input  logic                  IN_npu_we,
    input  logic [ADDR_WIDTH-1:0] IN_npu_addr,
    input  logic [         127:0] IN_npu_wdata,
    output logic [         127:0] OUT_npu_rdata
);

  // XPM True Dual-Port RAM (UltraRAM) Instantiation - MAXIMIZED
  xpm_memory_tdpram #(
      .MEMORY_SIZE     (URAM_DEPTH * 128),  // 114,688 * 128 = 14,680,064 bits (Exact Fit!)
      .MEMORY_PRIMITIVE("ultra"),
      .CLOCKING_MODE   ("common_clock"),

      // Port A Config (ACP Side)
      .WRITE_DATA_WIDTH_A(128),
      .BYTE_WRITE_WIDTH_A(128),
      .READ_DATA_WIDTH_A (128),
      .READ_LATENCY_A    (3),

      // Port B Config (NPU Pipeline Side)
      .WRITE_DATA_WIDTH_B(128),
      .BYTE_WRITE_WIDTH_B(128),
      .READ_DATA_WIDTH_B (128),
      .READ_LATENCY_B    (3),

      .USE_MEM_INIT(0)
  ) u_uram_fmap_kv (
      .sleep(1'b0),

      // --- Port A (ACP Interface) ---
      .clka          (clk_core),
      .rsta          (~rst_n_core),
      .ena           (1'b1),
      .wea           (IN_acp_we),
      .addra         (IN_acp_addr),
      .dina          (IN_acp_wdata),
      .douta         (OUT_acp_rdata),
      .injectsbiterra(1'b0),
      .injectdbiterra(1'b0),

      // --- Port B (NPU Pipeline Interface) ---
      .clkb    (clk_core),
      .rstb    (~rst_n_core),
      .enb     (1'b1),
      .web     (IN_npu_we),
      .addrb   (IN_npu_addr),
      .dinb    (IN_npu_wdata),
      .doutb   (OUT_npu_rdata),
      .sbiterrb(),
      .dbiterrb()
  );

endmodule
