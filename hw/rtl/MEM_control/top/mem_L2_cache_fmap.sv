`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

// ===| L2 Feature Map & KV Cache (URAM) |========================================
// True dual-port URAM: Depth x 128-bit wide.
//   Default Depth = 114688 entries = 1.75 MB
//
// Port A — ACP DMA path  (host DDR4 ↔ L2 via ACP)
// Port B — NPU compute   (GEMM / GEMV / CVO streaming R/W)
//
// READ_LATENCY = 3 (URAM registered output, meets 400 MHz timing)
// WRITE_MODE   = write_first (read-before-write on same address is undefined)
// ===============================================================================

module mem_L2_cache_fmap #(
    parameter int Depth = 114688   // 128-bit word entries (1.75 MB)
) (
    input  logic        clk_core,
    input  logic        rst_n_core,

    // ===| Port A — ACP host DMA |================================================
    input  logic         IN_acp_we,
    input  logic  [16:0] IN_acp_addr,
    input  logic [127:0] IN_acp_wdata,
    output logic [127:0] OUT_acp_rdata,

    // ===| Port B — NPU compute engines |=========================================
    input  logic         IN_npu_we,
    input  logic  [16:0] IN_npu_addr,
    input  logic [127:0] IN_npu_wdata,
    output logic [127:0] OUT_npu_rdata
);

  xpm_memory_tdpram #(
      // ===| Geometry |===
      .ADDR_WIDTH_A       (17),
      .ADDR_WIDTH_B       (17),
      .DATA_WIDTH_A       (128),
      .DATA_WIDTH_B       (128),
      .BYTE_WRITE_WIDTH_A (128),
      .BYTE_WRITE_WIDTH_B (128),
      .MEMORY_SIZE        (128 * Depth),

      // ===| Implementation |===
      .MEMORY_PRIMITIVE   ("ultra"),      // Force URAM on UltraScale+
      .CLOCKING_MODE      ("common_clock"),
      .READ_LATENCY_A     (3),
      .READ_LATENCY_B     (3),
      .WRITE_MODE_A       ("write_first"),
      .WRITE_MODE_B       ("write_first"),

      // ===| Init / Misc |===
      .MEMORY_INIT_FILE   ("none"),
      .MEMORY_INIT_PARAM  ("0"),
      .USE_MEM_INIT       (0),
      .AUTO_SLEEP_TIME    (0),
      .WAKEUP_TIME        ("disable_sleep"),
      .ECC_MODE           ("no_ecc"),
      .USE_EMBEDDED_CONSTRAINT(0)
  ) u_l2_uram (
      // Port A
      .clka           (clk_core),
      .rsta           (~rst_n_core),
      .ena            (1'b1),
      .wea            (IN_acp_we),
      .addra          (IN_acp_addr),
      .dina           (IN_acp_wdata),
      .douta          (OUT_acp_rdata),
      .regcea         (1'b1),
      .injectsbiterra (1'b0),
      .injectdbiterra (1'b0),
      .sbiterra       (),
      .dbiterra       (),

      // Port B
      .clkb           (clk_core),
      .rstb           (~rst_n_core),
      .enb            (1'b1),
      .web            (IN_npu_we),
      .addrb          (IN_npu_addr),
      .dinb           (IN_npu_wdata),
      .doutb          (OUT_npu_rdata),
      .regceb         (1'b1),
      .injectsbiterrb (1'b0),
      .injectdbiterrb (1'b0),
      .sbiterrb       (),
      .dbiterrb       ()
  );

endmodule
