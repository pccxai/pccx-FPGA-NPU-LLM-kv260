`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"

/**
 * Module: mem_Unified_L2_CACHE
 * Description:
 * Massive L2 Scratchpad utilizing 4 Independent Banks of Xilinx UltraRAM (URAM).
 * Absorbs 128-bit streams from FIFOs (HP0~HP3) and provides high-bandwidth
 * parallel read access to downstream Dispatchers / L1 Caches.
 */
module mem_Unified_L2_CACHE #(
    parameter int URAM_DEPTH = 4096,  // Example: 4096 * 128bit = 64KB per bank (Total 256KB)
    parameter int ADDR_WIDTH = $clog2(URAM_DEPTH)
) (
    input logic clk_core,
    input logic rst_n_core,

    // ===| Inputs from mem_BUFFER (Write Side - AXI Streams) |===
    axis_if.slave S_CORE_HP0_WEIGHT,
    axis_if.slave S_CORE_HP1_WEIGHT,
    axis_if.slave S_CORE_HP2_WEIGHT,
    axis_if.slave S_CORE_HP3_WEIGHT,

    // ===| Outputs to Dispatcher / L1 Cache (Read Side) |===
    // --- Bank 0 (HP0) Read Port ---
    input  logic [ADDR_WIDTH-1:0] IN_read_addr_hp0,
    input  logic                  IN_read_en_hp0,
    output logic [         127:0] OUT_read_data_hp0,

    // --- Bank 1 (HP1) Read Port ---
    input  logic [ADDR_WIDTH-1:0] IN_read_addr_hp1,
    input  logic                  IN_read_en_hp1,
    output logic [         127:0] OUT_read_data_hp1,

    // --- Bank 2 (HP2) Read Port ---
    input  logic [ADDR_WIDTH-1:0] IN_read_addr_hp2,
    input  logic                  IN_read_en_hp2,
    output logic [         127:0] OUT_read_data_hp2,

    // --- Bank 3 (HP3) Read Port ---
    input  logic [ADDR_WIDTH-1:0] IN_read_addr_hp3,
    input  logic                  IN_read_en_hp3,
    output logic [         127:0] OUT_read_data_hp3
);

  // [Bank 0] Logic & URAM (For HP0)
  logic [ADDR_WIDTH-1:0] write_ptr_hp0;
  logic                  write_en_hp0;

  // Always ready to sink data. Dispatcher must prevent overflow.
  assign S_CORE_HP0_WEIGHT.tready = 1'b1;
  assign write_en_hp0 = S_CORE_HP0_WEIGHT.tvalid & S_CORE_HP0_WEIGHT.tready;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) write_ptr_hp0 <= '0;
    else if (write_en_hp0) write_ptr_hp0 <= write_ptr_hp0 + 1'b1;
  end

  xpm_memory_sdpram #(
      .MEMORY_SIZE       (URAM_DEPTH * 128),
      .MEMORY_PRIMITIVE  ("ultra"),           // Forces UltraRAM
      .CLOCKING_MODE     ("common_clock"),
      .WRITE_DATA_WIDTH_A(128),
      .BYTE_WRITE_WIDTH_A(128),
      .READ_DATA_WIDTH_B (128),
      .READ_LATENCY_B    (3),                 // 3 cycles latency for 400MHz timing
      .USE_MEM_INIT      (0)
  ) u_uram_bank0 (
      .sleep(1'b0),
      // Port A (Write)
      .clka(clk_core),
      .ena(1'b1),
      .wea(write_en_hp0),
      .addra(write_ptr_hp0),
      .dina(S_CORE_HP0_WEIGHT.tdata),
      .injectsbiterra(1'b0),
      .injectdbiterra(1'b0),
      // Port B (Read)
      .clkb(clk_core),
      .rstb(~rst_n_core),
      .enb(IN_read_en_hp0),
      .addrb(IN_read_addr_hp0),
      .doutb(OUT_read_data_hp0),
      .sbiterrb(),
      .dbiterrb()
  );

  // [Bank 1] Logic & URAM (For HP1)
  logic [ADDR_WIDTH-1:0] write_ptr_hp1;
  logic                  write_en_hp1;

  assign S_CORE_HP1_WEIGHT.tready = 1'b1;
  assign write_en_hp1 = S_CORE_HP1_WEIGHT.tvalid & S_CORE_HP1_WEIGHT.tready;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) write_ptr_hp1 <= '0;
    else if (write_en_hp1) write_ptr_hp1 <= write_ptr_hp1 + 1'b1;
  end

  xpm_memory_sdpram #(
      .MEMORY_SIZE       (URAM_DEPTH * 128),
      .MEMORY_PRIMITIVE  ("ultra"),
      .CLOCKING_MODE     ("common_clock"),
      .WRITE_DATA_WIDTH_A(128),
      .BYTE_WRITE_WIDTH_A(128),
      .READ_DATA_WIDTH_B (128),
      .READ_LATENCY_B    (3),
      .USE_MEM_INIT      (0)
  ) u_uram_bank1 (
      .sleep(1'b0),
      .clka(clk_core),
      .ena(1'b1),
      .wea(write_en_hp1),
      .addra(write_ptr_hp1),
      .dina(S_CORE_HP1_WEIGHT.tdata),
      .injectsbiterra(1'b0),
      .injectdbiterra(1'b0),
      .clkb(clk_core),
      .rstb(~rst_n_core),
      .enb(IN_read_en_hp1),
      .addrb(IN_read_addr_hp1),
      .doutb(OUT_read_data_hp1),
      .sbiterrb(),
      .dbiterrb()
  );

  // [Bank 2] Logic & URAM (For HP2)
  logic [ADDR_WIDTH-1:0] write_ptr_hp2;
  logic                  write_en_hp2;

  assign S_CORE_HP2_WEIGHT.tready = 1'b1;
  assign write_en_hp2 = S_CORE_HP2_WEIGHT.tvalid & S_CORE_HP2_WEIGHT.tready;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) write_ptr_hp2 <= '0;
    else if (write_en_hp2) write_ptr_hp2 <= write_ptr_hp2 + 1'b1;
  end

  xpm_memory_sdpram #(
      .MEMORY_SIZE       (URAM_DEPTH * 128),
      .MEMORY_PRIMITIVE  ("ultra"),
      .CLOCKING_MODE     ("common_clock"),
      .WRITE_DATA_WIDTH_A(128),
      .BYTE_WRITE_WIDTH_A(128),
      .READ_DATA_WIDTH_B (128),
      .READ_LATENCY_B    (3),
      .USE_MEM_INIT      (0)
  ) u_uram_bank2 (
      .sleep(1'b0),
      .clka(clk_core),
      .ena(1'b1),
      .wea(write_en_hp2),
      .addra(write_ptr_hp2),
      .dina(S_CORE_HP2_WEIGHT.tdata),
      .injectsbiterra(1'b0),
      .injectdbiterra(1'b0),
      .clkb(clk_core),
      .rstb(~rst_n_core),
      .enb(IN_read_en_hp2),
      .addrb(IN_read_addr_hp2),
      .doutb(OUT_read_data_hp2),
      .sbiterrb(),
      .dbiterrb()
  );

  // [Bank 3] Logic & URAM (For HP3)
  logic [ADDR_WIDTH-1:0] write_ptr_hp3;
  logic                  write_en_hp3;

  assign S_CORE_HP3_WEIGHT.tready = 1'b1;
  assign write_en_hp3 = S_CORE_HP3_WEIGHT.tvalid & S_CORE_HP3_WEIGHT.tready;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) write_ptr_hp3 <= '0;
    else if (write_en_hp3) write_ptr_hp3 <= write_ptr_hp3 + 1'b1;
  end

  xpm_memory_sdpram #(
      .MEMORY_SIZE       (URAM_DEPTH * 128),
      .MEMORY_PRIMITIVE  ("ultra"),
      .CLOCKING_MODE     ("common_clock"),
      .WRITE_DATA_WIDTH_A(128),
      .BYTE_WRITE_WIDTH_A(128),
      .READ_DATA_WIDTH_B (128),
      .READ_LATENCY_B    (3),
      .USE_MEM_INIT      (0)
  ) u_uram_bank3 (
      .sleep(1'b0),
      .clka(clk_core),
      .ena(1'b1),
      .wea(write_en_hp3),
      .addra(write_ptr_hp3),
      .dina(S_CORE_HP3_WEIGHT.tdata),
      .injectsbiterra(1'b0),
      .injectdbiterra(1'b0),
      .clkb(clk_core),
      .rstb(~rst_n_core),
      .enb(IN_read_en_hp3),
      .addrb(IN_read_addr_hp3),
      .doutb(OUT_read_data_hp3),
      .sbiterrb(),
      .dbiterrb()
  );

endmodule
