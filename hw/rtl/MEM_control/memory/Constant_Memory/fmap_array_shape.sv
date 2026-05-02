`timescale 1ns / 1ps

// ===| Module: fmap_array_shape — fmap shape constant RAM (FF-based) |==========
// Purpose      : 64-entry × (3 × 17-bit) shape constant RAM for feature-map
//                tensor shape descriptors. MEMSET uops write triples
//                (X, Y, Z) here; LOAD uops read them by 6-bit pointer.
// Spec ref     : pccx v002 §3.3 (MEMSET), §5.4 (shape pointer routing).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low (synchronous clear of all 64 entries).
// Geometry     : 64 entries × 51 bits (= 3 × 17, see isa_pkg::shape_xyz_t
//                for the typed name; this module keeps the legacy three
//                17-bit ports for compile-time stability).
// Latency      : Write — 1 cycle. Read — 0 cycles (combinational).
// Throughput   : 1 write + 1 read per cycle.
// Reset state  : All entries cleared to 0.
// Notes        : Byte-for-byte duplicate of `weight_array_shape`. The
//                parameterised replacement
//                `MEM_control/memory/Constant_Memory/shape_const_ram.sv`
//                has been authored (not yet wired) and supersedes this
//                module once `mem_dispatcher.sv` migrates the two
//                instantiations. See Stage E analysis (REFACTOR_NOTES
//                §6.3.1) for the consolidation rationale.
// ===============================================================================

module fmap_array_shape
  import isa_pkg::*;
(
    input logic clk,
    input logic rst_n,

    // ===| write |===
    input logic        wr_en,
    input logic [ 5:0] wr_addr,
    input logic [16:0] wr_val0,  // shape: x
    input logic [16:0] wr_val1,  // shape: y
    input logic [16:0] wr_val2,  // shape: z

    // ===| read |===
    input ptr_addr_t rd_addr,
    output logic [16:0] rd_val0,  // shape: x
    output logic [16:0] rd_val1,  // shape: y
    output logic [16:0] rd_val2  // shape: z
);

  // 64 × 51 bit REGISTER array
  // [50:34] = val2 / [33:17] = val1 / [16:0] = val0
  logic [50:0] mem[0:63];

  // ===| write (sync to clk) |===
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int i = 0; i < 64; i++) mem[i] <= 51'd0;
    end else begin
      if (wr_en) begin
        mem[wr_addr] <= {wr_val2, wr_val1, wr_val0};
      end
    end
  end

  // ===| read (comb logic - latency 0) |===
  assign rd_val0 = mem[rd_addr][16:0];
  assign rd_val1 = mem[rd_addr][33:17];
  assign rd_val2 = mem[rd_addr][50:34];

endmodule
