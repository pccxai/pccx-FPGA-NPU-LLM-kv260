`timescale 1ns / 1ps

// ===| Module: shape_const_ram — parameterised shape constant RAM |=============
// Purpose      : Single source for the (X, Y, Z) tensor-shape constant RAM
//                used by fmap and weight MEMSET descriptors. Replaces the
//                byte-for-byte duplicate pair `fmap_array_shape` +
//                `weight_array_shape` (Stage E analysis §6.3.1, Stage C
//                decisions memo item 5).
//
//                **AUTHORED-BUT-UNWIRED** in this batch:
//                this file is intentionally NOT in `hw/vivado/filelist.f`
//                and `mem_dispatcher.sv` still instantiates the two
//                concrete modules. It is staged here so a future commit
//                can migrate `mem_dispatcher` to instantiate this
//                parameterised form and delete the two duplicates with a
//                single, focused review.
//
// Spec ref     : pccx v002 §3.3 (MEMSET), §5.4 (shape pointer routing).
// Clock        : clk @ 400 MHz.
// Reset        : rst_n active-low (synchronous clear of all entries).
// Geometry     : Depth × shape_xyz_t. With Depth = 64 the 51-bit storage
//                exactly matches the existing fmap_array_shape /
//                weight_array_shape footprint.
// Latency      : Write — 1 cycle. Read — 0 cycles (combinational).
// Throughput   : 1 write + 1 read per cycle.
// Reset state  : All entries cleared to 0.
// Notes        : Uses isa_pkg::shape_dim_t / shape_xyz_t typedefs (added
//                to isa_pkg in the prior commit). Port widths are
//                identical to the existing modules' 3 × 17-bit fan-out so
//                a parent migration is a one-line swap and a port name
//                change.
// Migration path:
//   1. Land this file in filelist.f (after isa_pkg) without removing
//      fmap_array_shape / weight_array_shape — confirms it lints clean.
//   2. In `mem_dispatcher.sv`, swap each instance of
//        fmap_array_shape   u_fmap_shape   (...);
//        weight_array_shape u_weight_shape (...);
//      for the parameterised form and rerun
//      `bash hw/sim/run_verification.sh`.
//   3. Once the two callers migrate, delete
//      fmap_array_shape.sv / weight_array_shape.sv and remove the two
//      filelist lines. `dead_module_inventory.md` updates reflect the
//      removal.
// ===============================================================================

module shape_const_ram
  import isa_pkg::*;
#(
    parameter int Depth = 64
) (
    input logic clk,
    input logic rst_n,

    // ===| write |===
    input logic                       wr_en,
    input logic [$clog2(Depth)-1:0]   wr_addr,
    input shape_xyz_t                 wr_xyz,    // packed { Z, Y, X }

    // ===| read |===
    input  ptr_addr_t                  rd_addr,  // 6-bit pointer (matches Depth=64)
    output shape_xyz_t                 rd_xyz    // packed { Z, Y, X }
);

  // 64 × 51-bit (3 × shape_dim_t) flop array. Vivado infers FFs because
  // the read path is combinational; consumers therefore see 0-cycle reads.
  shape_xyz_t mem [0:Depth-1];

  // ===| write (sync to clk) |===================================================
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int i = 0; i < Depth; i++) mem[i] <= '0;
    end else if (wr_en) begin
      mem[wr_addr] <= wr_xyz;
    end
  end

  // ===| read (comb logic - latency 0) |=========================================
  assign rd_xyz = mem[rd_addr];

endmodule
