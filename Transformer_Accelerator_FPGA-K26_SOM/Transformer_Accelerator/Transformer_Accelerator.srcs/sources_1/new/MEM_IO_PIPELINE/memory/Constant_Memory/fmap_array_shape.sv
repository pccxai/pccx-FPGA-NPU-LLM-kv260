`timescale 1ns / 1ps

// ============================================================
//  shape_ram
//  - Depth : 64 entries  (6-bit address)
//  - Width : 51 bits     (17-bit × 3 fields)
//
//  [write]  wr_en=1, wr_addr, wr_val{0,1,2} → next clk
//  [ read]  rd_addr → at same clk rd_val{0,1,2} out (comb logic)
// ============================================================

module fmap_array_shape (
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
