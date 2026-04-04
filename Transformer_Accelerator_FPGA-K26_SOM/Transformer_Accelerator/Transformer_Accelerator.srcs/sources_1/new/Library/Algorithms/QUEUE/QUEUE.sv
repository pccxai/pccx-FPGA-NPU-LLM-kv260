`timescale 1ns / 1ps
`include "algorithms.sv"

module QUEUE (
    fifo_if.owner q
);
  import algorithms_pkg::*;

  always_ff @(posedge q.clk) begin
    if (!q.rst_n) begin
      q.wr_ptr <= '0;
      q.rd_ptr <= '0;
    end else begin
      if (q.push_en && !q.full) begin
        q.mem[q.wr_ptr[q.PTR_W-1:0]] <= q.push_data;
        q.wr_ptr <= q.wr_ptr + 1'b1;
      end
      if (q.pop_en && !q.empty) q.rd_ptr <= q.rd_ptr + 1'b1;

      q.push_en <= 1'b0;
      q.pop_en  <= 1'b0;
    end
  end

endmodule
