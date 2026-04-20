`timescale 1ns / 1ps
// algorithms_pkg is compiled from Algorithms.sv; just import the symbols.

module QUEUE (
    IF_queue.owner q
);

  // Width of the pointer's index field — the interface stores the
  // parameter as `PTR_W` but modports can't export parameters, so
  // re-derive it here from the mem array depth.
  localparam int PTR_W = $clog2($size(q.mem));

  always_ff @(posedge q.clk) begin
    if (!q.rst_n) begin
      q.wr_ptr <= '0;
      q.rd_ptr <= '0;
    end else begin
      if (q.push_en && !q.full) begin
        q.mem[q.wr_ptr[PTR_W-1:0]] <= q.push_data;
        q.wr_ptr                   <= q.wr_ptr + 1'b1;
      end
      if (q.pop_en && !q.empty) begin
        q.rd_ptr <= q.rd_ptr + 1'b1;
      end
    end
  end

endmodule
