`timescale 1ns / 1ps
// algorithms_pkg is compiled from Algorithms.sv; just import the symbols.

// ===| Module: QUEUE — owner of an IF_queue interface (push/pop logic) |========
// Purpose      : Drive the push/pop handshake of an IF_queue interface.
//                The producer/consumer sit on the other side of the
//                interface; this module is the FIFO controller.
// Spec ref     : pccx v002 §4 (control plane primitives).
// Clock        : interface-supplied (q.clk). Single-clock FIFO.
// Reset        : interface-supplied (q.rst_n) active-low. Clears wr/rd ptrs.
// Latency      : 1 cycle (push_en → mem write); pop_data is combinational
//                (`pop_data = mem[rd_ptr]`).
// Throughput   : 1 push and 1 pop per cycle.
// Reset state  : wr_ptr = rd_ptr = 0.
// Errors       : Push when full or pop when empty are silently ignored.
//                (Stage C candidate: SVA on push&full / pop&empty.)
// Counters     : none.
// ===============================================================================
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
