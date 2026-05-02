// ===| Interface: IF_queue — FIFO "object" with producer/consumer modports |====
// Purpose      : Encapsulate FIFO storage + handshake plumbing behind a
//                single interface so producers and consumers don't have to
//                wire individual signals. Resembles an OOP "object" with
//                push() / pop() / clear() task methods.
// Spec ref     : pccx v002 §4 (control plane primitives).
// Parameters   : DATA_WIDTH (default 32), DEPTH (default 8).
// Geometry     : memory mem[0..DEPTH-1] × DATA_WIDTH bits. wr/rd pointers
//                are PTR_W+1 bits wide (extra bit distinguishes empty/full).
// Modports
//   producer : import push, output {push_data, push_en}, input {empty, full,
//              clk, rst_n}.  The producer never sees rd/wr pointers.
//   consumer : import pop,  output {pop_en}, input {empty, full, pop_data,
//              clk, rst_n}.
//   owner    : the FIFO module itself (`QUEUE`). Reads handshake signals,
//              updates pointers and `mem`. Has `ref` access to mem.
// Reset state  : push_en / pop_en cleared by clear() task.
// Notes        : modports cannot export parameters in SV2012, so `QUEUE`
//                re-derives PTR_W locally from $size(q.mem).
// ===============================================================================
interface IF_queue #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH      = 8
) (
    input logic clk,
    input logic rst_n
);

  localparam PTR_W = $clog2(DEPTH);

  // ── Storage ───────────────────────────────────
  logic [DATA_WIDTH-1:0] mem[0:DEPTH-1];
  logic [PTR_W:0] wr_ptr, rd_ptr;

  // ── Status flags ──────────────────────────────
  logic empty, full;
  assign empty = (wr_ptr == rd_ptr);
  assign full  = (wr_ptr[PTR_W] != rd_ptr[PTR_W]) && (wr_ptr[PTR_W-1:0] == rd_ptr[PTR_W-1:0]);

  // ── Push/Pop handshake signals ─────────────────
  logic [DATA_WIDTH-1:0] push_data;
  logic                  push_en;  // "push()" call
  logic [DATA_WIDTH-1:0] pop_data;
  logic                  pop_en;  // "pop()" call

  assign pop_data = mem[rd_ptr[PTR_W-1:0]];

  // ── "Methods" (tasks) ──────────────────────────
  task automatic push(input logic [DATA_WIDTH-1:0] wdata);
    push_data <= wdata;
    push_en   <= 1'b1;
  endtask

  task automatic pop();
    pop_en <= 1'b1;
  endtask

  task automatic clear();
    push_en <= 1'b0;
    pop_en  <= 1'b0;
  endtask

  // ── Modports ───── Access Control ──────────────
  // producer : only pushes
  modport producer(import push, input empty, full, clk, rst_n, output push_data, push_en);

  // consumer : only pops
  modport consumer(import pop, input empty, full, pop_data, clk, rst_n, output pop_en);

  // owner : the FIFO module itself. Reads producer/consumer handshake
  // signals, updates its own pointers + memory contents.
  modport owner(input  clk, rst_n, push_data, push_en, pop_en, full, empty,
                output wr_ptr, rd_ptr, ref mem);

endinterface
