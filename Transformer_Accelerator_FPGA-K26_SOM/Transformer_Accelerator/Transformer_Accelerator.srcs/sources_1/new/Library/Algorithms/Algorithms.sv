`timescale 1ns / 1ps
`ifndef ALGORITHMS_SV
`define ALGORITHMS_SV

package algorithms_pkg;

  /*─────────────────────────────────────────────
  QUEUE
  ─────────────────────────────────────────────*/
  typedef struct packed {
    logic empty;
    logic full;
  } queue_stat_t;

  /*─────────────────────────────────────────────
  STACK  (나중에)
  ─────────────────────────────────────────────*/
  // typedef struct packed { ... } stack_stat_t;

endpackage

`endif
