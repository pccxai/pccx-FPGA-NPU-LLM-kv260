`timescale 1ns / 1ps
`ifndef ALGORITHMS_SV
`define ALGORITHMS_SV

// ===| Package: algorithms_pkg — generic data-structure status types |==========
// Purpose      : Shared status struct types for the pccx queue / stack
//                infrastructure. Imported by IF_queue / QUEUE consumers
//                (e.g. AXIL_CMD_IN) so they speak the same status vocabulary.
// Spec ref     : pccx v002 §4 (control plane primitives).
// Provides
//   typedef queue_stat_t    : packed {empty, full} two-bit status word.
//   typedef stack_stat_t    : (reserved) — to be added when stack support
//                             is needed.
// ===============================================================================
package algorithms_pkg;

  /*─────────────────────────────────────────────
  QUEUE
  ─────────────────────────────────────────────*/
  typedef struct packed {
    logic empty;
    logic full;
  } queue_stat_t;

  /*─────────────────────────────────────────────
  STACK
  ─────────────────────────────────────────────*/
  // typedef struct packed { ... } stack_stat_t;

endpackage

`endif

