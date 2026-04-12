`ifndef NUMBERS_SVH
`define NUMBERS_SVH

// ===| Primitive Type Widths |===================================================
// Used by device_pkg.sv for algorithm-level type selection.
// All values are plain integers — no units, no semantics.
// ===============================================================================

`define N_SIZEOF_INT4   4   // INT4 weight width (bits)
`define N_BF16_SIZE    16   // BF16 activation width (bits)
`define N_FP32_SIZE    32   // FP32 accumulation width (bits)

`endif // NUMBERS_SVH
