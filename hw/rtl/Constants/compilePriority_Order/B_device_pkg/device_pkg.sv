`timescale 1ns / 1ps

// ===| Device Configuration Package |==========================================
// Selects the data type choices for this target design.
// This is where algorithm-level decisions (precision, pipeline count) are made.
//
// Compilation order: B — depends on A_const_svh (NUMBERS.svh).
// Naming convention: localparam uses PascalCase (linter: parameter-name-style).
// ===============================================================================

`include "NUMBERS.svh"

package device_pkg;

  // ===| Feature Map (Activation) Type |========================================
  // FmapType             : port-level precision  — BF16 (16-bit)
  // FmapTypeMixedPrecision: internal accumulation — FP32 (32-bit)
  localparam int FmapType                 = `N_BF16_SIZE;
  localparam int FmapTypeMixedPrecision   = `N_FP32_SIZE;

  // ===| Weight Type |===========================================================
  // INT4: 4-bit quantized weight, streamed from HP ports
  localparam int WeightType               = `N_SIZEOF_INT4;

  // ===| Pipeline Instance Counts |==============================================
  localparam int VecPipelineCnt           = 4;  // 4 x muV-Core (Vector Core)
  localparam int MatPipelineCnt           = 1;  // 1 x Matrix Core (32x32 systolic)

  // ===| Legacy aliases (snake_case) — keep until all RTL refs updated |=========
  localparam int GemvPipelineCnt          = VecPipelineCnt;
  localparam int GemmPipelineCnt          = MatPipelineCnt;

endpackage : device_pkg
