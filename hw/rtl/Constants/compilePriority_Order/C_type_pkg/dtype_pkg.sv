`timescale 1ns / 1ps

// ===| Data Type Package |=======================================================
// Numeric type constants for all data formats used in uXC.
// Merges and replaces: float_pkg.sv, float_emax_align_pkg.sv
//
// Compilation order: C — depends on A_const_svh, B_device_pkg.
// Naming convention: localparam uses PascalCase.
// ===============================================================================

package dtype_pkg;

  // ===| BF16 (Brain Float 16) |=================================================
  localparam int Bf16Width         = 16;  // total bit width
  localparam int Bf16ExpWidth      = 8;   // exponent bits
  localparam int Bf16MantWidth     = 7;   // mantissa bits (stored)

  // ===| Fixed-Point Mantissa (post-emax-alignment) |============================
  // After BF16 emax alignment:
  //   bits = 1 (sign) + 1 (implicit leading-1) + 7 (mantissa) + 18 (integer headroom) = 27
  // This fits in the DSP48E2 A-port (30-bit signed) with 3 bits to spare.
  localparam int FixedMantWidth    = 27;

  // ===| FP32 |==================================================================
  localparam int Fp32Width         = 32;

  // ===| INT4 (Weight) |=========================================================
  localparam int Int4Width         = 4;
  localparam int Int4Max           = 7;
  localparam int Int4Min           = -8;
  localparam int Int4Range         = 16;

  // ===| INT8 (Activation — W4A8 path) |=========================================
  localparam int Int8Width         = 8;

  // ===| DSP48E2 Accumulator |===================================================
  // P-register output: 48-bit signed integer
  localparam int DspPWidth         = 48;

endpackage : dtype_pkg
