`timescale 1ns / 1ps

// ===| Memory Architecture Package |============================================
// Derived memory parameters for the uXC NPU.
// All values are computed from device_pkg and kv260_device.svh — no magic numbers.
//
// Compilation order: C — depends on A_const_svh, B_device_pkg.
// Naming convention: localparam uses PascalCase.
// ===============================================================================

`include "kv260_device.svh"
`include "npu_arch.svh"

package mem_pkg;

  // ===| HP Port Configuration |=================================================
  // HP ports deliver weights to Vector Core (HP0/1/2) and Matrix Core (HP3).
  localparam int HpPortCnt            = `DEVICE_HP_PORT_CNT;        // 4
  localparam int HpSingleWidthBit     = `DEVICE_HP_SINGLE_WIDTH_BIT; // 128

  // Total aggregated weight bus width (all 4 HP lanes combined)
  localparam int HpTotalWidthBit      = HpPortCnt * HpSingleWidthBit; // 512

  // ===| Weight Count per HP Port Burst |=========================================
  // How many INT4 weights arrive per clock per HP port
  localparam int WeightBitWidth       = device_pkg::WeightType;       // 4
  localparam int HpSingleWeightCnt    = HpSingleWidthBit / WeightBitWidth; // 32
  localparam int HpTotalWeightCnt     = HpTotalWidthBit  / WeightBitWidth; // 128

  // ===| L2 Cache / FMap Cache Output Width |=====================================
  // Number of fixed-point mantissa values broadcast to the compute array per cycle
  // = ARRAY_SIZE_H (one per PE column)
  localparam int FmapL2CacheOutCnt    = `ARRAY_SIZE_H;   // 32

  // ===| XPM FIFO Depths |========================================================
  localparam int XpmFifoDepth         = `DEVICE_XPM_FIFO_DEPTH;      // 512
  localparam int XpmFifoDepthTiny     = `DEVICE_XPM_FIFO_DEPTH_TINY; // 16

  // ===| FMap L1 SRAM Cache |=====================================================
  localparam int FmapCacheDepth       = `FMAP_CACHE_DEPTH;   // 2048
  localparam int FmapAddrWidth        = `FMAP_ADDR_WIDTH;     // 11

endpackage : mem_pkg
