`timescale 1ns / 1ps

// ===| Vector Core (muV-Core) Configuration Package |===========================
// Defines the configuration struct and default parameters for the Vector Core.
// The Vector Core consists of 4 parallel muV-Cores, each performing GEMV
// (vector x matrix) operations using INT4 weights and BF16 feature maps.
//
// Replaces: GEMV_const_pkg.sv
// Compilation order: D — depends on A_, B_device_pkg, C_type_pkg.
// Naming convention: localparam uses PascalCase; struct fields use snake_case.
// ===============================================================================

package vec_core_pkg;

  // ===| Throughput / Batch Constants |==========================================
  // GEMV processes one fmap row (2048-dim) against one weight matrix column.
  // Batch size = number of weight rows consumed per invocation.
  localparam int Throughput       = 1;    // output elements per cycle per lane
  localparam int GemvBatch        = 512;  // weight rows processed per call
  localparam int GemvCycle        = 512;  // clock cycles per GEMV call
  localparam int GemvLineCnt      = mem_pkg::FmapL2CacheOutCnt; // = ARRAY_SIZE_H = 32

  // ===| Vector Core Configuration Struct |======================================
  // Passed as a parameter to GEMV_top and all sub-modules.
  // Use vec_cfg_t instead of raw integers — keeps instantiation self-documenting.
  typedef struct packed {
    // Pipeline topology
    int num_gemv_pipeline;     // number of parallel muV-Core lanes (= VecPipelineCnt)

    // Throughput
    int throughput;            // output elements per cycle
    int gemv_batch;            // weight rows per call
    int gemv_cycle;            // clocks per call

    // Data widths
    int fixed_mant_width;      // fixed-point mantissa width after emax alignment
    int weight_width;          // INT4 = 4
    int weight_cnt;            // weights per HP port per clock (= HpSingleWeightCnt)

    // Cache geometry
    int fmap_cache_out_cnt;    // FMap values broadcast per cycle (= ARRAY_SIZE_H)
    int fmap_type_mixed_precision; // output precision (FP32 = 32)
  } vec_cfg_t;

  // ===| Default Configuration (KV260 / Gemma 3N E4B target) |==================
  localparam vec_cfg_t VecCoreDefaultCfg = '{
    num_gemv_pipeline:          device_pkg::VecPipelineCnt,

    throughput:                 Throughput,
    gemv_batch:                 GemvBatch,
    gemv_cycle:                 GemvCycle,

    fixed_mant_width:           dtype_pkg::FixedMantWidth,
    weight_width:               mem_pkg::WeightBitWidth,
    weight_cnt:                 mem_pkg::HpSingleWeightCnt,

    fmap_cache_out_cnt:         mem_pkg::FmapL2CacheOutCnt,
    fmap_type_mixed_precision:  device_pkg::FmapTypeMixedPrecision
  };

  // ===| Legacy type alias |=====================================================
  // Old type name: gemv_cfg_t — kept so existing port declarations still compile
  // during the migration period. Remove once all GEMV_*.sv files are updated.
  typedef vec_cfg_t gemv_cfg_t;

endpackage : vec_core_pkg
