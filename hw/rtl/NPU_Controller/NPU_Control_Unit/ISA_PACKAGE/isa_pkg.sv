`timescale 1ns / 1ps

// ===| Package: isa_pkg — pccx v002 ISA master type vocabulary |================
// Purpose      : Master type package for the uCA (micro Compute Architecture)
//                ISA. Provides every type used to describe instructions,
//                routes, flags, and per-engine micro-ops. The architectural
//                vocabulary that every other module imports.
// Spec ref     : pccx v002 §3 (ISA encoding) — pccx/docs/v002/Architecture/.
// Dependencies : A_const_svh (npu_arch.svh `defines must be included first
//                outside this file; package itself avoids `include).
// Provides
//   Address types : dest_addr_t / src_addr_t / addr_t / ptr_addr_t / parallel_lane_t
//   Value types   : a_value_t / b_value_t / c_value_t / length_t
//   Shape types   : shape_dim_t / shape_xyz_t (shape constant RAM contract)
//   Direction enums: from_device_e, to_device_e, async_e, dest_cache_e
//   Flag structs   : flags_t (GEMV/GEMM), cvo_flags_t
//   Opcode enum    : opcode_e (5 ops: GEMV / GEMM / MEMCPY / MEMSET / CVO)
//   CVO function   : cvo_func_e (8 ops: EXP / SQRT / GELU / SIN / COS /
//                                       REDUCE_SUM / SCALE / RECIP)
//   Routing enum   : data_route_e (8 routes: host↔L2, L2↔L1_GEMM/GEMV/CVO,
//                                            engine_res→L2)
//   Instruction layouts: GEMV_op_x64_t, GEMM_op_x64_t, memcpy_op_x64_t,
//                        memset_op_x64_t, cvo_op_x64_t.
//   Micro-op structs   : gemm_control_uop_t, GEMV_control_uop_t,
//                        memory_control_uop_t, memory_set_uop_t,
//                        cvo_control_uop_t, acp_uop_t, npu_uop_t.
// Constants    : MemoryUopWidth (= 49 bits).
// Legacy       : isa_x32.svh / isa_memctrl.svh / isa_x64.svh are LEGACY —
//                types here supersede them. Do not extend the .svh files.
// ===============================================================================

package isa_pkg;

  // ===| Basic Address & Control Types |=========================================
  typedef logic [16:0] dest_addr_t;
  typedef logic [16:0] src_addr_t;
  typedef logic [16:0] addr_t;
  typedef logic [ 5:0] ptr_addr_t;       // shape / size pointer (6-bit index)
  typedef logic [ 4:0] parallel_lane_t;  // number of active parallel lanes

  // MEMSET value fields (16-bit each, per ISA §3.3)
  typedef logic [15:0] a_value_t;
  typedef logic [15:0] b_value_t;
  typedef logic [15:0] c_value_t;

  // CVO length (16-bit element count)
  typedef logic [15:0] length_t;

  // ===| Shape Types (shape constant RAM contract) |=============================
  // Both fmap_array_shape and weight_array_shape today expose three flat
  // 17-bit ports per access (wr_val0/1/2 and rd_val0/1/2) for the X / Y / Z
  // axes of the constant shape RAM. Naming the dimension and the triplet
  // makes future code clearer:
  //
  //   shape_dim_t : single-axis size (17-bit, matches the address-space
  //                 dimension used by dest_addr_t / src_addr_t).
  //   shape_xyz_t : a 3-axis bundle (Z is most-significant in packed order
  //                 so { Z, Y, X } maps to a familiar memory layout).
  //
  // These types back the parameterised shape_const_ram used by mem_dispatcher
  // for fmap and weight shape lookup.
  typedef logic [16:0] shape_dim_t;

  typedef struct packed {
    shape_dim_t z;
    shape_dim_t y;
    shape_dim_t x;
  } shape_xyz_t;

  // ===| Device Direction Enums |=================================================
  typedef enum logic {
    FROM_NPU  = 1'b0,
    FROM_HOST = 1'b1
  } from_device_e;

  typedef enum logic {
    TO_NPU  = 1'b0,
    TO_HOST = 1'b1
  } to_device_e;

  typedef enum logic {
    SYNC_OP  = 1'b0,
    ASYNC_OP = 1'b1
  } async_e;

  // ===| GEMV / GEMM Flags (6-bit, ISA §4) |=====================================
  typedef struct packed {
    logic findemax;   // [5] find & register e_max for output normalisation
    logic accm;       // [4] accumulate into destination (do not overwrite)
    logic w_scale;    // [3] apply weight scale factor during MAC
    logic [2:0] reserved;
  } flags_t;

  // ===| Opcode Table (4-bit, ISA §2) |==========================================
  typedef enum logic [3:0] {
    OP_GEMV   = 4'h0,
    OP_GEMM   = 4'h1,
    OP_MEMCPY = 4'h2,
    OP_MEMSET = 4'h3,
    OP_CVO    = 4'h4
  } opcode_e;

  // ===| Instruction Body (60-bit, opcode already stripped) |====================
  typedef logic [59:0] VLIW_instruction_x64;

  typedef struct packed {
    logic [59:0] instruction;
  } instruction_op_x64_t;

  // ===| Instruction Encodings (ISA §3) |========================================

  // GEMV / GEMM  (identical layout, ISA §3.1)  — 60 bits
  typedef struct packed {
    dest_addr_t     dest_reg;        // [59:43] 17-bit
    src_addr_t      src_addr;        // [42:26] 17-bit
    flags_t         flags;           // [25:20]  6-bit
    ptr_addr_t      size_ptr_addr;   // [19:14]  6-bit
    ptr_addr_t      shape_ptr_addr;  // [13: 8]  6-bit
    parallel_lane_t parallel_lane;   // [ 7: 3]  5-bit
    logic [2:0]     reserved;        // [ 2: 0]  3-bit
  } GEMV_op_x64_t;

  typedef GEMV_op_x64_t GEMM_op_x64_t;  // same layout

  // MEMCPY  (ISA §3.2)  — 60 bits
  typedef struct packed {
    from_device_e from_device;    // [59]      1-bit
    to_device_e   to_device;      // [58]      1-bit
    dest_addr_t   dest_addr;      // [57:41]  17-bit
    src_addr_t    src_addr;       // [40:24]  17-bit
    addr_t        aux_addr;       // [23: 7]  17-bit
    ptr_addr_t    shape_ptr_addr; // [ 6: 1]   6-bit
    async_e       async;          // [ 0]      1-bit
  } memcpy_op_x64_t;

  // MEMSET  (ISA §3.3)  — 60 bits
  typedef struct packed {
    logic [1:0] dest_cache;  // [59:58]  2-bit
    ptr_addr_t  dest_addr;   // [57:52]  6-bit
    a_value_t   a_value;     // [51:36] 16-bit
    b_value_t   b_value;     // [35:20] 16-bit
    c_value_t   c_value;     // [19: 4] 16-bit
    logic [3:0] reserved;    // [ 3: 0]  4-bit
  } memset_op_x64_t;

  // CVO  (ISA §3.4)  — 60 bits
  typedef struct packed {
    logic [ 3:0] cvo_func;   // [59:56]  4-bit
    src_addr_t   src_addr;   // [55:39] 17-bit
    addr_t       dst_addr;   // [38:22] 17-bit
    length_t     length;     // [21: 6] 16-bit
    logic [ 4:0] flags;      // [ 5: 1]  5-bit
    async_e      async;      // [ 0]     1-bit
  } cvo_op_x64_t;

  // ===| CVO Function Codes (ISA §3.4.1) |=======================================
  typedef enum logic [3:0] {
    CVO_EXP        = 4'h0,
    CVO_SQRT       = 4'h1,
    CVO_GELU       = 4'h2,
    CVO_SIN        = 4'h3,
    CVO_COS        = 4'h4,
    CVO_REDUCE_SUM = 4'h5,
    CVO_SCALE      = 4'h6,
    CVO_RECIP      = 4'h7
  } cvo_func_e;

  // ===| CVO Flags (5-bit, ISA §3.4.2) |=========================================
  typedef struct packed {
    logic sub_emax;     // [4] subtract e_max before operation
    logic recip_scale;  // [3] use reciprocal of scalar (divide instead of multiply)
    logic accm;         // [2] accumulate into dst
    logic [1:0] reserved;
  } cvo_flags_t;

  // ===| Memory Routing (ISA §5) |================================================
  // Each route encodes source[7:4] | dest[3:0] as an 8-bit enum.

  typedef enum logic [3:0] {
    data_to_host             = 4'h0,
    data_to_GLOBAL_cache     = 4'h1,
    data_to_L1_cache_GEMM_in = 4'h2,
    data_to_L1_cache_GEMV_in = 4'h3,
    data_to_CVO_in           = 4'h4
  } data_dest_e;

  typedef enum logic [3:0] {
    data_from_host              = 4'h0,
    data_from_GLOBAL_cache      = 4'h1,
    data_from_L1_cache_GEMM_res = 4'h2,
    data_from_L1_cache_GEMV_res = 4'h3,
    data_from_CVO_res           = 4'h4
  } data_source_e;

  typedef enum logic [7:0] {
    from_host_to_L2     = {data_from_host,              data_to_GLOBAL_cache    },
    from_L2_to_host     = {data_from_GLOBAL_cache,      data_to_host            },
    from_L2_to_L1_GEMM  = {data_from_GLOBAL_cache,      data_to_L1_cache_GEMM_in},
    from_L2_to_L1_GEMV  = {data_from_GLOBAL_cache,      data_to_L1_cache_GEMV_in},
    from_L2_to_CVO      = {data_from_GLOBAL_cache,      data_to_CVO_in          },
    from_GEMV_res_to_L2 = {data_from_L1_cache_GEMV_res, data_to_GLOBAL_cache    },
    from_GEMM_res_to_L2 = {data_from_L1_cache_GEMM_res, data_to_GLOBAL_cache    },
    from_CVO_res_to_L2  = {data_from_CVO_res,           data_to_GLOBAL_cache    }
  } data_route_e;

  typedef enum logic [1:0] {
    data_to_fmap_shape   = 2'h0,
    data_to_weight_shape = 2'h1
  } dest_cache_e;

  // ===| Micro-Op Structures (ISA §6) |==========================================

  localparam int MemoryUopWidth = 49;  // 8+17+17+6+1

  // GEMM control uop  (ISA §6.1)
  typedef struct packed {
    flags_t         flags;
    ptr_addr_t      size_ptr_addr;
    parallel_lane_t parallel_lane;
  } gemm_control_uop_t;

  // GEMV control uop  (same layout as GEMM)
  typedef struct packed {
    flags_t         flags;
    ptr_addr_t      size_ptr_addr;
    parallel_lane_t parallel_lane;
  } GEMV_control_uop_t;

  // Memory control uop  (ISA §6.2)
  typedef struct packed {
    data_route_e data_dest;       //  8-bit
    dest_addr_t  dest_addr;       // 17-bit
    src_addr_t   src_addr;        // 17-bit
    ptr_addr_t   shape_ptr_addr;  //  6-bit
    async_e      async;           //  1-bit
  } memory_control_uop_t;

  // Memory set uop  (ISA §6.3)
  typedef struct packed {
    dest_cache_e dest_cache;  //  2-bit
    ptr_addr_t   dest_addr;   //  6-bit
    a_value_t    a_value;     // 16-bit
    b_value_t    b_value;     // 16-bit
    c_value_t    c_value;     // 16-bit
  } memory_set_uop_t;

  // CVO control uop  (ISA §6.4)
  typedef struct packed {
    cvo_func_e  cvo_func;   //  4-bit
    src_addr_t  src_addr;   // 17-bit
    addr_t      dst_addr;   // 17-bit
    length_t    length;     // 16-bit
    cvo_flags_t flags;      //  5-bit
    async_e     async;      //  1-bit
  } cvo_control_uop_t;

  // ===| ACP / NPU Transfer uops (used by mem_dispatcher) |======================
  typedef struct packed {
    logic        write_en;
    logic [16:0] base_addr;
    logic [16:0] end_addr;
  } acp_uop_t;  // 35-bit

  typedef struct packed {
    logic        write_en;
    logic [16:0] base_addr;
    logic [16:0] end_addr;
  } npu_uop_t;  // 35-bit

endpackage : isa_pkg
