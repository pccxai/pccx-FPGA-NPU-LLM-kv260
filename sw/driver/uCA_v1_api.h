// ===| uCA API (High-Level Driver Interface) |====================================
// uCA: micro Compute Architecture — AI model acceleration API for FPGA NPU.
//
// This is the "CUDA equivalent" for the uCA NPU. Application code
// (sw/gemma3NE4B/ and future projects) should call only functions from
// this layer — never touch the HAL directly.
//
// This layer builds 64-bit VLIW instructions per the pccx v002 ISA and
// issues them via the HAL. The NPU frontend is fully decoupled: each
// uca_* call returns immediately after issuing to the instruction FIFO.
// Call uca_sync() to wait for all in-flight operations to complete.
//
// Encoding reference: https://pccxai.github.io/pccx/en/docs/v002/ISA/index.html
// ================================================================================

#ifndef UCA_V1_API_H
#define UCA_V1_API_H

#include <stdint.h>

// ===| Opcode Definitions |======================================================
// Must match ISA.md §2 and isa_x64.svh opcode_e
#define UCA_OP_GEMV    0x0
#define UCA_OP_GEMM    0x1
#define UCA_OP_MEMCPY  0x2
#define UCA_OP_MEMSET  0x3
#define UCA_OP_CVO     0x4

// ===| GEMV / GEMM Flags (6-bit) |===============================================
// Must match ISA.md §4 and flags_t in isa_x64.svh
#define UCA_FLAG_FINDEMAX  (1U << 5)  // Find e_max over output (for softmax)
#define UCA_FLAG_ACCM      (1U << 4)  // Accumulate into dest (do not overwrite)
#define UCA_FLAG_W_SCALE   (1U << 3)  // Apply weight scale factor during MAC

// ===| CVO Function Codes (4-bit) |==============================================
// Must match ISA.md §3.4.1 and cvo_func_e in isa_cvo.svh
#define UCA_CVO_EXP          0x0  // Element-wise exp(x)             — SFU
#define UCA_CVO_SQRT         0x1  // Element-wise sqrt(x)            — SFU
#define UCA_CVO_GELU         0x2  // Element-wise GELU(x)            — SFU
#define UCA_CVO_SIN          0x3  // Element-wise sin(x)             — CORDIC
#define UCA_CVO_COS          0x4  // Element-wise cos(x)             — CORDIC
#define UCA_CVO_REDUCE_SUM   0x5  // Sum reduction → scalar at dst   — SFU+Adder
#define UCA_CVO_SCALE        0x6  // Element-wise multiply by scalar — SFU
#define UCA_CVO_RECIP        0x7  // Element-wise 1/x                — SFU

// ===| CVO Flags (5-bit) |=======================================================
// Must match ISA.md §3.4.2 and cvo_flags_t in isa_cvo.svh
#define UCA_CVO_FLAG_SUB_EMAX      (1U << 4)  // Subtract e_max before operation
#define UCA_CVO_FLAG_RECIP_SCALE   (1U << 3)  // Use 1/scalar for SCALE op
#define UCA_CVO_FLAG_ACCM          (1U << 2)  // Accumulate into dst

// ===| Memory Route Codes |======================================================
// Must match ISA.md §5 and data_route_e in isa_memctrl.svh
// Upper nibble = from_device, lower nibble = to_device
#define UCA_ROUTE_HOST_TO_L2        0x01
#define UCA_ROUTE_L2_TO_HOST        0x10
#define UCA_ROUTE_L2_TO_L1_GEMM    0x12
#define UCA_ROUTE_L2_TO_L1_GEMV    0x13
#define UCA_ROUTE_GEMV_RES_TO_L2   0x31
#define UCA_ROUTE_GEMM_RES_TO_L2   0x21
#define UCA_ROUTE_CVO_RES_TO_L2    0x41

// ===| API Init |================================================================
int  uca_init(void);    // Calls uca_hal_init() and verifies NPU is responsive
void uca_deinit(void);

// ===| Compute: Vector Core (GEMV) |=============================================
// Issue a GEMV instruction (INT4 weight × BF16/INT8 activation → BF16 out).
//
//   dest_reg   : destination register / L2 address (17-bit)
//   src_addr   : source fmap address (17-bit)
//   flags      : OR of UCA_FLAG_* constants
//   size_ptr   : pointer to size descriptor in shape cache (6-bit)
//   shape_ptr  : pointer to shape descriptor in shape cache (6-bit)
//   lanes      : number of active parallel μV-Core lanes (5-bit, 1–4)
void uca_gemv(uint32_t dest_reg,   uint32_t src_addr,
              uint8_t  flags,      uint8_t  size_ptr,
              uint8_t  shape_ptr,  uint8_t  lanes);

// ===| Compute: Matrix Core (GEMM) |=============================================
// Issue a GEMM instruction (systolic 32×32 array).
// Same field layout as GEMV; differs only in opcode routing.
void uca_gemm(uint32_t dest_reg,   uint32_t src_addr,
              uint8_t  flags,      uint8_t  size_ptr,
              uint8_t  shape_ptr,  uint8_t  lanes);

// ===| Compute: CVO Core (Complex Vector Operations) |==========================
// Issue a CVO instruction to one of the 2× μCVO-Cores.
// Used for: softmax (EXP, REDUCE_SUM, SCALE), RMSNorm (SQRT, RECIP, SCALE),
//           activation functions (GELU), attention (SIN/COS for RoPE).
//
//   cvo_func   : one of UCA_CVO_* function codes
//   src_addr   : source address in L2 cache (17-bit)
//   dst_addr   : destination address in L2 cache (17-bit)
//   length     : number of elements to process (16-bit)
//   flags      : OR of UCA_CVO_FLAG_* constants
//   async      : 0=block until done, 1=fire-and-forget
void uca_cvo(uint8_t  cvo_func,   uint32_t src_addr,
             uint32_t dst_addr,   uint16_t length,
             uint8_t  flags,      uint8_t  async);

// ===| Memory: MEMCPY |=========================================================
// Issue a DMA transfer between host and NPU memory, or between NPU caches.
//
//   route      : one of UCA_ROUTE_* constants
//   dest_addr  : destination address (17-bit)
//   src_addr   : source address (17-bit)
//   shape_ptr  : pointer to shape descriptor (6-bit)
//   async      : 0=blocking, 1=fire-and-forget
void uca_memcpy(uint8_t  route,    uint32_t dest_addr,
                uint32_t src_addr, uint8_t  shape_ptr,
                uint8_t  async);

// ===| Memory: MEMSET |=========================================================
// Set shape descriptor values in the shape cache.
//
//   dest_cache : 0=fmap_shape cache, 1=weight_shape cache
//   dest_addr  : target pointer address in the shape cache (6-bit)
//   a, b, c    : values to write (16-bit each, typically dimension sizes)
void uca_memset(uint8_t  dest_cache, uint8_t  dest_addr,
                uint16_t a,          uint16_t b,  uint16_t c);

// ===| Synchronization |=========================================================
// Block until all issued instructions complete (polls UCA_STAT_BUSY).
// Returns 0 on success, -1 on timeout.
int uca_sync(uint32_t timeout_us);

#endif // UCA_V1_API_H
