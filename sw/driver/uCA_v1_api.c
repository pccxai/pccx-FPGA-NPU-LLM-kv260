// ===| uCA API Implementation |==================================================
// Builds 64-bit VLIW instructions from structured arguments and issues them
// to the NPU via the HAL. Encoding per docs/ISA.md.
// ================================================================================

#include "uCA_v1_api.h"
#include "uCA_v1_hal.h"

// ===| Instruction Builder Helpers |=============================================

// ===| build_compute_instr |===
// Packs GEMV or GEMM instruction into a 64-bit word.
// Layout (ISA.md §3.1):
//   [63:60] opcode        4-bit
//   [59:43] dest_reg     17-bit
//   [42:26] src_addr     17-bit
//   [25:20] flags         6-bit
//   [19:14] size_ptr      6-bit
//   [13:8]  shape_ptr     6-bit
//   [7:3]   lanes         5-bit
//   [2:0]   reserved      3-bit
static uint64_t build_compute_instr(uint8_t  opcode,    uint32_t dest_reg,
                                    uint32_t src_addr,  uint8_t  flags,
                                    uint8_t  size_ptr,  uint8_t  shape_ptr,
                                    uint8_t  lanes) {
    uint64_t instr = 0;
    instr |= ((uint64_t)(opcode    & 0xF)     << 60);
    instr |= ((uint64_t)(dest_reg  & 0x1FFFF) << 43);
    instr |= ((uint64_t)(src_addr  & 0x1FFFF) << 26);
    instr |= ((uint64_t)(flags     & 0x3F)    << 20);
    instr |= ((uint64_t)(size_ptr  & 0x3F)    << 14);
    instr |= ((uint64_t)(shape_ptr & 0x3F)    <<  8);
    instr |= ((uint64_t)(lanes     & 0x1F)    <<  3);
    return instr;
}

// ===| build_cvo_instr |===
// Packs a CVO instruction into a 64-bit word.
// Layout (ISA.md §3.4):
//   [63:60] opcode (UCA_OP_CVO = 4'h4)   4-bit
//   [59:56] cvo_func                      4-bit
//   [55:39] src_addr                     17-bit
//   [38:22] dst_addr                     17-bit
//   [21:6]  length                       16-bit
//   [5:1]   flags                         5-bit
//   [0]     async                         1-bit
static uint64_t build_cvo_instr(uint8_t  cvo_func,  uint32_t src_addr,
                                uint32_t dst_addr,  uint16_t length,
                                uint8_t  flags,     uint8_t  async) {
    uint64_t instr = 0;
    instr |= ((uint64_t)(UCA_OP_CVO & 0xF)   << 60);
    instr |= ((uint64_t)(cvo_func  & 0xF)    << 56);
    instr |= ((uint64_t)(src_addr  & 0x1FFFF)<< 39);
    instr |= ((uint64_t)(dst_addr  & 0x1FFFF)<< 22);
    instr |= ((uint64_t)(length    & 0xFFFF) <<  6);
    instr |= ((uint64_t)(flags     & 0x1F)   <<  1);
    instr |= ((uint64_t)(async     & 0x1)    <<  0);
    return instr;
}

// ===| API Init |================================================================
int uca_init(void) {
    return uca_hal_init();
}

void uca_deinit(void) {
    uca_hal_deinit();
}

// ===| Compute: Vector Core (GEMV) |=============================================
void uca_gemv(uint32_t dest_reg,   uint32_t src_addr,
              uint8_t  flags,      uint8_t  size_ptr,
              uint8_t  shape_ptr,  uint8_t  lanes) {
    uint64_t instr = build_compute_instr(UCA_OP_GEMV, dest_reg, src_addr,
                                         flags, size_ptr, shape_ptr, lanes);
    uca_hal_issue_instr(instr);
}

// ===| Compute: Matrix Core (GEMM) |=============================================
void uca_gemm(uint32_t dest_reg,   uint32_t src_addr,
              uint8_t  flags,      uint8_t  size_ptr,
              uint8_t  shape_ptr,  uint8_t  lanes) {
    uint64_t instr = build_compute_instr(UCA_OP_GEMM, dest_reg, src_addr,
                                         flags, size_ptr, shape_ptr, lanes);
    uca_hal_issue_instr(instr);
}

// ===| Compute: CVO Core (Complex Vector Operation Core) |=======================
void uca_cvo(uint8_t  cvo_func,  uint32_t src_addr,
             uint32_t dst_addr,  uint16_t length,
             uint8_t  flags,     uint8_t  async) {
    uint64_t instr = build_cvo_instr(cvo_func, src_addr, dst_addr,
                                      length, flags, async);
    uca_hal_issue_instr(instr);
}

// ===| Memory: MEMCPY |=========================================================
void uca_memcpy(uint8_t  route,    uint32_t dest_addr,
                uint32_t src_addr, uint8_t  shape_ptr,
                uint8_t  async) {
    // Layout (ISA.md §3.2):
    //   [63:60] opcode     4-bit
    //   [59]    from_dev   1-bit  (upper nibble of route)
    //   [58]    to_dev     1-bit  (lower nibble of route)
    //   [57:41] dest_addr 17-bit
    //   [40:24] src_addr  17-bit
    //   [23:7]  aux_addr  17-bit  (reserved, zero)
    //   [6:1]   shape_ptr  6-bit
    //   [0]     async      1-bit
    uint8_t from_dev = (route >> 4) & 0xF;
    uint8_t to_dev   = (route >> 0) & 0xF;

    uint64_t instr = 0;
    instr |= ((uint64_t)(UCA_OP_MEMCPY & 0xF) << 60);
    instr |= ((uint64_t)(from_dev  & 0x1)      << 59);
    instr |= ((uint64_t)(to_dev    & 0x1)      << 58);
    instr |= ((uint64_t)(dest_addr & 0x1FFFF)  << 41);
    instr |= ((uint64_t)(src_addr  & 0x1FFFF)  << 24);
    // aux_addr [23:7] left as zero
    instr |= ((uint64_t)(shape_ptr & 0x3F)     <<  1);
    instr |= ((uint64_t)(async     & 0x1)      <<  0);
    uca_hal_issue_instr(instr);
}

// ===| Memory: MEMSET |=========================================================
void uca_memset(uint8_t  dest_cache, uint8_t  dest_addr,
                uint16_t a,          uint16_t b,  uint16_t c) {
    // Layout (ISA.md §3.3):
    //   [63:60] opcode      4-bit
    //   [59:58] dest_cache  2-bit
    //   [57:52] dest_addr   6-bit
    //   [51:36] a_value    16-bit
    //   [35:20] b_value    16-bit
    //   [19:4]  c_value    16-bit
    //   [3:0]   reserved    4-bit
    uint64_t instr = 0;
    instr |= ((uint64_t)(UCA_OP_MEMSET & 0xF) << 60);
    instr |= ((uint64_t)(dest_cache & 0x3)     << 58);
    instr |= ((uint64_t)(dest_addr  & 0x3F)    << 52);
    instr |= ((uint64_t)(a          & 0xFFFF)  << 36);
    instr |= ((uint64_t)(b          & 0xFFFF)  << 20);
    instr |= ((uint64_t)(c          & 0xFFFF)  <<  4);
    uca_hal_issue_instr(instr);
}

// ===| Synchronization |=========================================================
int uca_sync(uint32_t timeout_us) {
    return uca_hal_wait_idle(timeout_us);
}
