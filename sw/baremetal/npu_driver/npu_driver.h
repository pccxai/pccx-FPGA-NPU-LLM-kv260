// PCCX(TM) - reusable AI accelerator project.
// SPDX-FileCopyrightText: 2026 Hyun Woo Kim
// SPDX-License-Identifier: Apache-2.0

#ifndef PCCX_BAREMETAL_NPU_DRIVER_H
#define PCCX_BAREMETAL_NPU_DRIVER_H

#include "xil_types.h"

#define PCCX_NPU_AXIL_BASE             0xA0000000UL
#define PCCX_NPU_CMDSTS_HP0_BASE       0xA0001000UL
#define PCCX_NPU_CMDSTS_HP1_BASE       0xA0002000UL
#define PCCX_NPU_CMDSTS_HP2_BASE       0xA0003000UL
#define PCCX_NPU_CMDSTS_HP3_BASE       0xA0004000UL
#define PCCX_NPU_CMDSTS_ACP_FMAP_BASE  0xA0005000UL
#define PCCX_NPU_CMDSTS_ACP_RESULT_BASE 0xA0006000UL

#define PCCX_NPU_REG_INSTRUCTION       0x000U
#define PCCX_NPU_REG_KICK              0x008U
#define PCCX_NPU_REG_STATUS            0x010U
#define PCCX_NPU_CMDSTS_FLAGS          0x014U

#define PCCX_NPU_KICK_WORD             0x8000000000000000ULL
#define PCCX_NPU_STATUS_BUSY           (1ULL << 0)
#define PCCX_NPU_STATUS_DONE           (1ULL << 1)
#define PCCX_NPU_STATUS_ERROR          (1ULL << 2)
#define PCCX_NPU_STATUS_TOKEN_VALID    (1ULL << 3)
#define PCCX_NPU_STATUS_TOKEN_SHIFT    32U

#define PCCX_HOST_OPCODE_RESET_KV      0x01U
#define PCCX_HOST_OPCODE_LOAD_WEIGHT   0x02U
#define PCCX_HOST_OPCODE_LOAD_PROMPT   0x03U
#define PCCX_HOST_OPCODE_NEXT_TOKEN    0x04U

typedef enum {
    PCCX_NPU_DMA_TO_NPU = 0,
    PCCX_NPU_DMA_FROM_NPU = 1,
} pccx_npu_dma_direction;

typedef struct {
    UINTPTR npu_base;
    UINTPTR cmdsts_hp0_base;
    UINTPTR cmdsts_hp1_base;
    UINTPTR cmdsts_hp2_base;
    UINTPTR cmdsts_hp3_base;
    UINTPTR cmdsts_acp_fmap_base;
    UINTPTR cmdsts_acp_result_base;
} pccx_npu_config;

typedef struct {
    const pccx_npu_config *cfg;
} pccx_npu;

typedef struct __attribute__((aligned(64))) {
    u64 src_addr;
    u64 dst_addr;
    u32 bytes;
    u32 flags;
    u32 route;
    u32 tag;
} pccx_npu_dma_desc;

extern const pccx_npu_config pccx_npu_default_config;

int pccx_npu_init(pccx_npu *npu, const pccx_npu_config *cfg);

void pccx_npu_write32(const pccx_npu *npu, u32 offset, u32 value);
u32 pccx_npu_read32(const pccx_npu *npu, u32 offset);
void pccx_npu_write64(const pccx_npu *npu, u32 offset, u64 value);
u64 pccx_npu_read64(const pccx_npu *npu, u32 offset);

void pccx_npu_issue_word(const pccx_npu *npu, u64 word);
u64 pccx_npu_read_status(const pccx_npu *npu);
int pccx_npu_wait_idle(const pccx_npu *npu, u32 poll_limit);
int pccx_npu_read_token(const pccx_npu *npu, u32 *token_out);

void pccx_npu_flush_range(const void *addr, u32 bytes);
void pccx_npu_invalidate_range(const void *addr, u32 bytes);

void pccx_npu_dma_desc_setup(pccx_npu_dma_desc *desc,
                             UINTPTR src_addr,
                             UINTPTR dst_addr,
                             u32 bytes,
                             pccx_npu_dma_direction direction,
                             u32 tag);

u64 pccx_npu_encode_reset_kv_cache(u32 session_id);
u64 pccx_npu_encode_load_weight(u8 weight_slot,
                                u8 descriptor_count,
                                u32 manifest_id,
                                u8 flags);
u64 pccx_npu_encode_load_prompt(u32 position, u32 token_id);
u64 pccx_npu_encode_next_token(u16 request_id,
                               u8 sampling,
                               u16 temperature_q8_8,
                               u16 top_p_q8_8);

#endif
