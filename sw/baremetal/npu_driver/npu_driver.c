// PCCX(TM) - reusable AI accelerator project.
// SPDX-FileCopyrightText: 2026 Hyun Woo Kim
// SPDX-License-Identifier: Apache-2.0

#include "npu_driver.h"

#include "xil_cache.h"
#include "xil_io.h"

const pccx_npu_config pccx_npu_default_config = {
    .npu_base = PCCX_NPU_AXIL_BASE,
    .cmdsts_hp0_base = PCCX_NPU_CMDSTS_HP0_BASE,
    .cmdsts_hp1_base = PCCX_NPU_CMDSTS_HP1_BASE,
    .cmdsts_hp2_base = PCCX_NPU_CMDSTS_HP2_BASE,
    .cmdsts_hp3_base = PCCX_NPU_CMDSTS_HP3_BASE,
    .cmdsts_acp_fmap_base = PCCX_NPU_CMDSTS_ACP_FMAP_BASE,
    .cmdsts_acp_result_base = PCCX_NPU_CMDSTS_ACP_RESULT_BASE,
};

static UINTPTR npu_addr(const pccx_npu *npu, u32 offset)
{
    return npu->cfg->npu_base + (UINTPTR)offset;
}

static u64 pack_host_command(u8 opcode, u64 operand)
{
    return ((u64)(opcode & 0xffU) << 56) | (operand & 0x00ffffffffffffffULL);
}

int pccx_npu_init(pccx_npu *npu, const pccx_npu_config *cfg)
{
    npu->cfg = cfg ? cfg : &pccx_npu_default_config;
    return 0;
}

void pccx_npu_write32(const pccx_npu *npu, u32 offset, u32 value)
{
    Xil_Out32(npu_addr(npu, offset), value);
}

u32 pccx_npu_read32(const pccx_npu *npu, u32 offset)
{
    return Xil_In32(npu_addr(npu, offset));
}

void pccx_npu_write64(const pccx_npu *npu, u32 offset, u64 value)
{
#if defined(__aarch64__)
    Xil_Out64(npu_addr(npu, offset), value);
#else
    Xil_Out32(npu_addr(npu, offset), (u32)(value & 0xffffffffULL));
    Xil_Out32(npu_addr(npu, offset + 4U), (u32)(value >> 32));
#endif
}

u64 pccx_npu_read64(const pccx_npu *npu, u32 offset)
{
#if defined(__aarch64__)
    return Xil_In64(npu_addr(npu, offset));
#else
    u64 lo = Xil_In32(npu_addr(npu, offset));
    u64 hi = Xil_In32(npu_addr(npu, offset + 4U));
    return lo | (hi << 32);
#endif
}

void pccx_npu_issue_word(const pccx_npu *npu, u64 word)
{
    pccx_npu_write64(npu, PCCX_NPU_REG_INSTRUCTION, word);
    pccx_npu_write64(npu, PCCX_NPU_REG_KICK, PCCX_NPU_KICK_WORD);
}

u64 pccx_npu_read_status(const pccx_npu *npu)
{
    return pccx_npu_read64(npu, PCCX_NPU_REG_STATUS);
}

int pccx_npu_wait_idle(const pccx_npu *npu, u32 poll_limit)
{
    while (poll_limit > 0U) {
        u64 status = pccx_npu_read_status(npu);
        if ((status & PCCX_NPU_STATUS_ERROR) != 0ULL) {
            return -2;
        }
        if ((status & PCCX_NPU_STATUS_BUSY) == 0ULL) {
            return 0;
        }
        poll_limit--;
    }
    return -1;
}

int pccx_npu_read_token(const pccx_npu *npu, u32 *token_out)
{
    u64 status = pccx_npu_read_status(npu);
    if ((status & PCCX_NPU_STATUS_TOKEN_VALID) == 0ULL) {
        return -1;
    }
    *token_out = (u32)(status >> PCCX_NPU_STATUS_TOKEN_SHIFT);
    return 0;
}

void pccx_npu_flush_range(const void *addr, u32 bytes)
{
    Xil_DCacheFlushRange((UINTPTR)addr, bytes);
}

void pccx_npu_invalidate_range(const void *addr, u32 bytes)
{
    Xil_DCacheInvalidateRange((UINTPTR)addr, bytes);
}

void pccx_npu_dma_desc_setup(pccx_npu_dma_desc *desc,
                             UINTPTR src_addr,
                             UINTPTR dst_addr,
                             u32 bytes,
                             pccx_npu_dma_direction direction,
                             u32 tag)
{
    desc->src_addr = (u64)src_addr;
    desc->dst_addr = (u64)dst_addr;
    desc->bytes = bytes;
    desc->flags = (direction == PCCX_NPU_DMA_TO_NPU) ? 0U : 1U;
    desc->route = (direction == PCCX_NPU_DMA_TO_NPU) ? 0U : 1U;
    desc->tag = tag;
    pccx_npu_flush_range(desc, (u32)sizeof(*desc));
}

u64 pccx_npu_encode_reset_kv_cache(u32 session_id)
{
    return pack_host_command(PCCX_HOST_OPCODE_RESET_KV, (u64)session_id);
}

u64 pccx_npu_encode_load_weight(u8 weight_slot,
                                u8 descriptor_count,
                                u32 manifest_id,
                                u8 flags)
{
    u64 operand = ((u64)weight_slot << 48)
                | ((u64)flags << 40)
                | ((u64)descriptor_count << 32)
                | (u64)manifest_id;
    return pack_host_command(PCCX_HOST_OPCODE_LOAD_WEIGHT, operand);
}

u64 pccx_npu_encode_load_prompt(u32 position, u32 token_id)
{
    u64 operand = (((u64)position & 0x00ffffffULL) << 32) | (u64)token_id;
    return pack_host_command(PCCX_HOST_OPCODE_LOAD_PROMPT, operand);
}

u64 pccx_npu_encode_next_token(u16 request_id,
                               u8 sampling,
                               u16 temperature_q8_8,
                               u16 top_p_q8_8)
{
    u64 operand = ((u64)request_id << 40)
                | ((u64)sampling << 32)
                | ((u64)temperature_q8_8 << 16)
                | (u64)top_p_q8_8;
    return pack_host_command(PCCX_HOST_OPCODE_NEXT_TOKEN, operand);
}
