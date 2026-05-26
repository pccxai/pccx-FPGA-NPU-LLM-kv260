// PCCX(TM) - reusable AI accelerator project.
// SPDX-FileCopyrightText: 2026 Hyun Woo Kim
// SPDX-License-Identifier: Apache-2.0

#include "npu_driver/npu_driver.h"

#include "xil_types.h"

#define GEMMA_WEIGHT_DESC_COUNT       4U
#define GEMMA_WEIGHT_STAGE_BYTES      (4U * 1024U * 1024U)
#define GEMMA_ACTIVATION_STAGE_BYTES  (1U * 1024U * 1024U)
#define GEMMA_MAX_PROMPT_TOKENS       16U
#define GEMMA_MAX_NEW_TOKENS          1U
#define GEMMA_SESSION_ID              1U
#define GEMMA_MANIFEST_ID             0x000e4b00U
#define GEMMA_POLL_LIMIT              1000000U

extern u8 __pccx_weight_buffer_start[];
extern u8 __pccx_weight_buffer_end[];
extern u8 __pccx_activation_buffer_start[];
extern u8 __pccx_activation_buffer_end[];

static u8 g_weight_stage[GEMMA_WEIGHT_STAGE_BYTES]
    __attribute__((section(".pccx_weights"), aligned(64)));
static u8 g_activation_stage[GEMMA_ACTIVATION_STAGE_BYTES]
    __attribute__((section(".pccx_activations"), aligned(64)));
static pccx_npu_dma_desc g_weight_desc[GEMMA_WEIGHT_DESC_COUNT]
    __attribute__((section(".pccx_activations"), aligned(64)));
static u32 g_prompt_tokens[GEMMA_MAX_PROMPT_TOKENS]
    __attribute__((section(".pccx_activations"), aligned(64)));
static u32 g_generated_tokens[GEMMA_MAX_NEW_TOKENS]
    __attribute__((section(".pccx_activations"), aligned(64)));
static volatile u32 g_boot_cookie = 0x50434358U;

static int gemma_check_memory_layout(void)
{
    UINTPTR weight_start = (UINTPTR)__pccx_weight_buffer_start;
    UINTPTR weight_end = (UINTPTR)__pccx_weight_buffer_end;
    UINTPTR activation_start = (UINTPTR)__pccx_activation_buffer_start;
    UINTPTR activation_end = (UINTPTR)__pccx_activation_buffer_end;

    if ((weight_start + GEMMA_WEIGHT_STAGE_BYTES) > weight_end) {
        return -1;
    }
    if ((activation_start + GEMMA_ACTIVATION_STAGE_BYTES) > activation_end) {
        return -2;
    }
    return 0;
}

static u32 gemma_tokenize_ascii_prompt(const char *prompt, u32 *tokens, u32 max_tokens)
{
    u32 count = 0U;
    while ((prompt[count] != '\0') && (count < max_tokens)) {
        tokens[count] = (u32)(u8)prompt[count];
        count++;
    }
    return count;
}

static void gemma_prepare_weight_descriptors(void)
{
    u32 chunk = GEMMA_WEIGHT_STAGE_BYTES / GEMMA_WEIGHT_DESC_COUNT;
    u32 index;

    for (index = 0U; index < GEMMA_WEIGHT_DESC_COUNT; index++) {
        UINTPTR src = (UINTPTR)&g_weight_stage[index * chunk];
        UINTPTR dst = (UINTPTR)&__pccx_weight_buffer_start[index * chunk];
        pccx_npu_dma_desc_setup(&g_weight_desc[index],
                                src,
                                dst,
                                chunk,
                                PCCX_NPU_DMA_TO_NPU,
                                index);
    }
}

static int gemma_reset_session(const pccx_npu *npu)
{
    u64 command = pccx_npu_encode_reset_kv_cache(GEMMA_SESSION_ID);
    pccx_npu_issue_word(npu, command);
    return pccx_npu_wait_idle(npu, GEMMA_POLL_LIMIT);
}

static int gemma_load_weights(const pccx_npu *npu)
{
    u64 command;

    pccx_npu_flush_range(g_weight_stage, GEMMA_WEIGHT_STAGE_BYTES);
    pccx_npu_flush_range(g_weight_desc, (u32)sizeof(g_weight_desc));

    command = pccx_npu_encode_load_weight(0U,
                                          (u8)GEMMA_WEIGHT_DESC_COUNT,
                                          GEMMA_MANIFEST_ID,
                                          0U);
    pccx_npu_issue_word(npu, command);
    return pccx_npu_wait_idle(npu, GEMMA_POLL_LIMIT);
}

static int gemma_load_prompt(const pccx_npu *npu, const u32 *tokens, u32 token_count)
{
    u32 index;

    pccx_npu_flush_range(tokens, token_count * (u32)sizeof(tokens[0]));

    for (index = 0U; index < token_count; index++) {
        u64 command = pccx_npu_encode_load_prompt(index, tokens[index]);
        pccx_npu_issue_word(npu, command);
        if (pccx_npu_wait_idle(npu, GEMMA_POLL_LIMIT) != 0) {
            return -1;
        }
    }
    return 0;
}

static int gemma_forward_one_token(const pccx_npu *npu, u32 request_id, u32 *token_out)
{
    u64 command = pccx_npu_encode_next_token((u16)request_id,
                                             0U,
                                             0U,
                                             0x0100U);
    pccx_npu_issue_word(npu, command);
    if (pccx_npu_wait_idle(npu, GEMMA_POLL_LIMIT) != 0) {
        return -1;
    }
    return pccx_npu_read_token(npu, token_out);
}

static int gemma_readback(const pccx_npu *npu)
{
    (void)npu;
    pccx_npu_invalidate_range(g_generated_tokens, (u32)sizeof(g_generated_tokens));
    pccx_npu_invalidate_range(g_activation_stage, GEMMA_ACTIVATION_STAGE_BYTES);
    return 0;
}

int main(void)
{
    pccx_npu npu;
    u32 token_count;
    int rc;

    rc = pccx_npu_init(&npu, &pccx_npu_default_config);
    if (rc != 0) {
        return rc;
    }

    if (g_boot_cookie != 0x50434358U) {
        return -1;
    }

    rc = gemma_check_memory_layout();
    if (rc != 0) {
        return rc;
    }

    rc = gemma_reset_session(&npu);
    if (rc != 0) {
        return rc;
    }

    gemma_prepare_weight_descriptors();

    rc = gemma_load_weights(&npu);
    if (rc != 0) {
        return rc;
    }

    token_count = gemma_tokenize_ascii_prompt("PCCX", g_prompt_tokens, GEMMA_MAX_PROMPT_TOKENS);
    rc = gemma_load_prompt(&npu, g_prompt_tokens, token_count);
    if (rc != 0) {
        return rc;
    }

    rc = gemma_forward_one_token(&npu, 0U, &g_generated_tokens[0]);
    if (rc != 0) {
        return rc;
    }

    return gemma_readback(&npu);
}
