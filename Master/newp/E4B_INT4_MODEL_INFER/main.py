import numpy as np
import CPU_CORE
import safeTensor
import math
import IGPU_CORE
import gc
import sys


# see ram usage
import os
import psutil

ACCEL_MODE = "IGPU"
_IGPU_WEIGHT_KEYS = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"]

def hw_matmul(x, w, use_gelu=False):
    if ACCEL_MODE == "IGPU":
        return IGPU_CORE.igpu_matmul_gelu(x, w) if use_gelu else IGPU_CORE.igpu_matmul(x, w)
    else:
        if isinstance(w, tuple):
            packed, scale = w
            low = (packed & 0x0F).astype(np.int8)
            low[low > 7] -= 16
            high = (packed >> 4).astype(np.int8)
            high[high > 7] -= 16
            res = np.empty((packed.shape[0], packed.shape[1]*2), dtype=np.float32)
            res[:, 0::2] = low
            res[:, 1::2] = high
            w_real = res * scale[:, np.newaxis]
            out = np.dot(x, w_real.T)
        else:
            out = np.dot(x, w)
        return CPU_CORE.gelu(out) if use_gelu else out

def rms_norm(x, gamma):
    x_f32 = x.astype(np.float32)
    rms   = np.sqrt(np.mean(x_f32 ** 2) + 1e-6)
    return (x_f32 / rms) * gamma

def get_router_modalities(x, w_norm, w_router):
    x_n = rms_norm(x, w_norm) / 2048.0
    return np.tanh(np.dot(x_n, w_router))

def forward_one_token(token_id, pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                      W_ple_proj, altup_projs, K_cache, V_cache):

    # 1. 단어 사전 범위(256,000)를 넘지 않게 인덱스 안전장치 적용
    safe_token_id = int(min(token_id, W_ple_packed.shape[0] - 1))

    # ==========================================================
    # Phase 1: W_embed (mmap) 디스크에서 딱 1KB(1줄)만 로드해서 입력 벡터 x0 만들기
    # W_embed는 튜플이므로 [0]번이 packed, [1]번이 scale 입니다.
    # ==========================================================
    x0 = CPU_CORE.embedding(safe_token_id, W_embed[0], W_embed[1])
    x0 = x0 * math.sqrt(2048.0)
    # 4-Stream 라우팅 (xs 배열 초기화)
    xs = np.zeros((4, 2048), dtype=np.float32)
    xs[0] = x0
    for k in range(3):
        xs[k + 1] = np.dot(x0, altup_projs[k])
        
    # PLE (Per-Layer Embedding) 투영 및 정규화
    x_proj = hw_matmul(x0, W_ple_proj) / math.sqrt(2048.0)
    x_proj = x_proj.reshape(35, 256)
    x_proj_f32 = x_proj.astype(np.float32)
    rms_vals   = np.sqrt(np.mean(x_proj_f32 ** 2, axis=1, keepdims=True) + 1e-6)
    x_proj_normed = (x_proj_f32 / rms_vals) * norm_ple

    # ==========================================================
    # Phase 1.5: W_ple (mmap) 디스크에서 딱 4.5KB(1줄)만 로드해서 압축 풀기
    # ==========================================================
    unpacked_w_ple = CPU_CORE.embedding(safe_token_id, W_ple_packed, W_ple_scale)

    # 35개 레이어용으로 형태(Shape)를 바꾸고 스케일링 곱셈
    y = unpacked_w_ple.reshape(35, 256) * math.sqrt(256.0)

    # 최종 pli_all 계산 (이 값이 Phase 2 루프 35번 도는 내내 쓰입니다!)
    pli_all = (x_proj_normed + y) * (1.0 / math.sqrt(2.0))

    for i in range(35):
        modalities  = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
        coef_mat    = np.dot(W["altup_pred"][i], modalities).reshape(4, 4)
        xs_pred     = xs + np.dot(coef_mat, xs)

        x                 = xs_pred[0].copy()
        inputs_normalized = rms_norm(x, W["input_ln"][i])

        Q = hw_matmul(inputs_normalized, W["W_q"][i])
        K = hw_matmul(inputs_normalized, W["W_k"][i])
        V = hw_matmul(inputs_normalized, W["W_v"][i])

        Q, K = CPU_CORE.cpu_qk_norm(Q,   K, W["gamma_q"][i], W["gamma_k"][i])

        theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
        Q     = CPU_CORE.cpu_rope(Q, pos=pos, theta_base=theta)
        K     = CPU_CORE.cpu_rope(K, pos=pos, theta_base=theta)

        if i < 20:
            CPU_CORE.cpu_update_kv_cache(K, V, i, K_cache, V_cache)
            target_k_cache = K_cache[i]
            target_v_cache = V_cache[i]
        else:
            if i % 5 == 4:
                target_k_cache = K_cache[19]
                target_v_cache = V_cache[19]
            else:
                target_k_cache = K_cache[18]
                target_v_cache = V_cache[18]

        attn_raw    = CPU_CORE.cpu_gqa(Q, target_k_cache, target_v_cache)
        attn_output = hw_matmul(attn_raw, W["W_o"][i])
        laurel_x          = hw_matmul(inputs_normalized, W["laurel_left"][i])
        laurel_x          = hw_matmul(laurel_x, W["laurel_right"][i])
        laurel_out_normed = inputs_normalized + rms_norm(laurel_x, W["laurel_norm"][i])

        attn_output  = rms_norm(attn_output, W["post_attn_ln"][i])
        attn_output += x
        attn_output  = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))

        x_n2 = rms_norm(attn_output, W["pre_ffn_ln"][i])

        if i < 10:
            gate_out = hw_matmul(x_n2, W["W_gate"][i], use_gelu=False)
            up_out   = hw_matmul(x_n2, W["W_up"][i])
            cutoff      = np.mean(gate_out) + np.std(gate_out) * 1.6448536
            sparse_gate = np.maximum(gate_out - cutoff, 0.0)
            hidden = CPU_CORE.gelu(sparse_gate) * up_out
        else:
            gate_out = hw_matmul(x_n2, W["W_gate"][i], use_gelu=True)
            up_out   = hw_matmul(x_n2, W["W_up"][i])
            hidden = gate_out * up_out

        mlp_out  = hw_matmul(hidden, W["W_down"][i])
        outputs  = rms_norm(mlp_out, W["post_ffn_ln"][i])
        outputs += attn_output

        activated  = outputs * W["altup_scale"][i]
        innovation = activated - xs_pred[0]
        mod_corr   = get_router_modalities(activated, W["altup_rn"][i], W["altup_router"][i])
        corr_coefs = np.dot(W["altup_corr"][i], mod_corr) + 1.0

        xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation
        pli      = pli_all[i]
        gate_ple = CPU_CORE.gelu(hw_matmul(activated, W["ple_gate"][i])) * pli
        mapped   = rms_norm(hw_matmul(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
        xs_new[1:] += mapped
        xs = xs_new

    return xs

def decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head):
    target_mag = np.mean(xs[0] ** 2) ** 0.5
    unembedded = [xs[0]]
    for k in range(3):
        proj_x  = np.dot(xs[k + 1], altup_unprojs[k])
        new_mag = np.mean(proj_x ** 2) ** 0.5
        proj_x *= target_mag / max(new_mag, 1e-12)
        unembedded.append(proj_x)
    x_final = np.mean(np.stack(unembedded, axis=0), axis=0)
    x_final = rms_norm(x_final, W_final_norm)
    logits = IGPU_CORE.igpu_matmul(x_final, W_lm_head)
    return logits

def _sample(logits: np.ndarray, temperature: float, top_p: float,
            rep_penalty: float, generated: list) -> int:
    if rep_penalty != 1.0 and len(generated) > 0:
        for token in set(generated):
            if logits[token] < 0:
                logits[token] *= rep_penalty
            else:
                logits[token] /= rep_penalty
    if temperature == 0.0: return int(np.argmax(logits))
    logits = logits / max(temperature, 1e-8)
    logits_safe = logits - np.max(logits)
    probs  = np.exp(logits_safe) / np.sum(np.exp(logits_safe))
    if top_p < 1.0:
        sorted_idx  = np.argsort(probs)[::-1]
        cumsum      = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumsum - probs[sorted_idx] < top_p
        probs_filtered = np.zeros_like(probs)
        probs_filtered[sorted_idx[cutoff_mask]] = probs[sorted_idx[cutoff_mask]]
        if probs_filtered.sum() == 0: probs_filtered[sorted_idx[0]] = 1.0
        probs = probs_filtered / probs_filtered.sum()
    return int(np.random.choice(len(probs), p=probs))

# Function to print current RAM usage (RSS) in MB
def print_ram_usage(step_name):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    print(f"[{step_name}] RAM Usage: {rss_mb:.2f} MB")

def main():
    print_ram_usage("1. Before Model Load")
    
    TEMPERATURE    = 0.5    # 0.3은 너무 딱딱해서 반복에 빠지기 쉬움. 0.6으로 상향!
    TOP_P          = 0.9    
    REP_PENALTY    = 1.05   # 1.02 -> 1.15 로 대폭 상향! (반복 루프 절대 방어)
    MAX_NEW_TOKENS = 512

    IGPU_CORE.warmup()
    print("\nGemma 3N [INT4 Optimized] - Chat Mode")
    W_embed, W_ple_packed, W_ple_scale, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()
    


    print("[메모리] 가중치 VRAM 최적화 중...")
    IGPU_CORE.preload_and_free(W, _IGPU_WEIGHT_KEYS)
    IGPU_CORE._get_or_upload_weight(W_lm_head)

    print_ram_usage("2. After Model Load")

    # 전체 대화 히스토리 (간단한 형태)
    history_tokens = []
    K_cache = [None for _ in range(35)]
    V_cache = [None for _ in range(35)]
    cur_pos = 0
    print_ram_usage("3. After KV Cache Allocation")

    print("\n--- 대화를 시작합니다 (종료: 'exit' 또는 'quit') ---")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]: break
            if not user_input.strip(): continue

            # Chat Template 적용 (단순화)
            prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
            input_tokens = CPU_CORE.tokenize(prompt)
            
            print("Model: ", end="", flush=True)
            
            xs = None
            # Prefill (새로운 입력만 처리)
            for token_id in input_tokens:
                xs = forward_one_token(token_id, cur_pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                                       W_ple_proj, altup_projs, K_cache, V_cache)
                cur_pos += 1
            
            print_ram_usage("4. After Prefill")
            
            # Generation
            generated = []
            STOP_TOKENS = [1, 106]
            printed_text = ""  

            for _ in range(MAX_NEW_TOKENS):
                logits = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)

                # 💡 [수정] Gemma 3 전용 Final Logit Soft-capping (30.0)
                # 이 한 줄이 문법과 지능을 비약적으로 끌어올립니다.
                logits = 30.0 * np.tanh(logits / 30.0)


                next_token = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)
                
                if next_token in STOP_TOKENS: break
                
                generated.append(next_token)

                # 💡 [수정] UTF-8 한글 잘림 방지 및 특수토큰(<unused>) 숨기기
                # 지금까지 모인 전체 토큰을 디코딩하고, 특수 기호는 무시(skip)합니다.
                current_text = CPU_CORE.tokenizer.decode(generated, skip_special_tokens=True)
                
                # 새로 추가된 부분(새 글자)만 잘라서 화면에 출력합니다.
                new_text = current_text[len(printed_text):]
                print(new_text, end="", flush=True)
                
                # 출력한 텍스트 상태 업데이트
                printed_text = current_text

                xs = forward_one_token(next_token, cur_pos, W, W_embed, W_ple_packed, W_ple_scale,
                                       norm_ple, W_ple_proj, altup_projs, K_cache, V_cache)
                cur_pos += 1
                
            print() # New line after model response
            gc.collect() # Turn-based Memory Cleanup
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    print_ram_usage("5. After Generation")

    print("\n[완료] 대화가 종료되었습니다.")

#import size_check

if __name__ == "__main__":
    main()
    #size_check.debug()
