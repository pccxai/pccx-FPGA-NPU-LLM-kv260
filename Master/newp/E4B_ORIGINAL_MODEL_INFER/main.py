import numpy as np
import CPU_CORE
import safeTensor
import math
import IGPU_CORE

ACCEL_MODE = "IGPU"

# IGPU 경로를 타는 레이어 큰 행렬 키 목록
# preload_and_free가 이것들을 VRAM에 올린 뒤 float16 원본 해제
_IGPU_WEIGHT_KEYS = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"]

def hw_matmul(x, w, use_gelu=False):
    if ACCEL_MODE == "IGPU":
        return IGPU_CORE.igpu_matmul_gelu(x, w) if use_gelu else IGPU_CORE.igpu_matmul(x, w)
    else:
        out = np.dot(x, w)
        return CPU_CORE.gelu(out) if use_gelu else out

def rms_norm(x, gamma):
    # float32 (이전 최적화 유지)
    x_f32 = x.astype(np.float32)
    rms   = np.sqrt(np.mean(x_f32 ** 2) + 1e-6)
    return (x_f32 / rms) * gamma

def get_router_modalities(x, w_norm, w_router):
    x_n = rms_norm(x, w_norm) / 2048.0
    return np.tanh(np.dot(x_n, w_router))

def forward_one_token(token_id, pos, W, W_embed, W_ple, norm_ple,
                      W_ple_proj, altup_projs, K_cache, V_cache):

    x0 = CPU_CORE.embedding(token_id, W_embed)   # W_embed float16 → float32 변환 내부 처리
    xs = np.zeros((4, 2048), dtype=np.float32)
    xs[0] = x0
    for k in range(3):
        xs[k + 1] = np.dot(x0, altup_projs[k])

    x_proj = np.dot(x0, W_ple_proj) / math.sqrt(2048.0)
    x_proj = x_proj.reshape(35, 256)

    # pli_all 배치 계산 (이전 최적화 유지)
    x_proj_f32 = x_proj.astype(np.float32)
    rms_vals   = np.sqrt(np.mean(x_proj_f32 ** 2, axis=1, keepdims=True) + 1e-6)
    x_proj_normed = (x_proj_f32 / rms_vals) * norm_ple

    # W_ple이 float16이어도 즉시 float32로 변환
    y = W_ple[min(token_id, W_ple.shape[0] - 1)].astype(np.float32).reshape(35, 256) * math.sqrt(256.0)
    pli_all = (x_proj_normed + y) * (1.0 / math.sqrt(2.0))

    for i in range(35):
        modalities  = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
        coef_mat    = np.dot(W["altup_pred"][i], modalities).reshape(4, 4)
        xs_pred     = xs + np.dot(coef_mat, xs)

        x                 = xs_pred[0].copy()
        inputs_normalized = rms_norm(x, W["input_ln"][i])

        # ==========================================
        # QKV 투영 및 RoPE
        # W["W_q"][i] 등은 WeightProxy 또는 float16 배열
        # igpu_matmul이 두 경우 모두 처리
        # ==========================================
        Q = hw_matmul(inputs_normalized, W["W_q"][i])
        K = hw_matmul(inputs_normalized, W["W_k"][i])
        V = hw_matmul(inputs_normalized, W["W_v"][i])

        Q, K = CPU_CORE.cpu_qk_norm(Q, K, W["gamma_q"][i], W["gamma_k"][i])

        theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
        Q     = CPU_CORE.cpu_rope(Q, pos=pos, theta_base=theta)
        K     = CPU_CORE.cpu_rope(K, pos=pos, theta_base=theta)

        # ==========================================
        # KV 캐시 라우팅 (기존 로직 유지)
        # ==========================================
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
        laurel_x          = np.dot(inputs_normalized, W["laurel_left"][i])
        laurel_x          = np.dot(laurel_x, W["laurel_right"][i])
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

        # 벡터화 (이전 최적화 유지)
        xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation

        pli      = pli_all[i]
        gate_ple = CPU_CORE.gelu(np.dot(activated, W["ple_gate"][i])) * pli
        mapped   = rms_norm(np.dot(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
        xs_new[1:] += mapped   # 브로드캐스트 (이전 최적화 유지)

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

    # 🚀 메모리 최적화: W_lm_head가 float16 (1GB) → float32 (2GB) copy 제거
    # x_final을 float16으로 내려 dot 계산 → 결과만 float32로 복원
    # 임시 메모리: (vocab,) float16 = 512KB (기존 방식 대비 극도로 작음)
    #logits = np.dot(x_final.astype(np.float16), W_lm_head).astype(np.float32)
    #logits -= np.max(logits)
    # 🚀 수정된 부분: np.max(logits) 빼는 부분 삭제 (페널티 정상 작동을 위함)
    logits = np.dot(x_final.astype(np.float16), W_lm_head).astype(np.float32)
    return logits

def main():
    MAX_NEW_TOKENS = 50
    TEMPERATURE    = 0.6
    TOP_P          = 0.9
    REP_PENALTY    = 1.15

    IGPU_CORE.warmup()

    print("Gemma 3N [iGPU/NPU Ready] - Flawless Generation")
    W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()

    # ============================================================
    # 🚀 핵심 메모리 최적화:
    # 모든 IGPU 경로 가중치를 VRAM(INT16)에 선업로드 후 float16 원본 해제
    # 효과: layers["W_q"][i] 등 float16 배열 ~3.5GB → WeightProxy(작은 객체)로 교체
    # ============================================================
    print("[메모리] IGPU 가중치 VRAM 선업로드 시작 (float16 원본 해제 예정)...")
    IGPU_CORE.preload_and_free(W, _IGPU_WEIGHT_KEYS)

    prompt = "<start_of_turn>user\n안녕 하세요<end_of_turn>\n<start_of_turn>model\n"
    input_tokens = CPU_CORE.tokenize(prompt)

    # 🚀 KV 캐시 None 초기화 (cpu_update_kv_cache가 None 여부로 최초/concat 판단)
    K_cache = [None for _ in range(35)]
    V_cache = [None for _ in range(35)]

    print(f"Prefill: {len(input_tokens)} 토큰 처리 중...")
    xs = None
    for pos, token_id in enumerate(input_tokens):
        xs = forward_one_token(token_id, pos, W, W_embed, W_ple, norm_ple,
                               W_ple_proj, altup_projs, K_cache, V_cache)

    print("\n[생성 시작]")
    print(prompt, end="", flush=True)

    STOP_TOKENS = [1, 106]
    generated   = []
    cur_pos     = len(input_tokens)

    logits     = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)
    next_token = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)

    for _ in range(MAX_NEW_TOKENS):
        if next_token in STOP_TOKENS:
            break

        generated.append(next_token)
        decoded = CPU_CORE.tokenizer.decode([next_token])
        print(decoded, end="", flush=True)

        xs         = forward_one_token(next_token, cur_pos, W, W_embed, W_ple,
                                       norm_ple, W_ple_proj, altup_projs, K_cache, V_cache)
        logits     = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)
        next_token = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)
        cur_pos   += 1

    print(f"\n\n[완료] 총 {len(generated)}개 토큰 생성")

def _sample(logits: np.ndarray, temperature: float, top_p: float,
            rep_penalty: float, generated: list) -> int:
    if rep_penalty != 1.0 and len(generated) > 0:
        for token in set(generated):
            if logits[token] < 0:
                logits[token] *= rep_penalty
            else:
                logits[token] /= rep_penalty

    if temperature == 0.0 or (temperature == 1.0 and top_p == 1.0):
        return int(np.argmax(logits))

    logits = logits / max(temperature, 1e-8)
    logits_safe = logits - np.max(logits)
    probs  = np.exp(logits_safe) / np.sum(np.exp(logits_safe))

    if top_p < 1.0:
        sorted_idx  = np.argsort(probs)[::-1]
        cumsum      = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumsum - probs[sorted_idx] < top_p

        probs_filtered = np.zeros_like(probs)
        probs_filtered[sorted_idx[cutoff_mask]] = probs[sorted_idx[cutoff_mask]]

        if probs_filtered.sum() == 0:
            probs_filtered[sorted_idx[0]] = 1.0

        probs = probs_filtered / probs_filtered.sum()

    return int(np.random.choice(len(probs), p=probs))

if __name__ == "__main__":
    main()
