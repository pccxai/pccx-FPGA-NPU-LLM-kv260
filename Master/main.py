import numpy as np
import CPU_CORE
import safeTensor
import math
import IGPU_CORE
import SYS_CONFIG  

MAX_SEQ_LEN = 1024

def hw_matmul(x, w, use_gelu=False):
    if SYS_CONFIG.ACCEL_MODE == "IGPU":
        return IGPU_CORE.igpu_matmul_gelu(x, w) if use_gelu else IGPU_CORE.igpu_matmul(x, w)
    else:
        if isinstance(w, dict) and "packed" in w:
            raise NotImplementedError("CPU 모드에서는 INT4 연산을 지원하지 않습니다. SYS_CONFIG.ACCEL_MODE='IGPU'를 사용하세요!")
        out = np.dot(x, w)
        return CPU_CORE.gelu(out) if use_gelu else out

def rms_norm(x, gamma):
    x_f64 = x.astype(np.float64)
    rms   = np.sqrt(np.mean(x_f64 ** 2) + 1e-6)
    return (x_f64 / rms).astype(np.float32) * gamma

def get_router_modalities(x, w_norm, w_router):
    x_n = rms_norm(x, w_norm) / 2048.0
    return np.tanh(np.dot(x_n, w_router))

def forward_one_token(token_id, pos, W, W_embed, W_ple, norm_ple,
                      W_ple_proj, altup_projs, K_cache, V_cache):

    x = CPU_CORE.embedding(token_id, W_embed)
    
    use_altup = altup_projs is not None and len(altup_projs) > 0
    use_ple = W_ple is not None

    if use_altup:
        xs = np.zeros((4, x.shape[-1]), dtype=np.float32)
        xs[0] = x
        for k in range(3):
            xs[k + 1] = hw_matmul(x, altup_projs[k])
    else:
        xs = [x]

    if use_ple:
        x_proj = np.dot(x, W_ple_proj) / math.sqrt(2048.0)
        x_proj = x_proj.reshape(30, 256) 
        x_proj_normed = np.stack([rms_norm(x_proj[i], norm_ple) for i in range(30)])
        y = W_ple[min(token_id, W_ple.shape[0] - 1)].astype(np.float32).reshape(30, 256) * math.sqrt(256.0)
        pli_all = (x_proj_normed + y) * (1.0 / math.sqrt(2.0))

    for i in range(30):
        if use_altup:
            modalities  = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
            coef_mat = np.dot(modalities, W["altup_pred"][i]).reshape(4, 4)
            xs_pred     = xs + np.dot(coef_mat, xs)
            x_current   = xs_pred[0].copy()
        else:
            x_current   = xs[0]

        inputs_normalized = rms_norm(x_current, W["input_ln"][i])

        Q = hw_matmul(inputs_normalized, W["W_q"][i])

        # ──────────────────────────────────────────────────────────────
        # 정적(Static) KV 캐시 업데이트 및 라우팅
        # ──────────────────────────────────────────────────────────────
        if i < 20:
            K = hw_matmul(inputs_normalized, W["W_k"][i])
            V = hw_matmul(inputs_normalized, W["W_v"][i])

            if W["gamma_q"][i] is not None:
                Q, K = CPU_CORE.cpu_qk_norm(Q, K, W["gamma_q"][i], W["gamma_k"][i])

            theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
            Q = CPU_CORE.cpu_rope(Q, pos=pos, theta_base=theta)
            K = CPU_CORE.cpu_rope(K, pos=pos, theta_base=theta)

            # 정적 캐시에 저장 후 pos만큼 잘라서 넘김
            CPU_CORE.cpu_update_kv_cache_static(K, V, i, pos, K_cache, V_cache)
            target_k_cache = K_cache[i, :pos+1]
            target_v_cache = V_cache[i, :pos+1]

        else:
            # Q만 정규화 및 RoPE 적용
            if W["gamma_q"][i] is not None:
                Q_r = Q.reshape(-1, 256)
                q_rms = np.sqrt(np.mean(Q_r.astype(np.float64)**2, axis=1, keepdims=True) + 1e-6)
                Q = ((Q_r / q_rms).astype(np.float32) * W["gamma_q"][i]).flatten()

            theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
            Q = CPU_CORE.cpu_rope(Q, pos=pos, theta_base=theta)

            # E2B 아키텍처: 18번, 19번 캐시 재사용
            idx = 19 if (i % 5 == 4) else 18
            target_k_cache = K_cache[idx, :pos+1]
            target_v_cache = V_cache[idx, :pos+1]

        attn_raw    = CPU_CORE.cpu_gqa_static(Q, target_k_cache, target_v_cache)
        attn_output = hw_matmul(attn_raw, W["W_o"][i])
        
        if W["laurel_left"][i] is not None:
            laurel_x          = hw_matmul(inputs_normalized, W["laurel_left"][i])
            laurel_x          = hw_matmul(laurel_x, W["laurel_right"][i])
            laurel_out_normed = inputs_normalized + rms_norm(laurel_x, W["laurel_norm"][i])
            attn_output  = rms_norm(attn_output, W["post_attn_ln"][i])
            attn_output += x_current
            attn_output  = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))
        else:
            attn_output  = rms_norm(attn_output, W["post_attn_ln"][i])
            attn_output += x_current

        x_n2 = rms_norm(attn_output, W["pre_ffn_ln"][i])
        
        # FFN 환각 로직 완전 철거! 전 레이어 정상 GeLU로 통일
        if W["W_gate"][i] is not None:
            gate_out = hw_matmul(x_n2, W["W_gate"][i], use_gelu=True)
            up_out   = hw_matmul(x_n2, W["W_up"][i])
            hidden   = gate_out * up_out
            mlp_out  = hw_matmul(hidden, W["W_down"][i])
        else:
            mlp_out = 0

        outputs  = rms_norm(mlp_out, W["post_ffn_ln"][i])
        outputs += attn_output

        if use_altup:
            activated    = outputs * W["altup_scale"][i]
            innovation   = activated - xs_pred[0]
            mod_corr     = get_router_modalities(activated, W["altup_rn"][i], W["altup_router"][i])
            corr_coefs   = np.dot(W["altup_corr"][i], mod_corr) + 1.0

            xs_new = xs_pred.copy()
            for k in range(4):
                xs_new[k] = xs_pred[k] + corr_coefs[k] * innovation

            if use_ple and W["ple_gate"][i] is not None:
                pli      = pli_all[i]
                gate_ple = CPU_CORE.gelu(np.dot(activated, W["ple_gate"][i])) * pli
                mapped   = rms_norm(np.dot(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
                for k in range(1, 4):
                    xs_new[k] += mapped
            xs = xs_new
        else:
            xs[0] = outputs

    return xs 

def decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head):
    if altup_unprojs is not None and len(altup_unprojs) > 0:
        target_mag  = np.mean(xs[0] ** 2) ** 0.5
        unembedded  = [xs[0]]
        for k in range(3):
            proj_x  = np.dot(xs[k + 1], altup_unprojs[k])
            new_mag = np.mean(proj_x ** 2) ** 0.5
            proj_x *= target_mag / max(new_mag, 1e-12)
            unembedded.append(proj_x)
        x_final = np.mean(np.stack(unembedded, axis=0), axis=0)
    else:
        x_final = xs[0]

    x_final = rms_norm(x_final, W_final_norm)
    logits  = np.dot(x_final, W_lm_head)
    
    return logits

def _sample(logits: np.ndarray, temperature: float, top_p: float, rep_penalty: float, generated: list) -> int:
    if rep_penalty != 1.0 and len(generated) > 0:
        for token in set(generated):
            if logits[token] < 0:
                logits[token] *= rep_penalty
            else:
                logits[token] /= rep_penalty

    logits_safe = logits - np.max(logits)
    
    if temperature == 0.0 or (temperature == 1.0 and top_p == 1.0):
        return int(np.argmax(logits_safe))

    logits_safe = logits_safe / max(temperature, 1e-8)
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

def main():
    TEMPERATURE = 0.0   # greedy decoding으로 먼저 테스트
    TOP_P       = 1.0
    STOP_TOKENS = [1]   # 107(<end_of_turn>) 임시 제거
    REP_PENALTY = 1.15  
    MAX_NEW_TOKENS = 50   

    IGPU_CORE.warmup()

    print("Gemma 3N E2B [INT4 W4A16] - Fixed Architecture & Static KV Cache")
    W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()

    prompt = "<start_of_turn>user\n안녕 하세요<end_of_turn>\n<start_of_turn>model\n"
    input_tokens = CPU_CORE.tokenize(prompt)

    # 메모리 할당 병목 완전 제거 (Static Array)
    K_cache = np.zeros((30, MAX_SEQ_LEN, 2, 256), dtype=np.float32)
    V_cache = np.zeros((30, MAX_SEQ_LEN, 2, 256), dtype=np.float32)

    print(f"Prefill: {len(input_tokens)} 토큰 처리 중...")
    xs = None
    for pos, token_id in enumerate(input_tokens):
        xs = forward_one_token(token_id, pos, W, W_embed, W_ple, norm_ple,
                               W_ple_proj, altup_projs, K_cache, V_cache)

    print("\n[생성 시작]")
    print(prompt, end="", flush=True)

    generated   = []
    cur_pos     = len(input_tokens)

    logits      = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)
    next_token  = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)

    for _ in range(MAX_NEW_TOKENS):
        if next_token in STOP_TOKENS:
            break

        generated.append(next_token)
        decoded = CPU_CORE.tokenizer.decode([next_token])
        print(decoded, end="", flush=True)

        xs          = forward_one_token(next_token, cur_pos, W, W_embed, W_ple,
                                        norm_ple, W_ple_proj, altup_projs, K_cache, V_cache)
        logits      = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)
        next_token  = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)
        cur_pos    += 1

    print(f"\n\n[완료] 총 {len(generated)}개 토큰 생성")

if __name__ == "__main__":
    main()