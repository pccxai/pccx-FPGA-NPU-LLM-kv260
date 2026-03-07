import numpy as np
import CPU_CORE
import safeTensor
import math

def rms_norm(x, gamma):
    x_f64 = x.astype(np.float64)
    rms = np.sqrt(np.mean(x_f64**2) + 1e-6)
    return (x_f64 / rms).astype(np.float32) * gamma

def get_router_modalities(x, w_norm, w_router):
    x_n = rms_norm(x, w_norm)
    x_n = x_n / 2048.0
    return np.tanh(np.dot(x_n, w_router))

def main():
    print("🚀 Gemma 3N [CPU Golden Reference] - The Google Holy Grail Applied!")
    W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, W_final_norm, W_lm_head, W = safeTensor.load_local_weights()

    K_cache = [[] for _ in range(35)]
    V_cache = [[] for _ in range(35)]

    prompt = "<start_of_turn>user\n안녕 하세요<end_of_turn>\n<start_of_turn>model\n"
    
    input_tokens = CPU_CORE.tokenize(prompt)    # input_tokens = CPU_CORE.tokenize(prompt)
    
    for pos, token_id in enumerate(input_tokens):
        if pos == len(input_tokens) - 1:
            print(f"\n🔥 [Decoding] Last Token ID: {token_id} (pos={pos})")
            
        x0 = CPU_CORE.embedding(token_id, W_embed)
        
        xs = np.zeros((4, 2048), dtype=np.float32)
        xs[0] = x0
        for k in range(3):
            xs[k+1] = np.dot(x0, altup_projs[k])
            
        # 🔥 충격적인 구글 공식 3: PLE Projection은 sqrt(2048)로 스케일 다운!
        x_proj = np.dot(x0, W_ple_proj) / math.sqrt(2048.0)
        x_proj = x_proj.reshape(35, 256)
        
        x_proj_normed = np.zeros_like(x_proj)
        for i in range(35):
            x_proj_normed[i] = rms_norm(x_proj[i], norm_ple)
            
        y = W_ple[min(token_id, W_ple.shape[0]-1)].astype(np.float32).reshape(35, 256) * math.sqrt(256.0)
        pli_all = (x_proj_normed + y) * (1.0 / math.sqrt(2.0))

        for i in range(35):
            modalities = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
            coef_flat = np.dot(W["altup_pred"][i], modalities)
            coef_mat = coef_flat.reshape(4, 4)

            out_pred = np.dot(coef_mat, xs)
            xs_pred = xs + out_pred

            x = xs_pred[0].copy()
            inputs_normalized = rms_norm(x, W["input_ln"][i])

            Q = np.dot(inputs_normalized, W["W_q"][i])
            K = np.dot(inputs_normalized, W["W_k"][i])
            V = np.dot(inputs_normalized, W["W_v"][i])

            Q, K = CPU_CORE.cpu_qk_norm(Q, K, W["gamma_q"][i], W["gamma_k"][i])
            
            # 🔥 충격적인 구글 공식 4: RoPE 각도가 5층마다 바뀐다!
            theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
            Q = CPU_CORE.cpu_rope(Q, pos=pos, theta_base=theta)
            K = CPU_CORE.cpu_rope(K, pos=pos, theta_base=theta)

            CPU_CORE.cpu_update_kv_cache(K, V, i, K_cache, V_cache)
            attn_raw = CPU_CORE.cpu_gqa(Q, K_cache[i], V_cache[i])
            attn_output = np.dot(attn_raw, W["W_o"][i])

            laurel_x = np.dot(inputs_normalized, W["laurel_left"][i])
            laurel_x = np.dot(laurel_x, W["laurel_right"][i])
            laurel_x_normed = rms_norm(laurel_x, W["laurel_norm"][i])
            laurel_out_normed = inputs_normalized + laurel_x_normed

            attn_output = rms_norm(attn_output, W["post_attn_ln"][i])
            attn_output += x
            attn_output = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))

            x_n2 = rms_norm(attn_output, W["pre_ffn_ln"][i])
            gate_out = np.dot(x_n2, W["W_gate"][i])
            up_out   = np.dot(x_n2, W["W_up"][i])
            
            # 🔥 충격적인 구글 공식 5: 앞의 10개 층은 하위 95% 데이터를 강제 소멸! (Gaussian Top-K)
            if i < 10:
                std_multiplier = 1.6448536 # 상위 5% 커트라인
                gate_mean = np.mean(gate_out)
                gate_std = np.std(gate_out)
                cutoff = gate_mean + gate_std * std_multiplier
                sparse_gate = np.maximum(gate_out - cutoff, 0.0) # ReLU
                hidden = CPU_CORE.gelu(sparse_gate) * up_out
            else:
                hidden = CPU_CORE.gelu(gate_out) * up_out
                
            mlp_out  = np.dot(hidden, W["W_down"][i])

            outputs = rms_norm(mlp_out, W["post_ffn_ln"][i])
            outputs += attn_output

            activated = outputs * W["altup_scale"][i]
            innovation = activated - xs_pred[0]

            modalities_corr = get_router_modalities(activated, W["altup_rn"][i], W["altup_router"][i])
            corr_coefs = np.dot(W["altup_corr"][i], modalities_corr) + 1.0

            xs_new = xs_pred.copy()
            for k in range(4):
                xs_new[k] = xs_pred[k] + corr_coefs[k] * innovation

            pli = pli_all[i]
            gate_ple = np.dot(activated, W["ple_gate"][i])
            gate_ple = CPU_CORE.gelu(gate_ple) * pli
            mapped_ple = np.dot(gate_ple, W["ple_proj"][i])
            mapped_ple = rms_norm(mapped_ple, W["ple_post_ln"][i])

            for k in range(1, 4): 
                xs_new[k] += mapped_ple

            xs = xs_new

        if pos == len(input_tokens) - 1:
            target_mag = np.mean(xs[0]**2)**0.5
            unembedded = [xs[0]]
            
            for k in range(3):
                proj_x = np.dot(xs[k+1], altup_unprojs[k])
                new_mag = np.mean(proj_x**2)**0.5
                proj_x *= target_mag / max(new_mag, 1e-12)
                unembedded.append(proj_x)
                
            x_final = np.mean(np.stack(unembedded, axis=0), axis=0)
            x_final = rms_norm(x_final, W_final_norm)
            
            logits = np.dot(x_final, W_lm_head)
            
            # 🔥 마지막 공식: Final Softcap도 삭제!
            logits_safe = logits - np.max(logits)
            probs = np.exp(logits_safe) / np.sum(np.exp(logits_safe))
            
            next_token = CPU_CORE.cpu_sample_token(probs)
            
            print(f"\n✅ Top 5 Tokens: {np.argsort(probs)[-5:][::-1]}")
            print(f"✅ Generated Next Token ID: {next_token}")
            print(f"🎉 Decoded Text: {CPU_CORE.tokenizer.decode([next_token])}")

if __name__ == "__main__":
    main()