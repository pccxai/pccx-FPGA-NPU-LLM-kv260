import numpy as np
import math
from transformers import AutoTokenizer
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_id = os.path.join(base_dir, "local_gemma_3n")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

def tokenize(text):
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    print(f"[CPU] Tokenized IDs: {tokens}")
    return tokens

def embedding(token_id, W_embed_real):
    x = W_embed_real[token_id].astype(np.float32)
    x = x * math.sqrt(2048.0) 
    return x 

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))

def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    Q_reshaped = Q.reshape(-1, 256)
    K_reshaped = K.reshape(-1, 256)

    q_rms = np.sqrt(np.mean(Q_reshaped.astype(np.float64)**2, axis=1, keepdims=True) + 1e-6)
    k_rms = np.sqrt(np.mean(K_reshaped.astype(np.float64)**2, axis=1, keepdims=True) + 1e-6)

    Q_norm = (Q_reshaped / q_rms).astype(np.float32) * gamma_q   
    K_norm = (K_reshaped / k_rms).astype(np.float32) * gamma_k   

    return Q_norm.flatten(), K_norm.flatten()

def cpu_rope(x, pos, theta_base): # 동적 theta_base 적용
    dim = 256 
    num_heads = len(x) // dim
    x_reshaped = x.reshape(num_heads, dim)
    out = np.zeros_like(x_reshaped, dtype=np.float32)
    
    for h in range(num_heads):
        half = dim // 2
        for i in range(half):
            freq = 1.0 / (theta_base ** ((2 * i) / dim))
            val = pos * freq
            cos_val = math.cos(val)
            sin_val = math.sin(val)
            
            x0 = x_reshaped[h, i]           
            x1 = x_reshaped[h, i + half]    
            
            out[h, i]        = x0 * cos_val - x1 * sin_val
            out[h, i + half] = x1 * cos_val + x0 * sin_val
            
    return out.flatten() 

def cpu_update_kv_cache(K_rope, V, layer_idx, K_cache, V_cache):
    K_cache[layer_idx].append(K_rope)
    V_cache[layer_idx].append(V)

def cpu_gqa(Q_rope, K_cache_layer, V_cache_layer):
    Q_reshaped = Q_rope.reshape(8, 256)
    K_mat = np.array(K_cache_layer).reshape(-1, 2, 256) 
    V_mat = np.array(V_cache_layer).reshape(-1, 2, 256) 
    
    attn_out = np.zeros((8, 256), dtype=np.float32)
    
    for q_head in range(8):
        kv_head = q_head // 4 
        # 충격적인 구글 공식 1: sqrt(256)으로 절대 나누지 않음!
        scores = np.dot(K_mat[:, kv_head, :].astype(np.float32), Q_reshaped[q_head].astype(np.float32)) 
        
        # 충격적인 구글 공식 2: Softcap 없음! 바로 Softmax!
        scores_safe = scores - np.max(scores)
        probs = np.exp(scores_safe) / np.sum(np.exp(scores_safe))
        
        head_out = np.dot(probs, V_mat[:, kv_head, :].astype(np.float32))
        attn_out[q_head] = head_out
        
    return attn_out.flatten()

def cpu_sample_token(probs):
    return int(np.argmax(probs))