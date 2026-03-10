import numpy as np
import math
from transformers import AutoTokenizer
import SYS_CONFIG  

tokenizer = AutoTokenizer.from_pretrained(SYS_CONFIG.MODEL_DIR, local_files_only=True)

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

# 클로드 지적 수용: RoPE Numpy 벡터화 (속도 수십 배 향상)
def cpu_rope(x, pos, theta_base): 
    dim = 256
    num_heads = len(x) // dim
    x_reshaped = x.reshape(num_heads, dim)
    half = dim // 2
    
    i = np.arange(half)
    freqs = 1.0 / (theta_base ** ((2 * i) / dim))
    angles = pos * freqs  
    cos_v = np.cos(angles)
    sin_v = np.sin(angles)
    
    x0 = x_reshaped[:, :half]
    x1 = x_reshaped[:, half:]
    
    out = np.empty_like(x_reshaped, dtype=np.float32)
    out[:, :half] = x0 * cos_v - x1 * sin_v
    out[:, half:] = x1 * cos_v + x0 * sin_v
    return out.flatten()

def cpu_update_kv_cache_static(K_rope, V, layer_idx, pos, K_cache, V_cache):
    K_cache[layer_idx, pos] = K_rope.reshape(2, 256)
    V_cache[layer_idx, pos] = V.reshape(2, 256)

def cpu_gqa_static(Q_rope, K_cache_slice, V_cache_slice):
    Q_reshaped = Q_rope.reshape(8, 256)
    K_mat = K_cache_slice 
    V_mat = V_cache_slice 
    
    attn_out = np.zeros((8, 256), dtype=np.float32)
    
    for q_head in range(8):
        kv_head = q_head // 4 
        
        # 스케일링 (1 / 256.0)
        scores = np.dot(K_mat[:, kv_head, :].astype(np.float32), Q_reshaped[q_head].astype(np.float32)) / 16.0        
        # Attention Softcapping (50.0)
        scores = np.tanh(scores / 50.0) * 50.0
        
        scores_safe = scores - np.max(scores)
        probs = np.exp(scores_safe) / np.sum(np.exp(scores_safe))
        
        head_out = np.dot(probs, V_mat[:, kv_head, :].astype(np.float32))
        attn_out[q_head] = head_out
        
    return attn_out.flatten()