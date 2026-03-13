import numpy as np
import math
from transformers import AutoTokenizer
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
# Note: config is in local_gemma_3n_int4 now
model_id = os.path.join(base_dir, "local_gemma_3n_int4")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

def tokenize(text):
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    print(f"[CPU] Tokenized IDs: {tokens}")
    return tokens

def embedding(token_id, W_packed, W_scale):
    #if isinstance(W_packed, np.ndarray):
    #packed, scale = W_embed_data
    # packed: [vocab, hidden//2], scale: [vocab]
    row_packed = W_packed[token_id]
    row_scale = W_scale[token_id]
    
    # Unpack uint8 to two int4
    low = row_packed & 0x0F
    high = (row_packed >> 4) & 0x0F
    
    # Sign extend
    low_s = low.astype(np.int8)
    low_s[low_s > 7] -= 16
    high_s = high.astype(np.int8)
    high_s[high_s > 7] -= 16
    
    # Interleave
    res = np.empty(len(row_packed) * 2, dtype=np.float32)
    res[0::2] = low_s
    res[1::2] = high_s
    
    x = res * row_scale
    #else:
    #    x = W_embed_data[token_id].astype(np.float32)
        
    # Gemma 3N scaling
    #x = x * math.sqrt(2048.0)
    return x

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))

def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    Q_reshaped = Q.reshape(-1, 256)
    K_reshaped = K.reshape(-1, 256)
    q_rms = np.sqrt(np.mean(Q_reshaped.astype(np.float32)**2, axis=1, keepdims=True) + 1e-6)
    k_rms = np.sqrt(np.mean(K_reshaped.astype(np.float32)**2, axis=1, keepdims=True) + 1e-6)
    Q_norm = (Q_reshaped.astype(np.float32) / q_rms) * gamma_q
    K_norm = (K_reshaped.astype(np.float32) / k_rms) * gamma_k
    return Q_norm.flatten(), K_norm.flatten()

_rope_freq_cache: dict = {}

def _get_rope_freqs(theta_base: float, dim: int = 256) -> np.ndarray:
    if theta_base not in _rope_freq_cache:
        half  = dim // 2
        i_arr = np.arange(half, dtype=np.float32)
        freqs = (1.0 / (theta_base ** (2.0 * i_arr / dim))).astype(np.float32)
        _rope_freq_cache[theta_base] = freqs
    return _rope_freq_cache[theta_base]

def cpu_rope(x, pos, theta_base):
    dim       = 256
    num_heads = len(x) // dim
    half      = dim // 2
    x_reshaped = x.reshape(num_heads, dim)
    x0 = x_reshaped[:, :half]
    x1 = x_reshaped[:, half:]
    freqs    = _get_rope_freqs(theta_base)
    angles   = (pos * freqs).astype(np.float32)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    out = np.empty_like(x_reshaped)
    out[:, :half] = x0 * cos_vals - x1 * sin_vals
    out[:, half:] = x1 * cos_vals + x0 * sin_vals
    return out.flatten()

def cpu_update_kv_cache(K_rope, V, layer_idx, K_cache, V_cache):
    K_new = K_rope.astype(np.float16)[np.newaxis]
    V_new = V.astype(np.float16)[np.newaxis]
    if K_cache[layer_idx] is None:
        K_cache[layer_idx] = K_new
        V_cache[layer_idx] = V_new
    else:
        K_cache[layer_idx] = np.concatenate([K_cache[layer_idx], K_new], axis=0)
        V_cache[layer_idx] = np.concatenate([V_cache[layer_idx], V_new], axis=0)

def cpu_gqa(Q_rope, K_cache_layer, V_cache_layer):
    Q_reshaped = Q_rope.reshape(2, 4, 256).astype(np.float32)
    K_mat = K_cache_layer.astype(np.float32).reshape(-1, 2, 256)
    V_mat = V_cache_layer.astype(np.float32).reshape(-1, 2, 256)
    K_t = K_mat.transpose(1, 2, 0)
    scores = np.matmul(Q_reshaped, K_t)
    scores_safe = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_safe)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    V_t = V_mat.transpose(1, 0, 2)
    attn_out = np.matmul(probs, V_t)
    return attn_out.flatten()

def cpu_sample_token(probs):
    return int(np.argmax(probs))
