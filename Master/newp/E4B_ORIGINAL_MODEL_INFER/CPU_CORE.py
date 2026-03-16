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
    # Safely convert to .astype(float32) even if W_embed_real is float16
    x = W_embed_real[token_id].astype(np.float32)
    x = x * math.sqrt(2048.0)
    return x

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))

def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    Q_reshaped = Q.reshape(-1, 256)
    K_reshaped = K.reshape(-1, 256)

    # float32 (keep previous optimizations)
    q_rms = np.sqrt(np.mean(Q_reshaped.astype(np.float32)**2, axis=1, keepdims=True) + 1e-6)
    k_rms = np.sqrt(np.mean(K_reshaped.astype(np.float32)**2, axis=1, keepdims=True) + 1e-6)

    Q_norm = (Q_reshaped.astype(np.float32) / q_rms) * gamma_q
    K_norm = (K_reshaped.astype(np.float32) / k_rms) * gamma_k

    return Q_norm.flatten(), K_norm.flatten()

# ================================================================
# RoPE frequency cache (retains previous optimizations)
# ================================================================
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

# ================================================================
# Memory optimization: store KV cache as float16
#
# Decrease K, V vectors to float16 to reduce cache memory by half.
# Restore float32 in cpu_gqa for calculation → maintain precision.
# Initial value: None (initialized as [None]*35 in main.py)
# ================================================================
def cpu_update_kv_cache(K_rope, V, layer_idx, K_cache, V_cache):
    """
    K_rope, convert V to float16 and add to KV cache.
    If None, create first, otherwise concat.
    """
    # Save half the memory by saving as float16
    K_new = K_rope.astype(np.float16)[np.newaxis]  # (1, dim)
    V_new = V.astype(np.float16)[np.newaxis]        # (1, dim)

    if K_cache[layer_idx] is None:
        K_cache[layer_idx] = K_new
        V_cache[layer_idx] = V_new
    else:
        K_cache[layer_idx] = np.concatenate([K_cache[layer_idx], K_new], axis=0)
        V_cache[layer_idx] = np.concatenate([V_cache[layer_idx], V_new], axis=0)


def cpu_gqa(Q_rope, K_cache_layer, V_cache_layer):
    # Change the form Q from (8, 256) to (2, 4, 256)
    # 2 is the number of KV heads, 4 is the number of Q heads mapped per KV head
    Q_reshaped = Q_rope.reshape(2, 4, 256).astype(np.float32)

    # Shape change after K, V cache restoration (seq_len, 2, 256)
    K_mat = K_cache_layer.astype(np.float32).reshape(-1, 2, 256)
    V_mat = V_cache_layer.astype(np.float32).reshape(-1, 2, 256)

    # Change K_mat axis for matrix multiplication: (2, 256, seq_len)
    K_t = K_mat.transpose(1, 2, 0)
    
    # Process matrix multiplication in one step without a Python for statement -> Result type: (2, 4, seq_len)
    scores = np.matmul(Q_reshaped, K_t)

    # Application of Google formula: Softmax calculation processed at once
    scores_safe = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_safe)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Change V_mat axis for matrix multiplication: (2, seq_len, 256)
    V_t = V_mat.transpose(1, 0, 2)
    
    # Calculate attention output in one step -> Result type: (2, 4, 256)
    attn_out = np.matmul(probs, V_t)

    # Spread out and return as a 1-dimensional vector
    return attn_out.flatten()

def cpu_sample_token(probs):
    return int(np.argmax(probs))
