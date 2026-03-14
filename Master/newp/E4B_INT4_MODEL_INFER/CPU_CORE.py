import numpy as np
from transformers import AutoTokenizer
import os
import ctypes 

base_dir = os.path.dirname(os.path.abspath(__file__))
# Note: config is in local_gemma_3n_int4 now
model_id = os.path.join(base_dir, "local_gemma_3n_int4")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

# -----------------------------------------------------------
# C - DLL porting (load and init set)

dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)


# ><><><><><><><><Parameters><><><><><><><><

# gelu function 
c_lib.run_gelu_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]
c_lib.run_gelu_inplace.restype = None

# int4 unpacking
c_lib.run_unpack_int4_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),  
    ctypes.c_float,                                                        
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int                                                           
]
c_lib.run_gelu_inplace.restype = None

# ROPE function
c_lib.run_rope_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # x 배열
    ctypes.c_int,   # pos
    ctypes.c_float, # theta_base
    ctypes.c_int,   # num_heads
    ctypes.c_int    # dim
]
c_lib.run_rope_inplace.restype = None

# ><><><><><><><><><><><><><><><><><><><><><

# -----------------------------------------------------------


def tokenize(text):
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    print(f"[CPU] Tokenized IDs: {tokens}")
    return tokens

# cpp ver
def embedding(token_id, W_packed, W_scale):
    # 1. get 1d data from disk using mmap
    row_packed = np.ascontiguousarray(W_packed[token_id])
    
    # cast scale vaule to float
    row_scale = float(W_scale[token_id])
    
    packed_length = row_packed.size
    
    # 2. 파이썬에서는 데이터를 담을 '결과용 빈 깡통 배열(C-Contiguous)'만 0초만에 딱 하나 만들어 줌
    out_f32 = np.empty(packed_length * 2, dtype=np.float32)
    
    # 3. C++ 커널에 주소만 던져주면 알아서 비트 쪼개고 채움
    c_lib.run_unpack_int4_inplace(row_packed, ctypes.c_float(row_scale), out_f32, packed_length)
    
    return out_f32

''' python ver
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
'''
# python ver
'''
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))
'''
def gelu(x):
    # flatten
    x_flat = np.ascontiguousarray(x.flatten().astype(np.float32))

    # pass value by reference
    c_lib.run_gelu_inplace(x_flat, x_flat.size)

    # return to original shape
    return x_flat.reshape(x.shape)


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
    dim = 256
    num_heads = len(x) // dim
    
    # 1. 혹시 모를 메모리 꼬임 방지를 위해 float32 1차원 연속 배열로 준비
    x_flat = np.ascontiguousarray(x.astype(np.float32).flatten())
    
    # 2. C++ 커널에 주소 던져서 In-place 회전 (알아서 덮어써짐)
    c_lib.run_rope_inplace(x_flat, int(pos), float(theta_base), int(num_heads), int(dim))
    
    return x_flat

'''python ver
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
'''

def cpu_update_kv_cache(K_rope, V, token_cnt,layer_idx, K_cache, V_cache):
    '''K_new = K_rope.astype(np.float16)[np.newaxis]
    V_new = V.astype(np.float16)[np.newaxis]
    print("K_new.shape",K_new.shape, "V_new.shape = ",V_new.shape)
    if K_cache[layer_idx] is None:
        K_cache[layer_idx] = K_new
        V_cache[layer_idx]   = V_new
    else:
        K_cache[layer_idx] = np.concatenate([K_cache[layer_idx], K_new], axis=0)
        V_cache[layer_idx] = np.concatenate([V_cache[layer_idx], V_new], axis=0)
    '''
    #K_new = K_rope.astype(np.float16)[np.newaxis]
    #V_new = V.astype(np.float16)[np.newaxis]
    
    #K_cache[layer_idx, token_cnt, : ] = K_new
    #V_cache[layer_idx, token_cnt, : ] = V_new


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
