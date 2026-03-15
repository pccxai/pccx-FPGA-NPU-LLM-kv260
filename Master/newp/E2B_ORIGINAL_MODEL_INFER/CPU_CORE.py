import numpy as np
import math
from transformers import AutoTokenizer
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
# E2B 모델 폴더 이름에 맞춰 경로 수정
model_id = os.path.join(base_dir, "[Original Model]gemma3NE2B")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

def tokenize(text):
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    print(f"[CPU] Tokenized IDs: {tokens}")
    return tokens

def embedding(token_id, W_embed_real):
    # W_embed_real이 float16이어도 .astype(float32)로 안전하게 변환
    x = W_embed_real[token_id].astype(np.float32)
    x = x * math.sqrt(2048.0)
    return x

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))

def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    Q_reshaped = Q.reshape(-1, 256)
    K_reshaped = K.reshape(-1, 256)

    # float32 (이전 최적화 유지)
    q_rms = np.sqrt(np.mean(Q_reshaped.astype(np.float32)**2, axis=1, keepdims=True) + 1e-6)
    k_rms = np.sqrt(np.mean(K_reshaped.astype(np.float32)**2, axis=1, keepdims=True) + 1e-6)

    Q_norm = (Q_reshaped.astype(np.float32) / q_rms) * gamma_q
    K_norm = (K_reshaped.astype(np.float32) / k_rms) * gamma_k

    return Q_norm.flatten(), K_norm.flatten()

# ================================================================
# RoPE 주파수 캐시 (이전 최적화 유지)
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
#  메모리 최적화: KV 캐시를 float16으로 저장
#
# K, V 벡터를 float16으로 내려서 캐시 메모리 절반으로 감소.
# cpu_gqa에서 float32로 복원해 연산 → 정밀도 유지.
# 초기값: None (main.py에서 [None]*30 (E2B 레이어 수)로 초기화)
# ================================================================
def cpu_update_kv_cache(K_rope, V, layer_idx, K_cache, V_cache):
    """
    K_rope, V를 float16으로 변환해 KV 캐시에 추가.
    None이면 최초 생성, 아니면 concat.
    """
    # float16으로 저장해 메모리 절반 절약
    K_new = K_rope.astype(np.float16)[np.newaxis]  # (1, dim)
    V_new = V.astype(np.float16)[np.newaxis]        # (1, dim)

    if K_cache[layer_idx] is None:
        K_cache[layer_idx] = K_new
        V_cache[layer_idx] = V_new
    else:
        K_cache[layer_idx] = np.concatenate([K_cache[layer_idx], K_new], axis=0)
        V_cache[layer_idx] = np.concatenate([V_cache[layer_idx], V_new], axis=0)


def cpu_gqa(Q_rope, K_cache_layer, V_cache_layer):
    # Q를 (8, 256)에서 (2, 4, 256)으로 형태 변경
    # 2는 KV 헤드 개수, 4는 KV 헤드 하나당 매핑되는 Q 헤드 개수
    Q_reshaped = Q_rope.reshape(2, 4, 256).astype(np.float32)

    # K, V 캐시 복원 후 형태 변경 (seq_len, 2, 256)
    K_mat = K_cache_layer.astype(np.float32).reshape(-1, 2, 256)
    V_mat = V_cache_layer.astype(np.float32).reshape(-1, 2, 256)

    # 행렬곱을 위해 K_mat 축 변경: (2, 256, seq_len)
    K_t = K_mat.transpose(1, 2, 0)
    
    # 파이썬 for 문 없이 행렬곱 한 방에 처리 -> 결과 형태: (2, 4, seq_len)
    scores = np.matmul(Q_reshaped, K_t)

    # 구글 공식 적용: Softmax 연산 한 번에 처리
    scores_safe = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_safe)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 행렬곱을 위해 V_mat 축 변경: (2, seq_len, 256)
    V_t = V_mat.transpose(1, 0, 2)
    
    # 어텐션 출력 한 방에 계산 -> 결과 형태: (2, 4, 256)
    attn_out = np.matmul(probs, V_t)

    # 다시 1차원 벡터로 쫙 펴서 리턴
    return attn_out.flatten()

def cpu_sample_token(probs):
    return int(np.argmax(probs))
