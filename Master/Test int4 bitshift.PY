# 별도 스크립트로 실행: test_int4.py
import numpy as np
import sys
sys.path.insert(0, 'Master')
import SYS_CONFIG
from safetensors.torch import load_file
import glob, IGPU_CORE

# 가중치 로드
tensors = {}
for f in sorted(glob.glob(SYS_CONFIG.MODEL_DIR + '/*.safetensors')):
    from safetensors.torch import load_file
    import torch
    t = load_file(f, device='cpu')
    for k, v in t.items():
        tensors[k] = v.numpy().copy() if v.dtype == torch.uint8 else v.float().numpy().copy()

# 레이어 0 q_proj 가져오기
packed = tensors['model.language_model.layers.0.self_attn.q_proj.weight.weight_int4']
scales = tensors['model.language_model.layers.0.self_attn.q_proj.weight.scales']

print(f"packed shape: {packed.shape}, scales shape: {scales.shape}")

# 테스트 입력
x = np.random.randn(2048).astype(np.float32)

# IGPU INT4 결과
igpu_result = IGPU_CORE.igpu_matmul(x, {"packed": packed, "scales": scales})

# Numpy로 직접 디퀀타이즈해서 계산 (정답)
out_f, in_f = packed.shape[0], packed.shape[1] * 2
W_dequant = np.zeros((out_f, in_f), dtype=np.float32)
for i in range(packed.shape[1]):
    byte = packed[:, i].astype(np.int32)
    low = byte & 0xF
    low = np.where(low > 7, low - 16, low).astype(np.float32)
    high = (byte >> 4) & 0xF
    high = np.where(high > 7, high - 16, high).astype(np.float32)
    g = (i * 2) // 32
    W_dequant[:, i*2]   = low  * scales[:, g]
    W_dequant[:, i*2+1] = high * scales[:, g]

numpy_result = W_dequant @ x

print(f"IGPU  결과 norm: {np.linalg.norm(igpu_result):.4f}, 범위: [{igpu_result.min():.3f}, {igpu_result.max():.3f}]")
print(f"Numpy 결과 norm: {np.linalg.norm(numpy_result):.4f}, 범위: [{numpy_result.min():.3f}, {numpy_result.max():.3f}]")
print(f"최대 오차: {np.max(np.abs(igpu_result - numpy_result)):.6f}")
print(f"상관계수: {np.corrcoef(igpu_result, numpy_result)[0,1]:.6f}")