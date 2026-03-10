import numpy as np
import torch
import math
import glob
from safetensors.torch import load_file
import sys
sys.path.insert(0, '/home/hwkim/Desktop/github/TinyNPU-RTL/Master')
import SYS_CONFIG, CPU_CORE, IGPU_CORE, safeTensor
from main import rms_norm

IGPU_CORE.warmup()
W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, W_final_norm, W_lm_head, W = safeTensor.load_local_weights()

print("원본 가중치 전체 로딩 중 (잠깐 기다려주세요)...")
orig = {}
for f in sorted(glob.glob('Master/gemma3NE2B/*.safetensors')):
    t = load_file(f, device='cpu')
    for k, v in t.items():
        if 'language_model' in k:
            orig[k] = v.float().numpy()
    del t
print("로딩 완료")

P = "model.language_model."

def ref_rms_norm(x, gamma):
    x64 = x.astype(np.float64)
    rms = np.sqrt(np.mean(x64**2) + 1e-6)
    return (x64 / rms).astype(np.float32) * gamma

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))

def qk_norm(Q, K, gq, gk):
    Qr = Q.reshape(-1, 256)
    Kr = K.reshape(-1, 256)
    qr = np.sqrt(np.mean(Qr.astype(np.float64)**2, axis=1, keepdims=True) + 1e-6)
    kr = np.sqrt(np.mean(Kr.astype(np.float64)**2, axis=1, keepdims=True) + 1e-6)
    return (Qr/qr).astype(np.float32)*gq, (Kr/kr).astype(np.float32)*gk

FULL_ATTN_LAYERS = {4, 9, 14, 19, 24, 29}
SLIDING_WINDOW = 512
MAX_SEQ_LEN = 1024

token_id = 2
pos = 0

# 초기화
x_ours = CPU_CORE.embedding(token_id, W_embed)
x_ref  = orig[P+'embed_tokens.weight'][token_id].astype(np.float32) * math.sqrt(2048.0)

K_cache = np.zeros((30, MAX_SEQ_LEN, 2, 256), dtype=np.float32)
V_cache = np.zeros((30, MAX_SEQ_LEN, 2, 256), dtype=np.float32)

for i in range(30):
    lp = P + f'layers.{i}.'
    theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0

    # ── 우리 구현 ──
    x_n_ours = rms_norm(x_ours, W["input_ln"][i])
    Q_o = IGPU_CORE.igpu_matmul(x_n_ours, W["W_q"][i])
    K_o = IGPU_CORE.igpu_matmul(x_n_ours, W["W_k"][i])
    V_o = IGPU_CORE.igpu_matmul(x_n_ours, W["W_v"][i])
    if W["gamma_q"][i] is not None:
        Q_o, K_o = CPU_CORE.cpu_qk_norm(Q_o, K_o, W["gamma_q"][i], W["gamma_k"][i])
    Q_o = CPU_CORE.cpu_rope(Q_o, pos=pos, theta_base=theta)
    K_o = CPU_CORE.cpu_rope(K_o, pos=pos, theta_base=theta)
    CPU_CORE.cpu_update_kv_cache_static(K_o, V_o, i, pos, K_cache, V_cache)
    if i in FULL_ATTN_LAYERS:
        tk = K_cache[i, :pos+1]; tv = V_cache[i, :pos+1]
    else:
        s = max(0, pos+1-SLIDING_WINDOW)
        tk = K_cache[i, s:pos+1]; tv = V_cache[i, s:pos+1]
    attn_o = CPU_CORE.cpu_gqa_static(Q_o, tk, tv)
    o_o = IGPU_CORE.igpu_matmul(attn_o, W["W_o"][i])

    if W["laurel_left"][i] is not None:
        ll = IGPU_CORE.igpu_matmul(x_n_ours, W["laurel_left"][i])
        lr = IGPU_CORE.igpu_matmul(ll, W["laurel_right"][i])
        laurel_n = x_n_ours + rms_norm(lr, W["laurel_norm"][i])
        o_n = rms_norm(o_o, W["post_attn_ln"][i])
        attn_f = (o_n + x_ours + laurel_n) * (1.0/math.sqrt(2.0))
    else:
        o_n = rms_norm(o_o, W["post_attn_ln"][i])
        attn_f = o_n + x_ours

    x_n2 = rms_norm(attn_f, W["pre_ffn_ln"][i])
    gate = IGPU_CORE.igpu_matmul_gelu(x_n2, W["W_gate"][i])
    up   = IGPU_CORE.igpu_matmul(x_n2, W["W_up"][i])
    mlp  = IGPU_CORE.igpu_matmul(gate*up, W["W_down"][i])
    x_ours = rms_norm(mlp, W["post_ffn_ln"][i]) + attn_f

    # ── 원본 구현 ──
    x_n_ref = ref_rms_norm(x_ref, orig[lp+'input_layernorm.weight'])
    Q_r = x_n_ref @ orig[lp+'self_attn.q_proj.weight'].T
    K_r = x_n_ref @ orig[lp+'self_attn.k_proj.weight'].T
    V_r = x_n_ref @ orig[lp+'self_attn.v_proj.weight'].T
    if lp+'self_attn.q_norm.weight' in orig:
        Q_r, K_r = qk_norm(Q_r, K_r, orig[lp+'self_attn.q_norm.weight'], orig[lp+'self_attn.k_norm.weight'])
    # pos=0 단일토큰: attn_out = V 그대로
    V_r2 = V_r.reshape(2,256)
    attn_r = np.array([V_r2[h//4] for h in range(8)]).flatten()
    o_r = attn_r @ orig[lp+'self_attn.o_proj.weight'].T

    if lp+'laurel.linear_left.weight' in orig:
        ll_r = x_n_ref @ orig[lp+'laurel.linear_left.weight'].T
        lr_r = ll_r @ orig[lp+'laurel.linear_right.weight'].T
        laurel_nr = x_n_ref + ref_rms_norm(lr_r, orig[lp+'laurel.post_laurel_norm.weight'])
        o_rn = ref_rms_norm(o_r, orig[lp+'post_attention_layernorm.weight'])
        attn_fr = (o_rn + x_ref + laurel_nr) * (1.0/math.sqrt(2.0))
    else:
        o_rn = ref_rms_norm(o_r, orig[lp+'post_attention_layernorm.weight'])
        attn_fr = o_rn + x_ref

    x_n2r = ref_rms_norm(attn_fr, orig[lp+'pre_feedforward_layernorm.weight'])
    gate_r = gelu(x_n2r @ orig[lp+'mlp.gate_proj.weight'].T)
    up_r   = x_n2r @ orig[lp+'mlp.up_proj.weight'].T
    mlp_r  = (gate_r*up_r) @ orig[lp+'mlp.down_proj.weight'].T
    x_ref  = ref_rms_norm(mlp_r, orig[lp+'post_feedforward_layernorm.weight']) + attn_fr

    corr = np.corrcoef(x_ours, x_ref)[0,1]
    print(f"레이어 {i:2d}: 상관계수={corr:.6f}, ours_norm={np.linalg.norm(x_ours):.2f}, ref_norm={np.linalg.norm(x_ref):.2f}")

print("\n=== 최종 logits 비교 ===")
def top5(x, label):
    xn = ref_rms_norm(x, W_final_norm)
    logits = np.tanh(np.dot(xn, W_lm_head)/30.0)*30.0
    t5 = np.argsort(logits)[-5:][::-1]
    print(f"[{label}]", [(int(tid), repr(CPU_CORE.tokenizer.decode([int(tid)]))) for tid in t5])

top5(x_ours, "우리 구현")
top5(x_ref,  "원본 참조")