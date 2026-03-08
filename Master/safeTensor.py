import numpy as np
import os
import glob
from safetensors.torch import load_file
import torch

import SYS_CONFIG  

def get_w_opt(tensors, base_name):
    # 1. 양자화된 INT4 (.weight.weight_int4)
    if base_name + ".weight.weight_int4" in tensors:
        return {
            "packed": tensors[base_name + ".weight.weight_int4"], 
            "scales": tensors[base_name + ".weight.scales"]
        }
    # 2. 이름이 깔끔한 양자화 버전
    elif base_name + ".weight_int4" in tensors:
        return {
            "packed": tensors[base_name + ".weight_int4"], 
            "scales": tensors[base_name + ".scales"]
        }
    # 3. 원본 float 가중치
    elif base_name + ".weight" in tensors:
        return tensors[base_name + ".weight"].T
    return None

def get_t_opt(tensors, name):
    return tensors.get(name, None)

def load_local_weights(model_dir=SYS_CONFIG.MODEL_DIR):
    print("Loading Gemma 3N [INT4] Weights (Auto-Detecting Architecture)...")
    tensors = {}
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    
    if len(st_files) == 0:
        raise FileNotFoundError(f"경로에 safetensors 파일이 없습니다: {model_dir}")
    
    for filename in st_files:
        pt_tensors = load_file(filename, device="cpu")
        for k, v in pt_tensors.items():
            if v.dtype == torch.uint8:
                tensors[k] = v.numpy().copy() 
            else:
                tensors[k] = v.float().numpy().copy()
        del pt_tensors

    P = "model.language_model."
    
    W_embed = get_t_opt(tensors, P + "embed_tokens.weight")
    if W_embed is None:
        raise KeyError("embed_tokens.weight 없음! 경로 확인 필요.")

    W_ple    = get_t_opt(tensors, P + "embed_tokens_per_layer.weight")
    norm_ple = get_t_opt(tensors, P + "per_layer_projection_norm.weight")
    
    ple_proj_raw = get_t_opt(tensors, P + "per_layer_model_projection.weight")
    W_ple_proj   = ple_proj_raw.T if ple_proj_raw is not None else None

    # ── altup_projections: INT4 packed이므로 get_w_opt 사용 ──
    # 실제 키: model.language_model.altup_projections.0.weight.weight_int4
    altup_projs   = []
    altup_unprojs = []
    if P + "altup_projections.0.weight.weight_int4" in tensors:
        print("  -> [E4B 감지] altup_projections INT4 로딩 중...")
        altup_projs = [
            get_w_opt(tensors, P + f"altup_projections.{i}") for i in range(3)
        ]
        # unembed은 float (INT4 아님)
        altup_unprojs = [
            tensors[P + f"altup_unembed_projections.{i}.weight"].T for i in range(3)
        ]
    elif P + "altup_projections.0.weight" in tensors:
        print("  -> [E4B 감지] altup_projections float 로딩 중...")
        altup_projs   = [tensors[P + f"altup_projections.{i}.weight"].T for i in range(3)]
        altup_unprojs = [tensors[P + f"altup_unembed_projections.{i}.weight"].T for i in range(3)]
    else:
        print("  -> [E2B 감지] altup 없음")
    
    W_final_norm = get_t_opt(tensors, P + "norm.weight")
    
    lm_head_raw = get_t_opt(tensors, P + "lm_head.weight")
    if lm_head_raw is not None:
        W_lm_head = lm_head_raw.T
    else:
        print("  -> [Weight Tying 감지] lm_head.weight가 없어 embed_tokens.weight를 재사용합니다.")
        W_lm_head = W_embed.T

    layers = {
        "W_q": [], "W_k": [], "W_v": [], "W_o": [],
        "gamma_q": [], "gamma_k": [],
        "input_ln": [], "post_attn_ln": [], "pre_ffn_ln": [], "post_ffn_ln": [],
        "W_gate": [], "W_up": [], "W_down": [],
        "ple_gate": [], "ple_proj": [], "ple_post_ln": [],
        "laurel_left": [], "laurel_right": [], "laurel_norm": [],
        "altup_rn": [], "altup_router": [], "altup_pred": [],
        "altup_scale": [], "altup_corr": []
    }

    for i in range(30):
        lp = f"{P}layers.{i}."
        
        layers["W_q"].append(get_w_opt(tensors, lp + "self_attn.q_proj"))
        layers["W_k"].append(get_w_opt(tensors, lp + "self_attn.k_proj"))
        layers["W_v"].append(get_w_opt(tensors, lp + "self_attn.v_proj"))
        layers["W_o"].append(get_w_opt(tensors, lp + "self_attn.o_proj"))
        
        layers["gamma_q"].append(get_t_opt(tensors, lp + "self_attn.q_norm.weight"))
        layers["gamma_k"].append(get_t_opt(tensors, lp + "self_attn.k_norm.weight"))

        layers["input_ln"].append(get_t_opt(tensors, lp + "input_layernorm.weight"))
        layers["post_attn_ln"].append(get_t_opt(tensors, lp + "post_attention_layernorm.weight"))
        layers["pre_ffn_ln"].append(get_t_opt(tensors, lp + "pre_feedforward_layernorm.weight"))
        layers["post_ffn_ln"].append(get_t_opt(tensors, lp + "post_feedforward_layernorm.weight"))

        layers["W_gate"].append(get_w_opt(tensors, lp + "mlp.gate_proj"))
        layers["W_up"].append(get_w_opt(tensors, lp + "mlp.up_proj"))
        layers["W_down"].append(get_w_opt(tensors, lp + "mlp.down_proj"))

        layers["ple_gate"].append(get_w_opt(tensors, lp + "per_layer_input_gate"))
        layers["ple_proj"].append(get_w_opt(tensors, lp + "per_layer_projection"))
        layers["ple_post_ln"].append(get_t_opt(tensors, lp + "post_per_layer_input_norm.weight"))

        layers["laurel_left"].append(get_w_opt(tensors, lp + "laurel.linear_left"))
        layers["laurel_right"].append(get_w_opt(tensors, lp + "laurel.linear_right"))
        layers["laurel_norm"].append(get_t_opt(tensors, lp + "laurel.post_laurel_norm.weight"))

        layers["altup_rn"].append(get_t_opt(tensors, lp + "altup.router_norm.weight"))
        layers["altup_router"].append(get_w_opt(tensors, lp + "altup.modality_router"))

        # ── 수정된 키 이름 3개 ──
        # altup.modality_predictor → altup.prediction_coefs
        layers["altup_pred"].append(get_w_opt(tensors, lp + "altup.prediction_coefs"))
        # altup.scale.weight → altup.correct_output_scale (스칼라 벡터)
        layers["altup_scale"].append(get_t_opt(tensors, lp + "altup.correct_output_scale"))
        # altup.corrector → altup.correction_coefs
        layers["altup_corr"].append(get_w_opt(tensors, lp + "altup.correction_coefs"))

    # 로딩 결과 요약
    none_counts = {k: sum(1 for v in vals if v is None) for k, vals in layers.items()}
    problem_keys = {k: v for k, v in none_counts.items() if v > 0}
    if problem_keys:
        print(f"  ⚠️  None인 레이어 가중치: {problem_keys}")
    else:
        print("  ✅ 모든 레이어 가중치 정상 로딩")
    
    print("가중치 로딩 및 아키텍처 매핑 완료.")
    return W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, W_final_norm, W_lm_head, layers