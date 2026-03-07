import numpy as np
import os
import torch
import glob
from safetensors.torch import load_file

base_dir = os.path.dirname(os.path.abspath(__file__))
default_model_dir = os.path.join(base_dir, "local_gemma_3n")

def load_local_weights(model_dir=default_model_dir):
    print("Loading FULL Gemma 3N E4B Weights (100% Clean Version)...")
    tensors = {}
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    
    for filename in st_files:
        pt_tensors = load_file(filename)
        for k, v in pt_tensors.items():
            tensors[k] = v.to(torch.float32).numpy()
            
    P = "model.language_model."
    
    W_embed = tensors[P + "embed_tokens.weight"]
    W_ple = tensors[P + "embed_tokens_per_layer.weight"]
    
    # 모든 + 1.0 완벽하게 멸종시킴!
    norm_ple = tensors[P + "per_layer_projection_norm.weight"] 
    W_ple_proj = tensors[P + "per_layer_model_projection.weight"].T

    altup_projs = [tensors[P + f"altup_projections.{i}.weight"].T for i in range(3)]
    altup_unprojs = [tensors[P + f"altup_unembed_projections.{i}.weight"].T for i in range(3)]
    
    W_final_norm = tensors[P + "norm.weight"]
    W_lm_head = W_embed.T.copy()
    
    layers = {"W_q":[], "W_k":[], "W_v":[], "W_o":[], "gamma_q":[], "gamma_k":[],
              "input_ln":[], "post_attn_ln":[], "pre_ffn_ln":[], "post_ffn_ln":[],
              "W_gate":[], "W_up":[], "W_down":[],
              "ple_gate":[], "ple_proj":[], "ple_post_ln":[],
              "laurel_left":[], "laurel_right":[], "laurel_norm":[],
              "altup_rn":[], "altup_router":[], "altup_pred":[], "altup_corr":[], "altup_scale":[]}

    for i in range(35):
        lp = P + f"layers.{i}."
        sa = lp + "self_attn."

        layers["W_q"].append(tensors[sa + "q_proj.weight"].T)
        layers["W_k"].append(tensors[sa + "k_proj.weight"].T)
        layers["W_v"].append(tensors[sa + "v_proj.weight"].T)
        layers["W_o"].append(tensors[sa + "o_proj.weight"].T)
        
        layers["gamma_q"].append(tensors[sa + "q_norm.weight"]) 
        layers["gamma_k"].append(tensors[sa + "k_norm.weight"]) 
        
        layers["input_ln"].append(tensors[lp + "input_layernorm.weight"]) 
        layers["post_attn_ln"].append(tensors[lp + "post_attention_layernorm.weight"]) 
        layers["pre_ffn_ln"].append(tensors[lp + "pre_feedforward_layernorm.weight"]) 
        layers["post_ffn_ln"].append(tensors[lp + "post_feedforward_layernorm.weight"]) 

        layers["W_gate"].append(tensors[lp + "mlp.gate_proj.weight"].T)
        layers["W_up"].append(tensors[lp + "mlp.up_proj.weight"].T)
        layers["W_down"].append(tensors[lp + "mlp.down_proj.weight"].T)

        layers["ple_gate"].append(tensors[lp + "per_layer_input_gate.weight"].T)
        layers["ple_proj"].append(tensors[lp + "per_layer_projection.weight"].T)
        layers["ple_post_ln"].append(tensors[lp + "post_per_layer_input_norm.weight"]) 

        layers["laurel_left"].append(tensors[lp + "laurel.linear_left.weight"].T)
        layers["laurel_right"].append(tensors[lp + "laurel.linear_right.weight"].T)
        layers["laurel_norm"].append(tensors[lp + "laurel.post_laurel_norm.weight"]) 

        layers["altup_rn"].append(tensors[lp + "altup.router_norm.weight"]) 
        layers["altup_router"].append(tensors[lp + "altup.modality_router.weight"].T)
        layers["altup_pred"].append(tensors[lp + "altup.prediction_coefs.weight"])
        layers["altup_corr"].append(tensors[lp + "altup.correction_coefs.weight"])
        layers["altup_scale"].append(tensors[lp + "altup.correct_output_scale"])

    print("All Gemma 3N Weights Loaded & Formatted (Zero +1.0)!")
    return W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, W_final_norm, W_lm_head, layers