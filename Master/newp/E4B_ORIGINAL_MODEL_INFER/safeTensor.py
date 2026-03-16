import numpy as np
import os
import gc
import torch
import glob
import re
from safetensors.torch import load_file
import IGPU_CORE

base_dir = os.path.dirname(os.path.abspath(__file__))
default_model_dir = os.path.join(base_dir, "local_gemma_3n")

# A suffix list of large matrices taking the IGPU path.
_BIG_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)

def load_local_weights(model_dir=default_model_dir):
    print("Loading FULL Gemma Weights (Memory Optimized Version)...")
    
    # Allocate space for 35 layers in advance
    num_layers = 35
    layers = {
        "W_q": [None]*num_layers, "W_k": [None]*num_layers, "W_v": [None]*num_layers, "W_o": [None]*num_layers,
        "gamma_q": [None]*num_layers, "gamma_k": [None]*num_layers,
        "input_ln": [None]*num_layers, "post_attn_ln": [None]*num_layers, "pre_ffn_ln": [None]*num_layers, "post_ffn_ln": [None]*num_layers,
        "W_gate": [None]*num_layers, "W_up": [None]*num_layers, "W_down": [None]*num_layers,
        "ple_gate": [None]*num_layers, "ple_proj": [None]*num_layers, "ple_post_ln": [None]*num_layers,
        "laurel_left": [None]*num_layers, "laurel_right": [None]*num_layers, "laurel_norm": [None]*num_layers,
        "altup_rn": [None]*num_layers, "altup_router": [None]*num_layers, "altup_pred": [None]*num_layers, "altup_corr": [None]*num_layers, "altup_scale": [None]*num_layers,
    }
    
    globals_dict = {}
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    
    # Extract layer number with regular expression
    layer_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.(.*)")
    
    for filename in st_files:
        print(f"  Reading {os.path.basename(filename)}...")
        pt_tensors = load_file(filename)
        
        for k, v in pt_tensors.items():
            # 1. Determine dtype (avoid wasting memory)
            is_big = any(k.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)
            dtype = torch.float16 if is_big else torch.float32
            
            if "embed_tokens.weight" in k or "embed_tokens_per_layer.weight" in k:
                dtype = torch.float16
                
            arr = v.to(dtype).numpy()
            
            # 2. Check the tensors needed for Transpose in advance and fix them in continuous memory.
            needs_transpose = False
            if "per_layer_model_projection.weight" in k or "altup_projections" in k or "altup_unembed_projections" in k:
                needs_transpose = True
            elif "q_proj.weight" in k or "k_proj.weight" in k or "v_proj.weight" in k or "o_proj.weight" in k:
                needs_transpose = True
            elif "gate_proj.weight" in k or "up_proj.weight" in k or "down_proj.weight" in k:
                needs_transpose = True
            elif "per_layer_input_gate.weight" in k or "per_layer_projection.weight" in k:
                needs_transpose = True
            elif "laurel.linear_left.weight" in k or "laurel.linear_right.weight" in k:
                needs_transpose = True
            elif "altup.modality_router.weight" in k:
                needs_transpose = True
                
            if needs_transpose:
                arr = np.ascontiguousarray(arr.T)
            else:
                arr = np.ascontiguousarray(arr)
            
            # 3. Direct assignment of layer parameters (no intermediate copies)
            match = layer_pattern.match(k)
            if match:
                layer_idx = int(match.group(1))
                sub_key = match.group(2)
                
                if sub_key == "self_attn.q_proj.weight": layers["W_q"][layer_idx] = arr
                elif sub_key == "self_attn.k_proj.weight": layers["W_k"][layer_idx] = arr
                elif sub_key == "self_attn.v_proj.weight": layers["W_v"][layer_idx] = arr
                elif sub_key == "self_attn.o_proj.weight": layers["W_o"][layer_idx] = arr
                elif sub_key == "self_attn.q_norm.weight": layers["gamma_q"][layer_idx] = arr
                elif sub_key == "self_attn.k_norm.weight": layers["gamma_k"][layer_idx] = arr
                elif sub_key == "input_layernorm.weight": layers["input_ln"][layer_idx] = arr
                elif sub_key == "post_attention_layernorm.weight": layers["post_attn_ln"][layer_idx] = arr
                elif sub_key == "pre_feedforward_layernorm.weight": layers["pre_ffn_ln"][layer_idx] = arr
                elif sub_key == "post_feedforward_layernorm.weight": layers["post_ffn_ln"][layer_idx] = arr
                elif sub_key == "mlp.gate_proj.weight": layers["W_gate"][layer_idx] = arr
                elif sub_key == "mlp.up_proj.weight": layers["W_up"][layer_idx] = arr
                elif sub_key == "mlp.down_proj.weight": layers["W_down"][layer_idx] = arr
                elif sub_key == "per_layer_input_gate.weight": layers["ple_gate"][layer_idx] = arr
                elif sub_key == "per_layer_projection.weight": layers["ple_proj"][layer_idx] = arr
                elif sub_key == "post_per_layer_input_norm.weight": layers["ple_post_ln"][layer_idx] = arr
                elif sub_key == "laurel.linear_left.weight": layers["laurel_left"][layer_idx] = arr
                elif sub_key == "laurel.linear_right.weight": layers["laurel_right"][layer_idx] = arr
                elif sub_key == "laurel.post_laurel_norm.weight": layers["laurel_norm"][layer_idx] = arr
                elif sub_key == "altup.router_norm.weight": layers["altup_rn"][layer_idx] = arr
                elif sub_key == "altup.modality_router.weight": layers["altup_router"][layer_idx] = arr
                elif sub_key == "altup.prediction_coefs.weight": layers["altup_pred"][layer_idx] = arr
                elif sub_key == "altup.correction_coefs.weight": layers["altup_corr"][layer_idx] = arr
                elif sub_key == "altup.correct_output_scale": layers["altup_scale"][layer_idx] = arr
            else:
                globals_dict[k] = arr
        
        # Key point: Read file by file and release memory immediately
        del pt_tensors
        gc.collect()

    # Global weight theorem
    P = "model.language_model."
    W_embed = globals_dict[P + "embed_tokens.weight"]
    W_ple = globals_dict[P + "embed_tokens_per_layer.weight"]
    norm_ple = globals_dict[P + "per_layer_projection_norm.weight"]
    W_ple_proj = globals_dict[P + "per_layer_model_projection.weight"]
    
    altup_projs = [globals_dict[P + f"altup_projections.{i}.weight"] for i in range(3)]
    altup_unprojs = [globals_dict[P + f"altup_unembed_projections.{i}.weight"] for i in range(3)]
    W_final_norm = globals_dict[P + "norm.weight"]

    W_lm_head = np.ascontiguousarray(W_embed.T)

    # Also free any remaining temporary dictionaries.
    del globals_dict
    gc.collect()

    print("All Weights Loaded Memory-Efficiently!")
    
    return (W_embed, W_ple, norm_ple, W_ple_proj,
            altup_projs, altup_unprojs,
            W_final_norm, W_lm_head, layers)

