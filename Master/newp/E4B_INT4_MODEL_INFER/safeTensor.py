'''

import numpy as np
import os
import gc
import torch
import glob
import re
from safetensors.torch import load_file

# see ram usage
import os
import psutil

def print_ram_usage(step_name):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    print(f"[{step_name}] RAM Usage: {rss_mb:.2f} MB")




base_dir = os.path.dirname(os.path.abspath(__file__))
default_model_dir = os.path.join(base_dir, "local_gemma_3n_int4")

# A suffix list of large matrices taking the IGPU path.
_BIG_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    "embed_tokens.weight",
    "embed_tokens_per_layer.weight",      # W_ple (8.75GB -> 4.37GB)
    "per_layer_input_gate.weight",        # ple_gate (140MB -> 70MB)
    "per_layer_model_projection.weight",  # ple_proj (140MB -> 70MB)
    "laurel.linear_left.weight",          # laurel_left (35MB -> 17MB)
    "laurel.linear_right.weight",         # laurel_right (35MB -> 17MB)
)

def load_local_weights(model_dir=default_model_dir):
    print(f"Loading INT4 Quantized Gemma Weights from {model_dir}...")
    
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
    
    layer_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.(.*)")
    
    # Store scales separately temporarily
    scales = {}
    print_ram_usage("1")
    for filename in st_files:
        print(f"  Reading {os.path.basename(filename)}...")
        pt_tensors = load_file(filename)
        
        print_ram_usage("1-?")

        # First, collect all scales
        for k in list(pt_tensors.keys()):
            if k.endswith(".scale"):
                scales[k[:-6]] = pt_tensors.pop(k).numpy()
        
        for k, v in pt_tensors.items():
            is_quantized = k in scales
            
            if is_quantized:
                # v is uint8 packed, scales[k] is float32 scale
                # We store them as a tuple (packed, scale)
                arr = v.numpy() # packed uint8
                scale = scales[k]
                
                # Transpose needed for MatMul efficiency in IGPU_CORE?
                # Actually, our quantization was on [N, M]. 
                # Let's keep it as is, and IGPU_CORE will handle it.
                val = (arr, scale)
            else:
                is_big = any(k.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)
                dtype = torch.float16 if is_big else torch.float32
                arr = v.to(dtype).numpy()
                
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
                val = arr

            match = layer_pattern.match(k)
            if match:
                layer_idx = int(match.group(1))
                sub_key = match.group(2)
                
                if sub_key == "self_attn.q_proj.weight": layers["W_q"][layer_idx] = val
                elif sub_key == "self_attn.k_proj.weight": layers["W_k"][layer_idx] = val
                elif sub_key == "self_attn.v_proj.weight": layers["W_v"][layer_idx] = val
                elif sub_key == "self_attn.o_proj.weight": layers["W_o"][layer_idx] = val
                elif sub_key == "self_attn.q_norm.weight": layers["gamma_q"][layer_idx] = val
                elif sub_key == "self_attn.k_norm.weight": layers["gamma_k"][layer_idx] = val
                elif sub_key == "input_layernorm.weight": layers["input_ln"][layer_idx] = val
                elif sub_key == "post_attention_layernorm.weight": layers["post_attn_ln"][layer_idx] = val
                elif sub_key == "pre_feedforward_layernorm.weight": layers["pre_ffn_ln"][layer_idx] = val
                elif sub_key == "post_feedforward_layernorm.weight": layers["post_ffn_ln"][layer_idx] = val
                elif sub_key == "mlp.gate_proj.weight": layers["W_gate"][layer_idx] = val
                elif sub_key == "mlp.up_proj.weight": layers["W_up"][layer_idx] = val
                elif sub_key == "mlp.down_proj.weight": layers["W_down"][layer_idx] = val
                elif sub_key == "per_layer_input_gate.weight": layers["ple_gate"][layer_idx] = val
                elif sub_key == "per_layer_projection.weight": layers["ple_proj"][layer_idx] = val
                elif sub_key == "post_per_layer_input_norm.weight": layers["ple_post_ln"][layer_idx] = val
                elif sub_key == "laurel.linear_left.weight": layers["laurel_left"][layer_idx] = val
                elif sub_key == "laurel.linear_right.weight": layers["laurel_right"][layer_idx] = val
                elif sub_key == "laurel.post_laurel_norm.weight": layers["laurel_norm"][layer_idx] = val
                elif sub_key == "altup.router_norm.weight": layers["altup_rn"][layer_idx] = val
                elif sub_key == "altup.modality_router.weight": layers["altup_router"][layer_idx] = val
                elif sub_key == "altup.prediction_coefs.weight": layers["altup_pred"][layer_idx] = val
                elif sub_key == "altup.correction_coefs.weight": layers["altup_corr"][layer_idx] = val
                elif sub_key == "altup.correct_output_scale": layers["altup_scale"][layer_idx] = val
            else:
                globals_dict[k] = val
        
        del pt_tensors
        gc.collect()

    print_ram_usage("2")

    P = "model.language_model."
    
    # 1. W_embed is also read using mmap and bundled into a tuple.
    W_embed_packed = np.load("Master/newp/E4B_INT4_MODEL_INFER/W_embed_packed.npy", mmap_mode='r')
    W_embed_scale  = np.load("Master/newp/E4B_INT4_MODEL_INFER/W_embed_scale.npy", mmap_mode='r')
    W_embed = (W_embed_packed, W_embed_scale) # Bottom line: packaging it as a tuple.

    # 2. W_ple is also read using mmap.
    W_ple_packed   = np.load("Master/newp/E4B_INT4_MODEL_INFER/W_ple_packed.npy", mmap_mode='r')
    W_ple_scale    = np.load("Master/newp/E4B_INT4_MODEL_INFER/W_ple_scale.npy", mmap_mode='r')

    norm_ple = globals_dict[P + "per_layer_projection_norm.weight"]
    W_ple_proj = globals_dict[P + "per_layer_model_projection.weight"]
    
    altup_projs = [globals_dict[P + f"altup_projections.{i}.weight"] for i in range(3)]
    altup_unprojs = [globals_dict[P + f"altup_unembed_projections.{i}.weight"] for i in range(3)]
    W_final_norm = globals_dict[P + "norm.weight"]

    # 3. LM Head refers to W_embed bound as a tuple.
    W_lm_head = W_embed

    del globals_dict
    gc.collect()

    print("All Weights Loaded (INT4 Support) ✓")
    
    return (W_embed, 
            W_ple_packed, 
            W_ple_scale,
            norm_ple, 
            W_ple_proj,
            altup_projs, 
            altup_unprojs,
            W_final_norm, 
            W_lm_head, 
            layers)
            
'''


import numpy as np
import os
import gc
import glob
import re
import psutil

def print_ram_usage(step_name):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    print(f"[{step_name}] RAM Usage: {rss_mb:.2f} MB")

base_dir = os.path.dirname(os.path.abspath(__file__))
# Now, rather than looking at the safetensors folder, we are looking at the split mmap_weights folder.
mmap_dir = os.path.join(base_dir, "mmap_weights")

def load_local_weights(model_dir=mmap_dir):
    print(f"Loading INT4 Quantized Gemma Weights via MMAP from {model_dir}...")
    
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
    layer_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.(.*)")
    
    # Extract all npy file names on disk
    all_files = glob.glob(os.path.join(model_dir, "*.npy"))
    all_keys = [os.path.basename(f)[:-4] for f in all_files]
    
    # Files ending in .scale are separated into a dictionary to find matches.
    scales = {k[:-6]: k for k in all_keys if k.endswith(".scale")}
    
    print_ram_usage("1. Start MMAP Virtual Mapping")
    
    for k in all_keys:
        if k.endswith(".scale"):
            continue # The scale is loaded when the main body (weight) is loaded.
            
        # RAM consumption 0MB! (only disk address is taken)
        val = np.load(os.path.join(model_dir, f"{k}.npy"), mmap_mode='r')
        
        # If it is a quantized (INT4) tensor with a scale value, it is grouped into a tuple (packed, scale).
        if k in scales:
            scale_val = np.load(os.path.join(model_dir, f"{scales[k]}.npy"), mmap_mode='r')
            val = (val, scale_val)
            
        match = layer_pattern.match(k)
        if match:
            layer_idx = int(match.group(1))
            sub_key = match.group(2)
            
            # [Core bug fix] 100% perfect match to the original name tag.
            if sub_key == "self_attn.q_proj.weight": layers["W_q"][layer_idx] = val
            elif sub_key == "self_attn.k_proj.weight": layers["W_k"][layer_idx] = val
            elif sub_key == "self_attn.v_proj.weight": layers["W_v"][layer_idx] = val
            elif sub_key == "self_attn.o_proj.weight": layers["W_o"][layer_idx] = val
            elif sub_key == "self_attn.q_norm.weight": layers["gamma_q"][layer_idx] = val
            elif sub_key == "self_attn.k_norm.weight": layers["gamma_k"][layer_idx] = val
            elif sub_key == "input_layernorm.weight": layers["input_ln"][layer_idx] = val
            elif sub_key == "post_attention_layernorm.weight": layers["post_attn_ln"][layer_idx] = val
            elif sub_key == "pre_feedforward_layernorm.weight": layers["pre_ffn_ln"][layer_idx] = val
            elif sub_key == "post_feedforward_layernorm.weight": layers["post_ffn_ln"][layer_idx] = val
            elif sub_key == "mlp.gate_proj.weight": layers["W_gate"][layer_idx] = val
            elif sub_key == "mlp.up_proj.weight": layers["W_up"][layer_idx] = val
            elif sub_key == "mlp.down_proj.weight": layers["W_down"][layer_idx] = val
            elif sub_key == "per_layer_input_gate.weight": layers["ple_gate"][layer_idx] = val
            elif sub_key == "per_layer_projection.weight": layers["ple_proj"][layer_idx] = val
            elif sub_key == "post_per_layer_input_norm.weight": layers["ple_post_ln"][layer_idx] = val
            elif sub_key == "laurel.linear_left.weight": layers["laurel_left"][layer_idx] = val
            elif sub_key == "laurel.linear_right.weight": layers["laurel_right"][layer_idx] = val
            elif sub_key == "laurel.post_laurel_norm.weight": layers["laurel_norm"][layer_idx] = val
            elif sub_key == "altup.router_norm.weight": layers["altup_rn"][layer_idx] = val
            elif sub_key == "altup.modality_router.weight": layers["altup_router"][layer_idx] = val
            elif sub_key == "altup.prediction_coefs.weight": layers["altup_pred"][layer_idx] = val
            elif sub_key == "altup.correction_coefs.weight": layers["altup_corr"][layer_idx] = val
            elif sub_key == "altup.correct_output_scale": layers["altup_scale"][layer_idx] = val
            else:
                globals_dict[k] = val
        else:
            globals_dict[k] = val

    print_ram_usage("2. MMAP Mapping Complete")

    P = "model.language_model."
    
    # Tuple and array decomposition (maintaining full compatibility with main.py)
    W_embed = globals_dict[P + "embed_tokens.weight"]
    W_ple_packed, W_ple_scale = globals_dict[P + "embed_tokens_per_layer.weight"]
    
    norm_ple = globals_dict[P + "per_layer_projection_norm.weight"]
    W_ple_proj = globals_dict[P + "per_layer_model_projection.weight"]
    
    altup_projs = [globals_dict[P + f"altup_projections.{i}.weight"] for i in range(3)]
    altup_unprojs = [globals_dict[P + f"altup_unembed_projections.{i}.weight"] for i in range(3)]
    W_final_norm = globals_dict[P + "norm.weight"]

    W_lm_head = W_embed

    print("All Weights Loaded (INT4 MMAP Support) ✓")
    return (W_embed, 
            W_ple_packed, 
            W_ple_scale,
            norm_ple, 
            W_ple_proj,
            altup_projs, 
            altup_unprojs,
            W_final_norm, 
            W_lm_head, 
            layers)