import sys
import numpy as np
import safeTensor

def get_real_memory_size(obj):
    """
    It even digs into Numpy arrays nested within tuples and lists to accurately calculate the actual VRAM/RAM occupancy.
    """
    total = sys.getsizeof(obj)  # Default shell size

    if isinstance(obj, np.ndarray):
        # If it is a NumPy array, add up the actual data byte size.
        total += obj.nbytes
    elif isinstance(obj, (list, tuple)):
        # In the case of a list or tuple, the contents are taken out one by one and the size is added recursively.
        for item in obj:
            total += get_real_memory_size(item)  # <- Key point: Drilling down to the end of the internal elements with recursive calls.
            
    return total

def format_memory_size(total_bytes):
    gb_size = total_bytes / (1024 ** 3)
    mb_size = total_bytes / (1024 ** 2)
    mb_bit_size = (total_bytes * 8) / (1024 ** 2)
    return f"{gb_size:.6f} | {mb_size:.3f} | {mb_bit_size:.3f} "

def calculate_memory_usage(obj):
    total_bytes = get_real_memory_size(obj)
    return format_memory_size(total_bytes)

def inspect_matrix_structure(name, obj):
    """
    Recursively delve into the nested structures of lists, tuples, and numpy arrays
    Returns the actual N x M dimensions and data type (int4, float32, etc.).
    """
    def _get_shape_and_type(item):
        if isinstance(item, list):
            if len(item) == 0:
                return "Empty List"
            # In the case of a layer list, only the structure of the first element (Layer 0) is checked as representative.
            return f"List[{len(item)}] ──>  { _get_shape_and_type(item[0]) }"
            
        elif isinstance(item, tuple):
            # If it is a tuple (usually a quantized matrix: (Packed_Weight, Scale))
            inner = ", ".join([_get_shape_and_type(sub) for sub in item])
            return f"Tuple( {inner} )"
            
        elif isinstance(item, np.ndarray):
            shape_str = " x ".join(map(str, item.shape))
            dtype_str = str(item.dtype)
            
            # In the quantization logic (quantize.py), the uint8 two-dimensional array is INT4 packed.
            # Since 1 byte (uint8) contains 2 INT4s, the actual number of columns is doubled.
            if dtype_str == "uint8" and len(item.shape) == 2:
                real_cols = item.shape[1] * 2
                return f"[ matrix: {shape_str} , type: {dtype_str} , (INT4 dimension: {item.shape[0]} x {real_cols}) ]"
            
            return f"[ matrix: {shape_str} , type: {dtype_str} ]"
            
        else:
            return f"[ type: {type(item).__name__} ]"

    return _get_shape_and_type(obj)
    

def debug():
    W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()
    
    print(f"|name|matrix|GB|MB|Mb")
    print(f"|---|---|---|---|---|")
    
    # W Prints the items inside the dictionary
    for key in ["altup_rn", "altup_router", "altup_pred", "input_ln", "W_q", "W_k", "W_v", 
                "gamma_q", "gamma_k", "W_o", "laurel_left", "laurel_right", "laurel_norm", 
                "post_attn_ln", "pre_ffn_ln", "W_gate", "W_up", "W_down", "post_ffn_ln", 
                "altup_scale", "altup_corr", "ple_gate", "ple_proj", "ple_post_ln"]:
        val = W[key]
        type_val = type(val)
        type_0 = type(val[0]) if isinstance(val, (list, tuple)) and len(val) > 0 else "-"
        print(f"{key} | {inspect_matrix_structure(key, val)} | {calculate_memory_usage(val)}")
        

    # Print independent variables
    for name, val in [("W_embed", W_embed), ("W_ple", W_ple), ("norm_ple", norm_ple), 
                      ("W_ple_proj", W_ple_proj), ("altup_projs", altup_projs), 
                      ("altup_unprojs", altup_unprojs), ("W_final_norm", W_final_norm), 
                      ("W_lm_head", W_lm_head)]:
        print(f"{name} | {inspect_matrix_structure(name, val)} | {calculate_memory_usage(val)}")

#if __name__ == "__main__":
#    debug()

import os
import glob
import gc
import torch
import numpy as np
from safetensors.torch import load_file

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "local_gemma_3n_int4")
out_dir = os.path.join(base_dir, "mmap_weights")

os.makedirs(out_dir, exist_ok=True)

st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
print(f"Split total {len(st_files)} Safetensors files into individual npy...")

count = 0
for st_file in st_files:
    print(f" Converting: {os.path.basename(st_file)}")
    tensors = load_file(st_file)
    
    # Check whether there is a scale file in advance to determine whether it is INT4 or not.
    scale_keys = [k for k in tensors.keys() if k.endswith(".scale")]
    quantized_bases = [k[:-6] for k in scale_keys]
    
    for k, val in tensors.items():
        if val.dtype == torch.bfloat16:
            val = val.to(torch.float32)
            
        arr = val.numpy()
        
        # [Core bug fix] INT4 (quantization) tensor is dimensionally twisted, so it is never flipped.
        is_quantized = k in quantized_bases or k.endswith(".scale")
        needs_transpose = False
        
        if not is_quantized:
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
            
        out_path = os.path.join(out_dir, f"{k}.npy")
        np.save(out_path, arr)
        count += 1
        
    del tensors
    gc.collect()

print(f" A total of {count} conversions completed! (INT4 protection fully applied)")
