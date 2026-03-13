import os
import torch
import numpy as np
from safetensors.torch import load_file, save_file
import gc
import glob

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_MODEL_DIR = "/home/hwkim/Desktop/github/TinyNPU-RTL/Master/newp/E4B_ORIGINAL_MODEL_INFER/local_gemma_3n"
SAVE_DIR = os.path.join(BASE_DIR, "local_gemma_3n_int4")

_BIG_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    "embed_tokens.weight", # Embedding is also big


    "embed_tokens_per_layer.weight",
    "per_layer_input_gate.weight",
    "per_layer_model_projection.weight",
    "laurel.linear_left.weight",
    "laurel.linear_right.weight",
)

def quantize_to_int4(weight):
    # weight: [N, M] (numpy array, float16/32)
    # Returns: quantized_packed (uint8 [N, M//2]), scale (float32 [N])
    
    # Per-row quantization
    N, M = weight.shape
    if M % 2 != 0:
        # Pad M if necessary, but usually transformer weights are powers of 2
        pass
    
    # Float32 for precision during quantization
    w_f32 = weight.astype(np.float32)
    
    # Symmetric quantization: scale = max(abs(w)) / 7.0
    # Range: [-7, 7] (uses 4 bits)
    # Actually, we can use [-8, 7] or [0, 15] with offset.
    # Let's use [-8, 7] symmetric-ish.
    
    max_vals = np.max(np.abs(w_f32), axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-8)
    scale = (max_vals / 7.0).flatten()
    
    # Quantize
    w_q = np.round(w_f32 / max_vals * 7.0).astype(np.int8)
    w_q = np.clip(w_q, -8, 7)
    
    # Pack to uint8
    # w_q: [N, M]
    # packed: [N, M//2]
    # We'll use low 4 bits for first, high 4 bits for second.
    # To handle negative: (w_q & 0x0F)
    w_q_low = w_q[:, 0::2] & 0x0F
    w_q_high = w_q[:, 1::2] & 0x0F
    packed = (w_q_low | (w_q_high << 4)).astype(np.uint8)
    
    return packed, scale

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    st_files = sorted(glob.glob(os.path.join(ORIGINAL_MODEL_DIR, "*.safetensors")))
    
    for filename in st_files:
        print(f"Processing {os.path.basename(filename)}...")
        tensors = load_file(filename)
        quantized_tensors = {}
        
        for name, tensor in tensors.items():
            is_big = any(name.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)
            
            if is_big and len(tensor.shape) == 2:
                print(f"  Quantizing {name} {tensor.shape}...")
                weight_np = tensor.to(torch.float32).numpy()
                packed, scale = quantize_to_int4(weight_np)
                
                quantized_tensors[name] = torch.from_numpy(packed)
                quantized_tensors[name + ".scale"] = torch.from_numpy(scale)
            else:
                # Keep original
                quantized_tensors[name] = tensor
            
        save_path = os.path.join(SAVE_DIR, os.path.basename(filename))
        print(f"Saving to {save_path}...")
        save_file(quantized_tensors, save_path)
        
        # Clean up
        del tensors
        del quantized_tensors
        gc.collect()

    print("Quantization complete!")

if __name__ == "__main__":
    main()
