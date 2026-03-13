import os
import torch
import numpy as np
from safetensors.torch import load_file, save_file
import gc
import glob

# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# E2B 원본 모델 경로
ORIGINAL_MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "E2B_ORIGINAL_MODEL_INFER", "[Original Model]gemma3NE2B"))
SAVE_DIR = os.path.join(BASE_DIR, "local_gemma_3n_int4")

_BIG_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    "embed_tokens.weight", 
)

def quantize_to_int4(weight):
    # Per-row quantization
    N, M = weight.shape
    
    # Float32 for precision during quantization
    w_f32 = weight.astype(np.float32)
    
    # Symmetric-ish quantization: range [-8, 7]
    max_vals = np.max(np.abs(w_f32), axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-8)
    scale = (max_vals / 7.0).flatten()
    
    # Quantize
    w_q = np.round(w_f32 / max_vals * 7.0).astype(np.int8)
    w_q = np.clip(w_q, -8, 7)
    
    # Pack to uint8
    w_q_low = w_q[:, 0::2] & 0x0F
    w_q_high = (w_q[:, 1::2] & 0x0F) << 4
    packed = (w_q_low | w_q_high).astype(np.uint8)
    
    return packed, scale

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print(f"Quantizing models from {ORIGINAL_MODEL_DIR}...")
    st_files = sorted(glob.glob(os.path.join(glob.escape(ORIGINAL_MODEL_DIR), "*.safetensors")))
    
    if not st_files:
        print(f"Error: No safetensors found in {ORIGINAL_MODEL_DIR}")
        return

    for filename in st_files:
        print(f"Processing {os.path.basename(filename)}...")
        tensors = load_file(filename)
        quantized_tensors = {}
        
        for name, tensor in tensors.items():
            # 텍스트 모델 가중치만 처리
            if "language_model" not in name:
                continue

            is_big = any(name.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)
            
            if is_big and len(tensor.shape) == 2:
                print(f"  Quantizing {name} {tensor.shape}...")
                weight_np = tensor.to(torch.float32).numpy()
                packed, scale = quantize_to_int4(weight_np)
                
                quantized_tensors[name] = torch.from_numpy(packed)
                quantized_tensors[name + ".scale"] = torch.from_numpy(scale.astype(np.float32))
            else:
                # Keep original as float16 to save space
                quantized_tensors[name] = tensor.to(torch.float16)
            
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
