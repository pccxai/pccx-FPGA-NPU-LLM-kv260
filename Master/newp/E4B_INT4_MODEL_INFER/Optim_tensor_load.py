import sys
import numpy as np
import safeTensor

def get_real_memory_size(obj):
    """
    튜플, 리스트 안에 중첩된 Numpy 배열까지 모두 파고들어서 실제 VRAM/RAM 점유율을 정확히 계산합니다.
    """
    total = sys.getsizeof(obj)  # 기본 껍데기 크기

    if isinstance(obj, np.ndarray):
        # 넘파이 배열인 경우 실제 데이터 바이트 크기 합산
        total += obj.nbytes
    elif isinstance(obj, (list, tuple)):
        # 리스트나 튜플인 경우, 안의 내용물을 하나씩 꺼내서 재귀적으로 크기를 더함
        for item in obj:
            total += get_real_memory_size(item)  # <- 핵심: 재귀 호출로 내부 요소 끝까지 파고듦!
            
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
    리스트, 튜플, numpy 배열의 중첩 구조를 재귀적으로 파고들어
    실제 N x M 차원과 데이터 타입(int4, float32 등)을 반환합니다.
    """
    def _get_shape_and_type(item):
        if isinstance(item, list):
            if len(item) == 0:
                return "Empty List"
            # 레이어 리스트인 경우 첫 번째 원소(Layer 0)의 구조만 대표로 확인
            return f"List[{len(item)}] ──>  { _get_shape_and_type(item[0]) }"
            
        elif isinstance(item, tuple):
            # 튜플인 경우 (보통 양자화된 행렬: (Packed_Weight, Scale))
            inner = ", ".join([_get_shape_and_type(sub) for sub in item])
            return f"Tuple( {inner} )"
            
        elif isinstance(item, np.ndarray):
            shape_str = " x ".join(map(str, item.shape))
            dtype_str = str(item.dtype)
            
            # 양자화 로직(quantize.py)에서 uint8 2차원 배열은 INT4가 패킹된 상태임.
            # 1바이트(uint8)에 INT4 2개가 들어있으므로 실제 열(Column) 개수는 2배.
            if dtype_str == "uint8" and len(item.shape) == 2:
                real_cols = item.shape[1] * 2
                return f"[ 행렬: {shape_str} , 타입: {dtype_str} , (INT4 차원: {item.shape[0]} x {real_cols}) ]"
            
            return f"[ 행렬: {shape_str} , 타입: {dtype_str} ]"
            
        else:
            return f"[ 타입: {type(item).__name__} ]"

    return _get_shape_and_type(obj)
    

def debug():
    W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()
    
    print(f"|name|matrix|GB|MB|Mb")
    print(f"|---|---|---|---|---|")
    
    # W 딕셔너리 내부 항목들 출력
    for key in ["altup_rn", "altup_router", "altup_pred", "input_ln", "W_q", "W_k", "W_v", 
                "gamma_q", "gamma_k", "W_o", "laurel_left", "laurel_right", "laurel_norm", 
                "post_attn_ln", "pre_ffn_ln", "W_gate", "W_up", "W_down", "post_ffn_ln", 
                "altup_scale", "altup_corr", "ple_gate", "ple_proj", "ple_post_ln"]:
        val = W[key]
        type_val = type(val)
        type_0 = type(val[0]) if isinstance(val, (list, tuple)) and len(val) > 0 else "-"
        print(f"{key} | {inspect_matrix_structure(key, val)} | {calculate_memory_usage(val)}")
        

    # 독립 변수들 출력
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
print(f"🚀 총 {len(st_files)}개의 Safetensors 파일을 개별 npy로 쪼갭니다...")

count = 0
for st_file in st_files:
    print(f"📂 변환 중: {os.path.basename(st_file)}")
    tensors = load_file(st_file)
    
    # 미리 scale 파일이 있는지 확인하여 INT4 여부 판별
    scale_keys = [k for k in tensors.keys() if k.endswith(".scale")]
    quantized_bases = [k[:-6] for k in scale_keys]
    
    for k, val in tensors.items():
        if val.dtype == torch.bfloat16:
            val = val.to(torch.float32)
            
        arr = val.numpy()
        
        # 💡 [핵심 버그 수정] INT4(양자화) 텐서는 차원이 꼬이므로 절대 뒤집지 않음!
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

print(f"✅ 총 {count}개 변환 완료! (INT4 보호 완벽 적용)")
