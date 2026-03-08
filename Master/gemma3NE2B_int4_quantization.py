import os
import glob
import torch  # bfloat16을 읽기 위해 PyTorch 필수!
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file  # 저장도 PyTorch 텐서 기반으로 변경

# 아까 만든 양자화 및 패킹 함수 (Numpy 기반 유지)
def quantize_and_pack_int4(weight_tensor, group_size=32):
    out_features, in_features = weight_tensor.shape
    reshaped_weight = weight_tensor.reshape(out_features, in_features // group_size, group_size)
    
    max_abs = np.max(np.abs(reshaped_weight), axis=-1, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-9)
    scales = max_abs / 7.0
    
    quantized_weight = np.round(reshaped_weight / scales)
    quantized_weight = np.clip(quantized_weight, -8, 7).astype(np.int8)
    
    quantized_flat = quantized_weight.reshape(out_features, in_features)
    packed_shape = (out_features, in_features // 2)
    packed_weight = np.zeros(packed_shape, dtype=np.uint8)
    
    for i in range(packed_shape[1]):
        low_4bit = quantized_flat[:, 2*i] & 0x0F
        high_4bit = quantized_flat[:, 2*i + 1] & 0x0F
        packed_weight[:, i] = (high_4bit << 4) | low_4bit
        
    scales_flat = scales.reshape(out_features, in_features // group_size)
    
    return packed_weight, scales_flat

def process_gemma_weights(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    safetensor_files = glob.glob(os.path.join(input_dir, "*.safetensors"))
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"총 {len(safetensor_files)}개의 safetensors 파일을 변환 시작합니다...")

    for file_path in safetensor_files:
        file_name = os.path.basename(file_path)
        print(f"\n[{file_name}] 처리 중...")
        
        quantized_tensors = {}
        
        # framework="pt"로 변경해서 PyTorch 기반으로 읽어옴 (bfloat16 네이티브 지원)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            
            for key in keys:
                # 1. 원본 텐서를 가져옴 (이 시점에서는 torch.bfloat16)
                raw_tensor = f.get_tensor(key)
                
                # 2. 양자화 타겟인지 확인
                is_target = any(keyword in key for keyword in target_keywords)
                if len(raw_tensor.shape) != 2:
                    is_target = False

                if is_target:
                    # 3. 양자화를 위해 float32로 캐스팅하고 Numpy로 변환
                    tensor_np = raw_tensor.float().numpy()
                    print(f"  -> [양자화 O] {key} : 원본 형태 {tensor_np.shape}")
                    
                    packed_w_np, scale_w_np = quantize_and_pack_int4(tensor_np, group_size=32)
                    
                    # 4. 저장하기 위해 다시 PyTorch 텐서로 감싸기
                    quantized_tensors[key + ".weight_int4"] = torch.from_numpy(packed_w_np)
                    quantized_tensors[key + ".scales"] = torch.from_numpy(scale_w_np)
                else:
                    # 5. 양자화 안 하는 놈들은 용량 절약을 위해 bfloat16 그대로 유지해서 넘김
                    print(f"  -> [양자화 X (원본유지)] {key} : 원본 형태 {raw_tensor.shape}")
                    quantized_tensors[key] = raw_tensor

        # safetensors.torch.save_file을 이용해 저장
        output_path = os.path.join(output_dir, "quantized_" + file_name)
        save_file(quantized_tensors, output_path)
        print(f"저장 완료: {output_path}")


# 사용 예시 (경로는 네 환경에 맞게 수정해)
input_directory = "Master/gemma3NE4B"   # 다운받은 safetensors가 있는 폴더
output_directory = "Master/gemma3NE4B_INT4_Q"      # 압축된 파일을 저장할 폴더

process_gemma_weights(input_directory, output_directory)