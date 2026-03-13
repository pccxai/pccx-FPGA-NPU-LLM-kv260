import sys
import numpy as np
import safeTensor

def get_total_size(obj):
    total = sys.getsizeof(obj)  # 리스트 자체 크기 먼저 시작
    
    for item in obj:
        if isinstance(item, np.ndarray):
            # 넘파이 배열: 데이터 크기(nbytes) + 객체 헤더 크기
            total += item.nbytes + sys.getsizeof(item)
        else:
            # 튜플 등 일반 객체
            total += sys.getsizeof(item)
    
    # --- 여기서부터 루프 밖으로 빼야 전체 합산본으로 계산됩니다 ---
    gb_size = total / (1024 ** 3)
    mb_bit_size = (total * 8) / (1024 ** 2)
    Mb_size = total / (1024 * 1024)
    return f"{gb_size:.10f} | {Mb_size:.6f} | {mb_bit_size:.4f}"

def calculate_memory_usage(obj):
    # 1. 기본 객체 자체의 헤더 크기 측정
    total = sys.getsizeof(obj)
    
    # 2. 내부 요소 순회 (리스트, 튜플 등 반복 가능한 객체인 경우)
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, np.ndarray):
                # 넘파이 배열: 실제 데이터(nbytes) + 객체 관리용 헤더 크기
                total += item.nbytes + sys.getsizeof(item)
            else:
                # 그 외 내부 객체(또 다른 튜플, 숫자 등)
                total += sys.getsizeof(item)
                
    # 3. 입력 객체가 단일 Numpy 배열인 경우 처리
    elif isinstance(obj, np.ndarray):
        total = obj.nbytes + sys.getsizeof(obj)

    # 단위 변환 계산
    gb_size = total / (1024 ** 3)
    mb_size = total / (1024 ** 2)
    mb_bit_size = (total * 8) / (1024 ** 2) # Megabit (Mb) 단위
    
    return f"{gb_size:.10f} GB | {mb_size:.6f} MB | {mb_bit_size:.4f} Mb (bits)"



def debug():
    W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()
    print(f"""|name|type|[0] type|GB size|MB size|mb_bit_size|\n
|---|---|---|---|---|---|""")
    
    dictKey = []
    for name1 in dictKey:
        print(f"{name1} | {type(W[name1])} | {type(W[name1][0])} | {get_total_size(W[name1])} |\n")
    
    print(f"""
W_embed | {type(W_embed)} | - | {calculate_memory_usage(W_embed)} |\n
W_ple | {type(W_ple)} | - | {calculate_memory_usage(W_ple)} |\n
norm_ple | {type(norm_ple)} | - | {calculate_memory_usage(norm_ple)} |\n
W_ple_proj | {type(W_ple_proj)} | - | {calculate_memory_usage(W_ple_proj)} |\n
altup_projs | {type(altup_projs)} | - | {calculate_memory_usage(altup_projs)} |\n
altup_unprojs | {type(altup_unprojs)} | - | {calculate_memory_usage(altup_unprojs)} |\n
W_final_norm | {type(W_final_norm)} | - | {calculate_memory_usage(W_final_norm)} |\n
W_lm_head | {type(W_lm_head)} | - | {calculate_memory_usage(W_lm_head)} |\n
""")
import numpy as np
from safetensors.torch import load_file
import glob
import os
import gc

# 💡 회원님의 safeTensor.py 와 동일한 절대경로 로직 적용!
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "local_gemma_3n_int4")

# 폴더 안의 safetensors 파일들 찾기
st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

if len(st_files) == 0:
    print(f"❌ 에러: {model_dir} 경로에서 safetensors 파일을 한 개도 찾지 못했습니다.")
    print("스크립트 파일이 main.py와 같은 위치(최상단 폴더)에 있는지 확인해 주세요!")
    exit(1)

target_keys = {
    "model.language_model.embed_tokens.weight": "W_embed_packed.npy",
    "model.language_model.embed_tokens.weight.scale": "W_embed_scale.npy",
    "model.language_model.embed_tokens_per_layer.weight": "W_ple_packed.npy",
    "model.language_model.embed_tokens_per_layer.weight.scale": "W_ple_scale.npy",
}

found_count = 0

print(f"🔍 [{model_dir}] 폴더에서 수색을 시작합니다. (발견된 파일: {len(st_files)}개)")

for st_file in st_files:
    print(f"\n📂 열어보는 중: {os.path.basename(st_file)}")
    tensors = load_file(st_file)
    
    for key in list(target_keys.keys()):
        if key in tensors:
            npy_filename = os.path.join(base_dir, target_keys[key])
            print(f"  👉 타겟 발견! [{key}]")
            print(f"     ➔ {target_keys[key]} 파일로 저장 중... ⏳")
            
            # numpy 배열로 변환해서 저장
            np.save(npy_filename, tensors[key].numpy())
            
            del target_keys[key]
            found_count += 1
            
    del tensors
    gc.collect()

print("\n" + "-" * 50)
if found_count == 4:
    print("🎉 완벽합니다! 4개의 필수 텐서를 모두 성공적으로 분리했습니다.")
    print("이제 이 스크립트가 있는 폴더에 4개의 .npy 파일이 생성되었습니다.")
else:
    print(f"⚠️ 일부 텐서를 찾지 못했습니다. 못 찾은 항목: {list(target_keys.keys())}")