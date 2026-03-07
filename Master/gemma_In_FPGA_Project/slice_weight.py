import numpy as np
import os

def slice_weight_for_npu(weight_name, weight_matrix, block_size=64, save_dir="./npu_bram_tiles"):
    """
    거대한 LLM 가중치 행렬을 NPU Systolic Array(64x64) 규격에 맞게 쪼개고(Tiling), 
    필요시 0으로 Padding하여 BRAM에 올릴 수 있는 Bin 파일로 저장하는 함수.
    """
    out_features, in_features = weight_matrix.shape
    print(f" Target Layer: [{weight_name}] | Original Shape: {out_features}x{in_features}")

    # 1. 64 단위로 딱 떨어지게 패딩(Memory Alignment) 계산
    # CUDA에서 Pitch 메모리 할당하는 개념과 동일함!
    pad_out = (block_size - (out_features % block_size)) % block_size
    pad_in  = (block_size - (in_features % block_size)) % block_size

    if pad_out > 0 or pad_in > 0:
        print(f"Memory Alignment 적용: Zero-padding (+{pad_out}, +{pad_in}) 추가")
        # numpy.pad를 이용해 오른쪽과 아래쪽에 0을 채움
        aligned_matrix = np.pad(weight_matrix, ((0, pad_out), (0, pad_in)), mode='constant', constant_values=0)
    else:
        aligned_matrix = weight_matrix

    aligned_out, aligned_in = aligned_matrix.shape
    
    # Grid 크기 계산 (몇 개의 64x64 블록이 나오는지)
    grid_y = aligned_out // block_size
    grid_x = aligned_in // block_size
    print(f"Grid 구성: {grid_y} x {grid_x} Blocks (Total: {grid_y * grid_x} 타일)")

    # 2. 64x64 블록으로 슬라이싱 (Tiling)
    # 메모리상에서 64x64 덩어리로 예쁘게 연속되도록 shape을 꼬아줌 (Numpy 마법!)
    tiles = aligned_matrix.reshape(grid_y, block_size, grid_x, block_size)
    tiles = tiles.swapaxes(1, 2)  # shape: (grid_y, grid_x, block_size, block_size)

    # 3. 하드웨어(BRAM)에 올리기 좋게 파일로 저장
    os.makedirs(save_dir, exist_ok=True)
    
    for y in range(grid_y):
        for x in range(grid_x):
            tile_data = tiles[y, x]
            
            # 실제 보드에서는 .bin (Raw Binary)로 넘기는 게 제일 빠르지만,
            # 지금은 Phase 2 파이썬 검증 단계니까 .npy로 저장할게!
            file_name = f"{weight_name}_block_Y{y}_X{x}.npy"
            np.save(os.path.join(save_dir, file_name), tile_data)
            
    print(f"슬라이싱 완료! 저장 위치: {save_dir}/\n")
    return tiles

if __name__ == "__main__":
    # --- [테스트 시나리오] ---
    # Gemma 3N의 Attention Query Projection 가중치라고 가정해 보자!
    # 예시: in_features = 2048, out_features = 2048
    # 근데 엣지 케이스 테스트를 위해 64로 안 떨어지는 이상한 숫자(2000x1500)를 넣어볼게.
    
    dummy_q_proj = np.random.randn(2000, 1500).astype(np.float32)
    
    # NPU 슬라이서 가동!
    slice_weight_for_npu(
        weight_name="layers.0.self_attn.q_proj", 
        weight_matrix=dummy_q_proj, 
        block_size=64
    )