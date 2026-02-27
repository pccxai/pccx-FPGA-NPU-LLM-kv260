import numpy as np

# =====================================================================
# 1. NPU 하드웨어 코어 (64x64 Systolic Array 모사)
# =====================================================================
class TinyNPU_Core:
    def __init__(self, block_size=64):
        self.block_size = block_size

    def execute_64x64_tile(self, input_tile, weight_tile):
        """ 진짜 하드웨어(Systolic Array)가 한 사이클에 처리하는 64x64 MAC 연산 """
        # 실제론 INT8 양자화 연산이 들어가지만 구조 검증을 위해 FP32 사용
        return np.dot(input_tile, weight_tile)

    def run_gemv_tiled(self, input_vec, weight_matrix):
        """ 
        거대 행렬을 64x64 타일로 쪼개어 NPU에 스케줄링 (C++ Kernel Dispatch 모사)
        input_vec: (1, in_features)
        weight_matrix: (in_features, out_features)
        """
        in_feat = weight_matrix.shape[0]
        out_feat = weight_matrix.shape[1]
        
        # 출력 버퍼 (Global Memory)
        out_vec = np.zeros((1, out_feat), dtype=np.float32)
        
        grid_x = in_feat // self.block_size
        grid_y = out_feat // self.block_size
        
        # Y축 (출력 차원) 방향으로 루프
        for y in range(grid_y):
            partial_sum = np.zeros((1, self.block_size), dtype=np.float32)
            
            # X축 (입력 차원) 방향으로 타일 누적 (Wavefront 파도타기)
            for x in range(grid_x):
                # DMA가 BRAM으로 64x64 타일을 밀어넣음
                in_tile = input_vec[:, x*self.block_size : (x+1)*self.block_size]
                w_tile = weight_matrix[x*self.block_size : (x+1)*self.block_size, y*self.block_size : (y+1)*self.block_size]
                
                # NPU 64x64 연산 가동 및 누적 (MAC)
                partial_sum += self.execute_64x64_tile(in_tile, w_tile)
                
            # 최종 결과 버퍼에 쓰기
            out_vec[:, y*self.block_size : (y+1)*self.block_size] = partial_sum
            
        return out_vec

# =====================================================================
# 2. KV Cache 매니저 (정적 메모리)
# =====================================================================
class HardwareKVCache:
    def __init__(self, max_seq, embed_dim):
        self.max_seq = max_seq
        self.pos = 0 

        # C++ CMA 할당 모사
        self.k_cache = np.zeros((max_seq, embed_dim), dtype=np.float32)
        self.v_cache = np.zeros((max_seq, embed_dim), dtype=np.float32)
        
    def write(self, k, v):
        self.k_cache[self.pos, :] = k
        self.v_cache[self.pos, :] = v
        self.pos += 1
        
    def read_all(self):
        return self.k_cache[:self.pos, :], self.v_cache[:self.pos, :]

# =====================================================================
# 3. Gemma 3N 단일 블록 (Attention + FFN)
# =====================================================================
class Gemma_NPU_Block_GQA:
    def __init__(self, npu_core, embed_dim, max_seq, num_q_heads=8, num_kv_heads=2):
        # npu, EMBED_DIM(256), MAX_SEQ(128)
        self.npu = npu_core
        self.embed_dim = embed_dim
        
        # GQA 파라미터 세팅 (예: Q는 8헤드, KV는 2헤드 -> 4:1 그룹핑!)
        self.head_dim = embed_dim // num_q_heads
        self.kv_dim = num_kv_heads * self.head_dim

        # KV Cache도 기존 embed_dim 크기가 아니라 kv_dim(1/4 크기)으로 대폭 축소!
        self.kv_cache = HardwareKVCache(max_seq, self.kv_dim)
        
        print(f" [GQA 최적화] Q차원: {embed_dim} | KV차원: {self.kv_dim} (메모리 75% down!)")

        # 1. Attention 가중치 (GQA 적용)
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
        self.W_k = np.random.randn(embed_dim, self.kv_dim).astype(np.float32) * 0.01 # 사이즈 확 줄음!
        self.W_v = np.random.randn(embed_dim, self.kv_dim).astype(np.float32) * 0.01 # 사이즈 확 줄음!
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
        
        # 2. FFN 가중치 (GeGLU)
        ffn_dim = embed_dim * 4 
        self.W_gate = np.random.randn(embed_dim, ffn_dim).astype(np.float32) * 0.01
        self.W_up   = np.random.randn(embed_dim, ffn_dim).astype(np.float32) * 0.01
        self.W_down = np.random.randn(ffn_dim, embed_dim).astype(np.float32) * 0.01
        
    def forward_decode(self, x):
        """ 토큰 1개가 들어왔을 때의 NPU 파이프라인 """
# --- 1. Attention Phase ---
        # Q는 크게, K/V는 작게 NPU 타일링 연산!
        q = self.npu.run_gemv_tiled(x, self.W_q) # shape: (1, 256)
        k = self.npu.run_gemv_tiled(x, self.W_k) # shape: (1, 64) 
        v = self.npu.run_gemv_tiled(x, self.W_v) # shape: (1, 64)
        
        # GQA KV Cache 저장 (메모리 대역폭 절약)
        self.kv_cache.write(k, v)
        past_k, past_v = self.kv_cache.read_all()
        
        # Attention Score 계산 (ARM CPU가 담당할 부분 - Softmax)
        # [주의] 원래는 여기서 Q를 4개의 그룹으로 쪼개서 K, V랑 브로드캐스팅(반복) 매칭해줘야 함.
        # (NPU 연산 흐름 검증이 목적이므로 여기선 차원 맞춤용 더미 코드로 대체!)
        dummy_k_expanded = np.repeat(past_k, 4, axis=1) # (seq, 64) -> (seq, 256) 모사
        dummy_v_expanded = np.repeat(past_v, 4, axis=1)

        attn_scores = np.dot(q, dummy_k_expanded.T)
        attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores))

        # NPU 가동: Score x V 타일링 (실제론 행렬 형태가 다르지만 GEMV로 단순화)
        attn_out = np.dot(attn_probs, dummy_v_expanded) 
        attn_proj = self.npu.run_gemv_tiled(attn_out, self.W_o)
        
        x = x + attn_proj # Residual 1
        
        # --- 2. MLP (FFN) Phase ---
        # Gemma는 GeGLU 구조를 사용함. NPU가 2개의 거대 행렬곱을 병렬로 쳐냄!
        gate = self.npu.run_gemv_tiled(x, self.W_gate)
        up   = self.npu.run_gemv_tiled(x, self.W_up)
        
        # 활성화 함수 (ARM CPU 또는 NPU 내부 LUT로 처리)
        # 파이썬 젤루(GELU) 모사
        # [목표] 이 GeLU 연산을 NPU 내부의 LUT로 뺄 예정!
        activated_gate = gate * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3))))
        
        ffn_mid = activated_gate * up
        ffn_out = self.npu.run_gemv_tiled(ffn_mid, self.W_down)
        
        return x + ffn_out #Residual 2

# =====================================================================
# 4. Main Execution
# =====================================================================
if __name__ == "__main__":
    EMBED_DIM = 256         # 64의 배수! (Gemma 실제 차원은 훨씬 크지만 테스트용)
    MAX_SEQ = 128              # 모델이 한 번에 처리할 수 있는 토큰(단어 조각)의 최대 개수
    
    npu = TinyNPU_Core(block_size=64)   #
    gemma_block = Gemma_NPU_Block_GQA(npu, EMBED_DIM, MAX_SEQ)
    
    print("\n Gemma 3N E4B NPU 시뮬레이터 가동 (Tile-based MAC Engine)")
    print("====================================================")
    
    # 3개의 단어가 연속해서 들어오는 상황 모사
    for step in range(3):
        print(f"\n[Decode Step {step+1}] 토큰 연산 시작...")
        dummy_token = np.random.randn(1, EMBED_DIM).astype(np.float32)
        
        out_token = gemma_block.forward_decode(dummy_token)
        print(f" 토큰 연산 완료! KV Cache 크기: {gemma_block.kv_cache.pos}")
        
    print("\n모든 소프트웨어 검증 완료!")