import numpy as np

# =====================================================================
# [Hardware 1] NPU 하드웨어 코어 (npu_core_top_NxN.sv 모사)
# =====================================================================
class TinyNPU_Core:
    def __init__(self, block_size=32): # 우리 Verilog (ARRAY_SIZE=32)에 맞춤!
        self.block_size = block_size

    def execute_32x32_tile(self, input_tile, weight_tile):
        """ 
        [Verilog 매핑]: systolic_NxN.sv (1,024개 PE 파도타기) 
        - 32클럭 동안 512비트 데이터를 먹고 쏟아내는 물리적 과정 모사
        """
        return np.dot(input_tile, weight_tile)

    def run_gemv_tiled(self, input_vec, weight_matrix):
        """ 
        [Verilog 매핑]: npu_core_top_NxN.sv 의 FSM + ping_pong_bram.sv
        - 거대한 LLM 가중치를 32x32 사이즈로 잘라서 BRAM에 DMA로 쏘고(Write),
        - NPU에게 "start_mac = 1"을 날리는 스케줄링 과정 모사!
        """
        in_feat = weight_matrix.shape[0]
        out_feat = weight_matrix.shape[1]
        
        out_vec = np.zeros((1, out_feat), dtype=np.float32)
        grid_x = in_feat // self.block_size
        grid_y = out_feat // self.block_size
        
        for y in range(grid_y):
            partial_sum = np.zeros((1, self.block_size), dtype=np.float32)
            for x in range(grid_x):
                in_tile = input_vec[:, x*self.block_size : (x+1)*self.block_size]
                w_tile = weight_matrix[x*self.block_size : (x+1)*self.block_size, y*self.block_size : (y+1)*self.block_size]
                
                # NPU 32x32 연산 가동 및 누적 (MAC)
                partial_sum += self.execute_32x32_tile(in_tile, w_tile)
                
            out_vec[:, y*self.block_size : (y+1)*self.block_size] = partial_sum
            
        return out_vec

# =====================================================================
# [Hardware 2] KV Cache 매니저 (보드 외부의 진짜 DDR4 RAM 모사)
# =====================================================================
class HardwareKVCache:
    def __init__(self, max_seq, embed_dim):
        self.max_seq = max_seq
        self.pos = 0 
        self.k_cache = np.zeros((max_seq, embed_dim), dtype=np.float32)
        self.v_cache = np.zeros((max_seq, embed_dim), dtype=np.float32)
        
    def write(self, k, v):
        self.k_cache[self.pos, :] = k
        self.v_cache[self.pos, :] = v
        self.pos += 1
        
    def read_all(self):
        return self.k_cache[:self.pos, :], self.v_cache[:self.pos, :]

# =====================================================================
# 3. Gemma 3N 단일 블록 (Attention + FFN) - RMSNorm 100% 탑재 버젼!
# =====================================================================
class Gemma_NPU_Block_GQA:
    def __init__(self, npu_core, embed_dim, max_seq, num_q_heads=8, num_kv_heads=2):
        self.npu = npu_core
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_q_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.kv_cache = HardwareKVCache(max_seq, self.kv_dim)
        
        # [가중치 세팅] Attention & FFN
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
        self.W_k = np.random.randn(embed_dim, self.kv_dim).astype(np.float32) * 0.01 
        self.W_v = np.random.randn(embed_dim, self.kv_dim).astype(np.float32) * 0.01 
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
        
        ffn_dim = embed_dim * 4 
        self.W_gate = np.random.randn(embed_dim, ffn_dim).astype(np.float32) * 0.01
        self.W_up   = np.random.randn(embed_dim, ffn_dim).astype(np.float32) * 0.01
        self.W_down = np.random.randn(ffn_dim, embed_dim).astype(np.float32) * 0.01

        #  [신규 추가] RMSNorm 학습 가중치 (초기값은 1로 세팅)
        self.rms_w1 = np.ones(embed_dim, dtype=np.float32) # Pre-Norm용
        self.rms_w2 = np.ones(embed_dim, dtype=np.float32) # Post-Norm용

    def rmsnorm_hardware_sim(self, x, weight, eps=1e-6):
        """ 
        [HW 매핑 타겟]: 8만 개의 남는 LUT를 쏟아부을 RMSNorm 가속기 모델 
        """
        # 1. 제곱의 평균 (Mean of squares) -> HW 누산기(Accumulator)로 처리 가능
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)
        
        # 2.  대망의 역제곱근 (Inverse Square Root) 
        # -> 소프트웨어는 np.sqrt() 한방에 끝나지만, 
        # 하드웨어로는 'Quake 3 Fast InvSqrt 알고리즘'이나 '테일러 급수'로 깎아야 함!
        inv_sqrt = 1.0 / np.sqrt(mean_sq + eps)
        
        # 3. 최종 정규화 및 가중치 곱셈 -> HW 벡터 곱셈기(Vector Multiplier)로 처리
        return (x * inv_sqrt) * weight

    def forward_decode(self, x):
        # -------------------------------------------------------------
        # Phase 1. Attention (어텐션 블록)
        # -------------------------------------------------------------
        # [완벽 구현] Pre-Norm (Attention 들어가기 전에 꾹꾹 눌러주기)
        
        # [미래의 HW 목표 1] RMSNorm (Pre-Norm)
        # 현재 코드엔 생략됨! 여기서 역제곱근(1/sqrt(x))을 계산해야 함!
        # x_norm = self.rmsnorm_hardware(x)
        x_norm = self.rmsnorm_hardware_sim(x, self.rms_w1)

        # NPU 가동: Q, K, V 행렬곱 (npu_core_top_NxN.sv 호출)
        q = self.npu.run_gemv_tiled(x_norm, self.W_q) 
        k = self.npu.run_gemv_tiled(x_norm, self.W_k)  
        v = self.npu.run_gemv_tiled(x_norm, self.W_v) 
        
        # DDR4 접근: KV Cache 저장
        self.kv_cache.write(k, v)
        past_k, past_v = self.kv_cache.read_all()
        
        # CPU 연산: Q x K^T (Attention Score 계산)
        dummy_k_expanded = np.repeat(past_k, 4, axis=1) 
        dummy_v_expanded = np.repeat(past_v, 4, axis=1)
        attn_scores = np.dot(q, dummy_k_expanded.T)

        # [미래의 HW 목표 2] Softmax 가속기
        # CPU가 지수함수(exp) 계산하느라 터져나가는 구간! 하드웨어로 빼야 함!
        attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores))

        # CPU 연산: Score x V
        attn_out = np.dot(attn_probs, dummy_v_expanded) 

        # NPU 가동: 최종 Attention 결과 출력용 행렬곱
        attn_proj = self.npu.run_gemv_tiled(attn_out, self.W_o)
        
        x = x + attn_proj # Residual 1
        
        # -------------------------------------------------------------
        # Phase 2. MLP (FFN 블록)
        # -------------------------------------------------------------
        # [완벽 구현] Post-Norm (MLP 들어가기 전에 한 번 더 꾹꾹 눌러주기)

        # [미래의 HW 목표 1-2] RMSNorm (Post-Norm)
        # Attention 끝나고 또 루트 계산해야 함!
        # x_norm2 = self.rmsnorm_hardware(x)
        x_norm2 = self.rmsnorm_hardware_sim(x, self.rms_w2)

        # NPU 가동: Gate & Up 행렬곱 병렬 처리
        gate = self.npu.run_gemv_tiled(x_norm2, self.W_gate)
        up   = self.npu.run_gemv_tiled(x_norm2, self.W_up)
        
        #  [우리가 만든 하드웨어!] GeLU LUT (gelu_lut.sv) 
        # 1클럭 만에 64KB ROM에서 정답을 읽어오는 마법의 구간!
        # 수식: gate * (0.5 * (1 + np.tanh...))
        activated_gate = gate * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3))))
        
        # CPU 연산: 단순 요소별 곱셈 (이건 HW로 1클럭 컷 가능)
        ffn_mid = activated_gate * up

        # NPU 가동: 마지막 Down 행렬곱
        ffn_out = self.npu.run_gemv_tiled(ffn_mid, self.W_down)
        
        return x + ffn_out # Residual 2
    
    def hardware_gelu_lut(self, x):
        """ [Verilog 매핑]: gelu_lut.sv 의 동작을 명시적으로 표현 """
        # 실제론 np.tanh() 수식을 쓰지만, 하드웨어 관점에선 
        # "16비트 주소로 ROM에서 8비트 정답을 퍼온다"는 의미!
        return x * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))