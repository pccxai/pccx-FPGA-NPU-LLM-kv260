import numpy as np

# =====================================================================
# [Hardware 1] NPU hardware core (npu_core_top_NxN.sv copy)
# =====================================================================
class TinyNPU_Core:
    def __init__(self, block_size=32): # Customized to the Verilog (ARRAY_SIZE=32).
        self.block_size = block_size

    def execute_32x32_tile(self, input_tile, weight_tile):
        """
        [Verilog mapping]: systolic_NxN.sv (surfing 1,024 PEs)
        - Simulation of the physical process of eating and pouring out 512 bits of data over 32 clocks
        """
        return np.dot(input_tile, weight_tile)

    def run_gemv_tiled(self, input_vec, weight_matrix):
        """
        [Verilog mapping]: FSM of npu_core_top_NxN.sv + ping_pong_bram.sv
        - Cut the huge LLM weights into 32x32 sizes and write them to BRAM using DMA.
        - Simulate the scheduling process of sending “start_mac = 1” to the NPU.
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

                # NPU 32x32 compute operation and accumulation (MAC)
                partial_sum += self.execute_32x32_tile(in_tile, w_tile)

            out_vec[:, y*self.block_size : (y+1)*self.block_size] = partial_sum

        return out_vec

# =====================================================================
# [Hardware 2] KV Cache Manager (replica of real DDR4 RAM off-board)
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
# 3. Gemma 3N single block (Attention + FFN) - RMSNorm 100% equipped version.
# =====================================================================
class Gemma_NPU_Block_GQA:
    def __init__(self, npu_core, embed_dim, max_seq, num_q_heads=8, num_kv_heads=2):
        self.npu = npu_core
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_q_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.kv_cache = HardwareKVCache(max_seq, self.kv_dim)

        # [Weight Setting] Attention & FFN
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
        self.W_k = np.random.randn(embed_dim, self.kv_dim).astype(np.float32) * 0.01
        self.W_v = np.random.randn(embed_dim, self.kv_dim).astype(np.float32) * 0.01
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01

        ffn_dim = embed_dim * 4
        self.W_gate = np.random.randn(embed_dim, ffn_dim).astype(np.float32) * 0.01
        self.W_up   = np.random.randn(embed_dim, ffn_dim).astype(np.float32) * 0.01
        self.W_down = np.random.randn(ffn_dim, embed_dim).astype(np.float32) * 0.01

        # [New addition] RMSNorm learning weight (initial value set to 1)
        self.rms_w1 = np.ones(embed_dim, dtype=np.float32) # For Pre-Norm
        self.rms_w2 = np.ones(embed_dim, dtype=np.float32) # For Post-Norm

    def rmsnorm_hardware_sim(self, x, weight, eps=1e-6):
        """
        [HW Mapping Target]: RMSNorm accelerator model to pour in 80,000 remaining LUTs
        """
        # 1. Mean of squares -> Can be processed with HW Accumulator
        mean_sq = np.mean(x**2, axis=-1, keepdims=True)

        # 2. The long-awaited Inverse Square Root
        # -> The software is completed in one go with np.sqrt(),
        # In terms of hardware, it must be reduced with the 'Quake 3 Fast InvSqrt algorithm' or 'Taylor series'.
        inv_sqrt = 1.0 / np.sqrt(mean_sq + eps)

        # 3. Final normalization and weight multiplication -> Processed with HW Vector Multiplier
        return (x * inv_sqrt) * weight

    def forward_decode(self, x):
        # -------------------------------------------------------------
        # Phase 1. Attention (Attention Block)
        # -------------------------------------------------------------
        # [Perfect Implementation] Pre-Norm (Press firmly before entering Attention)

        # [Future HW Goal 1] RMSNorm (Pre-Norm)
        # Omitted in the current code! Here we need to calculate the inverse square root (1/sqrt(x)).
        # x_norm = self.rmsnorm_hardware(x)
        x_norm = self.rmsnorm_hardware_sim(x, self.rms_w1)

        # NPU operation: Q, K, V matrix multiplication (call npu_core_top_NxN.sv)
        q = self.npu.run_gemv_tiled(x_norm, self.W_q)
        k = self.npu.run_gemv_tiled(x_norm, self.W_k)
        v = self.npu.run_gemv_tiled(x_norm, self.W_v)

        # DDR4 access: KV Cache storage
        self.kv_cache.write(k, v)
        past_k, past_v = self.kv_cache.read_all()

        # CPU calculation: Q x K^T (Attention Score calculation)
        dummy_k_expanded = np.repeat(past_k, 4, axis=1)
        dummy_v_expanded = np.repeat(past_v, 4, axis=1)
        attn_scores = np.dot(q, dummy_k_expanded.T)

        # [Future HW Goal 2] Softmax accelerator
        # The section where the CPU explodes from calculating the exponential function (exp)! Need to remove it with hardware.
        attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores))

        # CPU calculation: Score x V
        attn_out = np.dot(attn_probs, dummy_v_expanded)

        # NPU operation: Matrix multiplication for output of final attention result
        attn_proj = self.npu.run_gemv_tiled(attn_out, self.W_o)

        x = x + attn_proj # Residual 1

        # -------------------------------------------------------------
        # Phase 2. MLP (FFN block)
        # -------------------------------------------------------------
        # [Perfect Implementation] Post-Norm (Press once more before entering MLP)

        # [Future HW Goals 1-2] RMSNorm (Post-Norm)
        # After Attention, the user have to calculate the route again.
        # x_norm2 = self.rmsnorm_hardware(x)
        x_norm2 = self.rmsnorm_hardware_sim(x, self.rms_w2)

        # NPU operation: Gate & Up matrix multiplication parallel processing
        gate = self.npu.run_gemv_tiled(x_norm2, self.W_gate)
        up   = self.npu.run_gemv_tiled(x_norm2, self.W_up)

        # [Hardware we made!] GeLU LUT (gelu_lut.sv)
        # The magic section that reads the correct answer from 64KB ROM in just 1 clock.
        # Formula: gate * (0.5 * (1 + np.tanh...))
        activated_gate = gate * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3))))

        # CPU operation: Simple element-by-element multiplication (this can be cut by 1 clock with HW)
        ffn_mid = activated_gate * up

        # NPU Operation: Last Down Matrix Multiplication
        ffn_out = self.npu.run_gemv_tiled(ffn_mid, self.W_down)

        return x + ffn_out # Residual 2

    def hardware_gelu_lut(self, x):
        """ [Verilog mapping]: Explicitly express the behavior of gelu_lut.sv """
        # In reality, the np.tanh() formula is used, but from a hardware perspective,
        # It means “fetch the 8-bit answer from ROM with a 16-bit address”.
        return x * (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))