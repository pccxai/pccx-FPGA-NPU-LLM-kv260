# Collection of FPGA acceleration functions

# =====================================================================
# 1. General-purpose NPU matrix multiplication engine (Ping-Pong BRAM Overlapping)
# =====================================================================
import numpy as np
import MMIO

def run_npu_matmul(x_vec, weight_mat, mean_sq_val, use_gelu=False):
    if MMIO.SIMULATION_MODE:
        # [PC Simulation] Perfectly simulates NPU hardware operation (Mocking)
        # 1. RMSNorm inverse square root scaling
        inv_sqrt = 1.0 / np.sqrt(float(mean_sq_val) + 1e-6)
        
        # Key fix: Overflow when accumulating 2048 dimensions with FP16! Be sure to upgrade to FP32 for calculations.
        x_f32 = x_vec.astype(np.float32) * inv_sqrt
        w_f32 = weight_mat.astype(np.float32)
        
        # 2. Systolic Array huge matrix multiplication (safely with FP32)
        out = np.dot(x_f32, w_f32)
        
        # 3. GeLU (it's already FP32 so just count it)
        if use_gelu:
            out = 0.5 * out * (1 + np.tanh(np.sqrt(2 / np.pi) * (out + 0.044715 * (out**3))))
            
        return out.astype(np.float16)   
    """
    [FPGA] Q, K, V, O, Gate, Up, Down, LM_Head Core engine that processes all large matrix multiplications.
    x_vec: [2048] dimension 1D vector
    weight_mat: [2048, Output_Dim] dimension 2D matrix
    """
    input_dim = 2048
    output_dim = weight_mat.shape[1]
    final_out = np.zeros(output_dim, dtype=np.int16)
    
    num_ic_tiles = input_dim // 32
    num_oc_tiles = output_dim // 32
    total_tiles = num_ic_tiles * num_oc_tiles
    
    # ---------------------------------------------------------
    # [A] Kernel parameter setup (MMIO register control)
    # ---------------------------------------------------------
    # 0x08: Plug in denominator scalar for RMSNorm
    MMIO.npu_control.write(0x08, int(mean_sq_val))
    
    # 0x10: Setting whether to activate 1-Cycle GeLU hardware (Bit 0)
    mode_flag = 0x01 if use_gelu else 0x00
    MMIO.npu_control.write(0x10, mode_flag)
    
    # ---------------------------------------------------------
    # [B] Prologue (preloaded in tile 0 ping buffer)
    # ---------------------------------------------------------
    np.copyto(MMIO.ping_token, x_vec[0:32])
    np.copyto(MMIO.ping_weight, weight_mat[0:32, 0:32])
    
    MMIO.npu_control.write(0x0C, 0) # DMA switch -> Ping(0)
    MMIO.dma.sendchannel.transfer(MMIO.ping_token)
    MMIO.dma.sendchannel.transfer(MMIO.ping_weight)
    MMIO.dma.sendchannel.wait()
    
    # ---------------------------------------------------------
    # [C] Main Ping Pong Pipeline Loop
    # ---------------------------------------------------------
    for tile_idx in range(total_tiles):
        oc = tile_idx // num_ic_tiles
        ic = tile_idx % num_ic_tiles
        
        # Before starting a new Output Channel calculation, initialize the accumulator (ACC) to 0.
        if ic == 0:
            MMIO.npu_control.write(0x00, 0x02) # ACC_CLEAR bit (Bit 1) ON.
            
        next_idx = tile_idx + 1
        next_oc = next_idx // num_ic_tiles
        next_ic = next_idx % num_ic_tiles
        is_ping_turn = (tile_idx % 2 == 0)
        
        # --- 1. DMA background transfer (Prefetch) ---
        if next_idx < total_tiles:
            MMIO.ping_token = x_vec[next_ic*32 : (next_ic+1)*32]
            MMIO.ping_weight = weight_mat[next_ic*32 : (next_ic+1)*32, next_oc*32 : (next_oc+1)*32]
            
            if is_ping_turn:
                # Before oken transmission begins: 0 (Token) is recorded at address 0x14
                MMIO.npu_control.write(0x14, 0)
                MMIO.dma.sendchannel.transfer(MMIO.pong_token)
                MMIO.dma.sendchannel.wait() # Standby to prevent stream mixing

                # eight Before starting transmission: 1 (Weight) recorded at address 0x14
                MMIO.npu_control.write(0x14, 1)
                MMIO.dma.sendchannel.transfer(MMIO.pong_weight)
            else:
                MMIO.npu_control.write(0x14, 0)
                MMIO.dma.sendchannel.transfer(MMIO.ping_token)
                MMIO.dma.sendchannel.wait()

                MMIO.npu_control.write(0x14, 1)
                MMIO.dma.sendchannel.transfer(MMIO.ping_weight)

        # --- 2. NPU calculation kick! ---
        # Since I added pulse logic, once I use 1, it turns off automatically.
        MMIO.npu_control.write(0x00, 0x01) 
        
        # --- 3. Wait for operation completion (Polling bug fixed: 0x04 -> 0x10) ---
        while (MMIO.npu_control.read(0x10) & 0x010000) == 0: 
            # Since bit 16 of 0x10 is w_npu_done, AND operation with 0x010000.
            pass     
               
        # --- 4. Receive results (only from the last tile after 64 accumulations!) ---
        if ic == num_ic_tiles - 1:
            MMIO.dma.recvchannel.transfer(MMIO.result_buf)
            MMIO.dma.recvchannel.wait()
            final_out[oc*32 : (oc+1)*32] = np.array(MMIO.result_buf)
            
        # --- 5. Wait for DMA transfer to complete before proceeding to next loop ---
        if next_idx < total_tiles:
            MMIO.dma.sendchannel.wait()

    return final_out

# =====================================================================
# 2. Shell function wrapping (Big Picture function name and mapping)
# =====================================================================
def npu_matmul(x, weight, mean_sq):
    """ General huge matrix product """
    return run_npu_matmul(x, weight, mean_sq, use_gelu=False)

def npu_matmul_gelu(x, W_gate, mean_sq):
    """ FFN block only: 1-Cycle GeLU hardware activated after matrix multiplication! """
    return run_npu_matmul(x, W_gate, mean_sq, use_gelu=True)

# =====================================================================
# 3. Softmax acceleration function
# =====================================================================
def npu_softmax(logits):
    if MMIO.SIMULATION_MODE:
        # [PC Simulation] NPU Softmax IP simulation
        logits_safe = logits - np.max(logits)
        probs = np.exp(logits_safe) / np.sum(np.exp(logits_safe))
        return probs.astype(np.float16)
    """
    [FPGA] Converting 256,000 scores from LM Head into probability values
    """
    probs = np.zeros_like(logits, dtype=np.float16)
    
    # Turn on the Softmax_EN bit (Bit 1) in register 0x10.
    MMIO.npu_control.write(0x10, 0x02) 
    
    # Since Softmax is not a matrix multiplication, it is divided into 32 units and only NPU Softmax IP is passed.
    for i in range(0, len(logits), 32):
        np.copyto(MMIO.ping_token, logits[i:i+32])
        
        # Data transfer and NPU kick (simple vector operations)
        MMIO.npu_control.write(0x0C, 0)
        MMIO.dma.sendchannel.transfer(MMIO.ping_token)
        MMIO.dma.sendchannel.wait()
        
        MMIO.npu_control.write(0x00, 0x01)
        while (MMIO.npu_control.read(0x04) & 0x01) == 0:
            pass
            
        MMIO.dma.recvchannel.transfer(MMIO.result_buf)
        MMIO.dma.recvchannel.wait()
        
        probs[i:i+32] = np.array(MMIO.result_buf)
        
    return probs