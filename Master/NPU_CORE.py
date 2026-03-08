# Collection of FPGA acceleration functions

# =====================================================================
# 1. 범용 NPU 행렬곱 엔진 (Ping-Pong BRAM Overlapping)
# =====================================================================
import numpy as np
import Master.SYS_CONFIG as SYS_CONFIG

def run_npu_matmul(x_vec, weight_mat, mean_sq_val, use_gelu=False):
    if SYS_CONFIG.SIMULATION_MODE:
        #  [PC 시뮬레이션] NPU 하드웨어 동작을 완벽히 모사 (Mocking)
        # 1. RMSNorm 역제곱근 스케일링
        inv_sqrt = 1.0 / np.sqrt(float(mean_sq_val) + 1e-6)
        
        #  핵심 수정: FP16으로 2048차원 누산하면 오버플로우! 무조건 FP32로 올려서 계산!
        x_f32 = x_vec.astype(np.float32) * inv_sqrt
        w_f32 = weight_mat.astype(np.float32)
        
        # 2. Systolic Array 거대 행렬곱 (FP32로 안전하게)
        out = np.dot(x_f32, w_f32)
        
        # 3. GeLU (이미 FP32이므로 그냥 계산)
        if use_gelu:
            out = 0.5 * out * (1 + np.tanh(np.sqrt(2 / np.pi) * (out + 0.044715 * (out**3))))
            
        return out.astype(np.float16)   
    """
    [FPGA] Q, K, V, O, Gate, Up, Down, LM_Head 모든 거대 행렬곱을 처리하는 코어 엔진.
    x_vec: [2048] 차원 1D 벡터
    weight_mat: [2048, Output_Dim] 차원 2D 행렬
    """
    input_dim = 2048
    output_dim = weight_mat.shape[1]
    final_out = np.zeros(output_dim, dtype=np.int16)
    
    num_ic_tiles = input_dim // 32
    num_oc_tiles = output_dim // 32
    total_tiles = num_ic_tiles * num_oc_tiles
    
    # ---------------------------------------------------------
    # [A] 커널 파라미터 셋업 (MMIO 레지스터 제어)
    # ---------------------------------------------------------
    # 0x08: RMSNorm용 분모 스칼라 꽂기
    SYS_CONFIG.npu_control.write(0x08, int(mean_sq_val))
    
    # 0x10: 1-Cycle GeLU 하드웨어 발동 여부 세팅 (Bit 0)
    mode_flag = 0x01 if use_gelu else 0x00
    SYS_CONFIG.npu_control.write(0x10, mode_flag)
    
    # ---------------------------------------------------------
    # [B] 프롤로그 (0번 타일 Ping 버퍼에 선탑재)
    # ---------------------------------------------------------
    np.copyto(SYS_CONFIG.ping_token, x_vec[0:32])
    np.copyto(SYS_CONFIG.ping_weight, weight_mat[0:32, 0:32])
    
    SYS_CONFIG.npu_control.write(0x0C, 0) # DMA 스위치 -> Ping(0)
    SYS_CONFIG.dma.sendchannel.transfer(SYS_CONFIG.ping_token)
    SYS_CONFIG.dma.sendchannel.transfer(SYS_CONFIG.ping_weight)
    SYS_CONFIG.dma.sendchannel.wait()
    
    # ---------------------------------------------------------
    # [C] 메인 핑퐁 파이프라인 루프
    # ---------------------------------------------------------
    for tile_idx in range(total_tiles):
        oc = tile_idx // num_ic_tiles
        ic = tile_idx % num_ic_tiles
        
        # 새로운 Output Channel 계산 시작 전, 누산기(ACC) 0으로 초기화!
        if ic == 0:
            SYS_CONFIG.npu_control.write(0x00, 0x02) # ACC_CLEAR 비트(Bit 1) ON!
            
        next_idx = tile_idx + 1
        next_oc = next_idx // num_ic_tiles
        next_ic = next_idx % num_ic_tiles
        is_ping_turn = (tile_idx % 2 == 0)
        
        # --- 1. DMA 백그라운드 전송 (Prefetch) ---
        if next_idx < total_tiles:
            SYS_CONFIG.ping_token = x_vec[next_ic*32 : (next_ic+1)*32]
            SYS_CONFIG.ping_weight = weight_mat[next_ic*32 : (next_ic+1)*32, next_oc*32 : (next_oc+1)*32]
            
            if is_ping_turn:
                # oken 전송 시작 전: 0x14 번지에 0 (Token) 기록
                SYS_CONFIG.npu_control.write(0x14, 0)
                SYS_CONFIG.dma.sendchannel.transfer(SYS_CONFIG.pong_token)
                SYS_CONFIG.dma.sendchannel.wait() # 스트림 섞임 방지용 대기

                # eight 전송 시작 전: 0x14 번지에 1 (Weight) 기록
                SYS_CONFIG.npu_control.write(0x14, 1)
                SYS_CONFIG.dma.sendchannel.transfer(SYS_CONFIG.pong_weight)
            else:
                SYS_CONFIG.npu_control.write(0x14, 0)
                SYS_CONFIG.dma.sendchannel.transfer(SYS_CONFIG.ping_token)
                SYS_CONFIG.dma.sendchannel.wait()

                SYS_CONFIG.npu_control.write(0x14, 1)
                SYS_CONFIG.dma.sendchannel.transfer(SYS_CONFIG.ping_weight)

        # --- 2. NPU 연산 킥! ---
        # 펄스 로직 넣었으니까 이제 1 한 번 쓰면 알아서 꺼짐!
        SYS_CONFIG.npu_control.write(0x00, 0x01) 
        
        # --- 3. 연산 완료 대기 (Polling 버그 수정: 0x04 -> 0x10) ---
        while (SYS_CONFIG.npu_control.read(0x10) & 0x010000) == 0: 
            # 0x10의 16번 비트가 w_npu_done 이니까 0x010000과 AND 연산!
            pass     
               
        # --- 4. 결과 수신 (64번 누산이 끝난 마지막 타일에서만!) ---
        if ic == num_ic_tiles - 1:
            SYS_CONFIG.dma.recvchannel.transfer(SYS_CONFIG.result_buf)
            SYS_CONFIG.dma.recvchannel.wait()
            final_out[oc*32 : (oc+1)*32] = np.array(SYS_CONFIG.result_buf)
            
        # --- 5. 다음 루프 넘어가기 전 DMA 전송 완료 대기 ---
        if next_idx < total_tiles:
            SYS_CONFIG.dma.sendchannel.wait()

    return final_out

# =====================================================================
# 2. 껍데기 함수 랩핑 (Big Picture의 함수명과 매핑)
# =====================================================================
def npu_matmul(x, weight, mean_sq):
    """ 일반적인 거대 행렬곱 """
    return run_npu_matmul(x, weight, mean_sq, use_gelu=False)

def npu_matmul_gelu(x, W_gate, mean_sq):
    """ FFN 블록 전용: 행렬곱 후 1-Cycle GeLU 하드웨어 발동! """
    return run_npu_matmul(x, W_gate, mean_sq, use_gelu=True)

# =====================================================================
# 3. Softmax 가속 함수
# =====================================================================
def npu_softmax(logits):
    if SYS_CONFIG.SIMULATION_MODE:
        #  [PC 시뮬레이션] NPU Softmax IP 모사
        logits_safe = logits - np.max(logits)
        probs = np.exp(logits_safe) / np.sum(np.exp(logits_safe))
        return probs.astype(np.float16)
    
    """
    [FPGA] LM Head에서 나온 256,000개의 점수를 확률값으로 변환
    """
    probs = np.zeros_like(logits, dtype=np.float16)
    
    # 0x10 레지스터에 Softmax_EN 비트(Bit 1) 켜기!
    SYS_CONFIG.npu_control.write(0x10, 0x02) 
    
    # Softmax는 행렬곱이 아니므로 32개 단위로 쪼개서 NPU Softmax IP만 통과시킴
    for i in range(0, len(logits), 32):
        np.copyto(SYS_CONFIG.ping_token, logits[i:i+32])
        
        # 데이터 전송 및 NPU 킥 (단순 벡터 연산)
        SYS_CONFIG.npu_control.write(0x0C, 0)
        SYS_CONFIG.dma.sendchannel.transfer(SYS_CONFIG.ping_token)
        SYS_CONFIG.dma.sendchannel.wait()
        
        SYS_CONFIG.npu_control.write(0x00, 0x01)
        while (SYS_CONFIG.npu_control.read(0x04) & 0x01) == 0:
            pass
            
        SYS_CONFIG.dma.recvchannel.transfer(SYS_CONFIG.result_buf)
        SYS_CONFIG.dma.recvchannel.wait()
        
        probs[i:i+32] = np.array(SYS_CONFIG.result_buf)
        
    return probs