import numpy as np
import sys
import os

# =====================================================================
# 1. 시스템 전역 설정 (System Configuration)
# =====================================================================
# 현재 스크립트(Master 폴더)의 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 가중치(safetensors)와 토크나이저(tokenizer.json 등)가 들어있는 타겟 폴더
MODEL_DIR = os.path.join(BASE_DIR, "gemma3NE2B_INT4_Q")

# 가속기 모드 선택 ("IGPU", "CPU", "FPGA")
ACCEL_MODE = "IGPU"

# PC 시뮬레이션 모드 토글 (보드에 올릴 땐 False로 변경!)
SIMULATION_MODE = True

# =====================================================================
# 2. 하드웨어 리소스 할당 (MMIO & DMA)
# =====================================================================
if not SIMULATION_MODE:
    from pynq import Overlay, allocate
    print("FPGA Bitstream Loading...")
    overlay = Overlay("gemma_npu.bit")

    npu_control = overlay.gemma_npu_axi_slave_0
    dma = overlay.axi_dma_0

    ping_token  = allocate(shape=(32,), dtype=np.int16)
    ping_weight = allocate(shape=(32, 32), dtype=np.int16)
    pong_token  = allocate(shape=(32,), dtype=np.int16)
    pong_weight = allocate(shape=(32, 32), dtype=np.int16)
    result_buf  = allocate(shape=(32,), dtype=np.int16)
    print("Hardware Init Complete!")
else:
    print("[PC Simulation Mode] Bypassing FPGA Hardware...")
    npu_control = None
    dma = None
    
    # pynq allocate 대신 일반 Numpy 메모리 할당
    ping_token  = np.zeros((32,), dtype=np.int16)
    ping_weight = np.zeros((32, 32), dtype=np.int16)
    pong_token  = np.zeros((32,), dtype=np.int16)
    pong_weight = np.zeros((32, 32), dtype=np.int16)
    result_buf  = np.zeros((32,), dtype=np.int16)