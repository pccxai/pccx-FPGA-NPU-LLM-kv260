import numpy as np
import sys

# Toggle PC simulation mode (change to False when board arrives!)
SIMULATION_MODE = True

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
    
    # Allocate regular Numpy memory instead of pynq allocate (equivalent to using regular malloc instead of cudaMallocHost)
    ping_token  = np.zeros((32,), dtype=np.int16)
    ping_weight = np.zeros((32, 32), dtype=np.int16)
    pong_token  = np.zeros((32,), dtype=np.int16)
    pong_weight = np.zeros((32, 32), dtype=np.int16)
    result_buf  = np.zeros((32,), dtype=np.int16)
    print("Mock Hardware Init Complete!")