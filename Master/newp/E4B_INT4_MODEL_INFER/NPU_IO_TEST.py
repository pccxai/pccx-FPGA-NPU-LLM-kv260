import numpy as np
import time

# Toggle PC simulation mode (change to False when board arrives & bitstream is ready)
SIMULATION_MODE = True

def main():
    print("=== [KV260 NPU V2] 5-DMA & AXI GPIO I/O Test ===")
    
    if not SIMULATION_MODE:
        from pynq import Overlay, allocate
        print("Loading Bitstream...")
        # TODO: 실제 합성 완료된 bitstream 파일 이름으로 변경해야 함
        overlay = Overlay("NPU.bit") 
        
        # 1. Hardware IP Mapping (Vivado Block Design 이름 기준)
        print("Mapping Hardware IPs...")
        dma_fmap = overlay.DMA_FMAP
        dma_w0   = overlay.DMA_W_HP0
        dma_w1   = overlay.DMA_W_HP1
        dma_w2   = overlay.DMA_W_HP2
        dma_w3   = overlay.DMA_W_HP3
        
        # GPIOs (현재 BD상 axi_gpio_0/1을 각각 CMD/STAT 용도로 추정)
        # 만약 한 GPIO의 channel1, channel2로 나눴다면 overlay.axi_gpio_0.channel1 / channel2 로 수정
        gpio_cmd = overlay.axi_gpio_0.channel1
        gpio_stat = overlay.axi_gpio_1.channel1 
        
        # 2. Allocate DMA Buffers (PYNQ 메모리 할당)
        print("Allocating DMA Buffers...")
        # 임의의 길이 (실제 타일 크기 32x32에 맞춰 추후 조절)
        fmap_len = 1024
        weight_len = 2048
        
        # FMap: BF16 포맷을 나타내기 위해 uint16 사용
        fmap_in  = allocate(shape=(fmap_len,), dtype=np.uint16)
        fmap_out = allocate(shape=(fmap_len,), dtype=np.uint16)
        
        # Weights: INT4 가 2개씩 묶여 1바이트(uint8)가 됨
        w0_in = allocate(shape=(weight_len,), dtype=np.uint8)
        w1_in = allocate(shape=(weight_len,), dtype=np.uint8)
        w2_in = allocate(shape=(weight_len,), dtype=np.uint8)
        w3_in = allocate(shape=(weight_len,), dtype=np.uint8)
        
        # 3. Initialize with Dummy Data
        fmap_in[:] = np.arange(fmap_len, dtype=np.uint16)
        # 0xAA = 1010_1010 (INT4: -6, -6)
        w0_in[:] = np.ones(weight_len, dtype=np.uint8) * 0xAA 
        w1_in[:] = np.ones(weight_len, dtype=np.uint8) * 0xBB
        w2_in[:] = np.ones(weight_len, dtype=np.uint8) * 0xCC
        w3_in[:] = np.ones(weight_len, dtype=np.uint8) * 0xDD
        
        # 4. Trigger DMA Transfers (비동기 병렬 전송)
        print("Starting 5-Channel DMA Transfers...")
        dma_fmap.sendchannel.transfer(fmap_in)
        dma_w0.sendchannel.transfer(w0_in)
        dma_w1.sendchannel.transfer(w1_in)
        dma_w2.sendchannel.transfer(w2_in)
        dma_w3.sendchannel.transfer(w3_in)
        
        # 5. Wait for all sending to complete
        dma_fmap.sendchannel.wait()
        dma_w0.sendchannel.wait()
        dma_w1.sendchannel.wait()
        dma_w2.sendchannel.wait()
        dma_w3.sendchannel.wait()
        print("DMA Send Complete!")
        
        # 6. Trigger NPU execution via GPIO
        # mmio_npu_cmd: [0]=Start, [1]=Clear, [4:2]=Inst(VLIW)
        print("Pulsing NPU START via AXI GPIO...")
        gpio_cmd.write(0, 0) # Clear all bits
        gpio_cmd.write(1, 0) # Set Start bit (bit 0 = 1)
        gpio_cmd.write(0, 0) # Pulse down (auto-clear HW가 없더라도 여기서 0으로 내려줌)
        
        # 7. Poll for Completion
        print("Polling for NPU DONE...")
        # mmio_npu_stat: [0]=Done, [1]=FMap_Ready
        while True:
            stat = gpio_stat.read()
            if (stat & 0x01) != 0: # Check Done bit (bit 0)
                break
            time.sleep(0.001)
            
        print("NPU Execution Done!")
        
        # 8. Read back results
        print("Reading back FMAP Results from NPU...")
        dma_fmap.recvchannel.transfer(fmap_out)
        dma_fmap.recvchannel.wait()
        
        print(f"Result snippet (first 10 elements): {fmap_out[:10]}")
        print("=== Hardware I/O Test PASSED ===")
        
    else:
        print("[PC Simulation Mode] Skipping actual hardware I/O.")
        print("To run on KV260 board, set SIMULATION_MODE = False in code.")

if __name__ == "__main__":
    main()