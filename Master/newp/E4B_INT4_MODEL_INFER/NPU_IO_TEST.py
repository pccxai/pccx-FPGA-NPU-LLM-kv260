import numpy as np
import time

# Toggle PC simulation mode (change to False when board arrives & bitstream is ready)
SIMULATION_MODE = True

def main():
    print("=== [KV260 NPU V2] 5-DMA & AXI GPIO I/O Test ===")
    
    if not SIMULATION_MODE:
        from pynq import Overlay, allocate
        print("Loading Bitstream...")
        # TODO: Change to the name of the actual synthesized bitstream file
        overlay = Overlay("NPU.bit") 
        
        # 1. Hardware IP Mapping (Based on Vivado Block Design names)
        print("Mapping Hardware IPs...")
        dma_fmap = overlay.DMA_FMAP
        dma_w0   = overlay.DMA_W_HP0
        dma_w1   = overlay.DMA_W_HP1
        dma_w2   = overlay.DMA_W_HP2
        dma_w3   = overlay.DMA_W_HP3
        
        # GPIOs (Currently assumed to be axi_gpio_0/1 for CMD/STAT purposes in the BD)
        # If one GPIO is split into channel1 and channel2, modify it to overlay.axi_gpio_0.channel1 / channel2
        gpio_cmd = overlay.axi_gpio_0.channel1
        gpio_stat = overlay.axi_gpio_1.channel1 
        
        # 2. Allocate DMA Buffers (PYNQ memory allocation)
        print("Allocating DMA Buffers...")
        # Arbitrary length (to be adjusted later according to the actual tile size of 32x32)
        fmap_len = 1024
        weight_len = 2048
        
        # FMap: Uses uint16 to represent the BF16 format
        fmap_in  = allocate(shape=(fmap_len,), dtype=np.uint16)
        fmap_out = allocate(shape=(fmap_len,), dtype=np.uint16)
        
        # Weights: 2 INT4s are grouped to become 1 byte (uint8)
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
        
        # 4. Trigger DMA Transfers (Asynchronous parallel transmission)
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
        gpio_cmd.write(0, 0) # Pulse down (even without auto-clear HW, it goes down to 0 here)
        
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
