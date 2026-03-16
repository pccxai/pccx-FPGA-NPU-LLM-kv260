from pynq import Overlay, allocate
import numpy as np

class TinyNPUDriver:
    def __init__(self, bitstream_path="tinynpu.bit"):
        print("Burning NPU hardware on FPGA... (Bitstream Load)")
        self.overlay = Overlay(bitstream_path)

        # 1. Get hardware IP pointer (same as C++ structure pointer mapping)
        # Register for NPU control (AXI4-Lite)
        self.npu_ctrl = self.overlay.tiny_npu_0

        # Highway for data transmission (AXI DMA)
        self.dma = self.overlay.axi_dma_0

        # NPU control register offset (addresses we set when writing Verilog!)
        self.REG_CTRL      = 0x00 # Start, Done, Idle bits
        self.REG_TILE_SIZE = 0x10 # Tile size setting
        self.REG_MODE      = 0x14 # 0: GEMM, 1: GEMV

    def run_npu_gemv(self, input_vec, weight_tile):
        """ Driver function to push data to the NPU and perform matrix-vector multiplication (GEMV) """

        # [Step 1] CMA memory allocation (same as cudaMallocHost!)
        # The OS kernel tightly holds the ‘physically contiguous’ space in DDR4.
        # (This is how DMA scrapes all addresses at once without interruption)
        in_buffer = allocate(shape=input_vec.shape, dtype=np.float32)
        wt_buffer = allocate(shape=weight_tile.shape, dtype=np.float32)
        out_buffer = allocate(shape=(1, weight_tile.shape[1]), dtype=np.float32)

        # CPU memory -> Copy data to DMA dedicated memory
        np.copyto(in_buffer, input_vec)
        np.copyto(wt_buffer, weight_tile)

        # [Step 2] NPU register setting through MMIO (creating Command List)
        # In C++ volatile unsigned int* ptr = 0x40000010; *ptr = 64; It's the same as doing it.
        self.npu_ctrl.write(self.REG_TILE_SIZE, 64) # It's a 64x64 tile.
        self.npu_ctrl.write(self.REG_MODE, 1)       # Set to GEMV mode.

        # [Step 3] AXI DMA operation! (cudaMemcpyAsync)
        # The CPU immediately returns to the DMA controller saying, "Hey, start from this address and shoot this much to the NPU."
        self.dma.sendchannel.transfer(in_buffer)
        self.dma.sendchannel.transfer(wt_buffer)
        self.dma.recvchannel.transfer(out_buffer)

        # [Step 4] NPU calculation start trigger! (kernel launch)
        # Set bit 0 (Start) at address 0x00 to 1.
        self.npu_ctrl.write(self.REG_CTRL, 0x01)

        # [Step 5] Wait for synchronization (cudaDeviceSynchronize)
        # Infinite loop (Polling) until the 1st bit (Done) at address 0x00 becomes 1.
        while (self.npu_ctrl.read(self.REG_CTRL) & 0x02) == 0:
            pass

        # Computation complete! NPU controller reset
        self.npu_ctrl.write(self.REG_CTRL, 0x00)

        # Wait for DMA buffer and get results
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # Copy PYNQ buffer data to regular Numpy array and return
        result = np.array(out_buffer)

        # Memory leak prevention (free)
        in_buffer.free()
        wt_buffer.free()
        out_buffer.free()

        return result