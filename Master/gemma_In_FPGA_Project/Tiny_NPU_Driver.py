from pynq import Overlay, allocate
import numpy as np

class TinyNPUDriver:
    def __init__(self, bitstream_path="tinynpu.bit"):
        print("FPGA에 NPU 하드웨어 굽는 중... (Bitstream Load)")
        self.overlay = Overlay(bitstream_path)
        
        # 1. 하드웨어 IP 포인터 가져오기 (C++ 구조체 포인터 맵핑과 동일)
        # NPU 제어용 레지스터 (AXI4-Lite)
        self.npu_ctrl = self.overlay.tiny_npu_0 
        
        # 데이터 전송용 고속도로 (AXI DMA)
        self.dma = self.overlay.axi_dma_0

        # NPU 제어 레지스터 오프셋 (Verilog 짤 때 우리가 정한 주소들!)
        self.REG_CTRL      = 0x00 # Start, Done, Idle 비트
        self.REG_TILE_SIZE = 0x10 # 타일 크기 세팅 
        self.REG_MODE      = 0x14 # 0: GEMM, 1: GEMV

    def run_npu_gemv(self, input_vec, weight_tile):
        """ NPU에 데이터를 밀어넣고 행렬-벡터 곱(GEMV)을 수행하는 드라이버 함수 """
        
        # [Step 1] CMA 메모리 할당 (cudaMallocHost 랑 똑같음!)
        # OS 커널이 DDR4에서 '물리적으로 연속된' 공간을 꽉 잡아줌. 
        # (이래야 DMA가 주소 끊김 없이 한 번에 쫙 긁어감)
        in_buffer = allocate(shape=input_vec.shape, dtype=np.float32)
        wt_buffer = allocate(shape=weight_tile.shape, dtype=np.float32)
        out_buffer = allocate(shape=(1, weight_tile.shape[1]), dtype=np.float32)
        
        # CPU 메모리 -> DMA 전용 메모리로 데이터 복사
        np.copyto(in_buffer, input_vec)
        np.copyto(wt_buffer, weight_tile)

        # [Step 2] MMIO를 통한 NPU 레지스터 세팅 (Command List 작성)
        # C++에서 volatile unsigned int* ptr = 0x40000010; *ptr = 64; 하는 거랑 똑같음!
        self.npu_ctrl.write(self.REG_TILE_SIZE, 64) # 64x64 타일이다!
        self.npu_ctrl.write(self.REG_MODE, 1)       # GEMV 모드로 세팅!

        # [Step 3] AXI DMA 가동! (cudaMemcpy Async)
        # CPU는 DMA 컨트롤러한테 "야, 이 주소부터 이만큼 NPU로 쏴" 하고 바로 리턴됨
        self.dma.sendchannel.transfer(in_buffer)
        self.dma.sendchannel.transfer(wt_buffer)
        self.dma.recvchannel.transfer(out_buffer)

        # [Step 4] NPU 연산 시작 트리거 빵! (kernel launch)
        # 0x00 번지의 0번 비트(Start)를 1로 만듦
        self.npu_ctrl.write(self.REG_CTRL, 0x01)

        # [Step 5] 동기화 대기 (cudaDeviceSynchronize)
        # 0x00 번지의 1번 비트(Done)가 1이 될 때까지 무한 루프 (Polling)
        while (self.npu_ctrl.read(self.REG_CTRL) & 0x02) == 0:
            pass 

        # 연산 끝! NPU 컨트롤러 리셋
        self.npu_ctrl.write(self.REG_CTRL, 0x00)
        
        # DMA 버퍼 대기 및 결과물 가져오기
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        
        # PYNQ 버퍼 데이터를 일반 Numpy 배열로 복사 후 리턴
        result = np.array(out_buffer)
        
        # 메모리 누수 방지 (free)
        in_buffer.free()
        wt_buffer.free()
        out_buffer.free()
        
        return result