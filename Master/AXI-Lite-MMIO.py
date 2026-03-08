from pynq import Overlay, allocate
import numpy as np
import time

# 1. 비트스트림 및 블록 디자인(Overlay) 하드웨어 로드
# 보드에 네가 합성한 bit 파일과 hwh 파일을 밀어넣어서 FPGA 영역을 프로그래밍하는 단계야.
print("FPGA 오버레이 로딩 중...")
ol = Overlay("gemma3n_npu_design.bit")

# 2. IP 모듈 바인딩
# Vivado 블록 디자인에서 설정한 이름 그대로 파이썬 객체로 가져옴.
# 예를 들어 DMA IP 이름이 'axi_dma_0', NPU 제어용 AXI Lite 래퍼가 'npu_axi_wrapper_0'라고 가정함.
dma = ol.axi_dma_0
npu_axi = ol.npu_axi_wrapper_0

# 3. 물리적으로 연속된 DMA용 메모리 버퍼 할당 (CMA 영역)
# 데이터 타입은 하드웨어 버스 폭(예: 32bit)에 맞춰서 설정해야 해.
# Gemma의 bfloat16 데이터를 넘길 거라면 16비트 두 개를 32비트로 패킹해서 보내는 게 정석이야.
# 여기서는 32x32 Systolic Array 입력 사이즈에 맞춘다고 가정하고 (1024,) 크기로 잡을게.
print("물리 메모리(CMA) 버퍼 할당 중...")
input_buffer = allocate(shape=(1024,), dtype=np.uint32)
output_buffer = allocate(shape=(1024,), dtype=np.uint32)

# 1. 파이썬 단 데이터 변환 함수 (Float -> INT16 Fixed-Point Q4.12)
# Q4.12 포맷은 소수점 아래 12비트를 쓴다는 뜻이야.
# 스케일 팩터는 2^12 = 4096.0 이 됨.
SCALE_FACTOR = 4096.0

def float_to_q4_12(float_array):
    # 부동소수점 배열에 스케일 팩터를 곱함
    scaled = float_array * SCALE_FACTOR
    # np.clip으로 int16 범위를 넘지 않게 자른 뒤, int16으로 캐스팅
    # int16의 범위는 -32768 ~ 32767
    q_array = np.clip(scaled, -32768, 32767).astype(np.int16)
    return q_array

# 테스트용 float32 데이터 (Gemma 가중치나 입력값이라고 가정)
gemma_data_float = np.array([1.5, -0.75, 3.14, -2.5], dtype=np.float32)

# INT16으로 변환
gemma_data_int16 = float_to_q4_12(gemma_data_float)

# DMA로 쏘기 위해서는 32비트 버스에 맞춰서 16비트 데이터 2개를 하나로 묶어줘야 해.
# 짝수 인덱스는 하위 16비트, 홀수 인덱스는 상위 16비트로 패킹
packed_data = np.zeros(len(gemma_data_int16) // 2, dtype=np.uint32)
for i in range(len(packed_data)):
    low_16 = gemma_data_int16[2*i] & 0xFFFF
    high_16 = (gemma_data_int16[2*i + 1] & 0xFFFF) << 16
    packed_data[i] = high_16 | low_16

print("원본 Float:", gemma_data_float)
print("Q4.12 INT16:", gemma_data_int16)
print("DMA 전송용 32비트 패킹 데이터 (Hex):", [hex(x) for x in packed_data])

# 4. 입력 버퍼에 테스트 데이터 채우기
# 실제 프로젝트에서는 여기서 safetensors에서 읽어온 Gemma 가중치/입력값을 패킹해서 넣게 됨.
for i in range(1024):
    input_buffer[i] = i

# --- NPU 파이프라인 실행 루틴 ---

# 5. 하드웨어 초기화 (Clear)
# MMIO 0x04번지가 누산기(Accumulator) 및 상태 머신 초기화 레지스터라고 가정.
# 룰 5번 적용: 파이썬에서 1을 쓰면 하드웨어(Verilog) 단에서 다음 클럭에 알아서 0으로 내려야 함.
npu_axi.write(0x04, 1)

# 6. DMA를 통한 데이터 송신 (CPU -> BRAM)
print("DMA 데이터 전송 시작 (TX)...")
# sendchannel은 CPU에서 하드웨어로 데이터를 쏘는 채널이야.
dma.sendchannel.transfer(input_buffer)

# 데이터가 다 넘어갈 때까지 기다림. 내부적으로 캐시 플러시도 수행됨.
dma.sendchannel.wait()
print("DMA TX 완료. BRAM에 데이터 적재됨.")

# 7. NPU 연산 시작 트리거
# MMIO 0x00번지를 NPU Start 레지스터로 가정.
# BRAM에 데이터가 꽉 찼으니 NPU 내부 FSM에게 연산을 시작하라고 명령을 내림.
# 이것도 마찬가지로 오토 클리어 펄스로 동작해야 안전함.
npu_axi.write(0x00, 1)

# 8. 연산 완료 대기 (Polling)
# 룰 6번 적용: RTL 메모리 맵과 Python의 주소가 완벽히 일치해야 함.
# MMIO 0x10번지를 NPU Done 상태 레지스터로 가정.
# 값이 1이 될 때까지 무한 루프를 돌면서 하드웨어 상태를 체크함.
print("NPU 연산 진행 중... 폴링 대기")
while True:
    npu_status = npu_axi.read(0x10)
    if npu_status == 1:
        break
    time.sleep(0.001) # CPU 점유율이 100%로 치솟는 걸 방지하기 위해 아주 짧게 대기

print("NPU 연산 완료!")

# 9. DMA를 통한 결과 수신 (BRAM -> CPU)
# NPU가 연산을 끝내고 결과를 BRAM에 썼으므로, 이제 수신 채널로 가져옴.
dma.recvchannel.transfer(output_buffer)
dma.recvchannel.wait()
print("DMA RX 완료. 결과 데이터 메모리로 복사됨.")

# 10. 결과 검증 (샘플)
print("출력 버퍼 앞부분 확인:")
print(output_buffer[:10])

# 사용이 끝난 물리 메모리는 반드시 해제해줘야 함. 안 그러면 메모리 릭 발생.
input_buffer.close()
output_buffer.close()