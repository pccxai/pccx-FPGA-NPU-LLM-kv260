# Gemma 3N FPGA Custom NPU Architecture (400MHz Target)

이 문서는 Kria KV260 보드 위에서 Gemma 3N E2B 모델을 400MHz 풀스피드로 구동하기 위해 백지부터 새롭게 재설계된 커스텀 NPU의 핵심 아키텍처를 설명한다. (과거 Ping-Pong BRAM 및 AXI4-Lite 기반의 100MHz 설계는 폐기됨)

## 1. Top-Level Data Flow (컨베이어 벨트 아키텍처)

이 NPU의 핵심 철학은 **"메모리 대역폭의 극대화"**와 **"연산과 로딩의 완벽한 오버래핑(Zero-Bubble)"**이다.

### 1.1 Memory I/O Engine (`memIO_Engine.sv`)
* **Lane Orchestration:** 128-bit AXI-Stream 포트 4개(HP0~HP3)를 중앙에서 동적으로 통제한다. CPU의 개입 없이 포트의 방향(Input/Output)을 유연하게 전환(예: 4:0 -> 3:1)하여 병목을 제거한다.
* **Header Parsing:** 데이터 스트림의 첫 번째 128-bit를 헤더(Header)로 인식하여 해당 패킷이 Feature Map(BF16)인지, Weight(INT4)인지 판단하여 적절한 내부 버스(Lane)로 라우팅한다.
* **Per-Column e_max 추출:** 32개의 Feature Map 입력으로부터 각 열(Column)마다 고유한 지수(`e_max`) 32개를 뜯어내어 딜레이 라인(Delay Pipe)에 태워 보낸다. (연산 완료 후 결과 복원에 사용)

### 1.2 Elastic Buffering (XPM FIFO)
* 400MHz의 빡센 타이밍(WNS)을 견디고 CPU의 불규칙한 데이터 공급(Jitter)을 흡수하기 위해, Xilinx 전용 하드웨어 매크로인 `XPM_FIFO_AXIS`를 모든 입출력 포트에 배치(탄창 역할)하여 데이터를 안정적으로 공급한다.

## 2. Feature Map Cache & Distribution
* **SRAM 캐싱 (`stlc_fmap_cache.sv`):** Gemma의 Decode 단계(1x2048 GEMV)에서 Feature Map은 계속 재사용된다. 이를 위해 1x2048 크기의 XPM BRAM 캐시를 구축하였다.
* **BF16 to Fixed Pipeline:** 캐시에 저장되기 전, 파이프라인 쉬프터를 거쳐 BF16 데이터에서 가수를 추출 및 정렬(27-bit Mantissa)한다.
* **32-Lane Broadcast & Staggered Delay:** 캐시에서 읽어낸 1개의 데이터를 32개 열로 복사(Fan-out)한 뒤, 계단식(0~31클럭) 딜레이 라인을 거쳐 시스톨릭 어레이의 `V_in`(수직 입력)으로 비처럼 쏟아붓는다.

## 3. The Core: Unified GEMM/GEMV Systolic Array
32x32 크기의 PE(Processing Element) 어레이로, Xilinx DSP48E2를 극한까지 활용하여 설계되었다.

* **V_in (Vertical):** 위에서 아래로 Feature Map(Mantissa)과 Instruction이 흐른다.
* **H_in (Horizontal):** 왼쪽에서 오른쪽으로 INT4 가중치(Weight)가 흐른다.
* **Double Buffering & Latency Hiding:** 3-Stage FF 파이프라인을 두어, 현재 연산 중에도 백그라운드에서 다음 타일의 가중치가 이동한다.
* **Dual B-Register Freeze (Weight-Stationary):**
  - **GEMM 모드:** 가중치가 매 클럭 흐른다.
  - **GEMV 모드:** `i_w_load` 신호가 뜨면 가중치가 PE 내부의 `B2` 레지스터에 동결(Freeze)된다. 이후 수직으로 쏟아지는 Feature Map과 곱해지며, 32번의 덧셈이 수직으로 누적되어 `V_ACC_out`을 통해 행렬-벡터 곱 결과가 나온다.

## 4. Post-Processing (정규화 및 복원)
* **`stlc_result_normalizer.sv`:** 시스톨릭 어레이 하단에서 튀어나오는 32개의 48-bit 결과를 다시 BF16 포맷으로 되돌린다.
* **4-Stage Pipeline:** 
  1) Sign-Magnitude 변환 (음수 반전)
  2) Leading One Detection (LOD, 가장 높은 1 위치 찾기)
  3) Barrel Shift (가수 7번째 비트로 정렬)
  4) Exponent Update (아까 엔진에서 보내준 `e_max`와 LOD 결과를 조합해 최종 지수 도출)
* 이 정규화된 16-bit 결과들은 Result Packer를 통해 128-bit로 압축되어 DMA를 타고 CPU로 돌아간다.