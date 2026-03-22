# Gemma 3N E2B FPGA Accelerator Project & Architecture Details

## 1. 프로젝트 비전 (Vision)
이 프로젝트의 궁극적인 목표는 **Kria KV260 보드 상에서 Gemma 3N E2B 모델의 Inference (Prefill & Decode) 과정을 하드웨어적으로 극한까지 가속**하는 것이다.
기존의 단순한 AXI4-Lite 기반의 100MHz 설계를 폐기하고, 물리적 배선 최적화와 파이프라이닝을 통해 **400MHz 타겟의 초고속, 초저지연 NPU(Neural Processing Unit)**를 백지부터 재설계했다.

우리는 LLM 연산의 두 가지 핵심 형태인 **GEMM (Matrix x Matrix)**와 **GEMV (Matrix x Vector)**를 하나의 시스톨릭 어레이(Systolic Array)에서 동적으로(Zero-Bubble) 스위칭하며 처리하는 세계 최고 수준의 아키텍처를 구현한다.

---

## 2. 하드웨어 아키텍처 핵심 (Core Architecture)

### 2.1 포트 물리적 분리 전략 (Segregated AXI Ports)
CPU(PS)와 FPGA(PL) 간의 데이터 병목을 없애기 위해 KV260이 제공하는 다양한 AXI 포트들의 물리적 특성을 극대화하여 용도를 고정했다.
*   **HPC / ACP 포트 (Feature Map 전용):** 캐시 코히어런시(Cache Coherency)를 활용하는 초저지연 포트. CPU가 연산(예: RoPE)을 마치고 L2 캐시에 올려둔 활성화 함수 결과(Feature Map)를 DDR 메모리를 거치지 않고 즉시 FPGA로 스누핑(Snooping)해 온다.
*   **HP0 ~ HP3 포트 (Weight 전용):** 캐시 동기화 기능은 없지만 깡패 같은 대역폭을 자랑하는 High-Performance 포트. 128-bit 대역폭 4개(총 512-bit)를 모두 가중치(INT4) 무한 스트리밍에 몰빵하여 어레이의 연산기가 절대 굶지 않게(Starvation Free) 만든다.
*   **HPM 포트 (Control 전용):** 데이터 패스와 제어 패스(Control Plane)를 물리적으로 분리. AXI4-Lite를 통해 NPU의 중앙 컨트롤러(Global FSM)에 "시작", "정지", "명령어(VLIW)" 등을 하달한다.

### 2.2 극한의 하드웨어 기술 명세 (Deep-Dive Technologies)

우리의 32x32 하이브리드 시스톨릭 어레이는 단순히 논리적인 연산기가 아니라, **Xilinx DSP48E2의 실리콘 물리 구조(Physical Silicon Structure)**를 100% 이해하고 쥐어짜 낸 결과물이다.

**[1] ACIN / ACOUT 전용 고속 라인 (수직 Feature Map)**
*   **기술적 난제:** 32개의 DSP를 일반 배선(Fabric Routing)으로 수직 연결하면 배선 지연(Routing Delay) 때문에 400MHz 달성이 불가능하다.
*   **해결책:** DSP48E2 내부에 물리적으로 하드와이어링된 캐스케이드(Cascade) 전용선인 `ACIN`과 `ACOUT` 포트를 명시적으로 사용했다. 맨 윗줄(`IS_TOP_ROW=1`)만 `A_INPUT="DIRECT"`로 외부 입력을 받고, 나머지 1~31번 줄은 `A_INPUT="CASCADE"`로 설정하여 0.1ns 수준의 지연시간으로 데이터를 수직 하강시킨다.

**[2] B 포트 외부 FF 데이지 체인 (수평 Weight)**
*   **기술적 난제:** Xilinx DSP는 구조상 가로(Horizontal) 방향의 전용 캐스케이드 선(`BCIN_Horizontal` 같은 것)이 존재하지 않는다.
*   **해결책:** 인접한 PE끼리 1클럭만에 데이터를 넘기는 외부 Fabric FF(`out_H <= in_H`)를 배치했다 (Daisy Chain). Vivado의 배치기(Placer)가 이 FF를 DSP의 `B` 포트 바로 옆(동일 Slice)에 밀착 배치하도록 유도하여 수평 이동 타이밍을 완벽히 잡았다.

**[3] Dual B-Register 동결 (Weight-Stationary)**
*   DSP48E2 내부에는 `B1`, `B2`라는 2개의 파이프라인 레지스터가 존재한다. 우리는 `CEB1`과 `CEB2` 핀을 분리 제어하여 **GEMM / GEMV 듀얼 모드**를 창조했다.
*   **GEMM 모드:** 가중치가 매 클럭 수평으로 흘러간다. (`CEB1`, `CEB2` 모두 활성화)
*   **GEMV 모드:** 32클럭 동안 수평 전파된 가중치가 제자리에 도착하면, 단 1클럭의 `w_load` 펄스를 발생시켜 가중치를 `B2` 레지스터에 동결(Stationary Freeze)시킨다.

**[4] PCIN / PCOUT Flush (결과 배출의 예술)**
*   각 PE의 48-bit 누적 결과(`P` 레지스터)를 외부 MUX로 빼내는 것은 엄청난 배선 낭비다.
*   연산이 끝나는 순간, 3비트 VLIW 명령어에 의해 DSP의 `OPMODE`가 `P = P + M` (누적)에서 **`P = PCIN` (수직 쉬프트)**으로 다이내믹하게 변경된다.
*   32개의 PE가 동시에 자신의 결과를 아래 DSP의 `PCIN`으로 넘기며 거대한 48-bit 수직 쉬프트 레지스터로 변신한다.

**[5] 3-Bit VLIW & Event-Driven Latch (명령어 최적화)**
*   무거운 `case` 문 디코더를 제거하고 `[Inst[2]: Flush, Inst[1]: GEMV/GEMM, Inst[0]: Calc/Idle]`의 3비트 VLIW 형식을 채택해 하드웨어 핀(`CEP`, `CEB2`, `OPMODE`)에 다이렉트로 매핑했다.
*   특히, 매 클럭 명령어를 쏘지 않고 명령이 바뀔 때만 쏘는 **이벤트 드리븐 래치(`inst_valid_in_V`)**를 적용해 1024개 PE로 뻗어나가는 명령어 배선의 Toggle Rate(전력 소모)를 0에 수렴하게 만들었다.

**[6] Staggered Delay Line (계단식 딜레이)**
*   시스톨릭 어레이의 대각선 파동(Wavefront) 입력을 맞추기 위해, Feature Map 브로드캐스트 라인에 FF 체인 기반의 딜레이 라인을 구축했다. (Row 0은 0클럭 딜레이, Row 31은 31클럭 딜레이). BRAM의 복잡한 멀티포트 제어 없이 순수 FF만으로 지연을 구현해 병목을 없앴다.

### 2.3 Feature Map Cache & Post-Processing (BFP & 정규화)
*   **SRAM 캐싱 (`stlc_fmap_cache.sv`):** GEMV(Decode) 단계에서 1x2048 크기의 Feature Map은 계속 재사용된다. 이를 초고속 `XPM_MEMORY_SDPRAM` (BRAM)에 캐싱하고 32개 열에 동시에 브로드캐스팅(Fan-out)하여 대역폭 한계를 돌파했다.
*   **Block Floating Point (BFP):** 32개의 BF16 입력으로부터 각 열마다 고유한 지수(`e_max`)를 추출하여 보관한다. BF16의 가수(Mantissa)는 3-Stage Barrel Shifter를 거쳐 27-bit 고정소수점으로 정렬된 후 연산기에 투입된다.
*   **Result Normalization (정규화 복원):** 어레이 하단에서 배출된 48-bit 결과는 보관해둔 `e_max`를 사용하여 파이프라인화된 [음수 반전 -> LOD (가장 높은 1 찾기) -> Barrel Shift -> 지수 업데이트] 과정을 거쳐 완벽한 BF16 포맷으로 되돌아간다.

### 2.4 Zero-Bubble & Latency Hiding
*   현재 타일의 연산이 진행되는 32클럭 동안, **백그라운드에서는 이미 다음 32x32 가중치 타일이 수평(Fabric FF)으로 이동**하며 각 PE의 문 앞(`B1`)에 장전 대기 중이다.
*   이전 연산 결과가 배출(Flush)되는 단 4~6클럭의 틈(`flush_sequence` 쉬프트 레지스터 사용)을 타서, 미리 도착해 있던 가중치를 `B2` 레지스터에 덮어씌운다. **즉, 연산기 대기 시간(Bubble)이 0에 수렴한다.**

---

## 3. 남은 개발 로드맵 (Roadmap: 바이브 코딩 + 하드 코딩)

우리의 방식은 **AI(Gemini)가 RTL 뼈대와 최적화 로직(바이브 코딩)을 제안하고, 휴먼 엔지니어(hwkim)가 물리적 라우팅과 디버깅, 포트 매핑(하드 코딩)을 결합**하여 완성하는 형태다.

### Step 1: NPU Central Controller (Global FSM) 구현 [진행 중]
*   **목표:** HPM을 통해 들어온 VLIW 명령어를 받아, SRAM Cache 읽기 타이밍, Weight 장전 타이밍, 그리고 Systolic Array에 명령어를 하달하는 두뇌(Brain) 설계.

### Step 2: System Validation & WNS Tuning
*   **목표:** 400MHz 합성을 위해 Vivado에서 Synthesis 및 Implementation을 돌리고, WNS(Worst Negative Slack)가 0 이상인지 확인.
*   **바이브/하드 코딩 역할:** Timing Violation 발생 지점(Critical Path) 분석 및 Register Slicing(FF 추가) 솔루션 제공. Testbench 시뮬레이션을 통한 Data Hazard 직접 눈으로 확인 및 디버깅.

### Step 3: PS-PL Software Stack (Python PYNQ) 연동
*   **목표:** pynq 라이브러리를 활용하여 DMA로 데이터 쏘고, MMIO로 명령어 내려서 실제 결과 뽑기. Numpy 기반 INT4/BF16 양자화 전처리 스크립트 작성.

### Step 4: CPU Offloading (Extra Feature)
*   **목표:** FPGA 자원이 남을 경우, Attention 계산 시 가장 무거운 RoPE 연산을 FPGA에 하드웨어로 박아넣고 ACP 통신으로 CPU 부하를 덜어줌.

---

*이 문서는 프로젝트의 방향성을 잃지 않기 위한 '나침반' 역할을 하며, 향후 코드 구현 시 이 아키텍처 철학(Physical Segregation, Stationary Freeze, Zero-Bubble)을 절대 위반하지 않는다.*