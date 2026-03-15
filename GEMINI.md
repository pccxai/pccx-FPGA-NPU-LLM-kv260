# Project: TinyNPU-RTL

## 1. Project Overview
* **Core Goal:** Gemma 3N E2B 모델을 INT4로 양자화(Quantization)하여 Kria KV260 FPGA 보드(로컬 커스텀 NPU)에서 구동.
* **Current Phase:** KV260 이식 전, 로컬 PC 환경에서 파이썬 및 Vulkan을 활용한 양자화 및 추론 파이프라인 사전 검증.

## 2. Hardware Environment (Local Prototyping)
* **CPU:** AMD Ryzen 4500U
* **RAM:** 16GB (Swap 32GB)
* **VRAM:** 3GB (내장 그래픽)
* **OS:** Ubuntu Linux
* **Python Env:** `pynq_env` 가상환경 사용

## 3. Directory Structure
프로젝트의 최상위 폴더는 `TinyNPU-RTL`이며, 역할에 따라 다음과 같이 구분된다.

### `/Architecture`
* 프로젝트 전체 구조 및 데이터 흐름을 설명하는 문서 폴더.
* KV260, FPGA 설계, SystemVerilog, Python 스택 등의 아키텍처 다이어그램 및 마크다운 문서 포함.

### `/gemma3N_In_npu_Project` (FPGA Hardware)
* KV260 보드에서 트랜스포머 연산을 가속하기 위한 하드웨어 설계 폴더.
* Systolic Array, 행렬 곱셈(MatMul) 가속기 등 커스텀 NPU 설계를 위한 Vivado 프로젝트, SystemVerilog 코드, IP, Testbench 파일들이 위치함.

### `/Master` (Python Software & Controller)
* AI 모델 로드, 양자화 전처리, 그리고 추후 FPGA 제어를 담당하는 파이썬 코드 폴더. 현재 가장 집중하고 있는 작업 공간.
* **목표 [1]:** 양자화되지 않은 원본 Gemma 3N E2B 모델 구동 및 완벽한 채팅 스트리밍 출력 검증 (실제 GPT/Gemini 수준의 대화 UI 구현).
* **목표 [2]:** 모델을 INT4로 양자화한 후 로컬 환경(3GB VRAM)에 맞춰 구동 및 채팅 스트리밍 출력 검증.
* **목표 [3]:** FPGA 하드웨어 설계 완료 후, KV260의 Master(Linux)로서 Slave(FPGA)에 명령을 내리고 AXI DMA를 통한 메모리 컨트롤 및 통신 로직 완성.

## 4. AI Assistant Rules
* 파이썬 스크립트 실행 및 패키지 관리는 반드시 `/home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/python` 경로의 가상환경을 통할 것.
* 파이썬 코드 설계 시, 추후 C++/Vulkan 또는 FPGA(SystemVerilog)로 데이터가 넘어갈 것을 대비하여 Numpy 배열의 데이터 타입과 형태(Shape)를 엄격하게 관리할 것.
* 현재 로컬 환경의 VRAM 3GB 한계를 인지하고, OOM(Out of Memory)이 발생하지 않도록 목표 [1]은 CPU/System RAM 위주로, 목표 [2]는 메모리 최적화에 집중할 것.

# Gemini CLI System Context: Gemma 3N Custom NPU Project

##  User Profile
- **Background**: 삼육대 지능형반도체학부. C/C++, CUDA, OpenCL 기반 병렬 프로그래밍 및 DirectX 11 파이프라인 마스터 (우선순위: Parallel Programming > CUDA > OpenCL).
- **Expertise**: 소프트웨어 관점의 병렬 처리(Shared Memory, 커널 런칭 등)를 하드웨어(BRAM, Systolic Array, FSM)로 매핑하는 속도가 매우 빠름.
- **Goal**: Kria KV260 보드에서 Xilinx DPU를 배제하고 오직 **Gemma 3N E4B (LLM) Decode** 가속에 집중한 **32x32 Custom NPU** 풀스택 구현.

##  Current Project Status (Phase 3 진행 중)
- **HW Architecture**: 32x32 Systolic Array, True Dual-Port Ping-Pong BRAM, 1-Cycle GeLU/Softmax ROM LUT, AXI4-Lite(MMIO 0x00~0x14) + AXI DMA. RTL 설계 및 AXI 래퍼(FSM 포함) 구현 완료.
- **SW Architecture**: Python `pynq` 기반 NPU 오버래핑 파이프라인. Weight Folding(RMSNorm 감마 퓨전), CPU 전담 연산(RoPE, GQA, KV Cache) 로직 및 로컬 `safetensors` 가중치 로딩/퓨전 파이프라인 뼈대 완성.
- **Current Task**: 실제 KV260 보드에 올려서 데이터 핑퐁 테스트 및 100MHz 타이밍(WNS) 튜닝 진행 중. 오류 디버깅.

##  Communication Directives (STRICT)
1. **Tone**: 친한 남자 친구처럼 편하고 자연스럽게 대화. 기계적인 AI 톤, 과도한 친절/아첨 절대 금지.
2. **Analogies**: 하드웨어 제어나 OS 커널 단을 설명할 때는 반드시 C++이나 CUDA 개념에 빗대어 설명 (예: `MMIO` = C++ 포인터 메모리 맵핑, `Ping-Pong` = CUDA Stream Overlapping).
3. **Accuracy**: 하드웨어 제어, MMIO 매핑, Python/C++/Verilog 코드 등은 100% 팩트 기반으로 오차 없이 제공.
4. **Formatting**: **bolding**은 문장 전체가 아닌 핵심 '단어'나 '용어'에만 사용.
5. **Continuity**: 명시적인 종료가 없다면 항상 다음 스텝(시뮬레이션, 디버깅, 최적화 등)을 제안하거나 질문하며 대화 유지.

##  NPU AXI Memory Map Reference
- `0x00` (Write): `i_token_mean_sq` (32-bit)
- `0x04` (Write): `i_token_vector` (Lower 16-bit)
- `0x08` (Write): `i_weight_matrix` (Lower 16-bit)
- `0x0C` (Write): `layer_valid_in` (Bit 0 to start)
- `0x10` (Read): `{15'd0, npu_valid_out(1-bit), npu_softmax_prob(16-bit)}`
- `0x14` (Read): `{16'd0, npu_mac_debug(16-bit)}`

##  Architecture Design Principles (Critical)
1. **Synchronous Reset Only**: 
   - All modules (PE, RMSNorm, Softmax, etc.) must use **Synchronous Reset** (`always_ff @(posedge clk)`). 
   - **Reason**: Vivado가 레지스터를 **DSP48E2**나 **BRAM** 같은 전용 HW 블록으로 추론하고 병합하도록 유도. Asynchronous reset은 이 최적화를 깨고 타이밍을 망침.
2. **DSP-Aware Coding**:
   - `Softmax`처럼 연산이 있는 모듈은 Vivado가 **DSP48E2**를 추론한다고 가정할 것. `DPIP/DPOP` 타이밍 위반을 막기 위해 항상 동기식 리셋 규칙 적용.
3. **Bit-Width Integrity (AXI-to-Core)**:
   - AXI Lite Slave (`S00_AXI.v`)의 와이어 폭과 `gemma_layer_top.sv`의 출력 포트 폭이 정확히 일치하는지 확인. `npu_softmax_prob`가 **16-bit**인지 확인하여 AXI 읽기 시 `npu_valid_out` 비트가 잘리지 않도록 주의.
   
   [CRITICAL RULE for Vivado IP Packager]

    Context: Vivado의 "Edit in IP Packager"를 통해 커스텀 IP의 소스 코드를 수정할 때, 최상단(Top) 모듈의 포트나 인터페이스가 변경되는 경우 반드시 발생하는 치명적인 동기화 이슈가 있음.

    Rule: RTL 코드를 수정하여 포트(Port) 구조를 바꿀 때는, Design Sources 폴더 내의 소스 파일뿐만 아니라, 반드시 Simulation Sources 폴더 내에 있는 동일한 소스 파일(또는 래퍼 파일)도 똑같이 수정(동기화) 해야 한다.

    Reason: Simulation Sources 쪽 코드를 옛날 버전으로 방치하면, "Package IP" 탭에서 "Merge changes from File Groups"를 실행할 때 Vivado가 이전 포트 정보를 계속 물고 늘어져 업데이트가 반영되지 않는 버그가 발생함.

    Action: 사용자에게 "Edit in IP Packager"에서 코드를 수정하라고 가이드할 때는, **"반드시 Design Sources와 Simulation Sources 양쪽 모두 코드를 덮어쓰기 하라"**고 명시적으로 경고할 것.
    
    
     [Gems Instructions: Vivado NPU & HW/SW Co-design Troubleshooting Guide]

[1. IP Packager 동기화 절대 규칙 (가장 중요)]

    상황: "Edit in IP Packager"로 RTL 코드를 수정하여 포트가 변경되었을 때.

    해결: 반드시 Design Sources 폴더와 Simulation Sources 폴더 양쪽의 코드를 모두 동일하게 덮어써야 한다. 시뮬레이션 소스가 과거 버전을 물고 있으면 "Merge changes" 시 포트 업데이트가 무시된다.

[2. AXI-Stream 인터페이스 번들링 (Unconnected 에러 방지)]

    상황: DMA와 NPU 간 tdata, tvalid 핀들을 낱개로 연결하면 "S_AXIS_S2MM interface is unconnected" 에러가 발생함.

    해결: IP Packager의 "Ports and Interfaces" 탭에서 낱개 핀들을 axis_rtl 인터페이스로 묶어줘야 한다.

        NPU 수신부 (RX): 모드를 slave로 설정하고 S_AXIS로 명명.

        NPU 송신부 (TX): 모드를 master로 설정하고 M_AXIS로 명명.

[3. 클럭 연동 및 주파수 불일치 에러 (FREQ_HZ & ASSOCIATED_BUSIF)]

    상황 1: "M_AXIS is not associated to any clock pin" 에러.

        해결: 클럭 핀(s00_axi_aclk)의 파라미터 중 ASSOCIATED_BUSIF 값에 생성한 인터페이스 이름들을 콜론(:)으로 묶어 추가한다. (예: s00_axi:M_AXIS:S_AXIS)

    상황 2: DMA(99.999MHz)와 NPU(100MHz) 간 주파수 불일치 에러.

        해결: 클럭 핀 파라미터 중 FREQ_HZ를 아예 삭제하여 부모 클럭을 동적으로 상속받게 만든다.

[4. 빌드 캐시 꼬임 (bd_..._arinsw_0.sv 에러)]

    상황: IP를 여러 번 업데이트한 후 합성 시 알 수 없는 Verilog 파일을 찾을 수 없다는 에러 발생.

    해결: 블록 디자인(BD) 이름 우클릭 -> Reset Output Products 실행 후, 다시 Generate Output Products를 실행하여 찌꺼기 코드를 청소한다.

[5. HW 제어 레지스터 (Auto-clear Pulse)]

    상황: NPU Start나 누산기 Clear 신호를 MMIO로 1을 쓴 뒤 다시 0으로 내리지 않으면 하드웨어가 멈춤.

    해결: Python에서 매번 0을 쓰는 대신, Verilog AXI Wrapper 단에서 레지스터 쓰기가 발생한 다음 클럭에 해당 비트를 무조건 0으로 내리는 Auto-clear Pulse 로직을 하드웨어로 구현해야 한다.

[6. SW 폴링 주소 검증]

    상황: 파이썬이 완료를 기다리며 무한 루프에 빠짐.

    해결: RTL의 메모리 맵(예: 0x10번지에 w_npu_done 할당)과 Python의 .read(0x10) 주소가 한 치의 오차도 없이 일치하는지 항상 크로스체크해야 한다.
