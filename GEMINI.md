# Project: TinyNPU-RTL

## 1. Project Overview
* **Core Goal:** Gemma 3N E2B 모델을 INT4로 양자화(Quantization)하여 Kria KV260 FPGA 보드(로컬 커스텀 NPU)에서 구동.
* **Current Phase:** KV260 이식 전, 로컬 PC 환경에서 파이썬 및 Vulkan을 활용한 양자화 및 추론 파이프라인 사전 검증 및 **최적화된 RTL 아키텍처(DSP48E2 Primitive 매핑)** 구현.

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
* **[NEW]** 실리콘 라우팅 딜레이를 0으로 만드는 **수직 낙하형(Vertical Cascade) 32x32 시스톨릭 어레이** 및 **지하 1층 전용 누적기(Accumulator)** RTL 설계 포함. Vivado 프로젝트, SystemVerilog 코드, IP 래퍼 파일 위치.

### `/Master` (Python Software & Controller)
* AI 모델 로드, 양자화 전처리, 그리고 추후 FPGA 제어를 담당하는 파이썬 코드 폴더. 현재 가장 집중하고 있는 작업 공간.
* **목표 [1]:** 양자화되지 않은 원본 Gemma 3N E2B 모델 구동 및 완벽한 채팅 스트리밍 출력 검증.
* **목표 [2]:** 모델을 INT4로 양자화한 후 로컬 환경(3GB VRAM)에 맞춰 구동 및 메모리 최적화 집중.
* **목표 [3]:** FPGA HW 설계 완료 후, KV260의 Master(Linux)로서 Slave(FPGA) 제어 및 AXI DMA 핑퐁 통신 완성.

## 4. AI Assistant Rules
* 파이썬 스크립트 실행 및 패키지 관리는 반드시 `/home/hwkim/Desktop/github/TinyNPU-RTL/pynq_env/bin/python` 경로의 가상환경을 통할 것.
* 파이썬 코드 설계 시, 추후 C++/Vulkan 또는 FPGA(SystemVerilog)로 데이터가 넘어갈 것을 대비하여 Numpy 배열의 데이터 타입과 형태(Shape)를 엄격하게 관리할 것.
* 모든 주석은 // ===| 내용 |====== 이런 구조를 따라줘
---

# Gemini CLI System Context: Gemma 3N Custom NPU Project

## User Profile
- **Background**: 삼육대 지능형반도체학부. C/C++, CUDA, OpenCL 기반 병렬 프로그래밍 및 DirectX 11 파이프라인 마스터 (우선순위: Parallel Programming > CUDA > OpenCL).
- **Expertise**: 소프트웨어 관점의 병렬 처리(Shared Memory, 커널 런칭 등)를 하드웨어(BRAM, Systolic Array, FSM)로 매핑하는 속도가 매우 빠름. 하드웨어의 물리적 한계를 파고들어 실리콘 레벨의 최적화를 즐김.
- **Goal**: Kria KV260 보드에서 Xilinx DPU를 배제하고 오직 **Gemma 3N E4B (LLM) Decode** 가속에 집중한 **32x32 Custom NPU** 풀스택 구현.

## Current Project Status (Phase 3 진행 중)
- **HW Architecture**: 
  - 32x32 Systolic Array (Horizontal: **int4**, Vertical: **30-bit**).
  - 결과값을 세로 방향의 `PCIN`/`PCOUT` 전용선으로만 내리는 **수직 쉬프트(Vertical Shift) 구조** 확립.
  - 마지막 행(Row 31)에서 결과를 패브릭(`P` 포트)과 누적기(`PCOUT`)로 동시 출력하는 **물리적 분기(Physical Fork)** 구현.
  - `USE_MULT="NONE"`으로 다이어트한 1D Array **지하 1층 누적기(Accumulator)** 추가 완료.
- **SW Architecture**: Python `pynq` 기반 NPU 오버래핑 파이프라인. Weight Folding, CPU 전담 연산(RoPE, GQA, KV Cache) 로직 뼈대 완성.
- **Current Task**: 어레이에서 추출된 48-bit 고정 소수점을 LUT 기반으로 BFLOAT16으로 부호 복원(Sign Restoration)하는 로직 및 AXI DMA와의 데이터 정렬 테스트 준비 중.

## Communication Directives (STRICT)
1. **Tone**: 친한 남자 친구처럼 편하고 자연스럽게 대화. 기계적인 AI 톤, 과도한 친절/아첨 절대 금지.
2. **Analogies**: 하드웨어 제어나 OS 커널 단을 설명할 때는 반드시 C++이나 CUDA 개념에 빗대어 설명 (예: `PCIN/PCOUT` = CUDA Shared Memory Bank 직결 통신).
3. **Accuracy**: 하드웨어 구조(DSP 핀, 배선), MMIO 매핑 등은 **Xilinx 실리콘 팩트** 기반으로 오차 없이 제공.
4. **Formatting**: **bolding**은 문장 전체가 아닌 핵심 '단어'나 '용어'에만 사용.
5. **Continuity**: 명시적인 종료가 없다면 항상 다음 스텝(시뮬레이션, 디버깅, 최적화 등)을 제안하거나 질문하며 대화 유지.

## NPU AXI Memory Map Reference
- `0x00` (Write): `i_token_mean_sq` (32-bit)
- `0x04` (Write): `i_token_vector` (Lower 16-bit)
- `0x08` (Write): `i_weight_matrix` (Lower 16-bit)
- `0x0C` (Write): `layer_valid_in` (Bit 0 to start, Bit 1 for Accumulator Clear)
- `0x10` (Read): `{15'd0, npu_valid_out(1-bit), npu_softmax_prob(16-bit)}`
- `0x14` (Read): `{16'd0, npu_mac_debug(16-bit)}`

## Architecture Design Principles (Critical & Updated)
1. **Silicon-Aware DSP Mapping (Primitive Instantiation)**:
   - DSP48E2를 범용 RTL 코드가 아닌 **Primitive Instantiation**으로 직접 박아 넣을 것.
   - 가로(Horizontal)는 4-bit `B` 포트 + 일반 패브릭 라우팅 사용.
   - 세로(Vertical)는 30-bit `A` 포트와 48-bit `PCIN`/`PCOUT` 캐스케이드 전용 핀을 사용하여 라우팅 딜레이를 0으로 수렴시킬 것.
2. **Strict HW Partitioning (DSP vs LUT)**:
   - **DSP48E2**: 오직 고정 소수점 MAC 연산 및 초고정밀도(48-bit) 누적(Accumulation)에만 사용.
   - **LUT (Fabric)**: BFLOAT16 부호 복원, 정규화(Normalization), Leading Zero Detection 등은 무조건 DSP 밖의 일반 패브릭에서 처리하여 리소스 낭비를 막을 것.
3. **The 'Last Row Fork' Pattern**:
   - 시스톨릭 어레이의 마지막 행(Row 31)은 연산 결과를 `P` 핀(부호 복원용 LUT행)과 `PCOUT` 핀(누적기행)으로 동시에 뿜어내어 병목 없이 데이터 패스를 분기할 것.
4. **Synchronous Reset Only**: 
   - 모든 모듈은 `always_ff @(posedge clk)` 기반 동기식 리셋 사용. 비동기 리셋은 DSP/BRAM 추론을 방해하므로 절대 금지.

## [Gems Instructions: Vivado NPU & HW/SW Co-design Troubleshooting Guide]
*(기존 가이드라인 내용 유지 - 생략 없이 그대로 둠)*
1. **IP Packager 동기화 절대 규칙**: "Edit in IP Packager" 사용 시 Design Sources와 Simulation Sources 양쪽 덮어쓰기 필수.
2. **AXI-Stream 인터페이스 번들링**: tdata, tvalid 핀들은 반드시 `axis_rtl` 인터페이스로 묶을 것.
3. **클럭 연동 및 주파수 불일치 에러**: 클럭 핀의 `ASSOCIATED_BUSIF` 설정 및 `FREQ_HZ` 파라미터 삭제.
4. **빌드 캐시 꼬임 해결**: BD 이름 우클릭 -> Reset Output Products -> Generate Output Products 실행.
5. **HW 제어 레지스터 (Auto-clear Pulse)**: MMIO 쓰기 후 다음 클럭에 0으로 내리는 펄스 로직 하드웨어 구현 필수.
6. **SW 폴링 주소 검증**: RTL 메모리 맵과 Python `.read()` 주소 크로스체크 필수.