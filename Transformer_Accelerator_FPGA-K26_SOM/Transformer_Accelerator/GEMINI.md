# Gemini CLI System Context: Gemma 3N Custom NPU Project

## 1. Project Overview & Rules
* **Core Goal:** Gemma 3N E2B 모델 구동을 위한 Kria KV260 FPGA 기반 로컬 커스텀 NPU(32x32 Systolic Array) 풀스택 설계.
* **Architecture Strategy:** Block Floating Point (BFP) 기법을 사용하여, Feature Map은 BF16의 Mantissa를, Weight는 INT4를 사용. 400MHz 타겟의 하드웨어 타이밍 최적화 및 Pipelining 적극 도입.
* **Comment Formatting (CRITICAL):**
  - 모든 주석은 영어로 작성.
  - 다음 세 가지 포맷 중 하나를 반드시 따를 것:
    `// ===| 내용 |=======`
    `// =| 내용 |=`
    `// <><><><><><><> 내용 <><><><><>`

## 2. Updated NPU Architecture (400MHz Target)
### 2.1 Unified GEMM & GEMV Systolic Array
* **Vertical (V_in):** BF16 Mantissa (Feature Map) 하강 경로 및 Instruction 파이프라인.
* **Horizontal (H_in):** INT4 Weights 이동 경로. (Weight Stationary 로드를 위한 파이프라인 존재)
* **Dual B-Register Freeze:** Xilinx DSP48E2 내부의 B1, B2 레지스터를 분리 제어. `i_w_load` 신호를 통해 가중치를 동결(Freeze)시켜 행렬-벡터 곱(GEMV) 디코드 연산을 지원. (또한 일반적인 GEMM도 유연하게 지원)
* **Double Buffering:** 연산 중에도 백그라운드에서 다음 타일의 가중치가 이동하며 3-Stage Pipeline을 통해 Zero-Bubble 로딩 구현.

### 2.2 Memory I/O Engine (`memIO_Engine.sv`)
* **Lane Orchestration:** 128-bit AXI-Stream 포트 4개를 동적으로 Input/Output 모드로 스위칭 (4:0, 3:1, 2:2 등).
* **Header Routing:** 첫 번째 패킷 헤더를 파싱하여 FMap 캐시 또는 가중치 디스패처로 데이터를 라우팅.
* **Per-Column e_max:** 32개의 BF16 입력으로부터 각 열(Column)마다 고유한 지수(`e_max`)를 추출하여 딜레이 라인에 태워 보냄.

### 2.3 Feature Map Cache & Post-Processing
* **FMap SRAM Cache:** 1x2048 크기의 Feature Map을 XPM BRAM에 캐싱하고, 32개 열로 동시에 브로드캐스팅(Fan-out)하여 메모리 병목 현상 제거.
* **Result Normalization:** 시스톨릭 어레이의 각 열에서 나온 48-bit 결과를 딜레이된 개별 `e_max`를 사용하여 Sign-Magnitude -> LOD -> Shift -> Exp Update 단계를 거쳐 BF16 포맷으로 복원 (Pipelined).

## 3. Communication Directives (STRICT)
1. **Tone**: 친한 남자 친구처럼 편하고 자연스럽게 대화. 과도한 친절이나 기계적인 AI 톤 금지.
2. **Analogies**: 하드웨어 제어/메모리 구조 설명 시 C++/CUDA 개념에 빗대어 설명.
3. **Accuracy**: 100% 팩트 기반. 400MHz WNS 타이밍 달성을 위해 언제나 물리적 배선 지연을 고려한 FF 배치를 최우선으로 할 것.

## 4. Strict Rules: FPGA_debugging.txt
* 코드를 수정하거나 Vivado 에러가 발생할 때, 혹은 보드(KV260) 셋업 관련 특이사항을 발견할 때마다 반드시 프로젝트 내 `FPGA_debugging.txt` 파일에 기록(메모)할 것.
* **기록 양식:** [시도한 내용] - [발생한 문제/에러 로그] - [해결 방법 및 결과]
* 동일한 실수를 반복하지 않도록 작업 전 `FPGA_debugging.txt`를 확인하여 컨텍스트를 유지할 것.