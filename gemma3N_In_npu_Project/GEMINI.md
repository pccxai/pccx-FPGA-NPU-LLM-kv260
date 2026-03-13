# FPGA Hardware Directory Guide (Kria KV260 NPU)

## 1. Directory Purpose
* Kria KV260 보드에서 트랜스포머 모델(Gemma)의 행렬 곱셈 연산을 가속하기 위한 커스텀 NPU 하드웨어 설계 공간.

## 2. Core Components
* **Vivado Project:** 하드웨어 블록 디자인 및 비트스트림 생성.
* **SystemVerilog/RTL:** 시스톨릭 어레이(Systolic Array) 및 MAC 유닛 코어 로직 설계.
* **IP Cores:** AXI DMA, BRAM 등 마스터(Linux)와의 통신 및 메모리 제어 IP.
* **Testbench (TB):** 설계된 RTL 모듈의 기능 검증을 위한 시뮬레이션 환경.

## 3. Development Focus
* Master(Python/Linux) 환경에서 내려오는 제어 명령과 양자화된 INT4 가중치 데이터를 병목 없이 처리하는 하드웨어 아키텍처 완성.
