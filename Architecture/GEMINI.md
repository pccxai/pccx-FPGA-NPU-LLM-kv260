# Architecture Directory Guide (System Design & Flow)

## 1. Directory Purpose
* 프로젝트 전체 구조(소프트웨어-하드웨어 간 인터페이스) 및 데이터 흐름을 정의하는 마크다운 문서 보관 공간.
* 시스템 다이어그램, 메모리 맵, 통신 프로토콜 설계도 포함.

## 2. Key Documentation Areas
* **Python to FPGA Flow:** 파이썬(Master)에서 Numpy 전처리 후 AXI DMA를 통해 FPGA(Slave)로 데이터를 쏘고 받는 전체 파이프라인.
* **VRAM & RAM Management:** 3GB VRAM과 16GB 시스템 램 사이의 모델 가중치 분할 및 로드 전략.
* **Hardware Architecture:** KV260 NPU 내부 시스톨릭 어레이 및 MAC 유닛의 데이터 패스 구조.
