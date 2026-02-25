# 테스트벤치 시뮬레이션 흐름

세 개의 testbench가 각 모듈을 검증한다.

---

## 전체 검증 구조

```mermaid
flowchart LR
    %% 전체는 가로(1행 3열) 배치
    
    subgraph tb_mac_unit["<h3>tb_mac_unit<br/>(pe_unit 검증)</h3>"]
        direction TB
        %% 제목과의 간격을 위한 투명 노드
        sep1[ ] --- T1
        style sep1 fill:none,stroke:none
        
        T1["① 리셋 (rst_n=0 → 1)"] --> T2["② i_a=2, i_b=3 → acc=6"]
        T2 --> T3["③ i_a=4, i_b=5 → acc=26"]
        T3 --> T4["④ i_a=10, i_b=10 → acc=126"]
        T4 --> T5["⑤ $display로 결과 출력"]
        T5 --> W1["⚠️ i_valid 연결 안 됨\n→ 항상 누산됨"]
    end

    subgraph tb_ping_pong["<h3>tb_ping_pong<br/>(PP-bram 검증)</h3>"]
        direction TB
        %% 제목과의 간격을 위한 투명 노드
        sep2[ ] --- P1
        style sep2 fill:none,stroke:none

        P1["① sel=0, DMA→BRAM_0[0]=10"] --> P2["② DMA→BRAM_0[1]=20"]
        P2 --> P3["③ sel=1 (스위치!)"]
        P3 --> P4["④ NPU가 sys_addr=0 → rdata=10"]
        P4 --> P5["⑤ 동시에 DMA→BRAM_1[0]=30"]
        P5 --> W2["⚠️ 동기 읽기라\n1 사이클 후 데이터"]
    end

    subgraph tb_systolic["<h3>tb_systolic<br/>(systolic_2x2 검증)</h3>"]
        direction TB
        %% 제목과의 간격을 위한 투명 노드
        sep3[ ] --- S1
        style sep3 fill:none,stroke:none

        S1["① in_valid=1, 파도 1 투입"] --> S2["② 파도 2 투입"]
        S2 --> S3["③ 파도 3 투입"]
        S3 --> S4["④ in_valid=0 (flush)"]
        S4 --> S5["⑤ PE들이 순차적으로 결과 수렴"]
        S5 --> W3["⚠️ PE(1,1)은\n2 사이클 늦게 시작"]
    end

    %% 3열 배치를 유지하기 위한 투명 연결선
    tb_mac_unit ~~~ tb_ping_pong
    tb_ping_pong ~~~ tb_systolic
```

---

## 1. tb_mac_unit — pe_unit 단독 검증

PE 하나를 단독으로 검증한다. 매 클럭 입력을 넣고 누적 결과가 올바른지 확인.

```
시나리오:
  rst_n = 0 → 1  (리셋 해제)
  Cycle 1: i_a=2, i_b=3  → acc = 0 + 6  = 6
  Cycle 2: i_a=4, i_b=5  → acc = 6 + 20 = 26
  Cycle 3: i_a=10, i_b=10 → acc = 26 + 100 = 126 ✓
```

**알려진 이슈:** `tb_mac_unit.sv`에서 `i_valid` 포트가 연결되지 않아 항상 valid 상태로 동작한다. 실제 systolic 배열과 달리 valid 제어를 테스트하지 못한다.

---

## 2. tb_ping_pong — ping_pong_bram 더블버퍼 검증

핑퐁 버퍼의 핵심: DMA 쓰기와 NPU 읽기가 **동시에** 일어나는지 확인.

```
Phase 1 (sel=0):
  DMA → BRAM_0[0] = 10
  DMA → BRAM_0[1] = 20

Phase 2 (sel=1, 스위치!):
  NPU  → sys_addr=0 → (1 사이클 후) rdata = 10  ← BRAM_0에서 읽기
  DMA  → BRAM_1[0] = 30                          ← BRAM_1에 쓰기 (동시!)
  NPU  → sys_addr=1 → (1 사이클 후) rdata = 20
```

**동기식 읽기 주의:** `simple_bram`은 동기식이므로 주소 입력 후 **다음 클럭**에 데이터가 나온다.

---

## 3. tb_systolic — systolic_2x2 행렬 연산 검증

4사이클 파도를 흘려보내며 2×2 행렬 곱 결과를 검증한다.

| 사이클 | in_a_0 | in_a_1 | in_b_0 | in_b_1 | 이벤트 |
|--------|--------|--------|--------|--------|--------|
| 1 | 1 | 0 | 1 | 0 | PE(0,0) 첫 연산 시작 |
| 2 | 2 | 3 | 3 | 2 | 파도 확산: PE(0,0), (0,1), (1,0) 가동 |
| 3 | 0 | 4 | 0 | 4 | 전체 가동: PE(1,1) 첫 연산 |
| 4 | 0 | 0 | 0 | 0 | valid=0 (flush): PE(1,1) 마지막 연산 |

**valid 전파 지연:** PE(0,0) → PE(0,1), PE(1,0)까지는 **1 사이클** 늦고, PE(1,1)은 **2 사이클** 늦게 계산을 시작한다.

```mermaid
sequenceDiagram
    participant C as Clock
    participant PE00 as PE(0,0)
    participant PE01 as PE(0,1)
    participant PE10 as PE(1,0)
    participant PE11 as PE(1,1)

    C->>PE00: Cycle 1: A=1, B=1 → acc=1
    C->>PE00: Cycle 2: A=2, B=3 → acc=7
    C->>PE01: Cycle 2: A=1(from 00), B=2 → acc=2
    C->>PE10: Cycle 2: A=3, B=1(from 00) → acc=3
    C->>PE01: Cycle 3: A=2, B=4 → acc=10
    C->>PE10: Cycle 3: A=4, B=3 → acc=15
    C->>PE11: Cycle 3: A=3(from 10), B=2(from 01) → acc=6
    C->>PE11: Cycle 4: A=4, B=4 → acc=22 ✓
```

---

## 4. 검증 환경

**EDA Tool:** Xilinx Vivado 2025.2  
**Target:** xc7z020clg400-1 (PYNQ-Z2)  
**Simulator:** XSim (Vivado 내장) + Verilator (고속 C++ 기반)

```bash
# Verilator로 빠른 시뮬레이션
verilator --cc --exe --build tb_systolic.sv systolic_2x2.sv pe_unit.sv
./obj_dir/Vsystolic_2x2
```