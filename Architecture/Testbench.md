# Testbench Simulation Flow

Three testbenches verify each respective module.

---

## Overall Verification Structure

```mermaid
flowchart LR
    %% Overall horizontal (1 row 3 columns) arrangement

    subgraph tb_mac_unit["<h3>tb_mac_unit<br/>(pe_unit Verification)</h3>"]
        direction TB
        %% Transparent node for spacing from title
        sep1[ ] --- T1
        style sep1 fill:none,stroke:none

        T1["① Reset (rst_n=0 → 1)"] --> T2["② i_a=2, i_b=3 → acc=6"]
        T2 --> T3["③ i_a=4, i_b=5 → acc=26"]
        T3 --> T4["④ i_a=10, i_b=10 → acc=126"]
        T4 --> T5["⑤ Output result via $display"]
        T5 --> W1[" i_valid is unconnected\n→ Accumulates continuously"]
    end

    subgraph tb_ping_pong["<h3>tb_ping_pong<br/>(PP-bram Verification)</h3>"]
        direction TB
        %% Transparent node for spacing from title
        sep2[ ] --- P1
        style sep2 fill:none,stroke:none

        P1["① sel=0, DMA→BRAM_0[0]=10"] --> P2["② DMA→BRAM_0[1]=20"]
        P2 --> P3["③ sel=1 (Switch!)"]
        P3 --> P4["④ NPU sys_addr=0 → rdata=10"]
        P4 --> P5["⑤ Simultaneously DMA→BRAM_1[0]=30"]
        P5 --> W2[" Synchronous read\nso data appears 1 cycle later"]
    end

    subgraph tb_systolic["<h3>tb_systolic<br/>(systolic_2x2 Verification)</h3>"]
        direction TB
        %% Transparent node for spacing from title
        sep3[ ] --- S1
        style sep3 fill:none,stroke:none

        S1["① in_valid=1, Input Wave 1"] --> S2["② Input Wave 2"]
        S2 --> S3["③ Input Wave 3"]
        S3 --> S4["④ in_valid=0 (Flush)"]
        S4 --> S5["⑤ PEs sequentially converge results"]
        S5 --> W3[" PE(1,1) starts\n2 cycles later"]
    end

    %% Transparent connecting lines to maintain the 3-column layout
    tb_mac_unit ~~~ tb_ping_pong
    tb_ping_pong ~~~ tb_systolic
```

---

## 1. `tb_mac_unit` — `pe_unit` Standalone Verification

Verifies a single PE in isolation. Inputs are provided every clock, and it checks whether the accumulated result is correct.

```
Scenario:
  rst_n = 0 → 1  (Release Reset)
  Cycle 1: i_a=2, i_b=3  → acc = 0 + 6  = 6
  Cycle 2: i_a=4, i_b=5  → acc = 6 + 20 = 26
  Cycle 3: i_a=10, i_b=10 → acc = 26 + 100 = 126 ✓
```

**Known Issue:** The `i_valid` port is unconnected in `tb_mac_unit.sv`, meaning it operates in an always-valid state. It does not test the valid control differently from the actual systolic array.

---

## 2. `tb_ping_pong` — `ping_pong_bram` Double Buffer Verification

The core of the ping-pong buffer: verify that DMA writes and NPU reads occur **simultaneously**.

```
Phase 1 (sel=0):
  DMA → BRAM_0[0] = 10
  DMA → BRAM_0[1] = 20

Phase 2 (sel=1, Switch!):
  NPU  → sys_addr=0 → (1 cycle later) rdata = 10  ← Read from BRAM_0
  DMA  → BRAM_1[0] = 30                           ← Write to BRAM_1 (Simultaneous!)
  NPU  → sys_addr=1 → (1 cycle later) rdata = 20
```

**Synchronous Read Caution:** Because `simple_bram` is synchronous, data is output on the **next clock** after the address is inputted.

---

## 3. `tb_systolic` — `systolic_2x2` Matrix Multiplication Verification

Verifies the 2x2 matrix multiplication result by flowing waves over 4 cycles.

| Cycle | in_a_0 | in_a_1 | in_b_0 | in_b_1 | Event |
|--------|--------|--------|--------|--------|--------|
| 1 | 1 | 0 | 1 | 0 | PE(0,0) starts first computation |
| 2 | 2 | 3 | 3 | 2 | Wave spreads: PE(0,0), (0,1), (1,0) activate |
| 3 | 0 | 4 | 0 | 4 | Full activation: PE(1,1) first computation |
| 4 | 0 | 0 | 0 | 0 | valid=0 (flush): PE(1,1) final computation |

**Valid Propagation Delay:** Computations at PE(0,1) and PE(1,0) are delayed by **1 cycle** compared to PE(0,0). Computations at PE(1,1) start **2 cycles** later.

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

## 4. Verification Environment

- **EDA Tool:** Xilinx Vivado 2025.2
- **Target:** xc7z020clg400-1 (PYNQ-Z2)
- **Simulator:** XSim (Built-in Vivado) + Verilator (High-speed C++ based)

```bash
# Fast simulation with Verilator
verilator --cc --exe --build tb_systolic.sv systolic_2x2.sv pe_unit.sv
./obj_dir/Vsystolic_2x2
```