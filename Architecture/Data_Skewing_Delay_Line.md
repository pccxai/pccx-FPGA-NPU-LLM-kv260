### 3. `Data_Skewing_Delay_Line.md` (The secret of Wavefront execution)

# Data Skewing & Delay Line (Wavefront Execution)

## 1. Why Do We Need Delay?
In a Systolic Array, data shifts one step to the right and bottom in each clock cycle. If we inject the A-matrix data into all rows 'simultaneously' without any delay, the PEs at the bottom will compute with garbage values since the partial sums from the top haven't arrived yet.

To create a proper **Diagonal Wavefront**, **data destined for lower rows and rightmost columns must depart later.**

## 2. Shift Register (Delay Line) Structure
Calculating these precise timings via software-like FSM counters creates massive control complexity. Instead, we implemented physical delay circuits by cascading D-Flip-Flops (D-FF) like a conveyor belt.

```mermaid
graph LR
    subgraph "Data Skewing (A Matrix)"
        FSM["FSM (Simultaneous Fire!)"]
        
        FSM -->|Delay 0| PE00["PE (0,0)"]
        
        FSM -->|1 Cycle Delay| FF1["D-FF"]
        FF1 --> PE10["PE (1,0)"]
        
        FSM -->|2 Cycle Delay| FF2_1["D-FF"]
        FF2_1 --> FF2_2["D-FF"]
        FF2_2 --> PE20["PE (2,0)"]
    end
    
    style FF1 fill:#f96,stroke:#333
    style FF2_1 fill:#f96,stroke:#333
    style FF2_2 fill:#f96,stroke:#333
```