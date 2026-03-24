# Gemma 3N E2B FPGA Accelerator Project & Architecture Details

## 1. Project Vision
The ultimate goal of this project is to **accelerate the Inference (Prefill & Decode) process of the Gemma 3N E2B model to the extreme in hardware on the Kria KV260 board**.
We have discarded the conventional, simple AXI4-Lite based 100MHz design and rebuilt an **ultra-high-speed, ultra-low-latency Neural Processing Unit (NPU) targeting 400MHz** from scratch through physical routing optimization and pipelining.

We implement a world-class architecture that dynamically (Zero-Bubble) switches and processes the two core forms of LLM computation, **GEMM (Matrix x Matrix)** and **GEMV (Matrix x Vector)**, within a single Systolic Array.

---

## 2. Core Hardware Architecture

### 2.1 Segregated AXI Ports Strategy
To eliminate the data bottleneck between the CPU (PS) and FPGA (PL), we maximized the physical characteristics of the various AXI ports provided by the KV260 and fixed their purposes.
*   **HPC / ACP Port (Dedicated to Feature Maps):** An ultra-low latency port that utilizes Cache Coherency. After the CPU finishes an operation (e.g., RoPE) and places the activation function result (Feature Map) in the L2 cache, it is immediately snooped to the FPGA without going through DDR memory.
*   **HP0 ~ HP3 Ports (Dedicated to Weights):** High-Performance ports that boast massive bandwidth, although lacking cache synchronization. All four 128-bit bandwidths (totaling 512 bits) are devoted to infinite streaming of weights (INT4), making the array's arithmetic units absolutely Starvation-Free.
*   **HPM Port (Dedicated to Control):** Physically separates the Data Plane and the Control Plane. Issues "Start", "Stop", "Instructions (VLIW)", etc., to the NPU's Central Controller (Global FSM) via AXI4-Lite.

### 2.2 Extreme Hardware Technology Specifications (Deep-Dive Technologies)

Our 32x32 hybrid systolic array is not simply a logical arithmetic unit, but the result of 100% understanding and squeezing the **Physical Silicon Structure of the Xilinx DSP48E2**.

**[1] Dedicated High-Speed Lines for ACIN / ACOUT (Vertical Feature Map)**
*   **Technical Challenge:** Achieving 400MHz is impossible if 32 DSPs are connected vertically using generic Fabric Routing due to Routing Delay.
*   **Solution:** Explicitly used the `ACIN` and `ACOUT` ports, which are dedicated cascade lines physically hardwired inside the DSP48E2. Only the top row (`IS_TOP_ROW=1`) receives external input via `A_INPUT="DIRECT"`, and the remaining rows 1~31 are set to `A_INPUT="CASCADE"` to drop data vertically with a latency on the order of 0.1ns.

**[2] B Port External FF Daisy Chain (Horizontal Weight)**
*   **Technical Challenge:** Due to its structure, the Xilinx DSP does not have dedicated horizontal cascade lines (like `BCIN_Horizontal`).
*   **Solution:** Placed external Fabric FFs (`out_H <= in_H`) that pass data between adjacent PEs in 1 clock cycle (Daisy Chain). We guided the Vivado Placer to place these FFs right next to the DSP's `B` port (same Slice) to perfectly capture the horizontal movement timing.

**[3] Dual B-Register Freeze (Weight-Stationary)**
*   Inside the DSP48E2, there are 2 pipeline registers named `B1` and `B2`. We created a **Dual GEMM / GEMV Mode** by separately controlling the `CEB1` and `CEB2` pins.
*   **GEMM Mode:** Weights flow horizontally every clock. (Both `CEB1` and `CEB2` enabled)
*   **GEMV Mode:** When the weights, propagated horizontally for 32 clocks, arrive in place, a single 1-clock `w_load` pulse is generated to freeze the weights in the `B2` register (Stationary Freeze).

**[4] PCIN / PCOUT Flush (The Art of Discharging Results)**
*   Extracting the 48-bit accumulated result (`P` register) of each PE through external MUXes is a massive waste of wiring.
*   At the moment the operation finishes, a 3-bit VLIW instruction dynamically changes the DSP's `OPMODE` from `P = P + M` (Accumulate) to **`P = PCIN` (Vertical Shift)**.
*   32 PEs simultaneously pass their results to the `PCIN` of the DSP below, transforming into a massive 48-bit vertical shift register.

**[5] 3-Bit VLIW & Event-Driven Latch (Instruction Optimization)**
*   Removed the heavy `case` statement decoder and adopted a 3-bit VLIW format of `[Inst[2]: Flush, Inst[1]: GEMV/GEMM, Inst[0]: Calc/Idle]`, mapping it directly to hardware pins (`CEP`, `CEB2`, `OPMODE`).
*   In particular, applied an **Event-Driven Latch (`inst_valid_in_V`)** that shoots instructions only when they change, rather than every clock, converging the Toggle Rate (power consumption) of the instruction wiring branching out to 1024 PEs to 0.

**[6] Staggered Delay Line**
*   To match the diagonal Wavefront input of the systolic array, a delay line based on an FF chain was built on the Feature Map broadcast line. (Row 0 has 0 clock delay, Row 31 has 31 clock delay). Eliminated bottlenecks by implementing delay purely with FFs without the complex multi-port control of BRAM.

### 2.3 Feature Map Cache & Post-Processing (BFP & Normalization)
*   **SRAM Caching (`stlc_fmap_cache.sv`):** During the GEMV (Decode) phase, the 1x2048 Feature Map is reused continuously. This was cached in ultra-high-speed `XPM_MEMORY_SDPRAM` (BRAM) and broadcast (Fan-out) to 32 columns simultaneously to break through bandwidth limitations.
*   **Block Floating Point (BFP):** Extracts and stores a unique exponent (`e_max`) for each column from the 32 BF16 inputs. The mantissa of the BF16 passes through a 3-Stage Barrel Shifter, is aligned to 27-bit fixed-point, and then fed into the arithmetic unit.
*   **Result Normalization (Restoration):** The 48-bit result discharged from the bottom of the array is perfectly restored to the BF16 format via a pipelined process [Negative Inversion -> LOD (Find highest 1) -> Barrel Shift -> Exponent Update] using the stored `e_max`.

### 2.4 Zero-Bubble & Latency Hiding
*   During the 32 clocks while the operation of the current tile is in progress, **in the background, the next 32x32 weight tile is already moving horizontally (Fabric FF)** and waiting to be loaded at the doorstep (`B1`) of each PE.
*   Taking advantage of the brief 4~6 clock gap (`flush_sequence` shift register used) when the previous operation result is discharged (Flush), the pre-arrived weights overwrite the `B2` register. **In other words, the arithmetic unit wait time (Bubble) converges to 0.**

---

## 3. Remaining Development Roadmap (Roadmap: Vibe Coding + Hard Coding)

Our method is one where **AI (Gemini) proposes the RTL skeleton and optimization logic (Vibe Coding), and a human engineer (hwkim) completes it by combining physical routing, debugging, and port mapping (Hard Coding)**.

### Step 1: Implement NPU Central Controller (Global FSM) [In Progress]
*   **Goal:** Design the Brain that receives VLIW instructions coming through HPM and issues SRAM Cache read timing, Weight load timing, and instructions to the Systolic Array.

### Step 2: System Validation & WNS Tuning
*   **Goal:** Run Synthesis and Implementation in Vivado for 400MHz synthesis, and confirm that the WNS (Worst Negative Slack) is above 0.
*   **Vibe/Hard Coding Role:** Analyze Timing Violation occurrence points (Critical Paths) and provide Register Slicing (Adding FFs) solutions. Visually confirm Data Hazards through Testbench simulations and debug.

### Step 3: PS-PL Software Stack (Python PYNQ) Integration
*   **Goal:** Utilize the pynq library to shoot data via DMA, send instructions via MMIO, and extract actual results. Write a Numpy-based INT4/BF16 quantization preprocessing script.

### Step 4: CPU Offloading (Extra Feature)
*   **Goal:** If FPGA resources remain, hardwire the RoPE operation, the heaviest part of the Attention calculation, into the FPGA to relieve CPU load via ACP communication.

---

*This document serves as a 'compass' to prevent losing the direction of the project, and absolutely does not violate this architectural philosophy (Physical Segregation, Stationary Freeze, Zero-Bubble) during future code implementation.*
