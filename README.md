# TinyNPU-RTL: Gemma 3N E2B LLM Decode Accelerator

## 1. Project Overview
This project is a full-stack Edge AI accelerator designed explicitly to run **Gemma 3N E2B (LLM)** on the **Kria KV260** FPGA board. 
Bypassing standard vendor DPU solutions, we built a 100% **`custom 32x32 Systolic Array NPU`** focused entirely on maximizing the LLM's **Decode Phase (T=1)** performance using a Hardware/Software Co-design approach.

## 2. Hardware Architecture (RTL)
The NPU core is designed in **SystemVerilog** to achieve a 100MHz clock with 0 setup/hold timing violations.
* **1024-Core Systolic Array:** A 32x32 MAC grid optimized for massive Matrix-Vector Multiplication (GEMV) used in Q, K, V, and FFN projections.
* **True Dual-Port Ping-Pong BRAM:** Completely hides memory latency by overlapping AXI DMA data fetching with NPU matrix computation (similar to CUDA Streams).
* **On-the-fly RMSNorm Scaling:** Directly computes `inv_sqrt` via a highly optimized DSP48E2 pipeline and scales tokens in 1-cycle as they stream from BRAM.
* **1-Cycle Hardware Activation LUTs:** Completely offloads heavy non-linear functions (GeLU, Softmax) from the CPU to dedicated hardware ROM LUTs.

## 3. Software Architecture (Python / PYNQ)
The host CPU (ARM Cortex) runs a highly optimized Python pipeline communicating with the NPU via MMIO (`0x00 ~ 0x14` registers) and AXI DMA.
* **HW/SW Partitioning:** * **CPU:** Memory-bound and scalar-heavy tasks (Tokenization, Embedding, RoPE, KV Cache Management, Grouped-Query Attention).
  * **NPU:** Compute-bound tasks (Dense Matrix Projections, FFN Gate/Up/Down, GeLU, Softmax, LM Head).
* **Weight Folding (Gamma Fusion):** RMSNorm `gamma` parameters are mathematically pre-fused into the linear projection weights during the `safetensors` model loading phase, saving NPU pipeline stages and memory bandwidth.
* **Singleton MMIO Control:** A robust `MMIO.py` interface maps directly to physical AXI4-Lite registers, acting as the "Kernel Launcher" for the FPGA.

## 4. Directory Structure
```text
├── Architecture/         # Diagrams and detailed hardware specifications
├── src/                  # SystemVerilog RTL Sources
│   ├── gemma_layer_top.sv    # Top-level NPU Wrapper (AXI FSM + Routing)
│   ├── systolic_NxN.sv       # 32x32 MAC Array
│   ├── ping_pong_bram.sv     # Double Buffering Controller
│   └── rmsnorm_inv_sqrt.sv   # Hardware RMSNorm Inverse Square Root
├── software/             # Python Host Code (PYNQ)
│   ├── main.py               # Inference Pipeline Entry Point
│   ├── CPU_CORE.py           # Host-side Pre/Post-processing
│   ├── NPU_CORE.py           # FPGA Kernel Dispatcher (MMIO & DMA)
│   └── safeTensor.py         # Local Model Loader & Weight Folding

└── .gitignore

