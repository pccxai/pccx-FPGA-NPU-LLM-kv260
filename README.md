# TinyNPU-Gemma: Bare-Metal Gemma 3N LLM Accelerator on FPGA

![WIP](https://img.shields.io/badge/Status-Work_in_Progress-red)
![Vulkan](https://img.shields.io/badge/Vulkan-Raw_API-red?logo=vulkan)
![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.x-yellow?logo=python)
![Hardware](https://img.shields.io/badge/Target-FPGA_KV260-orange)

> **notice: Active Developement in Progress**
>
> This repository is currently under active development. The codebase, architecture, and features are subject to change. The RTL synthesis targeting the Xilinx KV260 FPGA (including DSP48E2 integration) is still a work in progress. It is not yet ready for use.


## Project Overview

TinyNPU-Gemma is a custom SystemVerilog-based Neural Processing Unit (NPU) engineered specifically to accelerate a quantized Gemma 3N (E2B/E4B) Large Language Model on the Xilinx Kria KV260 FPGA. The architecture is meticulously designed to push the physical constraints of the KV260 platform, which is equipped with 1,248 DSP48E2 slices and 144 Block RAMs (BRAMs).

This project encompasses a full-stack hardware-software co-design approach, integrating a SystemVerilog hardware accelerator, Python-based Golden Models for Trace-Driven Verification, CPU SIMD optimizations, and a high-performance AXI Direct Memory Access (AXI DMA) pipeline.

## System Architecture and Components

### 1. Custom ISA and Decoupled Dataflow Execution Pipeline

[![ISA Instruction Set Architecture](./images/ISA_screen_shot_0409.png)](https://docs.google.com/spreadsheets/d/e/2PACX-1vQOZ4tMXcdIpcdOCvneAx0r8wmRfmprogqkhbCTK2ythlzxp2GBromIiCi9J9yEz9G_ZO4o7BreDOoq/pubhtml?gid=584280668&single=true)

> **[Click Here to Explore the Full Custom ISA Specification (Google Sheets)]**(https://docs.google.com/spreadsheets/d/e/2PACX-1vQOZ4tMXcdIpcdOCvneAx0r8wmRfmprogqkhbCTK2ythlzxp2GBromIiCi9J9yEz9G_ZO4o7BreDOoq/pubhtml?gid=584280668&single=true)
>
> *Click the image or the link above to interactively view the detailed bit-level instruction formats and flag configurations.*
> 
> *The detailed bit-level instruction formats, flags, and memory addresses can be viewed interactively in the link above.*

The accelerator operates on a **Custom Instruction Set Architecture (ISA)** meticulously designed for LLM workload acceleration. To ensure versatile integration and compatibility with host systems, the NPU supports both **x86 and x64 modes**.

The accelerator operates on a **Custom Instruction Set Architecture (ISA)** meticulously designed for LLM workload acceleration. To ensure versatile integration and compatibility with host systems, the NPU supports both **x86 and x64 modes**. 

To maximize parallel execution and eliminate pipeline stalls, the architecture employs a strictly **Decoupled Dataflow** system, divided into two asynchronous stages:

* **Stage 1: Global Front-End:** The central `cu_npu_decoder` fetches and decodes the 64-bit custom instructions. It merely classifies the instructions by their target operation (e.g., Matrix-Matrix `MdotM` or Vector-Matrix `VdotM`) and immediately pushes them into small, dedicated **Instruction FIFOs (Instruction Queues)** located at the front of each execution pipeline. Once the instruction is dispatched, the global front-end moves to the next instruction without waiting for execution.
* **Stage 2: Local Dispatcher:** Each compute engine (featuring one `MdotM` engine and multiple `VdotM` engines) is paired with a highly lightweight `local_dispatcher`. This dispatcher pops instructions from its dedicated FIFO and checks strictly local execution conditions: *Are weights available in the Weight FIFO? Is the Feature Map (FMAP) data ready in the L1 Cache?* Once local dependencies are met, it fires the execution engine independently. This asynchronous approach ensures that a stall in one engine does not halt neighboring engines, enabling true parallel execution.

### 2. SystemVerilog NPU and Hardware Acceleration

The core compute engine is implemented in SystemVerilog, featuring a highly optimized **32x32 Systolic Array MAC Engine**. This engine is tailored to maximize the utilization of the Xilinx DSP48E2 slices, executing low-precision INT8/INT4 matrix multiplications with high throughput.

To aggressively alleviate memory bandwidth bottlenecks, the architecture implements a multi-tiered memory hierarchy that includes **L1 and L2 Caches**. These caches provide ultra-low latency access to frequently used weights and activations. They are backed by **Dual-Port Ping-Pong BRAMs** with a 512-bit wide data path. This configuration packs the upper 256-bit weights and lower 256-bit activations, enabling parallel read and write operations that perfectly hide memory copy latency behind hardware computation time.

Furthermore, custom hardware accelerators have been developed for critical non-linear functions:
* **RMSNorm Accelerator:** Executes in 1 clock cycle.
* **GeLU Accelerator:** Executes in 1 clock cycle.
* **Softmax Accelerator:** Executes in 3 clock cycles.

### DSP48E2 Architecture Utilization

The Systolic Array heavily relies on the DSP48E2 blocks to perform multiply-accumulate (MAC) operations efficiently.

![DSP48E2 Architecture](IMG_2852.jpeg)
*Figure 1: Internal architecture of the DSP48E2 slice utilized for the Systolic Array MAC Engine.*

### 3. AXI Direct Memory Access (AXI DMA)

Data movement between the Processing System (PS) and Programmable Logic (PL) is orchestrated via a high-performance **AXI DMA** interface. The DMA engine manages high-speed, zero-copy data transfers between the host CPU memory and the NPU's internal caches and BRAMs. By utilizing stream-based communication, the AXI DMA ensures that the Systolic Array is continuously fed with weights and activations, preventing pipeline stalls and reaching the physical memory bandwidth limits of the system.

### 4. CPU SIMD Optimizations and Python Golden Model

The software stack includes a highly optimized Python environment responsible for model quantization, preprocessing, and trace generation.
To accelerate software-side preprocessing and inference emulation, we leverage **CPU SIMD (Single Instruction, Multiple Data)** instructions and vectorized memory access patterns.

The bit-width strategy utilizes INT8/INT4 quantization for inputs and weights, maintaining intermediate 16-bit or 32-bit accumulation to prevent overflow before requantization.

Hardware verification relies strictly on **Trace-Driven Verification**, requiring a bit-true match (0% error rate) between the SystemVerilog RTL simulation and a Python (NumPy/PyTorch) golden model.

### 5. Gemma 3N Architecture Specifics

The accelerator is specifically tuned for the idiosyncratic architectural features of the Gemma 3N E4B/E2B model:

* **AltUp Router:** The router mixes four multi-streams (xs[0] to xs[3]). It applies a hyperbolic tangent scaling scaled by dimension (Tanh(Norm(x)/2048)*W), leaving the main stream (xs[0]) untouched.
* **RMSNorm Optimization:** The RMSNorm implementation enforces scale_plus_one=False, ensuring no arbitrary +1.0 is added to the weights, strictly conforming to the Gemma specification.
* **Top-K Extraction:** The software stack employs numpy.argpartition instead of numpy.argsort for Top-K extraction during sequence generation. This reduces algorithmic complexity from O(N log N) to O(N), significantly enhancing generation speed.
* **Gaussian Top-K Sparsity:** Feed-Forward Network (FFN) layers 0-9 apply Gaussian Top-K Sparsity (0.95), effectively zeroing out the bottom 95% via ReLU to induce sparse activation.
* **Cache Reusability:** Layers 20-34 bypass local KV cache updates, forcefully reusing caches from Layer 18 (Local) and Layer 19 (Global) to drastically reduce memory footprint.

## Verification and Testing

All hardware modules have been verified via Trace-Driven Verification. The verification suite ensures bit-exact equivalence between the Python Golden Model and the SystemVerilog RTL output.

## Current Status & Future Work

**Active Development Phase:**
* The software simulation and Vulkan-based hardware memory bottleneck profiling have been successfully verified.
* **Currently Working On:** Porting the verified dataflow pipeline to SystemVerilog/C++ for RTL synthesis targeting the **Xilinx KV260 FPGA**.
* **Hardware Integration:** Actively implementing the systolic array architecture optimizing the utilization of **DSP48E2** slices to maximize compute density and meet 400MHz timing constraints.

*Note: The hardware description files (e.g., `NPU_top.sv`, `stlc_global_fsm.sv`) in this repository are currently experimental and undergoing heavy testing.*