# TinyNPU-Vulkan-Engine: Bare-metal LLM Accelerator

![Vulkan](https://img.shields.io/badge/Vulkan-Raw_API-red?logo=vulkan)
![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.x-yellow?logo=python)
![Hardware](https://img.shields.io/badge/Target-FPGA_KV260-orange)

This project is a bare-metal hardware acceleration simulator for Large Language Model (LLM, Gemma 3N INT4) inference, built entirely from scratch without relying on high-level deep learning frameworks.

Designed with future FPGA (Xilinx KV260) and NPU (Neural Processing Unit) deployment in mind, this engine analyzes hardware memory bottlenecks at the software (C++/Vulkan) level and fully verifies the HLS Dataflow pipeline through simulation.

## Key Optimizations

To push the physical limits (Memory Bandwidth) of an integrated GPU (AMD Ryzen 4500U with Radeon Vega 6) in a UMA (Unified Memory Architecture) environment, the following optimization techniques were applied:

### 1. Raw Vulkan Compute Shader & Zero-Copy Memory
* Replaced heavy Python/Taichi wrappers with direct Raw Vulkan API control in C++.
* Mapped Zero-Copy Persistent Buffers using `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, effectively reducing VRAM data transfer overhead (`memcpy`) between the CPU (RAM) and iGPU to near zero.

### 2. 128-bit Vectorized Memory Access (uvec4)
* To resolve the iGPU's `Texture Addresser` bottleneck, memory access within the GLSL compute shader was vectorized from 32-bit (4-byte) units to 128-bit (`uvec4`, 16-byte) units.
* Maximized the efficiency of the GPU memory controller, reaching the physical bandwidth limit.

### 3. Asynchronous Ping-Pong Double Buffering (Dataflow)
* Implemented a Ping-Pong Buffer structure to simulate `#pragma HLS DATAFLOW` behavior in software.
* Utilized C++ `std::async` background threads. While the iGPU performs matrix multiplication with Buffer A, the CPU prefetches the next weight matrix into Buffer B. This perfectly hides memory copy latency behind hardware computation time.

### 4. Static KV Caching (Zero-Allocation)
* To prevent dynamic memory reallocation overhead and memory leaks in Python as the context length grows, a massive static cache buffer is pre-allocated during the initial model load.

---

## Hardware Profiling Results

Profiling on the AMD Ryzen 4500U proves that the engine perfectly reached the physical memory bandwidth limit (Memory Bound):

* Memory Clock: 82.14% (RAM is operating at peak speed to feed weights to the GPU).
* Texture Addresser: 62.50% (Significantly stabilized from 80% after the 128-bit `uvec4` optimization).
* Shader Clock: 13.56% (The iGPU computation cores are highly optimized and waiting for data, demonstrating an absolute Memory Wall).
* VRAM Usage: 19.15% (Minimal VRAM footprint achieved via Zero-Copy).

## Future Work
* Based on the Ping-Pong Buffer and Vectorized Memory Access architectures verified in this project, the codebase will be ported to Vitis HLS C++ and synthesized into RTL targeting the Xilinx KV260 FPGA.