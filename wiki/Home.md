# Welcome to the TinyNPU-RTL Wiki!

This wiki contains comprehensive documentation for the **TinyNPU-RTL** project, an initiative aimed at accelerating the inference of the Gemma 3N E2B LLM model on the Kria KV260 FPGA board.

## 📌 Contents

Below is a structured guide to the documentation available in this wiki, designed to provide a deep dive into the software architecture, mathematical foundations, and hardware optimizations of the Gemma 3N Custom NPU.

### 📖 [1. Gemma 3N (INT4 + AltUp) Pipeline Detailed Operation Flowchart](Pipeline_Flowchart.md)
*Formerly `dd.md` & `test.md`*
*   **Overview:** A step-by-step breakdown of the complete Gemma 3N E4B inference pipeline.
*   **Key Topics:**
    *   Core mathematical operations (Embedding, RMSNorm, GELU, RoPE).
    *   AltUp Router initialization and modality updates.
    *   Unscaled Grouped Query Attention (GQA) and KV Cache routing rules.
    *   Extreme FFN Sparsity (Layers 0~9).
    *   Layer-wise Augmented Residual Learning (LAuReL) and Per-Layer Embedding (PLE) injection.
    *   Logit decoding, Soft-capping, Repetition Penalty, and Top-P sampling.

### 💻 [2. Codebase Documentation (C++, Vulkan, Python)](Code_Documentation.md)
*Formerly `GEMMA_3N_E4B.md` Parts 1-6 & 8*
*   **Overview:** Detailed explanations of the software stack driving the local prototyping and PC simulation.
*   **Key Topics:**
    *   **C++ Acceleration Layer:** `my_accelerator.cpp` (AVX2/OpenMP SIMD kernels for GELU, RMSNorm, RoPE, Softmax).
    *   **Vulkan Compute Shaders:** `vulkan_core.cpp` and `gemv_int4_vector4.comp` (128-bit `uvec4` optimization, INT4 unpacking, Ping-Pong buffering).
    *   **CPU/IGPU Interfaces:** `CPU_CORE.py`, `CPU_MATRIX_CORE.py`, `IGPU_CORE.py` (Orchestrating hardware kernels).
    *   **Main Pipeline Orchestration:** `main.py` execution flow and state management.
    *   **Weight Loading & Conversion:** `safeTensor.py` (mmap zero-copy strategy) and `Optim_tensor_load.py` (Safetensors to `.npy` decomposition).
    *   **Quantization:** `quantize.py` (Per-row symmetric INT4 quantization) and KV Cache pre-allocation.
    *   **Chat Templates & Memory Usage:** `chat_template.jinja` serialization and `optim_tensor_size.md` baseline memory metrics.

### ⚙️ [3. FPGA NPU Engine & Architecture Design](FPGA_Architecture.md)
*Formerly `GEMMA_3N_E4B.md` Part 7 & `dd.md` mathematical optimizations*
*   **Overview:** The definitive guide to the custom 32x32 Systolic Array NPU designed for the Kria KV260.
*   **Key Topics:**
    *   **Hardware Control Layer:** `NPU_CORE.py` (MMIO register map, DMA transfers, Simulation Mode).
    *   **Systolic Array Design:** 32x32 Tiling structure, Dual B-Register Freeze (Weight-Stationary mode for GEMV/GEMM switching).
    *   **Hardware Optimizations (The "Zero-Cost" Secrets):**
        *   Skipping $\sqrt{2048.0}$ division via RMSNorm mathematical cancellation.
        *   Implementing $\sqrt{256.0}$ multiplication as a 0-cost 4-bit left shift.
        *   Offline Constant Folding for $\frac{1}{\sqrt{2.0}}$ scaling.
    *   **Data Path & Synchronization:** Ping-Pong BRAM pipeline, Latency Hiding, and In-place Memory Overwriting.

---
*Explore the links above to delve into the extreme silicon-level optimizations and software-hardware co-design principles that make TinyNPU-RTL possible.*
