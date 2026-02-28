# TinyNPU-Gemma: Transformer Hardware Accelerator (RTL)

This project documents the entire journey of building a **Full-Stack AI Accelerator**, specifically designed to run the **quantized Gemma 3-4B-IT(E4B) LLM** entirely on the physical hardware of a **Xilinx Kria KV260 FPGA board**.

It details how we scaled the design to a massive 32x32 Systolic Array and created custom math accelerators for non-linear operations (RMSNorm, Softmax, GeLU) to achieve a 38-clock latency data path.

## Table of Contents

This documentation alternates between bottom-up and top-down approaches to explain the design philosophy of TinyNPU. We highly recommend reading the files in the following order:

1. **[Overall Architecture (Architecture)](Architecture.md)**
   - The 4 Exodia Parts, the 38-clock pipeline latency, and the 512-bit data path optimized for the KV260.
2. **[The Heart of Computation (PE Unit)](PE_Unit.md)**
   - INT8 signed MAC operations, 16/32-bit accumulations, and 1-Cycle data forwarding.
3. **[Scalable Array (Systolic Array NxN)](Systolic_Array_NxN.md)**
   - Scaling to 32x32 to maximize the 1,024 DSP limit and wavefront execution strategies.
4. **[The Magic of Time (Data Skewing & Delay Line)](Data_Skewing_Delay_Line.md)**
   - Hardware automation of the wavefront execution using Shift Registers.
5. **[Breaking the Memory Bottleneck (Ping-Pong BRAM)](Ping-Pong_BRAM_Controller.md)**
   - 512-bit data packing, BRAM latency handling, and Double Buffering.
6. **[The Math Accelerators (RMSNorm, Softmax, GeLU)](Math_Accelerators.md)**
   - PWL approximation, Base-2 shifting, and LUT strategies for zero-CPU non-linear processing.
7. **[Full-Stack Integration (AXI4-Lite & PYNQ)](AXI4_Lite_and_PYNQ.md)**
   - Vivado IP packaging, MMIO communication, and Python Jupyter Notebook control.
8. **[Verification & Simulation (Testbench)](Testbench.md)**
   - Bit-True verification matching the Python (Numpy) Golden Model exactly.