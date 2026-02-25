# TinyNPU-RTL: From C++ to Silicon

This project documents the entire journey of building a **Full-Stack AI Accelerator**, starting from a simple matrix multiplication algorithm in C++, designing it into hardware (Verilog), and finally controlling it via Python software through the AXI interconnect.

## Table of Contents

This documentation alternates between bottom-up and top-down approaches to explain the design philosophy of TinyNPU. We highly recommend reading the files in the following order:

1. **[Overall Architecture (Architecture)](Architecture.md)**
   - The limitations of the Von Neumann architecture and the necessity of the Systolic Array.
2. **[The Heart of Computation (PE Unit)](PE_Unit.md)**
   - The principles of MAC operations and 1-Cycle data forwarding.
3. **[Scalable Array (Systolic Array NxN)](Systolic_Array_NxN.md)**
   - Evolution from a hardcoded 2x2 array to an NxN array using SystemVerilog `generate` statements.
4. **[The Magic of Time (Data Skewing & Delay Line)](Data_Skewing_Delay_Line.md)**
   - Hardware automation of the wavefront execution using Shift Registers.
5. **[Breaking the Memory Bottleneck (Ping-Pong BRAM)](Ping-Pong_BRAM_Controller.md)**
   - Double Buffering technique to overlap DMA transfers and NPU computations.
6. **[Full-Stack Integration (AXI4-Lite & PYNQ)](AXI4_Lite_and_PYNQ.md)**
   - Vivado IP packaging, MMIO communication, and Python Jupyter Notebook control.
7. **[Verification & Simulation (Testbench)](Testbench.md)**
   - Strategies for verifying hardware timing and waveforms.