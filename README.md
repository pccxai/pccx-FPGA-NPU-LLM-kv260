# TinyNPU-RTL: 2D Systolic Array CNN Accelerator

## 1. Project Overview
This project focuses on the RTL-level design and implementation of a lightweight Convolutional Neural Network (CNN) accelerator (NPU) tailored for Edge Devices. 
This project utilizes **SystemVerilog** to construct a highly optimized hardware architecture. 
The target platform is the **Zynq-7000** SoC (e.g., PYNQ-Z2). To overcome the inherent memory bandwidth bottlenecks of edge environments, the design aggressively adopts a **Systolic Array** architecture and **AXI DMA** for efficient data pipelining.

## 2. System Architecture
The system is built on a Hardware/Software Co-design approach, partitioning tasks between the Processing System (PS - ARM Cortex-A9) and Programmable Logic (PL - FPGA).

### A. Processing Element (PE)  
The PE is the fundamental computing unit, essentially a Multiply-Accumulate (MAC) block.
* **Combinational & Sequential Logic:** Performs multiplication combinationally and accumulates the result sequentially at every positive clock edge.
* **Data Flow Control:** Utilizes a `valid` signal to ensure only legitimate data is processed.
* **Forwarding:** Propagates internal data to adjacent PEs in the subsequent clock cycle, forming the basis of the pipeline.

### B. 2D Systolic Array
An architecture comprising multiple PEs arranged in a 2D grid.
* **Data Reuse:** Maximizes internal data reuse by flowing data continuously through the array like a wavefront, drastically reducing external memory access.
* **Scalability:** The initial implementation is a 2x2 array, structurally modeled to scale up to NxN configurations via parameterization.

### C. Memory Hierarchy & Bandwidth Optimization
* **Global Memory (DDR3):** Stores the large-scale original input feature maps and pre-trained weights.
* **AXI DMA:** A high-speed Direct Memory Access controller that fetches data from DDR3 to the PL without CPU intervention.
* **BRAM (Block RAM):** Ultra-fast on-chip SRAM, serving a role identical to Shared Memory in CUDA architectures.
* **Tiling & Ping-Pong Buffer:** To bypass the strict capacity limits of BRAM, large feature maps are divided into smaller tiles. A Ping-Pong buffer scheme is employed to overlap data transfer with computation.

## 3. Directory Structure
```text
├── src/
│   ├── pe_unit.sv        # PE module (MAC + Valid control + Data forwarding)
│   └── systolic_2x2.sv   # Top module (2x2 Array of PEs)
├── tb/
│   └── tb_systolic.sv    # Testbench for behavioral simulation
├── constrs/
│   └── pynq_z2.xdc       # Physical pin and timing constraints
├── .gitignore            # Excludes Vivado/Verilator generated junk files
└── README.md             # Project documentation (Main Page)  
```

## 4. Development Workflow & Tools
**Code Editor**: VS Code (Configured with SystemVerilog & TerosHDL extensions for linting and structural visualization)

**EDA Tool**: Xilinx Vivado 2025.2 (Used for RTL Synthesis, Implementation, and Waveform Simulation)

**Simulator**: Verilator (For high-speed, C++ based cycle-accurate simulation) / Vivado Behavioral Simulator

**Version Control**: Git & GitHub