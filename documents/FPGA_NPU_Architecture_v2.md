# Gemma 3N FPGA Custom NPU Architecture (400MHz Target)

This document describes the core architecture of a custom NPU redesigned from scratch to run the Gemma 3N E2B model at full 400MHz speed on the Kria KV260 board. (The previous 100MHz design based on Ping-Pong BRAM and AXI4-Lite has been discarded)

## 1. Top-Level Data Flow (Conveyor Belt Architecture)

The core philosophy of this NPU is **"Maximizing Memory Bandwidth"** and **"Perfect Overlapping of Computation and Loading (Zero-Bubble)"**.

### 1.1 Memory I/O Engine (`memIO_Engine.sv`)
* **Lane Orchestration:** Centrally and dynamically controls four 128-bit AXI-Stream ports (HP0~HP3). Eliminates bottlenecks by flexibly switching the direction (Input/Output) of ports (e.g., 4:0 -> 3:1) without CPU intervention.
* **Header Parsing:** Recognizes the first 128 bits of the data stream as a header, determines whether the packet is a Feature Map (BF16) or a Weight (INT4), and routes it to the appropriate internal bus (Lane).
* **Per-Column e_max Extraction:** Extracts 32 unique exponents (`e_max`) for each column from the 32 Feature Map inputs and sends them down the Delay Pipe. (Used to restore the result after computation is complete)

### 1.2 Elastic Buffering (XPM FIFO)
* To withstand the tight timing (WNS) of 400MHz and absorb the irregular data supply (Jitter) from the CPU, `XPM_FIFO_AXIS`, a Xilinx-specific hardware macro, is placed on all input/output ports (acting as a magazine) to supply data stably.

## 2. Feature Map Cache & Distribution
* **SRAM Caching (`stlc_fmap_cache.sv`):** During Gemma's Decode phase (1x2048 GEMV), the Feature Map is continuously reused. To this end, a 1x2048 size XPM BRAM cache was built.
* **BF16 to Fixed Pipeline:** Before being stored in the cache, the mantissa is extracted and aligned (27-bit Mantissa) from the BF16 data via a pipeline shifter.
* **32-Lane Broadcast & Staggered Delay:** A single piece of data read from the cache is copied (Fan-out) to 32 columns, and then poured down like rain into the `V_in` (Vertical Input) of the systolic array after passing through a staggered (0~31 clock) delay line.

## 3. The Core: Unified GEMM/GEMV Systolic Array
A 32x32 size PE (Processing Element) array, designed utilizing the Xilinx DSP48E2 to the extreme.

* **V_in (Vertical):** Feature Map (Mantissa) and Instructions flow from top to bottom.
* **H_in (Horizontal):** INT4 Weights flow from left to right.
* **Double Buffering & Latency Hiding:** A 3-Stage FF pipeline is in place, so that while the current operation is running, the weights of the next tile move in the background.
* **Dual B-Register Freeze (Weight-Stationary):**
  - **GEMM Mode:** Weights flow every clock.
  - **GEMV Mode:** When the `i_w_load` signal is active, the weights are frozen in the `B2` register inside the PE. Then, they are multiplied with the Feature Map pouring vertically, and 32 additions are accumulated vertically to output the matrix-vector multiplication result through `V_ACC_out`.

## 4. Post-Processing (Normalization and Restoration)
* **`stlc_result_normalizer.sv`:** Restores the 32 48-bit results popping out from the bottom of the systolic array back to the BF16 format.
* **4-Stage Pipeline:** 
  1) Sign-Magnitude Conversion (Negative Inversion)
  2) Leading One Detection (LOD, finding the position of the highest 1)
  3) Barrel Shift (Aligning to the 7th bit of the mantissa)
  4) Exponent Update (Deriving the final exponent by combining the `e_max` sent earlier by the engine and the LOD result)
* These normalized 16-bit results are compressed to 128-bit via the Result Packer and return to the CPU riding the DMA.
