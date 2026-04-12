# uXC Documentation Index

This directory contains all design documents for the uXC (micro eXcelerator Core) NPU project.

---

## 1. Hardware Architecture

| File | Description |
|------|-------------|
| [FPGA_NPU_Architecture_v2.md](FPGA_NPU_Architecture_v2.md) | Top-level NPU architecture — block diagram, memory hierarchy, data flow for all engines |
| [HW_Optimization_DSP48E2.md](HW_Optimization_DSP48E2.md) | DSP48E2-level optimization notes: constant folding, bit-shift tricks, RMSNorm scale cancellation |

---

## 2. ISA & Driver

| File | Description |
|------|-------------|
| [ISA.md](ISA.md) | 64-bit VLIW ISA specification: opcode table, instruction encoding, memory routing, uop structures |

---

## 3. Gemma 3N E4B Model Analysis

These documents describe the mathematical behavior and hardware constraints of the target model.
Read before implementing any compute pipeline.

| File | Description |
|------|-------------|
| [GEMMA_3N_E4B.md](GEMMA_3N_E4B.md) | Comprehensive Gemma 3N E4B model analysis: weights, layers, quantization, KV cache |
| [Gemma3N_Pipeline_EN.md](Gemma3N_Pipeline_EN.md) | Full pipeline mathematical specification (English) — token to logit, all 35 layers |
| [Attention_RoPE.md](Attention_RoPE.md) | Attention constraints: no scaling, no softcap, alternating RoPE theta |
| [FFN_Sparsity.md](FFN_Sparsity.md) | Gaussian Top-K sparsity (0.95) in FFN layers 0–9 |
| [PLE_LAuReL.md](PLE_LAuReL.md) | LAuReL parallel calibration and PLE shadow-stream injection rules |

---

## Reading Order

**Hardware engineers starting on RTL:**
1. `FPGA_NPU_Architecture_v2.md` — full system understanding
2. `ISA.md` — instruction set before touching the controller
3. `HW_Optimization_DSP48E2.md` — DSP tricks for compute pipelines

**Understanding the target workload:**
1. `Gemma3N_Pipeline_EN.md` — ground truth mathematical spec
2. `Attention_RoPE.md`, `FFN_Sparsity.md`, `PLE_LAuReL.md` — critical constraint details
3. `GEMMA_3N_E4B.md` — model internals deep-dive

---

## Project Structure

```
hw/rtl/          SystemVerilog RTL (synthesizable only, Vivado 2024.1)
sw/driver/       uXC HAL driver — AXI-Lite MMIO + high-level inference API
sw/gemma3NE4B/   Gemma 3N E4B inference application (calls uXC driver API)
docs/            This directory
```
