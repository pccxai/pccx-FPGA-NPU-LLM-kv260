# uXC — Bare-Metal Transformer Accelerator on FPGA

![WIP](https://img.shields.io/badge/Status-RTL_Complete-blue)
![SystemVerilog](https://img.shields.io/badge/RTL-SystemVerilog-green)
![Target](https://img.shields.io/badge/Target-Kria_KV260-orange)
![Quantization](https://img.shields.io/badge/Precision-W4A16_BF16-green)

Custom NPU for running the **Gemma 3N E4B** language model on the Xilinx Kria KV260 FPGA,
bare-metal at 400 MHz. No OS, no PetaLinux.

Software baseline: [llm-lite](https://github.com/hwkim-dev/llm-lite) (x64 CPU reference implementation).

---

## Architecture

```
AXI-Lite (HPM) ──► NPU Controller ──► Global Scheduler
                                              │
              ┌───────────────┬───────────────┼───────────────┐
              ▼               ▼               ▼               ▼
       Vector Core      Matrix Core       CVO Core      mem_dispatcher
       (GEMV_top)    (GEMM_systolic_top)  (CVO_top)         │
        HP2/3 weights   HP0/1 weights    stream via    L2 URAM cache
                                         CVO bridge   (114,688 × 128-bit)
              └───────────────┴────────────── ─ ─ ─ ─ ─ ─ ─ ┘
                       preprocess_fmap (ACP fmap in)
```

### Compute Engines

| Engine | Operation | Weights | Activation | Accumulator |
|--------|-----------|---------|-----------|-------------|
| Matrix Core | GEMM (prefill, projections) | INT4 via HP0/1 (32/clk) | BF16→27-bit fixed-pt | INT48 DSP48E2 |
| Vector Core | GEMV (autoregressive decode) | INT4 via HP2/3 (32/clk each) | BF16→27-bit fixed-pt | INT48 DSP48E2 |
| CVO Core | Non-linear ops (softmax, GELU, RoPE) | — | BF16 stream from L2 | BF16 |

### Memory Hierarchy

| Level | Technology | Width | Capacity |
|-------|-----------|-------|---------|
| L2 Global Cache | URAM True Dual-Port | 128-bit | 1.75 MB (14 URAMs) |
| Shape Constant RAM | BRAM | 17-bit × 3 | 64 shape entries each |
| FMap L1 Buffer | BRAM | 128-bit | 2048 entries (256 KB) |
| HP CDC FIFOs | XPM FIFO | 128-bit | 512-deep × 4 ports |

### Key Design Points

- **W4A16**: INT4 weights × BF16 activations → INT48 accumulator → BF16 normalizer → output
- **CVO DMA bridge** (`mem_CVO_stream_bridge`): L2 → 16-bit BF16 stream to CVO → L2 writeback, sequential with XPM FIFO result buffer (2048-deep)
- **e_max tracking**: Column-0 normalizer exponent is encoded as BF16 and fed to CVO for numerically stable softmax
- **AXI port usage**: HP0/1 = GEMM weights, HP2/3 = GEMV weights (32 INT4/clk each), ACP = fmap DMA in / result DMA out
- **No arbitration stalls** on L2 (true dual-port): CVO bridge wins port B over NPU DMA when active

---

## Custom ISA (64-bit VLIW)

5 opcodes. Each instruction is 64 bits: `[63:60]` opcode + `[59:0]` body.

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| `4'h0` | `OP_GEMV` | Vector × Matrix multiply |
| `4'h1` | `OP_GEMM` | Matrix × Matrix multiply |
| `4'h2` | `OP_MEMCPY` | Host DDR4 ↔ L2 DMA |
| `4'h3` | `OP_MEMSET` | Write shape constants to RAM |
| `4'h4` | `OP_CVO` | Element-wise non-linear op (exp/sqrt/GELU/sin/cos/reduce_sum/scale/recip) |

Full specification: [docs/ISA.md](docs/ISA.md)

### Softmax sequence (example — 4 instructions)
```
OP_GEMV  flags.findemax=1          ; compute attention scores, track e_max
OP_CVO   CVO_EXP  flags.sub_emax=1 ; exp(score - e_max) for each element
OP_CVO   CVO_REDUCE_SUM            ; Σ exp values → scalar at dst
OP_CVO   CVO_SCALE flags.recip_scale=1 ; divide each exp by the sum
```

---

## Repository Structure

```
hw/
  rtl/
    NPU_top.sv                  ← top-level wiring
    NPU_Controller/             ← VLIW frontend, decoder, Global Scheduler
    MAT_CORE/                   ← 32×32 systolic array, normalizer, packer
    VEC_CORE/                   ← GEMV pipeline (4 μV-Core lanes)
    CVO_CORE/                   ← CVO SFU + CORDIC unit
    PREPROCESS/                 ← BF16→fixed-pt pipeline, fmap cache
    MEM_control/                ← L2 cache, DMA dispatcher, CVO bridge, HP buffer
    Constants/                  ← `define macros + SystemVerilog packages (A→D)
    Library/                    ← BF16 math pkg, algorithms pkg, QUEUE
sw/
  driver/                       ← AXI-Lite MMIO HAL + inference API (skeleton)
  gemma3NE4B/                   ← Gemma 3N E4B application (submodule)
docs/                           ← Architecture, ISA, model analysis documents
```

---

## Constant / Package Hierarchy

Compilation order enforced by directory naming:

```
A_const_svh/   → `define only  (NUMBERS.svh, kv260_device.svh, npu_arch.svh)
B_device_pkg/  → device_pkg    (precision/type choices)
C_type_pkg/    → dtype_pkg, mem_pkg
D_pipeline_pkg → vec_core_pkg  (Vector Core config struct)
```

All RTL files include `GLOBAL_CONST.svh` which chains A through npu_arch.

---

## AXI Port Map

| KV260 Port | Direction | Width | Usage |
|-----------|-----------|-------|-------|
| HP-0 | Slave | 128-bit | Matrix Core (GEMM) weights |
| HP-1 | Slave | 128-bit | Matrix Core spare lane |
| HP-2 | Slave | 128-bit | Vector Core lane A (GEMV) |
| HP-3 | Slave | 128-bit | Vector Core lane B (GEMV) |
| ACP | Bi-directional | 128-bit | FMap DMA in + Result DMA out |
| HPM (AXI-Lite) | Slave | 32-bit | Instruction issue + status read |

---

## Implementation Status

| Block | Status |
|-------|--------|
| VLIW frontend + decoder | RTL complete |
| Global Scheduler | RTL complete |
| 32×32 Systolic Array (Matrix Core) | RTL complete |
| Result normalizer + packer | RTL complete |
| GEMV pipeline (Vector Core, 4 lanes) | RTL complete |
| CVO SFU (exp, sqrt, GELU, scale, recip, reduce_sum) | RTL complete |
| CVO CORDIC (sin, cos) | RTL complete |
| FMap preprocessing pipeline | RTL complete |
| L2 URAM cache + ACP DMA | RTL complete |
| CVO stream bridge | RTL complete |
| NPU top-level wiring | RTL complete |
| uXC driver (AXI-Lite HAL) | Skeleton only |
| Gemma 3N E4B application | Submodule |
| Simulation / verification | Not started |
| Vivado synthesis + timing closure | Not started |

---

## Documentation

See [docs/](docs/) for detailed specifications.

| Document | Description |
|----------|-------------|
| [docs/FPGA_NPU_Architecture_v2.md](docs/FPGA_NPU_Architecture_v2.md) | Full architecture reference |
| [docs/ISA.md](docs/ISA.md) | 64-bit ISA encoding, uop structures, memory routing |
| [docs/HW_Optimization_DSP48E2.md](docs/HW_Optimization_DSP48E2.md) | DSP48E2 optimization techniques |
| [docs/GEMMA_3N_E4B.md](docs/GEMMA_3N_E4B.md) | Target model internals |
| [docs/Gemma3N_Pipeline_EN.md](docs/Gemma3N_Pipeline_EN.md) | Layer-by-layer math specification |
