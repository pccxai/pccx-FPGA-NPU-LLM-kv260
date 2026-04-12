# uXC NPU Architecture — V2 (W4A16/BF16, 400 MHz Target)

Target board: **Xilinx Kria KV260** | Bare-metal (no OS) | 400 MHz core clock

---

## 1. System Overview

The uXC (micro eXcelerator Core) NPU is a custom neural processing unit designed to run
the **Gemma 3N E4B** model on the KV260 FPGA.

Compute paradigm: **W4A16** — INT4 weights × BF16 activations, accumulating into INT48
via DSP48E2 P-registers. Precision is promoted to BF16/FP32 only inside the CVO Core
for non-linear functions.

### 1.1 Top-Level Block Diagram

```
AXI-Lite (HPM) ─────────► ctrl_npu_frontend ──► ctrl_npu_decoder ──► Global_Scheduler
                                                                              │
                                  uop dispatch per engine                     │
                 ┌────────────────────┬──────────────────┬────────────────────┤
                 ▼                    ▼                   ▼                    ▼
          Vector Core           Matrix Core           CVO Core          mem_dispatcher
        (GEMV_top)           (GEMM_systolic_top)    (CVO_top)               │
         HP2/3 weights          HP0/1 weights      stream via          L2 URAM cache
              │                      │             mem_CVO_stream_bridge      │
              └──────────────────────┴─────────────────────────────────────────┘
                              preprocess_fmap (ACP fmap in)
```

---

## 2. Memory Hierarchy

| Level | Module | Technology | Width | Purpose |
|-------|--------|-----------|-------|---------|
| L2 Global Cache | `mem_L2_cache_fmap` (inside `mem_GLOBAL_cache`) | URAM (True Dual-Port) | 128-bit | Shared fmap / result storage |
| Shape Constant RAM | `fmap_array_shape`, `weight_array_shape` | BRAM | 17-bit × 3 | Tensor shape descriptors |
| FMap L1 SRAM | `fmap_cache` | BRAM | 128-bit | Broadcast buffer per column |
| HP CDC FIFO | `mem_HP_buffer` | XPM FIFO | 128-bit × 4 ports | AXI→core clock domain crossing |
| ACP CDC FIFO | `mem_BUFFER` | XPM FIFO | 128-bit | ACP DMA staging |

**L2 address convention:** 128-bit word units. Element address ÷ 8 = word address.  
**L2 capacity:** 114,688 × 128-bit words = 1.75 MB (14 URAMs).

---

## 3. AXI Port Assignment

| Port | Direction | Width | Usage |
|------|-----------|-------|-------|
| HP-0 | IN | 128-bit/clk | Matrix Core (GEMM) weight stream |
| HP-1 | IN | 128-bit/clk | Matrix Core (GEMM) weight stream (spare lane) |
| HP-2 | IN | 128-bit/clk | Vector Core (GEMV) weight lane A |
| HP-3 | IN | 128-bit/clk | Vector Core (GEMV) weight lane B |
| ACP  | Bi-directional | 128-bit | FMap DMA in / Result DMA out (coherent) |
| HPM  | AXI-Lite slave | 32-bit | Control plane — VLIW instruction issue |

All HP ports go through `mem_HP_buffer` (XPM async FIFO for AXI→core CDC) before
reaching the compute engines.

---

## 4. Vector Core (GEMV_top)

Handles all **GEMV** (vector × matrix) operations — dominant in autoregressive decoding.

**Configuration** (`VecCoreDefaultCfg`):
- 4 parallel μV-Core lanes
- INT4 weights, BF16 activations
- 32 weights per HP port per clock
- 32-element fmap broadcast per cycle (`ARRAY_SIZE_H`)

**Data flow:**
```
HP2 (128-bit) → unpack 32 × INT4 → lane A weight
HP3 (128-bit) → unpack 32 × INT4 → lane B weight
ACP fmap → preprocess_fmap → BF16→fixed-point → broadcast[0:31]
                                                         │
                                           GEMV_generate_lut
                                           (builds LUT per weight bit)
                                                         │
                                           GEMV_reduction_branch
                                           (INT4 × fixed-pt dot-product)
                                                         │
                                           GEMV_accumulate → output
```

**Weight unpacking** is done in `NPU_top.sv` via a `generate` loop:
```systemverilog
assign gemv_weight_A[w] = M_CORE_HP2_WEIGHT.tdata[w*4 +: 4];  // for each w in 0..31
```

---

## 5. Matrix Core (GEMM_systolic_top)

Handles **GEMM** (matrix × matrix) operations — prefill phase and projection layers.

**Configuration:**
- 32×32 systolic array
- INT4 weights from HP0, 128-bit/clk (32 weights/clk)
- BF16 → 27-bit fixed-point fmap from `preprocess_fmap`
- DSP48E2 B-port: INT4 (4-bit), A-port: fixed-point (27-bit → 30-bit padded)
- P-register: INT48 accumulator

**Control:** `global_inst[2:0]` = `flags[5:3]` = `{findemax, accm, w_scale}` from `GEMM_uop`.

**Output pipeline** (one per column, 32 total):
```
raw_res_sum[n] [48-bit]
      │
gemm_result_normalizer    ← sign-mag → LOD → barrel shift → BF16
      │
norm_res_seq[n] [16-bit BF16]
      │
FROM_gemm_result_packer   ← pack 8 × BF16 → 128-bit AXI-S word
      │
M_AXIS_ACP_RESULT
```

---

## 6. CVO Core (CVO_top)

Handles non-linear activation functions that require floating-point precision:
**exp, sqrt, GELU, sin, cos (CORDIC), reduce_sum, scale, recip**.

**Data flow:**
```
OP_CVO instruction → Global_Scheduler → CVO_uop
                                              │
                                        mem_dispatcher
                                              │
                              mem_CVO_stream_bridge (inside mem_dispatcher)
                              ├─ Phase 1 READ : L2[src_addr] → 8×BF16/cycle → CVO_top
                              └─ Phase 2 WRITE: CVO results ← XPM FIFO → L2[dst_addr]
```

**e_max encoding for softmax:**
```systemverilog
// BF16 = {sign=0, exp=delayed_emax_32[0], mant=7'b0}
// Encodes 2^(exp-127) — the exponent-max from the GEMM normalizer pipeline
assign cvo_emax_bf16 = {1'b0, delayed_emax_32[0], 7'b0};
```

**Softmax sequence (4 CVO ops):**
1. `OP_GEMV` with `FLAG_FINDEMAX` — find e_max over attention scores
2. `OP_CVO CVO_EXP` with `FLAG_SUB_EMAX` — exp(x − e_max) per element
3. `OP_CVO CVO_REDUCE_SUM` — Σ exp(xᵢ − e_max) (denominator)
4. `OP_CVO CVO_SCALE` with `FLAG_RECIP_SCALE` — divide each by denominator

---

## 7. FMap Preprocessing Pipeline

Before entering the systolic array or Vector Core, BF16 feature maps from the ACP port are:

1. Received via ACP → `S_AXIS_ACP_FMAP` interface
2. Buffered in a 128-bit XPM FIFO inside `mem_BUFFER` (AXI→core CDC)
3. Processed by `preprocess_bf16_fixed_pipeline`:
   - Takes 2 clocks × 16 BF16 elements = 32-element block
   - Finds global e_max across the block
   - Shifts each mantissa to align with e_max → 27-bit fixed-point
4. Cached in `fmap_cache` (BRAM, depth 2048)
5. Broadcast to all 32 PE columns when `i_rd_start` pulses (from `sram_rd_start_wire`)

---

## 8. Memory Dispatcher (`mem_dispatcher`)

Central DMA controller. Translates Global_Scheduler micro-ops into cache commands.

**Sub-modules:**
| Module | Role |
|--------|------|
| `mem_GLOBAL_cache` | L2 URAM TDP cache + ACP DMA state machine |
| `mem_L2_cache_fmap` | XPM TDP URAM macro (114,688 × 128-bit) |
| `mem_u_operation_queue` | Command FIFOs (ACP + NPU, 35-bit uop, depth 512) |
| `mem_CVO_stream_bridge` | L2 ↔ CVO 16-bit stream adapter (READ→WRITE sequentially) |
| `fmap_array_shape` | Shape constant RAM for fmap tensors |
| `weight_array_shape` | Shape constant RAM for weight tensors |

**LOAD uop priority:** GEMM > GEMV > MEMCPY > CVO

**Port B arbitration:** `cvo_bridge_busy` wins over NPU DMA state machine.

---

## 9. Control Plane

```
Host (AXI-Lite HPM) ──► AXIL_CMD_IN ──► ctrl_npu_frontend ──► ctrl_npu_decoder
                                                                       │
                                    ┌──────────┬────────┬─────────────┤
                                    ▼          ▼        ▼             ▼
                               GEMV FIFO  GEMM FIFO  CVO FIFO   MEM/MEMSET FIFO
                                    └──────────┴────────┴─────────────┘
                                                     │
                                             Global_Scheduler
                                             (single always_ff per uop type,
                                              GEMM > GEMV > MEMCPY > CVO priority)
```

See [ISA.md](ISA.md) for the full instruction encoding reference.

---

## 10. Implementation Status

| Block | Key Files | Status |
|-------|-----------|--------|
| NPU Controller (frontend + decoder) | `NPU_Controller/NPU_frontend/`, `ctrl_npu_decoder.sv` | RTL Complete |
| Global Scheduler | `Global_Scheduler.sv` | RTL Complete |
| Matrix Core — Systolic Array | `MAT_CORE/GEMM_systolic_top.sv` | RTL Complete |
| Matrix Core — Result Normalizer | `MAT_CORE/mat_result_normalizer.sv` | RTL Complete |
| Matrix Core — Result Packer | `MAT_CORE/FROM_mat_result_packer.sv` | RTL Complete |
| Vector Core (GEMV) | `VEC_CORE/GEMV_top.sv` | RTL Complete |
| CVO Core (SFU + CORDIC) | `CVO_CORE/CVO_top.sv` | RTL Complete |
| FMap Preprocessing | `PREPROCESS/preprocess_fmap.sv` | RTL Complete |
| Memory Dispatcher | `MEM_control/top/mem_dispatcher.sv` | RTL Complete |
| L2 URAM Cache | `MEM_control/top/mem_L2_cache_fmap.sv` | RTL Complete |
| CVO Stream Bridge | `MEM_control/top/mem_CVO_stream_bridge.sv` | RTL Complete |
| HP Weight CDC Buffer | `MEM_control/top/mem_HP_buffer.sv` | RTL Complete |
| NPU Top — Final Wiring | `NPU_top.sv` | RTL Complete |
| uXC Driver (HAL + API) | `sw/driver/` | Skeleton Only |
| Gemma 3N E4B Application | `sw/gemma3NE4B/` | Submodule |
| Verification / Simulation | — | Not Started |
