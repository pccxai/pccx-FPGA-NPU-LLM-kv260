# pccx — Bare-Metal Transformer Accelerator on Kria KV260

Open SystemVerilog NPU for experimental Gemma-class LLM acceleration on
AMD/Xilinx Kria KV260.

```text
PCCX KV260 Roadmap

RTL Alpha        ███████████████░░  85%
Verification     ███████████░░░░░  70%
Driver Bring-up  ███████░░░░░░░░░  45%
Bitstream        ████░░░░░░░░░░░░  25%
Public Release   ██░░░░░░░░░░░░░░  15%

Next milestone: v0.2.0 evidence pack
```

**Current status:** RTL alpha · timing closure and full runtime bring-up in progress.

This repo is the **bare-metal Kria KV260 implementation** of the
**pccx v002** NPU architecture. It hosts an early SystemVerilog RTL and
bare-metal driver source snapshot intended to close the loop between the
pccx architecture specification, source-level RTL inspection, early
verification setup, and planned KV260 bring-up.

This is not a timing-closed production bitstream release — Vivado
synthesis, trace-driven verification, and full Gemma 3N E4B application
wiring are still in progress or planned.

> ### Start with the architecture docs
>
> The design rationale, ISA, memory map, and model-mapping notes live on
> the **pccx documentation site**. This repo implements what that site
> specifies — read the spec first, then come back here for the RTL.
>
> **→ [pccx v002 — Architecture & ISA spec](https://pccxai.github.io/pccx/en/docs/v002/index.html)**
> &nbsp;·&nbsp; [Gemma 3N E4B on pccx v002](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_execution.html)
> &nbsp;·&nbsp; [한국어 문서](https://pccxai.github.io/pccx/ko/docs/v002/index.html)

Related repos: [pccx (spec)](https://github.com/pccxai/pccx) · [pccx-lab (profiler / simulator)](https://github.com/pccxai/pccx-lab) · [llm-bottleneck-lab (related research)](https://github.com/hkimw/llm-bottleneck-lab)

---

## Project status

**Public alpha** — `v0.1.0-alpha` is published as a prerelease. Core
RTL and ISA are stable; verification and KV260 bring-up are in
progress. This is not a timing-closed bitstream release. Feedback and
issues are welcome.

| Entry point | Link |
| --- | --- |
| Architecture & ISA spec | <https://pccxai.github.io/pccx/en/docs/v002/index.html> |
| RTL reference (this repo) | [`hw/rtl/`](hw/rtl/) — top wrapper [`NPU_top.sv`](hw/rtl/NPU_top.sv) |
| Releases | <https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/releases> |
| `v0.1.0-alpha` notes | [docs/releases/v0.1.0-alpha.md](docs/releases/v0.1.0-alpha.md) |
| Roadmap (project board) | <https://github.com/orgs/pccxai/projects/1> |
| Contributing | <https://github.com/pccxai/.github/blob/main/CONTRIBUTING.md> |
| How to cite | [CITATION.cff](CITATION.cff) |
| Verification check | `Sail typecheck` + `repo-validate` required on `main` |
| Discussions (board bring-up, RTL Q&A) | <https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/discussions> |
| Good first issues | <https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/labels/good%20first%20issue> |

> The repository **Wiki** is intentionally empty — RTL- and bring-up
> questions belong in **Discussions**, and the canonical architecture
> documentation lives on the [pccx Sphinx site](https://pccxai.github.io/pccx/en/docs/v002/index.html).

---

## What's here vs. what's in pccx

| Layer                                | Lives in                                   | Authoritative source                                                                     |
| ------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Architecture / ISA / driver spec     | `pccx/docs/v002/`                          | [pccx v002 docs](https://pccxai.github.io/pccx/en/docs/v002/index.html)               |
| Target-model pipeline (Gemma 3N E4B) | `pccx/docs/v002/Models/`                   | [Models section](https://pccxai.github.io/pccx/en/docs/v002/Models/index.html)        |
| RTL (SystemVerilog)                  | this repo — `hw/rtl/`                      | —                                                                                         |
| Bare-metal driver (C/C++)            | this repo — `sw/driver/`                   | API spec: [Drivers/api](https://pccxai.github.io/pccx/en/docs/v002/Drivers/api.html)  |
| Application (planned, v0.2.0)        | this repo — `sw/gemma3NE4B/` (not yet in tree) | —                                                                                     |

If you want to **read about how the accelerator works**, head to the
**[pccx v002 docs](https://pccxai.github.io/pccx/en/docs/v002/index.html)** —
that's the canonical source for every architectural decision in this repo.
If you want to **read the RTL or synthesize the bitstream**, stay here.

---

## Architecture Snapshot (pccx v002)

[![pccx v002 architecture](https://raw.githubusercontent.com/pccxai/pccx/main/assets/images/Architecture/v002/architecture_v002.png)](https://pccxai.github.io/pccx/en/docs/v002/Architecture/top_level.html)

*Click the diagram for the annotated top-level page on the pccx site.*

Three heterogeneous cores around a centralized L2 URAM cache:

| Core                      | Shape        | What it runs                                                       |
| ------------------------- | ------------ | ------------------------------------------------------------------ |
| **Systolic Array (GEMM)** | 32 × 16 × 2  | Prefill — ``Q · Kᵀ`` across the full context, FFN in prefill       |
| **GEMV Core**             | 32 × 1 × 4   | Decode — every projection in the autoregressive step               |
| **SFU (CVO)**             | 32 × 1 × 4   | Softmax · GELU · RMSNorm · RoPE · sin/cos · reduce · scale · recip |

- **Clock domains**: AXI 250 MHz ↔ core 400 MHz, crossed by async CDC FIFOs.
- **Weight path**: HP0/1 = GEMM weights, HP2/3 = GEMV weights, 128-bit/clk each.
- **Activation path**: host DDR4 → ACP DMA → L2 URAM (1.75 MB, true dual port).
- **Direct-connect FIFO**: GEMV → SFU, so softmax runs without an L2 round-trip.

Full rationale and numbers: [Top-Level →](https://pccxai.github.io/pccx/en/docs/v002/Architecture/top_level.html)
· [Design rationale →](https://pccxai.github.io/pccx/en/docs/v002/Architecture/rationale.html)

---

## Matrix Core (GEMM) — 32 × 16 × 2 Systolic Array

Weight-stationary 2D systolic layout. Activations broadcast along
columns, partial sums propagate vertically into the result accumulator.
Used only during prefill; idle during decode.

[![GEMM array layout](https://raw.githubusercontent.com/pccxai/pccx/main/assets/images/Architecture/v002/Processing_Elements_GEMM_1_v002.png)](https://pccxai.github.io/pccx/en/docs/v002/Architecture/gemm_core.html)

Inside each PE — a DSP48E2 wrapped with input flip-flops on both
Activation and Weight ports, and an accumulator with a P-register
output:

[![GEMM single PE](https://raw.githubusercontent.com/pccxai/pccx/main/assets/images/Architecture/v002/Processing_Elements_GEMM_2_v002.png)](https://pccxai.github.io/pccx/en/docs/v002/Architecture/gemm_core.html)

Details: [GEMM core →](https://pccxai.github.io/pccx/en/docs/v002/Architecture/gemm_core.html)
· [GEMM dataflow →](https://pccxai.github.io/pccx/en/docs/v002/ISA/dataflow.html)

---

## Vector Core (GEMV) — 32 × 1 × 4 Lanes

Four parallel GEMV lanes, each an 8-wide DSP pipeline fed by an
Activation broadcast and a Weight row. Outputs feed a reduction tree
that collapses partial products into the final vector entry register.
The primary compute path during autoregressive decode.

[![GEMV core layout](https://raw.githubusercontent.com/pccxai/pccx/main/assets/images/Architecture/v002/Processing_Elements_GEMV_1_v002.png)](https://pccxai.github.io/pccx/en/docs/v002/Architecture/gemv_core.html)

Per-cycle operand shapes — a 1×N activation row multiplied against an
N×N weight tile:

[![GEMV operand shapes](https://raw.githubusercontent.com/pccxai/pccx/main/assets/images/Architecture/v002/Processing_Elements_GEMV_2_v002.png)](https://pccxai.github.io/pccx/en/docs/v002/Architecture/gemv_core.html)

Details: [GEMV core →](https://pccxai.github.io/pccx/en/docs/v002/Architecture/gemv_core.html)
· [GEMV dataflow →](https://pccxai.github.io/pccx/en/docs/v002/ISA/dataflow.html)

---

## Why W4A8 with 1 DSP = 2 MAC

The DSP48E2 has a single 27×18 multiplier, not two. pccx v002 bit-packs
**two INT4 weights into port A** alongside a single INT8 activation on
port B, so each DSP emits **two MACs per cycle** into the 48-bit
accumulator with a 19-bit guard band between the two channels.

[![DSP48E2 W4A8 port layout](https://raw.githubusercontent.com/pccxai/pccx/main/assets/images/Architecture/v002/Processing_Elements_GEMM_4_v002.png)](https://pccxai.github.io/pccx/en/docs/v002/Architecture/dsp48e2_w4a8.html)

After accumulation, a sign-recovery step restores the upper channel
when the lower channel borrowed a carry:

[![Sign recovery SV snippet](https://raw.githubusercontent.com/pccxai/pccx/main/assets/images/Architecture/v002/Processing_Elements_GEMM_5_v002.png)](https://pccxai.github.io/pccx/en/docs/v002/Architecture/dsp48e2_w4a8.html)

- Maximum accumulations before draining the ACCM: **2^10 ≈ 1024** per
  channel (guard-band limited).
- For K > 1024 (e.g. Gemma 3N's FFN with K = 16384), the Global
  Scheduler drains the ACCM every 1024 cycles into a LUT-based adder
  tree and merges the partial sums.
- Peak: **2048 MAC × 400 MHz ≈ 819 GMAC/s** across the two systolic arrays.

Details: [DSP48E2 W4A8 bit-packing →](https://pccxai.github.io/pccx/en/docs/v002/Architecture/dsp48e2_w4a8.html)

---

## Target Gemma 3N E4B Execution Path

The target model (Google Gemma 3N E4B) has several deviations from a
textbook decoder that the scheduler has to honor. The short list:

| Feature                             | Effect                                                                                                                                                      |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AltUp 4 residual streams            | Four copies of ``xs`` live in L2; main stream ``xs[0]`` stays clean, shadow streams ``xs[1..3]`` receive depth-dependent updates.                           |
| Alternating RoPE θ (5-layer cycle)  | θ = 10 000 (local) or 1 000 000 (global), preloaded per-layer via ``MEMSET``.                                                                               |
| No attention scaling, no softcap    | Softmax sequence drops from 4 CVO instructions to 3.                                                                                                        |
| LAuReL parallel branch              | Two tiny GEMVs (``D × 64``, ``64 × D``) + a ``CVO_SCALE`` by ``1/sqrt(2)``.                                                                                 |
| PLE shadow-stream injection         | Per-Layer Embedding only touches ``xs[1..3]`` at the *end* of each layer; the main stream is untouched.                                                     |
| FFN Gaussian Top-K sparsity (L0–9)  | Replaces a sort with `Mean + 1.645·Std`; ~95 % of `gate_raw` becomes zero and ``W_down`` skips masked rows.                                                 |
| Cross-layer KV sharing              | Only layers 0–19 own their KV cache; layers 20–34 reuse layer 18 (local) or 19 (global). Cache shape is ``[20, L, 512]``, not ``[35, L, 512]``.             |

End-to-end decode flow, per-cycle overlap strategy, instruction-level
mapping, memory layout, and the performance budget all live in the
pccx Models section:

- [Gemma 3N overview](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_overview.html)
- [Full operator pipeline (embedding → sampling)](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_pipeline.html)
- [Attention & RoPE constraints](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_attention_rope.html)
- [PLE & LAuReL routing rules](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_ple_laurel.html)
- [FFN Gaussian Top-K sparsity](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_ffn_sparsity.html)
- [**Execution & scheduling on pccx v002**](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_execution.html)

---

## Custom 64-bit VLIW ISA

Five opcodes, 64 bits each: `[63:60]` opcode + `[59:0]` body.

| Opcode | Mnemonic   | Use                                                                      |
| ------ | ---------- | ------------------------------------------------------------------------ |
| `4'h0` | `OP_GEMV`  | Vector × Matrix — decode projections                                     |
| `4'h1` | `OP_GEMM`  | Matrix × Matrix — prefill Q·Kᵀ, A·V across full sequence                 |
| `4'h2` | `OP_MEMCPY`| Host DDR4 ↔ L2 DMA (ACP)                                                 |
| `4'h3` | `OP_MEMSET`| Write shape / size / scale constants to the Constant Cache              |
| `4'h4` | `OP_CVO`   | Element-wise non-linear (exp, sqrt, GELU, sin, cos, reduce_sum, scale, recip) |

Spec: [Per-instruction encoding →](https://pccxai.github.io/pccx/en/docs/v002/ISA/instructions.html)
· [Dataflow per opcode →](https://pccxai.github.io/pccx/en/docs/v002/ISA/dataflow.html)

The pipeline is **fully decoupled**: the front-end decodes and enqueues
into per-engine FIFOs, and each compute engine fires independently once
its local dependencies (weight stream, fmap ready) are satisfied. A
stall in one engine never halts another.

#### Example — softmax sequence (three CVO calls because Gemma 3N has no softcap)

```
GEMV   flags.findemax=1              ; Q · Kᵀ, track e_max
CVO    CVO_EXP  flags.sub_emax=1     ; exp(score - e_max)
CVO    CVO_REDUCE_SUM                ; Σ exp → scalar
CVO    CVO_SCALE flags.recip_scale=1 ; divide each exp by the sum
```

---

## KV Cache Strategy

KV bandwidth (not FLOPs) is what pins down `L` on KV260. At 32 K context
the cumulative cache would hit ~1.31 GB, and DDR4's ~10 GB/s puts
floor-to-floor read time above 130 ms per token. Three mitigations,
enforced at RTL / memory controller / driver level:

1. **KV quantization** — DRAM format is INT8 (default) or INT4. 2–4×
   bandwidth and capacity savings, aligned with the W4A8 compute path.
2. **Attention-sink + local-window eviction** — the driver retains only
   the first few tokens and a sliding recent window; middle tokens are
   evicted on a schedule, combined with Google Turbo-Quant-style
   requantization.
3. **Hard cap** — the KV ring-buffer ceiling is set at init
   (`max_tokens = 8192`). Wrap-around overwrites the oldest entries.
   This bounds both OOM risk and worst-case memory traffic.

Details: [KV cache strategy →](https://pccxai.github.io/pccx/en/docs/v002/Architecture/kv_cache.html)

---

## Roadmap — Two-Track

This repository hosts the RTL for **both** active tracks. As of
2026-04-20:

| Track | Target model | Planned target | Horizon | Shared RTL assets |
|-------|-------------|------|---------|-------------------|
| **v002 Extended** (this repo, `main`) | Gemma 3N E4B | **20 tok/s** target, evidence-gated | Week 1–49 | sparse weight fetcher (Phase G), EAGLE draft/verify dispatch (Phase H+), SSD scheduler (Phase I), tree mask generator (Phase J) |
| **v003** (future branch) | Gemma 4 E4B | **12–15 tok/s** target | Week 16–52 (parallel) | reuses v002 Phase G/H/I/J modules with hidden/layer/KV-head re-parameterization |

- v002 freeze → snapshotted into `pccx/codes/v002/` via the pccx
  version-cutover workflow (`tools/freeze_active.sh`).
- v003 lives on a later branch of this same repo; the pccx docs site
  will fork its own `v003/` docs tree at the same time.

Full phase-by-phase plan, decision points, compute budget, and Year 2
**Auto-Porting Pipeline α** vision:

**→ [Roadmap (EN)](https://pccxai.github.io/pccx/en/docs/roadmap.html)**
&nbsp;·&nbsp; [**한국어**](https://pccxai.github.io/pccx/ko/docs/roadmap.html)

---

## Implementation Status

| Block                                            | Status         |
| ------------------------------------------------ | -------------- |
| Custom VLIW ISA                                  | Spec complete  |
| VLIW frontend + decoder                          | RTL complete   |
| Global Scheduler                                 | RTL complete   |
| Systolic Array (Matrix Core, v001 32×32)         | RTL complete   |
| GEMV pipeline (Vector Core, 4 lanes)             | RTL complete   |
| CVO SFU + CORDIC                                 | RTL complete   |
| FMap preprocessing (BF16 → fixed-pt)             | RTL complete   |
| L2 URAM cache + ACP DMA                          | RTL complete   |
| CVO ↔ L2 stream bridge                           | RTL complete   |
| NPU top-level wiring                             | RTL complete   |
| Python golden model                              | Verified       |
| pccx v002 re-parameterization (1 DSP = 2 MAC)    | **In progress** |
| uXC driver (AXI-Lite HAL)                        | Skeleton       |
| Gemma 3N E4B application (`sw/gemma3NE4B/`)      | Planned (v0.2.0) — not yet in tree |
| Simulation / trace-driven verification           | xsim smoke suite active |
| Vivado synthesis + timing closure                | Synth attempted; no completed report yet. Timing closure pending. |

> The current `hw/rtl/` tree reflects the v001 parameterization
> (W4A16 / 32×32 array / 1 DSP = 1 MAC). Re-parameterization to the
> v002 target (W4A8 / 32×16 ×2 / 1 DSP = 2 MAC) is the current
> in-flight task. See the [design rationale](https://pccxai.github.io/pccx/en/docs/v002/Architecture/rationale.html#id3)
> for the 3.125× theoretical speedup.

---

## Repository Layout

```
hw/
  rtl/
    NPU_top.sv                ← top-level wiring
    NPU_Controller/           ← VLIW frontend, decoder, Global Scheduler
    MAT_CORE/                 ← Systolic array, normalizer, packer
    VEC_CORE/                 ← GEMV pipeline (4 μV-Core lanes)
    CVO_CORE/                 ← CVO SFU + CORDIC unit
    PREPROCESS/               ← BF16 → fixed-pt pipeline, fmap cache
    MEM_control/              ← L2 cache, DMA dispatcher, CVO bridge, HP buffer
    Constants/                ← `define macros + SystemVerilog packages (A → D)
    Library/                  ← BF16 math pkg, algorithms pkg, QUEUE
sw/
  driver/                     ← AXI-Lite MMIO HAL + inference API (skeleton)
  gemma3NE4B/                 ← Gemma 3N E4B application — planned for v0.2.0, not yet in tree
docs/                         ← Redirect stub only — full docs live on pccx
```

`docs/` in this repo is intentionally a **redirect stub**. All
architectural and model documentation now lives in the pccx repo /
GitHub Pages site.

---

## Verification workflow

The repo ships a minimal xsim harness at `hw/sim/` that every testbench
listed in `hw/sim/run_verification.sh` plugs into. One command runs the full suite and
emits a `.pccx` trace per bench for [pccx-lab][pccx-lab] to visualise:

```bash
hw/sim/run_verification.sh
```

See [docs/SIMULATION.md](docs/SIMULATION.md) for generated log paths,
PASS verdict rules, and the evidence checklist.

### Current testbench matrix

| Testbench | Module(s) under test | Verdict count |
|-----------|-----------------------|--------------:|
| `tb_shape_const_ram`                | `shape_const_ram` (reset / write / read contract)        |   15 |
| `tb_mem_dispatcher_shape_lookup`    | `mem_dispatcher` + `shape_const_ram` LOAD pointer routing |   11 |
| `tb_GEMM_dsp_packer_sign_recovery` | `GEMM_dsp_packer` + `GEMM_sign_recovery` (W4A8 dual-MAC) | 1024 |
| `tb_GEMM_fmap_staggered_delay`     | `GEMM_fmap_staggered_dispatch` (column stagger)          |   65 |
| `tb_GEMM_weight_dispatcher`        | `GEMM_weight_dispatcher` (upper / lower AND-valid)      |  128 |
| `tb_mat_result_normalizer`         | `mat_result_normalizer` (48 b 2sC → BF16 4-stage)       |  256 |
| `tb_FROM_mat_result_packer`        | `FROM_gemm_result_packer` (32 lanes → 4×128 b FSM)      |    4 |
| `tb_barrel_shifter_BF16`           | `barrel_shifter_BF16` (BF16 → 27 b fixed-point)         |  512 |
| `tb_ctrl_npu_decoder`              | `ctrl_npu_decoder` (4-bit opcode → one-hot valid)       |    6 |
| `tb_mem_u_operation_queue`         | `mem_u_operation_queue` (queue push / pop smoke)         |   32 |

Every bench emits the canonical `PASS:` line that
[pccx-lab][pccx-lab]'s `from_xsim_log` converter recognises — the
deterministic runner fails nonzero on a missing verdict or explicit
non-PASS verdict.

### Adding a new testbench

Two lines in `hw/sim/run_verification.sh`:

```bash
TB_DEPS[tb_new_module]="SUB_DIR/new_module.sv"
TB_CORE[tb_new_module]=N   # pick an unused core-id for the emitted trace
```

Then drop `hw/tb/tb_new_module.sv` with the canonical `$display` line on
success / failure. See
[pccx-lab's verification-workflow doc](https://pccxai.github.io/pccx/en/lab/verification-workflow.html)
for the end-to-end flow diagram and the Tauri IPC surface.

### What plugs into pccx-lab

| Surface                 | Command                                            |
|-------------------------|----------------------------------------------------|
| Full suite runner       | `run_verification` IPC → `run_verification.sh`     |
| Per-tb trace loader     | "Open" button per row (calls `load_pccx`)          |
| Synth utilisation       | `SynthStatusCard`  (parses `hw/build/reports/`)    |
| Roofline classification | `RooflineCard`     (runs on the loaded trace)      |
| Markdown summary        | `generate_markdown_report` IPC                     |

[pccx-lab]: https://github.com/pccxai/pccx-lab

---

## License

Apache 2.0 — same as pccx. This protects the architecture from patent
risk while keeping the ecosystem open for hardware research.
