# v002 full-system diagnosis before next synth

State: `IMPL_TIMING_NOT_CLEAN`.

This note records the static/system pass before the next expensive Vivado run. It intentionally does not claim timing closure, bitstream readiness, KV260 execution, Gemma 3N E4B hardware execution, or measured throughput.

## Top/module hierarchy

The OOC synthesis top is `NPU_top`. `npu_core_wrapper` is a packaging shim for a future block-design flow and is not the current synthesis top.

Main hierarchy:

- `NPU_top`
- `npu_controller_top` -> AXI-Lite frontend and `ctrl_npu_decoder`
- `Global_Scheduler` -> ISA body to GEMM/GEMV/MEM/CVO uops
- `mem_dispatcher` -> shape RAM, LOAD/MEMSET descriptor generation, L2/ACP/CVO boundary
- `mem_GLOBAL_cache` -> ACP CDC, L2 URAM, ACP/NPU FSMs
- `mem_L2_cache_fmap` -> XPM UltraRAM true dual-port memory
- `mem_CVO_stream_bridge` -> 128-bit L2 words to 16-bit CVO stream and result writeback
- `preprocess_fmap` -> L2 AXIS stream to BF16/fixed fmap broadcast
- `GEMM_systolic_top`, `GEMV_top`, `CVO_top` -> compute engines

## Data movement map

Program intent enters through AXI-Lite, is decoded by `ctrl_npu_decoder`, and is translated by `Global_Scheduler` into one-cycle uop pulses. The memory path resolves shape pointers in `mem_dispatcher`, converts element dimensions to 128-bit L2 word ranges, and queues ACP or NPU descriptors through `mem_u_operation_queue`.

Major paths:

- Host to L2: AXIS ACP input -> `mem_BUFFER` CDC -> `mem_GLOBAL_cache` port A -> `mem_L2_cache_fmap`.
- L2 to host: `mem_L2_cache_fmap` port A -> `mem_BUFFER` CDC -> AXIS ACP result.
- L2 to GEMM/GEMV fmap: `mem_GLOBAL_cache` port B -> `M_AXIS_L1_FMAP` -> `preprocess_fmap`.
- L2 to CVO and back: `mem_CVO_stream_bridge` owns direct port-B address/write/data while busy, streams BF16 elements into `CVO_top`, buffers results, and writes 128-bit words back to L2.

## Clock/reset assumptions

- Core clock target is 400 MHz (`core_clk`, 2.500 ns).
- AXI clock target is 250 MHz (`axi_clk`, 4.000 ns).
- `rst_n_core` and `rst_axi_n` are active-low.
- `i_clear` is propagated through the controller, preprocessing, GEMM, and CVO paths.
- The MEM subtree currently resets from `rst_n_core`/`rst_axi_n`; `mem_u_operation_queue` already documents that its optional counters do not soft-clear until `i_clear` is propagated through this subtree.

## Filelist/script/constraint alignment

Static filelist checks found:

- No missing entries in `hw/vivado/filelist.f`.
- No unlisted RTL `.sv` files under `hw/rtl`.
- No duplicate module definitions among `hw/rtl` and `hw/vivado` SystemVerilog modules.
- `NPU_top` remains the OOC synthesis top in `hw/vivado/synth.tcl`.
- `top-status` / `top-bitstream` are explicit full-top flow gates and do not turn OOC route evidence into a KV260 bitstream claim.

## Library reuse findings

Reusable components present and in use:

- `bf16_math_pkg` for BF16 helpers used by CVO and top-level imports.
- `algorithms_pkg`, `IF_queue`, and `QUEUE` for AXI-Lite command queue plumbing.
- `isa_pkg` / ISA subpackages for uop and instruction field types.
- `mem_pkg`, `device_pkg`, and `vec_core_pkg` for architecture constants and typed boundaries.
- Xilinx XPM FIFO/RAM primitives for CDC FIFOs, operation queues, result FIFO, and L2 URAM.

No new arithmetic helper RTL was added in this pass. The immediate fix reused the existing deep memory boundary instead of introducing a wrapper.

## Static bug risks found and fixed

Finding: `mem_dispatcher` had a CVO direct L2 port-B mux (`cvo_l2_*` / `final_npu_*`) but only muxed write data reached `mem_GLOBAL_cache`; the muxed write-enable and address were not connected to `mem_L2_cache_fmap`. This meant the CVO bridge could compute L2 read/write addresses without the URAM port-B wrapper seeing them.

Fix: `mem_GLOBAL_cache` now exposes an explicit `IN_npu_direct_*` port-B owner boundary. `mem_dispatcher` drives those signals from `mem_CVO_stream_bridge` while `cvo_bridge_busy` is asserted. The local NPU FSM is held while direct ownership is active.

Coverage: `tb_mem_dispatcher_shape_lookup` now checks that a CVO uop drives the URAM port-B address/write-enable boundary.

Finding from the single controlled synth re-entry: the direct boundary fix made the CVO bridge read-address adder visible to the URAM cascade address path. The current post-synth report for that intermediate tree shows WNS `-0.385 ns`, TNS `-34.307 ns`, with the top path from `mem_CVO_stream_bridge.rd_word_cnt` through `rd_base + rd_word_cnt` into `mem_L2_cache_fmap` port-B address.

Fix after that report: `mem_CVO_stream_bridge` now registers the L2 read command address (`rd_word_addr` / `rd_word_valid`) before driving the URAM port-B address. This preserves the external CVO stream contract while moving the read-address add off the URAM input cycle. This later fix has xsim coverage but has not had a second synth run in this batch, to avoid a synth/fail loop.

## Timing-risk classes

Current report-backed risks:

- `mem_CVO_stream_bridge` read-address to L2 URAM port-B: post-synth OOC WNS `-0.385 ns`, TNS `-34.307 ns` on the intermediate direct-boundary tree. A registered address cut was added after this report; next synth must verify it.
- `mem_dispatcher` shape/word-count DSP-to-DSP path: previous post-impl OOC WNS `-0.009 ns`, TNS `-0.273 ns`. That implementation report predates this batch's RTL changes and must be treated as stale.
- CVO BF16/SFU paths: previously reduced through explicit pipeline cuts; not the current top failing class.
- L2 URAM cascade/readout path: previously top failing class; now displaced by descriptor arithmetic.
- Clock fanout: current top path report shows high `clk_core` fanout, but this is not a user RTL bug by itself.
- Full-top uncertainty: current implementation evidence is OOC route only, so full KV260 clocking/BD/partpin effects remain unknown.

Deferred timing action: run one controlled synth on the final diagnosed tree. If the registered CVO read-address cut closes post-synth timing, then implementation can revisit the shape-word arithmetic path. If it does not, inspect the new top path before editing.

## Evidence coverage

Current regression covers:

- Shape constant RAM.
- `mem_dispatcher` shape lookup and CVO direct L2 port-B ownership smoke.
- GEMM pack/sign recovery, weight dispatch, fmap stagger, result normalization, result pack.
- BF16 barrel shifter.
- Decoder and runtime smoke program path through decoder/scheduler.
- CVO SFU smoke for reduce sum, GELU, EXP, RECIP, SCALE.
- Memory operation queue.

Coverage gaps that remain acceptable for this PR:

- Full CVO vector read/compute/writeback with initialized L2 contents.
- Full `NPU_top` AXI/AXIS end-to-end hardware traffic.
- Full KV260 block design and bitstream flow.
- Board execution, Gemma runtime, and throughput measurement.

## Current blockers

- The latest post-synth report was generated before the final registered CVO read-address cut and is not timing-clean for that intermediate tree.
- OOC post-impl timing remains not closed in the existing report, and that post-impl report is stale relative to this batch's RTL changes.
- Full top-level block design script is missing.
- Full top-level implementation has not run.
- Bitstream is not generated.
- KV260 board execution evidence is unavailable.
- Gemma 3N E4B hardware runtime evidence is unavailable.
- Measured throughput evidence is unavailable.
