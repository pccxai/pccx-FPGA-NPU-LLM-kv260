`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "mem_IO.svh"
`include "npu_interfaces.svh"

import isa_pkg::*;
import vec_core_pkg::*;
import bf16_math_pkg::*;

// ===| Module: NPU_top — pccx v002 SoC integration wrapper |====================
// Purpose      : Top-level integration of all v002 NPU subsystems on KV260.
// Spec ref     : pccx v002 §1 (architecture overview), §6 (KV260 target).
// Target       : Xilinx Kria KV260 (xck26-sfvc784-2LV-c), ZU5EV.
// Clock        : clk_core @ 400 MHz (compute), clk_axi @ 250 MHz (HP/AXIL).
// Reset        : rst_n_core / rst_axi_n active-low. Synchronous release.
// Soft-clear   : i_clear (active-high, sync) — combined with reset wherever
//                state is latched per the repo reset convention.
// Throughput   : Steady-state, dual-lane W4A8 systolic = 32 × 32 × 2 MAC/clk.
// Backpressure : HP weight FIFOs (mem_HP_buffer) provide CDC + skid; ACP fmap
//                FIFO (preprocess_fmap) holds at boundary when broadcast stalls.
//
// Architecture V2 (SystemVerilog Interface Version):
//   HPC0 / HPC1 : 256-bit Feature Map caching bus (ACP port).
//   HP0  ~ HP3  : High-throughput Weight streaming (128-bit each).
//   HPM  (MMIO) : Centralised control & VLIW instruction issuing (AXI-Lite).
//   ACP         : Coherent Result Output.
//
// Data paths (one active per ISA opcode at a time):
//   OP_GEMM  : ACP_FMAP → preprocess_fmap → systolic → normalizer → packer → ACP_RESULT
//   OP_GEMV  : ACP_FMAP → preprocess_fmap → GEMV_top (HP2/3 weights)
//   OP_MEMCPY: ACP DDR4 ↔ L2 (mem_dispatcher via ACP)
//   OP_MEMSET: Shape constant RAM write (mem_dispatcher)
//   OP_CVO   : L2 → CVO_top → L2 (mem_dispatcher ↔ CVO stream bridge)
//
// Status word (mmio_npu_stat[31:0], surfaced to AXIL_STAT_OUT):
//   bit 0 : BUSY  = fifo_full | cvo_busy | cvo_disp_busy
//   bit 1 : DONE  = CVO dispatcher operation complete pulse (one cycle)
//   bit 31:2 reserved.
// ===============================================================================

module NPU_top (
    // ===| Clock & Reset |=======================================================
    input logic clk_core,
    input logic rst_n_core,

    input logic clk_axi,
    input logic rst_axi_n,

    // ===| Soft Clear (synchronous, active-high) |===============================
    input logic i_clear,

    // ===| Control Plane (MMIO) |================================================
    axil_if.slave S_AXIL_CTRL,

    // ===| HP Weight Ports — Matrix Core (Systolic) |============================
    axis_if.slave S_AXI_HP0_WEIGHT,
    axis_if.slave S_AXI_HP1_WEIGHT,

    // ===| HP Weight Ports — Vector Core (GEMV) |================================
    axis_if.slave S_AXI_HP2_WEIGHT,
    axis_if.slave S_AXI_HP3_WEIGHT,

    // ===| ACP Feature Map / Result (Full-Duplex) |==============================
    axis_if.slave  S_AXIS_ACP_FMAP,
    axis_if.master M_AXIS_ACP_RESULT
);

  // ===| Internal Wires — HP Weight (Core-side, post-CDC FIFO) |================
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP0_WEIGHT ();
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP1_WEIGHT ();
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP2_WEIGHT ();
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP3_WEIGHT ();

  // ===| Internal Wires — Instruction Path |=====================================
  logic                GEMV_op_x64_valid_wire;
  logic                GEMM_op_x64_valid_wire;
  logic                memcpy_op_x64_valid_wire;
  logic                memset_op_x64_valid_wire;
  logic                cvo_op_x64_valid_wire;

  instruction_op_x64_t instruction;

  logic                fifo_full_wire;
  logic [31:0]         mmio_npu_stat;

  // ===| [1] NPU Controller |====================================================
  npu_controller_top #() u_npu_controller_top (
      .clk    (clk_core),
      .rst_n  (rst_n_core),
      .i_clear(i_clear),

      .S_AXIL_CTRL(S_AXIL_CTRL),

      .IN_enc_stat ({32'd0, mmio_npu_stat}),
      .IN_enc_valid(mmio_npu_stat[1]),

      .OUT_GEMV_op_x64_valid  (GEMV_op_x64_valid_wire),
      .OUT_GEMM_op_x64_valid  (GEMM_op_x64_valid_wire),
      .OUT_memcpy_op_x64_valid(memcpy_op_x64_valid_wire),
      .OUT_memset_op_x64_valid(memset_op_x64_valid_wire),
      .OUT_cvo_op_x64_valid   (cvo_op_x64_valid_wire),

      .OUT_op_x64(instruction)
  );

  // ===| [2] Global Scheduler |==================================================
  gemm_control_uop_t   GEMM_uop_wire;
  GEMV_control_uop_t   GEMV_uop_wire;
  memory_control_uop_t LOAD_uop_wire;
  memory_control_uop_t STORE_uop_wire;  // latched at issue; drives result writeback
  memory_set_uop_t     mem_set_uop;
  cvo_control_uop_t    CVO_uop_wire;
  logic                sram_rd_start_wire;  // one-cycle pulse: start fmap broadcast

  Global_Scheduler #() u_Global_Scheduler (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),

      .IN_GEMV_op_x64_valid  (GEMV_op_x64_valid_wire),
      .IN_GEMM_op_x64_valid  (GEMM_op_x64_valid_wire),
      .IN_memcpy_op_x64_valid(memcpy_op_x64_valid_wire),
      .IN_memset_op_x64_valid(memset_op_x64_valid_wire),
      .IN_cvo_op_x64_valid   (cvo_op_x64_valid_wire),

      .instruction(instruction),

      .OUT_GEMM_uop     (GEMM_uop_wire),
      .OUT_GEMV_uop     (GEMV_uop_wire),
      .OUT_LOAD_uop     (LOAD_uop_wire),
      .OUT_STORE_uop    (STORE_uop_wire),
      .OUT_mem_set_uop  (mem_set_uop),
      .OUT_CVO_uop      (CVO_uop_wire),
      .OUT_sram_rd_start(sram_rd_start_wire)
  );

  // ===| [3] Memory Dispatcher |=================================================
  // CVO stream wires: bridge ↔ CVO_top
  logic [15:0] cvo_disp_data_wire;
  logic        cvo_disp_valid_wire;
  logic        cvo_disp_ready_wire;
  logic [15:0] cvo_result_wire;
  logic        cvo_result_valid_wire;
  logic        cvo_result_ready_wire;
  logic        cvo_disp_busy_wire;
  logic        cvo_disp_done_wire;

  mem_dispatcher #() u_mem_dispatcher (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),

      .clk_axi  (clk_axi),
      .rst_axi_n(rst_axi_n),

      .S_AXIS_ACP_FMAP  (S_AXIS_ACP_FMAP),
      .M_AXIS_ACP_RESULT(M_AXIS_ACP_RESULT),

      .IN_LOAD_uop     (LOAD_uop_wire),
      .IN_mem_set_uop  (mem_set_uop),
      .IN_CVO_uop      (CVO_uop_wire),
      .IN_cvo_uop_valid(cvo_op_x64_valid_wire),

      .OUT_cvo_data     (cvo_disp_data_wire),
      .OUT_cvo_valid    (cvo_disp_valid_wire),
      .IN_cvo_data_ready(cvo_disp_ready_wire),

      .IN_cvo_result       (cvo_result_wire),
      .IN_cvo_result_valid (cvo_result_valid_wire),
      .OUT_cvo_result_ready(cvo_result_ready_wire),

      .OUT_fifo_full(fifo_full_wire),
      .OUT_cvo_busy (cvo_disp_busy_wire),
      .OUT_cvo_done (cvo_disp_done_wire)
  );

  // ===| [4] HP Weight Buffer (CDC FIFO: AXI → Core clock) |====================
  mem_HP_buffer #() u_HP_buffer (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),
      .clk_axi   (clk_axi),
      .rst_axi_n (rst_axi_n),

      .S_AXI_HP0_WEIGHT(S_AXI_HP0_WEIGHT),
      .S_AXI_HP1_WEIGHT(S_AXI_HP1_WEIGHT),
      .S_AXI_HP2_WEIGHT(S_AXI_HP2_WEIGHT),
      .S_AXI_HP3_WEIGHT(S_AXI_HP3_WEIGHT),

      .M_CORE_HP0_WEIGHT(M_CORE_HP0_WEIGHT),
      .M_CORE_HP1_WEIGHT(M_CORE_HP1_WEIGHT),
      .M_CORE_HP2_WEIGHT(M_CORE_HP2_WEIGHT),
      .M_CORE_HP3_WEIGHT(M_CORE_HP3_WEIGHT)
  );

  // ===| [5] FMap Preprocessing Pipeline |=======================================
  logic [`FIXED_MANT_WIDTH-1:0] fmap_broadcast       [0:`ARRAY_SIZE_H-1];
  logic                         fmap_broadcast_valid;
  logic [  `BF16_EXP_WIDTH-1:0] cached_emax_out      [0:`ARRAY_SIZE_H-1];

  preprocess_fmap #() u_fmap_pre (
      .clk    (clk_core),
      .rst_n  (rst_n_core),
      .i_clear(i_clear),

      .S_AXIS_ACP_FMAP(S_AXIS_ACP_FMAP),

      .i_rd_start(sram_rd_start_wire),

      .o_fmap_broadcast(fmap_broadcast),
      .o_fmap_valid    (fmap_broadcast_valid),
      .o_cached_emax   (cached_emax_out)
  );

  // ===| [6] Systolic Array Engine (Matrix Core) |================================
  // global_inst[2:0] = flags[5:3] = {findemax, accm, w_scale}
  logic [`DSP48E2_POUT_SIZE-1:0] raw_res_sum      [0:`ARRAY_SIZE_H-1];
  logic                          raw_res_sum_valid[0:`ARRAY_SIZE_H-1];
  logic [   `BF16_EXP_WIDTH-1:0] delayed_emax_32  [0:`ARRAY_SIZE_H-1];

  // ===| v002 dual-lane weight unpack |=========================================
  // HP0 / HP1 each carry a 128-bit AXIS word that holds 32 INT4 weights.
  // Slice each into a 32-element INT4 array for the systolic engine.
  localparam int WEIGHT_CNT = `HP_SINGLE_WIDTH / `INT4_WIDTH;  // 32
  logic [`INT4_WIDTH-1:0] hp0_weight_int4 [0:WEIGHT_CNT-1];
  logic [`INT4_WIDTH-1:0] hp1_weight_int4 [0:WEIGHT_CNT-1];
  genvar wi;
  generate
    for (wi = 0; wi < WEIGHT_CNT; wi++) begin : g_weight_unpack
      assign hp0_weight_int4[wi] =
          M_CORE_HP0_WEIGHT.tdata[wi*`INT4_WIDTH +: `INT4_WIDTH];
      assign hp1_weight_int4[wi] =
          M_CORE_HP1_WEIGHT.tdata[wi*`INT4_WIDTH +: `INT4_WIDTH];
    end
  endgenerate

  GEMM_systolic_top #() u_systolic_engine (
      .clk    (clk_core),
      .rst_n  (rst_n_core),
      .i_clear(i_clear),

      .global_weight_valid(M_CORE_HP0_WEIGHT.tvalid),
      .global_inst        (GEMM_uop_wire.flags[5:3]),
      .global_inst_valid  (GEMM_op_x64_valid_wire),

      .IN_fmap_broadcast      (fmap_broadcast),
      .IN_fmap_broadcast_valid(fmap_broadcast_valid),
      .IN_cached_emax_out     (cached_emax_out),

      // v002 dual-lane weights: HP0 → upper INT4 channel, HP1 → lower.
      // Both 128-bit AXIS streams are unpacked into 32 × INT4 arrays by a
      // simple bit-slice assign just before this instantiation.
      .IN_weight_upper      (hp0_weight_int4),
      .IN_weight_upper_valid(M_CORE_HP0_WEIGHT.tvalid),
      .IN_weight_upper_ready(M_CORE_HP0_WEIGHT.tready),
      .IN_weight_lower      (hp1_weight_int4),
      .IN_weight_lower_valid(M_CORE_HP1_WEIGHT.tvalid),
      .IN_weight_lower_ready(M_CORE_HP1_WEIGHT.tready),

      .raw_res_sum      (raw_res_sum),
      .raw_res_sum_valid(raw_res_sum_valid),
      .delayed_emax_32  (delayed_emax_32)
  );

  // ===| [7] Result Normalizers (one per systolic column) |======================
  logic [`BF16_WIDTH-1:0] norm_res_seq      [0:`ARRAY_SIZE_H-1];
  logic                   norm_res_seq_valid[0:`ARRAY_SIZE_H-1];

  genvar n;
  generate
    for (n = 0; n < `ARRAY_SIZE_H; n++) begin : gen_norm
      gemm_result_normalizer u_norm_seq (
          .clk      (clk_core),
          .rst_n    (rst_n_core),
          .data_in  (raw_res_sum[n]),
          .e_max    (delayed_emax_32[n]),
          .valid_in (raw_res_sum_valid[n]),
          .data_out (norm_res_seq[n]),
          .valid_out(norm_res_seq_valid[n])
      );
    end
  endgenerate

  // ===| [8] Result Packer |=====================================================
  logic [`AXI_STREAM_WIDTH-1:0] packed_res_data;
  logic                         packed_res_valid;

  FROM_gemm_result_packer #() u_packer (
      .clk          (clk_core),
      .rst_n        (rst_n_core),
      .row_res      (norm_res_seq),
      .row_res_valid(norm_res_seq_valid),
      .packed_data  (packed_res_data),
      .packed_valid (packed_res_valid),
      .packed_ready (1'b1),                // downstream accepts unconditionally for now
      .o_busy       ()
  );

  // ===| [9] Vector Core (GEMV) |================================================
  // Unpack 128-bit flat HP bus → 32 × INT4 per lane before feeding GEMV_top.
  // HP2 → lane A, HP3 → lane B;  C/D tied to zero (2-lane configuration).
  localparam int GemvWeightCnt = mem_pkg::HpSingleWeightCnt;  // 32 weights per port
  localparam int GemvWeightW = mem_pkg::WeightBitWidth;  // 4-bit INT4

  logic [GemvWeightW-1:0] gemv_weight_A[0:GemvWeightCnt-1];
  logic [GemvWeightW-1:0] gemv_weight_B[0:GemvWeightCnt-1];
  logic [GemvWeightW-1:0] gemv_weight_C[0:GemvWeightCnt-1];
  logic [GemvWeightW-1:0] gemv_weight_D[0:GemvWeightCnt-1];

  genvar w;
  generate
    for (w = 0; w < GemvWeightCnt; w++) begin : gen_gemv_unpack
      assign gemv_weight_A[w] = M_CORE_HP2_WEIGHT.tdata[w*GemvWeightW+:GemvWeightW];
      assign gemv_weight_B[w] = M_CORE_HP3_WEIGHT.tdata[w*GemvWeightW+:GemvWeightW];
      assign gemv_weight_C[w] = '0;
      assign gemv_weight_D[w] = '0;
    end
  endgenerate

  // num_recur: number of fmap accumulation rounds = vector length / broadcast width.
  // size_ptr_addr is a 6-bit shape pointer; actual cycle count resolved by scheduler.
  logic [16:0] gemv_num_recur;
  logic        gemv_activated_lane[0:VecCoreDefaultCfg.num_gemv_pipeline-1];

  assign gemv_num_recur      = {11'b0, GEMV_uop_wire.size_ptr_addr};
  assign gemv_activated_lane = '{default: 1'b0};

  GEMV_top #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_top (
      .clk  (clk_core),
      .rst_n(rst_n_core),

      .IN_weight_valid_A(M_CORE_HP2_WEIGHT.tvalid),
      .IN_weight_valid_B(M_CORE_HP3_WEIGHT.tvalid),
      .IN_weight_valid_C(1'b0),
      .IN_weight_valid_D(1'b0),

      .IN_weight_A(gemv_weight_A),
      .IN_weight_B(gemv_weight_B),
      .IN_weight_C(gemv_weight_C),
      .IN_weight_D(gemv_weight_D),

      .OUT_weight_ready_A(M_CORE_HP2_WEIGHT.tready),
      .OUT_weight_ready_B(M_CORE_HP3_WEIGHT.tready),
      .OUT_weight_ready_C(),
      .OUT_weight_ready_D(),

      .IN_fmap_broadcast      (fmap_broadcast),
      .IN_fmap_broadcast_valid(fmap_broadcast_valid),
      .IN_num_recur           (gemv_num_recur),
      .IN_cached_emax_out     (cached_emax_out),
      .IN_activated_lane      (gemv_activated_lane),

      .OUT_final_fmap_A(),
      .OUT_final_fmap_B(),
      .OUT_final_fmap_C(),
      .OUT_final_fmap_D(),

      .OUT_result_valid_A(),
      .OUT_result_valid_B(),
      .OUT_result_valid_C(),
      .OUT_result_valid_D()
  );

  // ===| [10] CVO Core |=========================================================
  // e_max BF16 encoding: value = 2^(exp - 127), mantissa implicit 1.0.
  //   delayed_emax_32[0] is the 8-bit exponent field from column-0 normalizer.
  //   Packed as BF16: {sign=0, exp=delayed_emax_32[0], mant=7'b0}.
  logic [15:0] cvo_emax_bf16;
  logic        cvo_busy_wire;

  assign cvo_emax_bf16 = {1'b0, delayed_emax_32[0], 7'b0};

  CVO_top u_CVO_top (
      .clk    (clk_core),
      .rst_n  (rst_n_core),
      .i_clear(i_clear),

      .IN_uop       (CVO_uop_wire),
      .IN_uop_valid (cvo_op_x64_valid_wire),
      .OUT_uop_ready(),

      // ===| L2 DMA stream — via mem_CVO_stream_bridge inside mem_dispatcher |===
      .IN_data       (cvo_disp_data_wire),
      .IN_data_valid (cvo_disp_valid_wire),
      .OUT_data_ready(cvo_disp_ready_wire),

      .OUT_result      (cvo_result_wire),
      .OUT_result_valid(cvo_result_valid_wire),
      .IN_result_ready (cvo_result_ready_wire),

      .IN_e_max(cvo_emax_bf16),

      .OUT_busy(cvo_busy_wire),
      .OUT_done(),
      .OUT_accm()
  );

  // ===| Status |================================================================
  // Aggregated NPU busy/done flags — intended for ctrl_npu_frontend IN_enc_stat.
  // Bit 0 : BUSY  (memory FIFO full | CVO engine active | CVO DMA bridge active)
  // Bit 1 : DONE  (CVO dispatcher operation complete pulse)
  assign mmio_npu_stat[0]    = fifo_full_wire | cvo_busy_wire | cvo_disp_busy_wire;
  assign mmio_npu_stat[1]    = cvo_disp_done_wire;
  assign mmio_npu_stat[31:2] = 30'd0;

endmodule
