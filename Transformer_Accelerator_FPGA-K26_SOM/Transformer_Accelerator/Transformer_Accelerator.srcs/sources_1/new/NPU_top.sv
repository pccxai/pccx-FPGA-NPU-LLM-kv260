`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "mem_IO.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

/**
 * Module: NPU_top
 * Target: Kria KV260 @ 400MHz
 *
 * Architecture V2 (SystemVerilog Interface Version):
 * - HPC0/HPC1: Combined to form 256-bit Feature Map caching bus.
 * - HP0~HP3: Dedicated to high-throughput Weight streaming.
 * - HPM (MMIO): Centralized control & VLIW Instruction issuing.
 * - ACP: Coherent Result Output.
 */
module NPU_top (
    // Clock & Reset
    input logic clk_core,
    input logic rst_n_core,

    input logic clk_axi,
    input logic rst_axi_n,

    // Control Plane (MMIO)
    input  logic [31:0] mmio_npu_vliw,
    output logic [31:0] mmio_npu_stat,

    axil_if.slave S_AXIL_CTRL,

    // AXI4-Stream Interfaces (Clean & Modern)
    // |================================|
    // | Weight M dot M Input (256-bit) |
    // | Systolic 128bit                |
    // | (V dot M)'s support 128bit     |
    // |================================|
    axis_if.slave S_AXI_HP0_WEIGHT,
    axis_if.slave S_AXI_HP1_WEIGHT,

    // | Weight V dot M Input (256-bit) |
    axis_if.slave S_AXI_HP2_WEIGHT,
    axis_if.slave S_AXI_HP3_WEIGHT,


    // ACP      = featureMAP in, out (Full-Duplex), read & write at same time
    axis_if.slave  S_AXIS_ACP_FMAP,   // Feature Map Input 0 (128-bit, HPC0)
    axis_if.master M_AXIS_ACP_RESULT  // Final Result Output (128-bit)

);

  memory_op_t memcpy_cmd_wire;

  logic memcpy_op_x64_valid_wire;
  memory_op_x64_t memcpy_op_x64_wire;

  logic vdotm_op_x64_valid_wire;
  vdotm_op_x64_t vdotm_op_x64_wire;

  logic mdotm_op_x64_valid_wire;
  mdotm_op_x64_t mdotm_op_x64_wire;

  logic fifo_full_wire;


  npu_controller_top #() u_npu_controller_top (
      .clk(clk_core),
      .rst_n(rst_n_core),
      .i_clear(i_clear),

      .S_AXIL_CTRL(S_AXIL_CTRL),

      // memcpy
      .OUT_memcpy_op_x64_valid(OUT_memcpy_op_x64_valid_wire),
      .memory_op_x64_t(memcpy_op_x64_wire),

      .OUT_vdotm_op_x64_valid(OUT_vdotm_op_x64_valid_wire),
      .vdotm_op_x64_t(vdotm_op_x64),

      .OUT_mdotm_op_x64_valid(OUT_mdotm_op_x64_valid_wire),
      .mdotm_op_x64_t(mdotm_op_x64)
  );

  memory_control_uop_t  mem_uop_wire;
  stlc_control_uop_t    stlc_uop_wire;
  vdotm_control_uop_t   vdotm_uop_wire;

  Global_Scheduler #() u_Global_Scheduler (
    .clk_core(clk_core),
    .rst_n_core(rst_n_core),

    .IN_memcpy_op_x64_valid(memcpy_op_x64_valid_wire),
    .memcpy_op_x64(memcpy_op_x64_wire),

    .IN_vdotm_op_x64_valid(vdotm_op_x64_valid_wire),
    .vdotm_op_x64(vdotm_op_x64_wire),

    .IN_mdotm_op_x64_valid(mdotm_op_x64_valid_wire),
    .mdotm_op_x64(mdotm_op_x64_wire),

    .OUT_mem_uop(mem_uop_wire),
    .OUT_stlc_uop(stlc_uop_wire),
    .OUT_vdotm_uop(vdotm_uop_wire)
  );

  mem_dispatcher #(
  ) u_mem_dispatcher (
    .clk_core(clk_core),
    .rst_n_core(rst_n_core),

    .clk_axi(clk_axi),
    .rst_axi_n(rst_axi_n),

    axis_if.slave(S_AXI_HP0_WEIGHT),
    axis_if.slave(S_AXI_HP1_WEIGHT),
    axis_if.slave(S_AXI_HP2_WEIGHT),
    axis_if.slave(S_AXI_HP3_WEIGHT),

    // ACP      = featureMAP in, out (Full-Duplex), read & write at same time
    axis_if.slave(S_AXIS_ACP_FMAP),  // Feature Map Input 0 (128-bit, HPC0)
    axis_if.master(M_AXIS_ACP_RESULT),  // Final Result Output (128-bit)

    .IN_mem_uop(mem_uop_wire),
    .OUT_fifo_full(fifo_full_wire)

  )

  /*
    // ===| Weight Pipeline Control (To/From Dispatcher) |=============
    input  logic [`ADDR_WIDTH_L2-1:0] IN_read_addr_hp [0:3],
    input  logic                      IN_read_en_hp   [0:3],
    output logic [             127:0] OUT_read_data_hp[0:3],

    // ===| FMAP/KV Pipeline Control (To/From Dispatcher) |============
    // ACP (External) Memory Map Control
    input logic [16:0] IN_acp_base_addr,  // Dispatcher tells where to store incoming FMAP
    input logic        IN_acp_rx_start,   // Trigger to accept ACP data

    // NPU (Internal) Compute Access (Port B)
    input  logic         IN_npu_we,
    input  logic [ 16:0] IN_npu_addr,
    input  logic [127:0] IN_npu_wdata,
    output logic [127:0] OUT_npu_rdata
*/

  // ===| FMap Preprocessing Pipeline (The Common Path) |=======
  logic [`FIXED_MANT_WIDTH-1:0] fmap_broadcast       [0:`ARRAY_SIZE_H-1];
  logic                         fmap_broadcast_valid;
  logic [  `BF16_EXP_WIDTH-1:0] cached_emax_out      [0:`ARRAY_SIZE_H-1];

  preprocess_fmap u_fmap_pre (
      .clk(clk_core),
      .rst_n(rst_n_core),
      .i_clear(npu_clear),

      // HPC Streaming Inputs
      .S_AXIS_ACP_FMAP(S_AXIS_ACP_FMAP),
      //.S_AXIS_FMAP1(S_AXIS_HPC1),

      // Control
      .i_rd_start(global_sram_rd_start),

      // Preprocessed Outputs (to Branch Engines)
      .o_fmap_broadcast(fmap_broadcast),
      .o_fmap_valid(fmap_broadcast_valid),
      .o_cached_emax(cached_emax_out)
  );

/*
  // [MOD-1]Weight Pipeline: M dot M
  // (HP0 -> Systolic Array)
  // (HP1 -> V dot M = support Systolic Array)

  // [MOD-2] Weight Pipeline: V dot M
  // (HP2 & HP3 -> V dot M)
  logic [`AXI_DATA_WIDTH-1:0] weight_fifo_data [0:`AXI_WEIGHT_PORT_CNT-1];
  logic                       weight_fifo_valid[0:`AXI_WEIGHT_PORT_CNT-1];
  logic                       weight_fifo_ready[0:`AXI_WEIGHT_PORT_CNT-1];

  weight_in_HP_port_fifo #(
      .DATA_WIDTH(`HP_PORT_SINGLE_WIDTH)
  ) u_weight_in_HP_port (
      .IN_HP0(S_AXI_HP0_WEIGHT),
      .IN_HP1(S_AXI_HP1_WEIGHT),
      .IN_HP2(S_AXI_HP2_WEIGHT),
      .IN_HP3(S_AXI_HP3_WEIGHT),

      .weight_fifo_data (weight_fifo_data),
      .weight_fifo_valid(weight_fifo_valid),
      .weight_fifo_ready(weight_fifo_ready)
  );


*/

  vdotm_top #(
      .line_lengt(32),
      .line_cnt(128),
      .fmap_line_cnt(32),
      .reduction_rate(4),
      .in_weight_size(`INT4),
      .in_fmap_size(`BF16),
      .in_fmap_e_size(`BF16_EXP),
      .in_fmap_m_size(`BF16_MANTISSA)
  ) u_vdotm_top (
      .clk  (clk_core),
      .rst_n(rst_n_core),

      // weight
      .i_valid(weight_fifo_data[]),
      .IN_weight(weight_fifo_data[]),

      .IN_fmap_broadcast(fmap_broadcast),
      .IN_fmap_broadcast_valid(fmap_broadcast_valid),

      // e_max (from Cache for Normalization alignment)
      .IN_cached_emax_out(cached_emax_out),

      .activated_lane(activated_lane),

      .OUT_final_fp32(),
      .OUT_final_valid()
  );



  // 3. Systolic Array Engine (Modularized)
  logic [`DSP48E2_POUT_SIZE-1:0] raw_res_sum      [0:`ARRAY_SIZE_H-1];
  logic                          raw_res_sum_valid[0:`ARRAY_SIZE_H-1];
  logic [   `BF16_EXP_WIDTH-1:0] delayed_emax_32  [0:`ARRAY_SIZE_H-1];

  stlc_systolic_top u_systolic_engine (
      .clk(clk_core),
      .rst_n(rst_n_core),
      .i_clear(npu_clear),

      .global_weight_valid(global_weight_valid),
      .global_inst(global_inst),
      .global_inst_valid(global_inst_valid),

      .fmap_broadcast(fmap_broadcast),
      .fmap_broadcast_valid(fmap_broadcast_valid),

      .cached_emax_out(cached_emax_out),

      // Weight Input from FIFO (Direct)
      .weight_fifo_data (weight_fifo_data[0]),
      .weight_fifo_valid(weight_fifo_valid[0]),
      .weight_fifo_ready(weight_fifo_ready[0]),

      .raw_res_sum(raw_res_sum),
      .raw_res_sum_valid(raw_res_sum_valid),
      .delayed_emax_32(delayed_emax_32)
  );

  // 4. Output Pipeline (Result Normalization -> Result Packer -> FIFO)
  // Normalizers
  logic [`BF16_WIDTH-1:0] norm_res_seq      [0:`ARRAY_SIZE_H-1];
  logic                   norm_res_seq_valid[0:`ARRAY_SIZE_H-1];


  genvar n;
  generate
    for (n = 0; n < `ARRAY_SIZE_H; n++) begin : gen_norm
      stlc_result_normalizer u_norm_seq (
          .clk(clk_core),
          .rst_n(rst_n_core),
          .data_in(raw_res_sum[n]),
          .e_max(delayed_emax_32[n]),
          .valid_in(raw_res_sum_valid[n]),
          .data_out(norm_res_seq[n]),
          .valid_out(norm_res_seq_valid[n])
      );
    end
  endgenerate

  // Packer
  logic [`AXI_DATA_WIDTH-1:0] packed_res_data;
  logic                       packed_res_valid;
  logic                       packed_res_ready;

  FROM_stlc_result_packer u_packer (
      .clk(clk_core),
      .rst_n(rst_n_core),
      .row_res(norm_res_seq),
      .row_res_valid(norm_res_seq_valid),
      .packed_data(packed_res_data),
      .packed_valid(packed_res_valid),
      .packed_ready(packed_res_ready),
      .o_busy(packer_busy_status)
  );


  /*
  // Output FIFO
  xpm_fifo_axis #(
      .FIFO_DEPTH(`XPM_FIFO_DEPTH),
      .TDATA_WIDTH(`AXI_DATA_WIDTH),
      .FIFO_MEMORY_TYPE("block"),
      .CLOCKING_MODE("independent_clock")
  ) u_output_fifo (
      .s_aclk(clk_core),
      .m_aclk(clk_core),
      .s_aresetn(rst_n_core),
      .s_axis_tdata(packed_res_data),
      .s_axis_tvalid(packed_res_valid),
      .s_axis_tready(packed_res_ready),
      .m_axis_tdata(m_axis_result_tdata),
      .m_axis_tvalid(m_axis_result_tvalid),
      .m_axis_tready(m_axis_result_tready)
  );

*/
  // Status Assignment
  assign mmio_npu_stat[1] = 1'b0;
  assign mmio_npu_stat[31:2] = 30'd0;

endmodule
