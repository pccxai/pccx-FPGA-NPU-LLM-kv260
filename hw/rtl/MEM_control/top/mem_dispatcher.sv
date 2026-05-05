// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "GLOBAL_CONST.svh"
`include "mem_IO.svh"

import isa_pkg::*;

// ===| Module: mem_dispatcher — L2/ACP/CVO data-movement boundary |==============
// Purpose      : Hide the L2 cache + ACP DMA + CVO stream bridge behind a
//                single uop-driven contract. NPU_top sees only command-level
//                inputs and busy/full status outputs.
// Spec ref     : pccx v002 §5 (memory hierarchy), §5.3 (CVO bridge).
// Clock        : Dual-clock — clk_core (uop side) + clk_axi (ACP / DMA side).
//                CDC handled inside mem_GLOBAL_cache and child FIFOs.
// Reset        : rst_n_core / rst_axi_n active-low.
// Responsibilities (Keller "deep module" boundary):
//   - Shape constant RAM (MEMSET: fmap / weight array shapes).
//   - ACP DMA path   : host DDR4 ↔ L2 (MEMCPY host↔NPU).
//   - NPU burst path : L2 → GEMM fmap / GEMV fmap broadcast.
//   - CVO stream     : L2 → CVO engine input; CVO output → L2
//                      (via mem_CVO_stream_bridge — port-B arbitration:
//                       CVO bridge wins while OUT_cvo_busy is asserted).
// L2 addressing : 128-bit word units (word N = bytes [16N .. 16N+15]).
// L2 port-B arb : final_npu_* mux selects CVO bridge while busy, else NPU DMA.
// Throughput   : One ACP descriptor + one NPU descriptor in flight at a time
//                (FIFO-gated by mem_u_operation_queue).
// Backpressure : OUT_fifo_full asserts upstream when either operation queue
//                cannot accept a new descriptor.
// Errors       : none — illegal LOAD_uop.data_dest values fall through default.
// Counters     : none. (Stage D candidates: ACP_in_words, NPU_burst_words,
//                CVO_bridge_busy_cycles, queue_full_cycles).
// Assertions   : (Stage C) acp_rx_start and npu_rx_start are one-cycle pulses;
//                cvo_bridge_busy and npu_rx_start mutually exclusive on
//                final_npu_we.
// ===============================================================================

module mem_dispatcher #() (
    input logic clk_core,
    input logic rst_n_core,
    input logic clk_axi,
    input logic rst_axi_n,

    // ===| AXI-Stream ACP (external) |==========================================
    axis_if.slave  S_AXIS_ACP_FMAP,
    axis_if.master M_AXIS_ACP_RESULT,
    axis_if.master M_AXIS_L1_FMAP,

    // ===| Engine uop inputs |===================================================
    input  memory_control_uop_t IN_LOAD_uop,
    input  memory_set_uop_t     IN_mem_set_uop,
    input  cvo_control_uop_t    IN_CVO_uop,
    input  logic                IN_cvo_uop_valid,

    // ===| CVO streaming ports (to/from CVO_top) |==============================
    output logic [15:0] OUT_cvo_data,
    output logic        OUT_cvo_valid,
    input  logic        IN_cvo_data_ready,

    input  logic [15:0] IN_cvo_result,
    input  logic        IN_cvo_result_valid,
    output logic        OUT_cvo_result_ready,

    // ===| Status |=============================================================
    output logic OUT_fifo_full,
    output logic OUT_cvo_busy
);

  // ===| FIFO full aggregation |=================================================
  logic acp_cmd_fifo_full;
  logic npu_cmd_fifo_full;
  logic cvo_bridge_busy;

  assign OUT_fifo_full = acp_cmd_fifo_full | npu_cmd_fifo_full;
  assign OUT_cvo_busy  = cvo_bridge_busy;

  // ===| Shape Constant RAM — FMap |=============================================
  // Split write- and read-side pointers so the MEMSET handler (write side)
  // and the LOAD-uop handler (read side) drive independent flops; sharing a
  // single signal across two always_ff blocks tripped xelab's
  // multi-driver check.
  logic        fmap_write_enable;
  logic [ 5:0] fmap_shape_wr_addr;
  logic [ 5:0] fmap_shape_rd_addr;
  logic [16:0] fmap_arr_shape_X;
  logic [16:0] fmap_arr_shape_Y;
  logic [16:0] fmap_arr_shape_Z;
  shape_xyz_t  fmap_shape_wr_xyz;
  shape_xyz_t  fmap_shape_rd_xyz;
  logic [16:0] fmap_read_arr_shape_X;
  logic [16:0] fmap_read_arr_shape_Y;
  logic [16:0] fmap_read_arr_shape_Z;

  // ===| Shape Constant RAM — Weight |===========================================
  logic        weight_write_enable;
  logic [ 5:0] weight_shape_wr_addr;
  logic [ 5:0] weight_shape_rd_addr;
  logic [16:0] weight_arr_shape_X;
  logic [16:0] weight_arr_shape_Y;
  logic [16:0] weight_arr_shape_Z;
  shape_xyz_t  weight_shape_wr_xyz;
  shape_xyz_t  weight_shape_rd_xyz;
  logic [16:0] weight_read_arr_shape_X;
  logic [16:0] weight_read_arr_shape_Y;
  logic [16:0] weight_read_arr_shape_Z;

  // ===| MEMSET handler |========================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      fmap_write_enable   <= 1'b0;
      weight_write_enable <= 1'b0;
      fmap_shape_wr_addr  <= '0;
      weight_shape_wr_addr <= '0;
      fmap_arr_shape_X    <= '0;
      fmap_arr_shape_Y    <= '0;
      fmap_arr_shape_Z    <= '0;
      weight_arr_shape_X  <= '0;
      weight_arr_shape_Y  <= '0;
      weight_arr_shape_Z  <= '0;
    end else begin
      fmap_write_enable   <= 1'b0;
      weight_write_enable <= 1'b0;

      case (IN_mem_set_uop.dest_cache)
        data_to_fmap_shape: begin
          fmap_shape_wr_addr <= IN_mem_set_uop.dest_addr;
          fmap_arr_shape_X   <= IN_mem_set_uop.a_value;
          fmap_arr_shape_Y   <= IN_mem_set_uop.b_value;
          fmap_arr_shape_Z   <= IN_mem_set_uop.c_value;
          fmap_write_enable  <= 1'b1;
        end

        data_to_weight_shape: begin
          weight_shape_wr_addr <= IN_mem_set_uop.dest_addr;
          weight_arr_shape_X   <= IN_mem_set_uop.a_value;
          weight_arr_shape_Y   <= IN_mem_set_uop.b_value;
          weight_arr_shape_Z   <= IN_mem_set_uop.c_value;
          weight_write_enable  <= 1'b1;
        end

        default: ;
      endcase
    end
  end

  // Shape RAM reads are combinational. Keep read pointers on the current
  // consumer input so a one-cycle LOAD uses its own shape_ptr_addr rather
  // than the previous MEMSET/LOAD address.
  always_comb begin
    fmap_shape_rd_addr   = IN_LOAD_uop.shape_ptr_addr;
    weight_shape_rd_addr = IN_mem_set_uop.dest_addr;
  end

  assign fmap_shape_wr_xyz.x = fmap_arr_shape_X;
  assign fmap_shape_wr_xyz.y = fmap_arr_shape_Y;
  assign fmap_shape_wr_xyz.z = fmap_arr_shape_Z;
  assign fmap_read_arr_shape_X = fmap_shape_rd_xyz.x;
  assign fmap_read_arr_shape_Y = fmap_shape_rd_xyz.y;
  assign fmap_read_arr_shape_Z = fmap_shape_rd_xyz.z;

  shape_const_ram u_fmap_shape (
      .clk   (clk_core),
      .rst_n (rst_n_core),
      .wr_en (fmap_write_enable),
      .wr_addr(fmap_shape_wr_addr),
      .wr_xyz(fmap_shape_wr_xyz),
      .rd_addr(fmap_shape_rd_addr),
      .rd_xyz(fmap_shape_rd_xyz)
  );

  assign weight_shape_wr_xyz.x = weight_arr_shape_X;
  assign weight_shape_wr_xyz.y = weight_arr_shape_Y;
  assign weight_shape_wr_xyz.z = weight_arr_shape_Z;
  assign weight_read_arr_shape_X = weight_shape_rd_xyz.x;
  assign weight_read_arr_shape_Y = weight_shape_rd_xyz.y;
  assign weight_read_arr_shape_Z = weight_shape_rd_xyz.z;

  shape_const_ram u_weight_shape (
      .clk   (clk_core),
      .rst_n (rst_n_core),
      .wr_en (weight_write_enable),
      .wr_addr(weight_shape_wr_addr),
      .wr_xyz(weight_shape_wr_xyz),
      .rd_addr(weight_shape_rd_addr),
      .rd_xyz(weight_shape_rd_xyz)
  );

  // ===| LOAD uop → ACP / NPU command translation |==============================
  // The shape RAM returns BF16 element dimensions, but ACP/NPU commands need
  // 128-bit word counts. Keep that arithmetic behind a registered boundary:
  //   s1: capture route/base/shape
  //   s2: compute X*Y
  //   s3: compute ceil(X*Y*Z/8)
  //   issue: enqueue ACP/NPU descriptor
  // This removes the prior shape_const_ram -> DSP -> DSP -> descriptor path
  // from the 400 MHz core cycle while preserving the external command contract.
  logic        IN_acp_rdy;
  acp_uop_t acp_uop;
  logic        acp_rx_start;

  logic        IN_npu_rdy;
  npu_uop_t npu_uop;
  logic        npu_rx_start;

  logic        load_s1_valid;
  logic        load_s1_to_acp;
  logic        load_s1_to_npu;
  logic        load_s1_write_en;
  logic [16:0] load_s1_base_addr;
  logic [16:0] load_s1_shape_x;
  logic [16:0] load_s1_shape_y;
  logic [16:0] load_s1_shape_z;

  logic        load_s2_valid;
  logic        load_s2_to_acp;
  logic        load_s2_to_npu;
  logic        load_s2_write_en;
  logic [16:0] load_s2_base_addr;
  logic [33:0] load_s2_shape_xy;
  logic [16:0] load_s2_shape_z;

  logic        load_s3_valid;
  logic        load_s3_to_acp;
  logic        load_s3_to_npu;
  logic        load_s3_write_en;
  logic [16:0] load_s3_base_addr;
  logic [16:0] load_s3_word_total;
  logic [50:0] load_s2_words_plus7;

  assign load_s2_words_plus7 = (load_s2_shape_xy * load_s2_shape_z) + 51'd7;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_rx_start      <= 1'b0;
      npu_rx_start      <= 1'b0;
      IN_acp_rdy        <= 1'b0;
      IN_npu_rdy        <= 1'b0;
      acp_uop           <= '0;
      npu_uop           <= '0;
      load_s1_valid     <= 1'b0;
      load_s1_to_acp    <= 1'b0;
      load_s1_to_npu    <= 1'b0;
      load_s1_write_en  <= 1'b0;
      load_s1_base_addr <= '0;
      load_s1_shape_x   <= '0;
      load_s1_shape_y   <= '0;
      load_s1_shape_z   <= '0;
      load_s2_valid     <= 1'b0;
      load_s2_to_acp    <= 1'b0;
      load_s2_to_npu    <= 1'b0;
      load_s2_write_en  <= 1'b0;
      load_s2_base_addr <= '0;
      load_s2_shape_xy  <= '0;
      load_s2_shape_z   <= '0;
      load_s3_valid     <= 1'b0;
      load_s3_to_acp    <= 1'b0;
      load_s3_to_npu    <= 1'b0;
      load_s3_write_en  <= 1'b0;
      load_s3_base_addr <= '0;
      load_s3_word_total <= '0;
    end else begin
      acp_rx_start <= 1'b0;
      npu_rx_start <= 1'b0;
      IN_acp_rdy   <= 1'b0;
      IN_npu_rdy   <= 1'b0;

      load_s1_valid     <= 1'b0;
      load_s1_to_acp    <= 1'b0;
      load_s1_to_npu    <= 1'b0;
      load_s1_write_en  <= 1'b0;
      load_s1_base_addr <= '0;
      load_s1_shape_x   <= fmap_read_arr_shape_X;
      load_s1_shape_y   <= fmap_read_arr_shape_Y;
      load_s1_shape_z   <= fmap_read_arr_shape_Z;

      case (IN_LOAD_uop.data_dest)
        // Host DDR4 → L2 (feature map DMA in)
        from_host_to_L2: begin
          load_s1_valid     <= 1'b1;
          load_s1_to_acp    <= 1'b1;
          load_s1_write_en  <= `PORT_MOD_E_WRITE;
          load_s1_base_addr <= IN_LOAD_uop.dest_addr;
        end

        // L2 → host DDR4 (result DMA out)
        from_L2_to_host: begin
          load_s1_valid     <= 1'b1;
          load_s1_to_acp    <= 1'b1;
          load_s1_write_en  <= `PORT_MOD_E_READ;
          load_s1_base_addr <= IN_LOAD_uop.src_addr;
        end

        // L2 → GEMM fmap broadcast
        from_L2_to_L1_GEMM: begin
          load_s1_valid     <= 1'b1;
          load_s1_to_npu    <= 1'b1;
          load_s1_write_en  <= `PORT_MOD_E_READ;
          load_s1_base_addr <= IN_LOAD_uop.src_addr;
        end

        // L2 → GEMV fmap broadcast
        from_L2_to_L1_GEMV: begin
          load_s1_valid     <= 1'b1;
          load_s1_to_npu    <= 1'b1;
          load_s1_write_en  <= `PORT_MOD_E_READ;
          load_s1_base_addr <= IN_LOAD_uop.src_addr;
        end

        // L2 → CVO input (handled by mem_CVO_stream_bridge below)
        from_L2_to_CVO: ;  // bridge watches IN_CVO_uop directly

        default: ;
      endcase

      load_s2_valid     <= load_s1_valid;
      load_s2_to_acp    <= load_s1_to_acp;
      load_s2_to_npu    <= load_s1_to_npu;
      load_s2_write_en  <= load_s1_write_en;
      load_s2_base_addr <= load_s1_base_addr;
      load_s2_shape_xy  <= load_s1_shape_x * load_s1_shape_y;
      load_s2_shape_z   <= load_s1_shape_z;

      load_s3_valid      <= load_s2_valid;
      load_s3_to_acp     <= load_s2_to_acp;
      load_s3_to_npu     <= load_s2_to_npu;
      load_s3_write_en   <= load_s2_write_en;
      load_s3_base_addr  <= load_s2_base_addr;
      load_s3_word_total <= load_s2_words_plus7[19:3];

      if (load_s3_valid && load_s3_to_acp) begin
        acp_uop <= '{
            write_en  : load_s3_write_en,
            base_addr : load_s3_base_addr,
            end_addr  : load_s3_base_addr + load_s3_word_total
        };
        acp_rx_start <= 1'b1;
        IN_acp_rdy   <= 1'b1;
      end

      if (load_s3_valid && load_s3_to_npu) begin
        npu_uop <= '{
            write_en  : load_s3_write_en,
            base_addr : load_s3_base_addr,
            end_addr  : load_s3_base_addr + load_s3_word_total
        };
        npu_rx_start <= 1'b1;
        IN_npu_rdy   <= 1'b1;
      end
    end
  end

  // ===| Operation queues |======================================================
  acp_uop_t OUT_acp_cmd;
  npu_uop_t OUT_npu_cmd;
  logic     OUT_acp_cmd_valid;
  logic     OUT_npu_cmd_valid;
  logic     acp_is_busy_wire;
  logic     npu_is_busy_wire;

  mem_u_operation_queue #() u_op_queue (
      .clk_core             (clk_core),
      .rst_n_core           (rst_n_core),
      .IN_acp_rdy           (IN_acp_rdy),
      .IN_acp_cmd           (acp_uop),
      .OUT_acp_cmd          (OUT_acp_cmd),
      .OUT_acp_cmd_valid    (OUT_acp_cmd_valid),
      .OUT_acp_cmd_fifo_full(acp_cmd_fifo_full),
      .IN_acp_is_busy       (acp_is_busy_wire),
      .IN_npu_rdy           (IN_npu_rdy),
      .IN_npu_cmd           (npu_uop),
      .OUT_npu_cmd          (OUT_npu_cmd),
      .OUT_npu_cmd_valid    (OUT_npu_cmd_valid),
      .OUT_npu_cmd_fifo_full(npu_cmd_fifo_full),
      .IN_npu_is_busy       (npu_is_busy_wire)
  );

  // ===| L2 cache controller |===================================================
  // CVO bridge drives L2 port B when active; otherwise port B is driven by
  // the NPU DMA state machine in mem_GLOBAL_cache.
  logic        cvo_l2_we;
  logic [16:0] cvo_l2_addr;
  logic [127:0] cvo_l2_wdata;
  logic [127:0] cvo_l2_rdata;

  logic [127:0] npu_l2_wdata;
  logic [127:0] npu_l2_rdata;

  // Direct NPU writes into L2 are reserved for the result-writeback phase.
  // Until that producer is wired, keep the inactive write data deterministic;
  // final_npu_we remains deasserted for current L2->L1 read routes.
  assign npu_l2_wdata = '0;

  // Route L2 rdata to the appropriate consumer
  assign cvo_l2_rdata = npu_l2_rdata;  // shared read bus

  mem_GLOBAL_cache #() u_l2_cache (
      .clk_core         (clk_core),
      .rst_n_core       (rst_n_core),
      .clk_axi          (clk_axi),
      .rst_axi_n        (rst_axi_n),

      .S_AXIS_ACP_FMAP  (S_AXIS_ACP_FMAP),
      .M_AXIS_ACP_RESULT(M_AXIS_ACP_RESULT),
      .M_AXIS_NPU_FMAP  (M_AXIS_L1_FMAP),

      // ACP control
      .IN_acp_write_en  (OUT_acp_cmd.write_en),
      .IN_acp_base_addr (OUT_acp_cmd.base_addr),
      .IN_acp_end_addr  (OUT_acp_cmd.end_addr),
      .IN_acp_rx_start  (OUT_acp_cmd_valid),
      .OUT_acp_is_busy  (acp_is_busy_wire),

      // NPU port B (CVO bridge or DMA state machine)
      .IN_npu_write_en  (OUT_npu_cmd.write_en),
      .IN_npu_base_addr (OUT_npu_cmd.base_addr),
      .IN_npu_end_addr  (OUT_npu_cmd.end_addr),
      .IN_npu_rx_start  (OUT_npu_cmd_valid),
      .OUT_npu_is_busy  (npu_is_busy_wire),

      // Direct port-B owner for CVO L2 read/write bursts.
      .IN_npu_direct_en   (cvo_bridge_busy),
      .IN_npu_direct_we   (cvo_l2_we),
      .IN_npu_direct_addr (cvo_l2_addr),
      .IN_npu_direct_wdata(cvo_l2_wdata),

      .IN_npu_wdata     (npu_l2_wdata),
      .OUT_npu_rdata    (npu_l2_rdata)
  );

  // ===| CVO Stream Bridge |=====================================================
  logic cvo_bridge_done;

  mem_CVO_stream_bridge u_cvo_bridge (
      .clk                (clk_core),
      .rst_n              (rst_n_core),

      .IN_cvo_uop         (IN_CVO_uop),
      .IN_cvo_uop_valid   (IN_cvo_uop_valid),
      .OUT_busy           (cvo_bridge_busy),
      .OUT_done           (cvo_bridge_done),

      // L2 port B direct access
      .OUT_l2_we          (cvo_l2_we),
      .OUT_l2_addr        (cvo_l2_addr),
      .OUT_l2_wdata       (cvo_l2_wdata),
      .IN_l2_rdata        (cvo_l2_rdata),

      // CVO data stream
      .OUT_cvo_data       (OUT_cvo_data),
      .OUT_cvo_valid      (OUT_cvo_valid),
      .IN_cvo_data_ready  (IN_cvo_data_ready),

      // CVO result stream
      .IN_cvo_result        (IN_cvo_result),
      .IN_cvo_result_valid  (IN_cvo_result_valid),
      .OUT_cvo_result_ready (OUT_cvo_result_ready)
  );

endmodule
