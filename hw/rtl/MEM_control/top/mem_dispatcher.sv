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
    output logic OUT_cvo_busy,
    output logic OUT_cvo_done
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

  // ===| Shape totals (word counts for DMA) |====================================
  logic [16:0] fmap_word_total;
  logic [16:0] weight_word_total;

  // Total BF16 elements → 128-bit words: ceil(X*Y*Z / 8)
  assign fmap_word_total   = (fmap_read_arr_shape_X   * fmap_read_arr_shape_Y   * fmap_read_arr_shape_Z   + 7) >> 3;
  assign weight_word_total = (weight_read_arr_shape_X * weight_read_arr_shape_Y * weight_read_arr_shape_Z + 7) >> 3;

  // ===| LOAD uop → ACP / NPU command translation |==============================
  logic    IN_acp_rdy;
  acp_uop_t acp_uop;
  logic    acp_rx_start;

  logic    IN_npu_rdy;
  npu_uop_t npu_uop;
  logic    npu_rx_start;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_rx_start <= 1'b0;
      npu_rx_start <= 1'b0;
      IN_acp_rdy   <= 1'b0;
      IN_npu_rdy   <= 1'b0;
    end else begin
      acp_rx_start <= 1'b0;
      npu_rx_start <= 1'b0;
      IN_acp_rdy   <= 1'b0;
      IN_npu_rdy   <= 1'b0;

      case (IN_LOAD_uop.data_dest)
        // Host DDR4 → L2 (feature map DMA in)
        from_host_to_L2: begin
          acp_uop <= '{
              write_en  : `PORT_MOD_E_WRITE,
              base_addr : IN_LOAD_uop.dest_addr,
              end_addr  : IN_LOAD_uop.dest_addr + 17'(fmap_word_total)
          };
          acp_rx_start <= 1'b1;
          IN_acp_rdy   <= 1'b1;
        end

        // L2 → host DDR4 (result DMA out)
        from_L2_to_host: begin
          acp_uop <= '{
              write_en  : `PORT_MOD_E_READ,
              base_addr : IN_LOAD_uop.src_addr,
              end_addr  : IN_LOAD_uop.src_addr + 17'(fmap_word_total)
          };
          acp_rx_start <= 1'b1;
          IN_acp_rdy   <= 1'b1;
        end

        // L2 → GEMM fmap broadcast
        from_L2_to_L1_GEMM: begin
          npu_uop <= '{
              write_en  : `PORT_MOD_E_READ,
              base_addr : IN_LOAD_uop.src_addr,
              end_addr  : IN_LOAD_uop.src_addr + 17'(fmap_word_total)
          };
          npu_rx_start <= 1'b1;
          IN_npu_rdy   <= 1'b1;
        end

        // L2 → GEMV fmap broadcast
        from_L2_to_L1_GEMV: begin
          npu_uop <= '{
              write_en  : `PORT_MOD_E_READ,
              base_addr : IN_LOAD_uop.src_addr,
              end_addr  : IN_LOAD_uop.src_addr + 17'(fmap_word_total)
          };
          npu_rx_start <= 1'b1;
          IN_npu_rdy   <= 1'b1;
        end

        // L2 → CVO input (handled by mem_CVO_stream_bridge below)
        from_L2_to_CVO: ;  // bridge watches IN_CVO_uop directly

        default: ;
      endcase
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

  logic        npu_l2_we;
  logic [16:0] npu_l2_addr;
  logic [127:0] npu_l2_wdata;
  logic [127:0] npu_l2_rdata;

  // Port B arbitration: CVO bridge wins when busy
  logic        final_npu_we;
  logic [16:0] final_npu_addr;
  logic [127:0] final_npu_wdata;

  always_comb begin
    if (cvo_bridge_busy) begin
      final_npu_we    = cvo_l2_we;
      final_npu_addr  = cvo_l2_addr;
      final_npu_wdata = cvo_l2_wdata;
    end else begin
      final_npu_we    = npu_l2_we;
      final_npu_addr  = npu_l2_addr;
      final_npu_wdata = npu_l2_wdata;
    end
  end

  // Route L2 rdata to the appropriate consumer
  assign cvo_l2_rdata = npu_l2_rdata;  // shared read bus

  mem_GLOBAL_cache #() u_l2_cache (
      .clk_core         (clk_core),
      .rst_n_core       (rst_n_core),
      .clk_axi          (clk_axi),
      .rst_axi_n        (rst_axi_n),

      .S_AXIS_ACP_FMAP  (S_AXIS_ACP_FMAP),
      .M_AXIS_ACP_RESULT(M_AXIS_ACP_RESULT),

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

      .IN_npu_wdata     (final_npu_wdata),
      .OUT_npu_rdata    (npu_l2_rdata)
  );

  // ===| CVO Stream Bridge |=====================================================
  logic cvo_bridge_done;

  assign OUT_cvo_done = cvo_bridge_done;

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
