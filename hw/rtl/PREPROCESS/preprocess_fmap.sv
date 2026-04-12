`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "npu_interfaces.svh"

/**
 * Module: gemm_fmap_preprocessor
 *
 * Role:
 * - Combined 256-bit FMap streaming from HPC0/HPC1.
 * - e_max (Exponent) extraction and caching for BFP.
 * - Mantissa shifting to Fixed-point.
 * - SRAM Caching for broadcasting to multiple compute engines (Branch point).
 */
module preprocess_fmap #(
    parameter fmap_width = `ACP_PORT_IN
) (
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // AXI4-Stream Interfaces from ACP
    axis_if.slave S_AXIS_ACP_FMAP,  // ACP (128-bit)

    // Control from Brain
    input logic i_rd_start,




    // Output to Branch Engines (Systolic / GEMV / CVO)
    output logic [`FIXED_MANT_WIDTH-1:0] o_fmap_broadcast[0:`ARRAY_SIZE_H-1],
    output logic                         o_fmap_valid,

    output logic [`BF16_EXP_WIDTH-1:0] o_cached_emax[0:`ARRAY_SIZE_H-1]
);

  // ===| Bridge & Alignment: 256-bit Feature Map |=======
  //logic [`ACP_PORT_IN:0] s_axis_fmap_combined_tdata;
  //logic                  s_axis_fmap_combined_tvalid;
  //logic                  s_axis_fmap_combined_tready;

  //assign s_axis_fmap_combined_tdata = S_AXIS_ACP_FMAP.tdata;
  //assign s_axis_fmap_combined_tvalid = S_AXIS_FMAP0.tvalid & S_AXIS_FMAP1.tvalid;

  //assign S_AXIS_FMAP0.tready = s_axis_fmap_combined_tready & S_AXIS_FMAP1.tvalid;
  //assign S_AXIS_FMAP1.tready = s_axis_fmap_combined_tready & S_AXIS_FMAP0.tvalid;

  // 256-bit FIFO for FMap
  logic [fmap_width:0] fmap_fifo_data;
  logic                fmap_fifo_valid;
  logic                fmap_fifo_ready;


  xpm_fifo_axis #(
      .FIFO_DEPTH(`XPM_FIFO_DEPTH),
      .TDATA_WIDTH(256),
      .FIFO_MEMORY_TYPE("block"),
      .CLOCKING_MODE("common_clock")
  ) u_fmap_fifo (
      .s_aclk(clk),
      .m_aclk(clk),
      .s_aresetn(rst_n),
      .s_axis_tdata(S_AXIS_ACP_FMAP.tdata),
      .s_axis_tvalid(S_AXIS_ACP_FMAP.tvalid),
      .s_axis_tready(S_AXIS_ACP_FMAP.tready),
      .m_axis_tdata(fmap_fifo_data),
      .m_axis_tvalid(fmap_fifo_valid),
      .m_axis_tready(fmap_fifo_ready)
  );

  // ===| e_max parsing & cache logic |=======
  logic [`BF16_EXP_WIDTH-1:0] active_emax[0:`ARRAY_SIZE_H-1];
  logic fmap_word_toggle;
  logic emax_group_valid;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      fmap_word_toggle <= 1'b0;
      emax_group_valid <= 1'b0;
    end else if (fmap_fifo_valid && fmap_fifo_ready) begin
      fmap_word_toggle <= ~fmap_word_toggle;
      for (int k = 0; k < 16; k++) begin
        if (fmap_word_toggle == 1'b0) active_emax[k] <= fmap_fifo_data[(k*16)+7+:8];
        else active_emax[k+16] <= fmap_fifo_data[(k*16)+7+:8];
      end
      emax_group_valid <= (fmap_word_toggle == 1'b1);
    end else begin
      emax_group_valid <= 1'b0;
    end
  end

  logic [`BF16_EXP_WIDTH-1:0] emax_cache_mem[0:1023][0:`ARRAY_SIZE_H-1];
  logic [9:0] emax_wr_addr, emax_rd_addr;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      emax_wr_addr <= 0;
    end else if (emax_group_valid) begin
      for (int i = 0; i < `ARRAY_SIZE_H; i++) begin
        emax_cache_mem[emax_wr_addr][i] <= active_emax[i];
      end
      emax_wr_addr <= emax_wr_addr + 1;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      emax_rd_addr <= 0;
    end else if (i_rd_start) begin
      emax_rd_addr <= 0;
    end
    for (int i = 0; i < `ARRAY_SIZE_H; i++) begin
      o_cached_emax[i] <= emax_cache_mem[emax_rd_addr][i];
    end
  end

  // ===| Mantissa Shifter & SRAM Cache |=======
  logic [431:0] fixed_fmap;
  logic         fixed_fmap_valid;
  logic         fmap_shifter_ready;

  preprocess_bf16_fixed_pipeline u_fmap_shifter (
      .clk(clk),
      .rst_n(rst_n),
      .s_axis_tdata(fmap_fifo_data),
      .s_axis_tvalid(fmap_fifo_valid),
      .s_axis_tready(fmap_shifter_ready),
      .m_axis_tdata(fixed_fmap),
      .m_axis_tvalid(fixed_fmap_valid),
      .m_axis_tready(1'b1)
  );
  assign fmap_fifo_ready = fmap_shifter_ready;

  logic [6:0] sram_wr_addr;
  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) sram_wr_addr <= 0;
    else if (fixed_fmap_valid) sram_wr_addr <= sram_wr_addr + 1;
  end


  fmap_cache #(
      .DATA_WIDTH(`FIXED_MANT_WIDTH),
      .WRITE_LANES(16),
      .CACHE_DEPTH(`FMAP_CACHE_DEPTH),
      .LANES(`ARRAY_SIZE_H)
  ) u_fmap_sram (
      .clk(clk),
      .rst_n(rst_n),
      .wr_data(fixed_fmap),
      .wr_valid(fixed_fmap_valid),
      .wr_addr(sram_wr_addr),
      .wr_en(1'b1),
      .rd_start(i_rd_start),
      .rd_data_broadcast(o_fmap_broadcast),
      .rd_valid(o_fmap_valid)
  );

endmodule
