`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "mem_IO.svh"

/**
 * Module: NPU_top
 * Target: Kria KV260 @ 400MHz
 * 
 * Architecture V2 (Segregated Physical Ports):
 * - HPC/ACP: Dedicated to low-latency Feature Map caching.
 * - HP0~HP3: Dedicated to high-throughput Weight streaming.
 * - HPM (MMIO): Centralized control & VLIW Instruction issuing.
 */
module NPU_top (
    // Clock & Reset (Must be associated with AXI interfaces for Vivado BD)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS_FMAP:S_AXI_WEIGHT:M_AXIS_RESULT, ASSOCIATED_RESET rst_n" *)
    input  logic clk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  logic rst_n,

    // Control Plane (Connect to AXI GPIO in Vivado Block Design)
    input  logic [31:0] mmio_npu_cmd,  // [0]=Start, [1]=Clear, [4:2]=Inst(VLIW)
    output logic [31:0] mmio_npu_stat, // [0]=Done, [1]=FMap_Ready

    // Feature Map Data Path (AXI4-Stream Slave)
    //[`AXI_DATA_WIDTH-1:0]
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_FMAP TDATA" *)
    input  logic [127:0] s_axis_fmap_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_FMAP TVALID" *)
    input  logic                       s_axis_fmap_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_FMAP TREADY" *)
    output logic                       s_axis_fmap_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_FMAP TLAST" *)
    input  logic                       s_axis_fmap_tlast,

    // Weight Data Path (AXI4-Stream Slave) - Perfectly done!
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXI_WEIGHT TDATA" *)
    input  logic      [511:0]          s_axis_weight_tdata_FLAT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXI_WEIGHT TVALID" *)
    input  logic      [3:0]            s_axis_weight_tvalid_FLAT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXI_WEIGHT TREADY" *)
    output logic      [3:0]            s_axis_weight_tready_FLAT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXI_WEIGHT TLAST" *)
    input  logic      [3:0]            s_axis_weight_tlast_FLAT,
    
    // Output Data Path (AXI4-Stream Master)
    // [`AXI_DATA_WIDTH-1:0]
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TDATA" *)
    output logic [127:0] m_axis_result_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TVALID" *)
    output logic                       m_axis_result_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TREADY" *)
    input  logic                       m_axis_result_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TLAST" *)
    output logic         m_axis_result_tlast

);

    // ===| Internal Array Mapping |=======
    logic [127:0] s_axis_weight_tdata  [0:3];
    logic         s_axis_weight_tvalid [0:3];
    logic         s_axis_weight_tready [0:3];
    genvar p;
    generate
        for (p = 0; p < 4; p++) begin : unpack_weight
            assign s_axis_weight_tdata[p]  = s_axis_weight_tdata_FLAT[p*128 +: 128];
            assign s_axis_weight_tvalid[p] = s_axis_weight_tvalid_FLAT[p];
            assign s_axis_weight_tready_FLAT[p] = s_axis_weight_tready[p];
        end
    endgenerate
    // ===| Internal Array Mapping |============================================


    // ===| Control Signals Extraction |=======
    logic npu_start;
    logic npu_clear;
    logic [2:0] npu_inst;
    
    assign npu_start = mmio_npu_cmd[`CMD_START_BIT];
    assign npu_clear = mmio_npu_cmd[`CMD_CLEAR_BIT];
    assign npu_inst  = mmio_npu_cmd[`CMD_INST_END_BIT:`CMD_INST_START_BIT];

    // Central Controller (Global FSM)
    logic global_weight_valid;
    logic global_sram_rd_start;
    logic [2:0] global_inst;
    logic global_inst_valid;

    stlc_global_fsm u_brain (
        .clk(clk), .rst_n(rst_n),
        .npu_start(npu_start),
        .npu_done(mmio_npu_stat[0]),
        
        .i_weight_valid(global_weight_valid),
        .sram_rd_start(global_sram_rd_start),
        .inst_out(global_inst),
        .inst_valid_out(global_inst_valid)
    );

    // 1. Feature Map Pipeline (HPC/ACP -> SRAM Cache)
    // FIFO for FMap
    logic [`AXI_DATA_WIDTH-1:0] fmap_fifo_data;
    logic                       fmap_fifo_valid;
    logic                       fmap_fifo_ready;

    xpm_fifo_axis #(
        .FIFO_DEPTH(`XPM_FIFO_DEPTH),
        .TDATA_WIDTH(`AXI_DATA_WIDTH),
        .FIFO_MEMORY_TYPE("block"),
        .CLOCKING_MODE("common_clock")
    ) u_fmap_fifo (
        .s_aclk(clk), .m_aclk(clk),
        .s_aresetn(rst_n),
        .s_axis_tdata(s_axis_fmap_tdata),
        .s_axis_tvalid(s_axis_fmap_tvalid),
        .s_axis_tready(s_axis_fmap_tready),
        .m_axis_tdata(fmap_fifo_data),
        .m_axis_tvalid(fmap_fifo_valid),
        .m_axis_tready(fmap_fifo_ready)
    );

    // Dynamic e_max extraction (32 columns) directly from the incoming 128-bit chunk
    // Note: Assuming 128-bit contains 8 words, you might need a mechanism to gather 
    // all 32 exponents over 4 cycles, or adapt if your data packing is different.
    // For simplicity, we define the wire here. Implementation will follow in a sub-module or FSM.
    logic [`BF16_EXP_WIDTH-1:0] active_emax [0:`ARRAY_SIZE_H-1];

    // Shifter
    logic [`FIXED_MANT_WIDTH-1:0] fixed_fmap;
    logic        fixed_fmap_valid;
    
    // Using Lane 0 (first 16 bits) for Shifter for now. Needs expansion for full 128-bit.
    stlc_bf16_fixed_pipeline u_fmap_shifter (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(fmap_fifo_data[`BF16_WIDTH-1:0]), 
        .s_axis_tvalid(fmap_fifo_valid),
        .m_axis_tdata(fixed_fmap),
        .m_axis_tvalid(fixed_fmap_valid),
        .m_axis_tready(1'b1) 
    );
    assign fmap_fifo_ready = 1'b1;

    // SRAM Cache
    logic [`FIXED_MANT_WIDTH-1:0] fmap_broadcast [0:`ARRAY_SIZE_H-1];
    logic        fmap_broadcast_valid;
    logic [`FMAP_ADDR_WIDTH-1:0] sram_wr_addr;

    // Temporary basic write logic
    always_ff @(posedge clk) begin
        if (!rst_n || npu_clear) sram_wr_addr <= 0;
        else if (fixed_fmap_valid) sram_wr_addr <= sram_wr_addr + 1;
    end

    stlc_fmap_cache #(
        .DATA_WIDTH(`FIXED_MANT_WIDTH),
        .CACHE_DEPTH(`FMAP_CACHE_DEPTH),
        .LANES(`ARRAY_SIZE_H)
    ) u_fmap_sram (
        .clk(clk), .rst_n(rst_n),
        .wr_data(fixed_fmap),
        .wr_valid(fixed_fmap_valid),
        .wr_addr(sram_wr_addr),
        .wr_en(1'b1), 
        .rd_start(global_sram_rd_start), 
        .rd_data_broadcast(fmap_broadcast),
        .rd_valid(fmap_broadcast_valid)
    );

    // Staggered Delay Line for FMap & Instructions
    logic [`FIXED_MANT_WIDTH-1:0] staggered_fmap [0:`ARRAY_SIZE_H-1];
    logic                         staggered_fmap_valid [0:`ARRAY_SIZE_H-1];
    logic [2:0]                   staggered_inst [0:`ARRAY_SIZE_H-1];
    logic                         staggered_inst_valid [0:`ARRAY_SIZE_H-1];

    stlc_fmap_staggered_delay #(
        .DATA_WIDTH(`FIXED_MANT_WIDTH), 
        .ARRAY_SIZE(`ARRAY_SIZE_H)
    ) u_delay_line (
        .clk(clk), .rst_n(rst_n),
        .fmap_in(fmap_broadcast),
        .fmap_valid(fmap_broadcast_valid),
        .global_inst(global_inst),
        .global_inst_valid(global_inst_valid),
        .row_data(staggered_fmap),
        .row_valid(staggered_fmap_valid),
        .row_inst(staggered_inst),
        .row_inst_valid(staggered_inst_valid)
    );

    // 2. Weight Pipeline (HP0~3 -> Systolic Array H_in)
    
    logic [`AXI_DATA_WIDTH-1:0] weight_fifo_data  [0:`AXI_WEIGHT_PORT_CNT-1];
    logic                       weight_fifo_valid [0:`AXI_WEIGHT_PORT_CNT-1];
    logic                       weight_fifo_ready [0:`AXI_WEIGHT_PORT_CNT-1];

    genvar i;
    generate
        for (i = 0; i < `AXI_WEIGHT_PORT_CNT; i++) begin : weight_fifos
            xpm_fifo_axis #(
                .FIFO_DEPTH(`XPM_FIFO_DEPTH),
                .TDATA_WIDTH(`AXI_DATA_WIDTH),
                .FIFO_MEMORY_TYPE("block"),
                .CLOCKING_MODE("common_clock")
            ) u_w_fifo (
                .s_aclk(clk), .s_aresetn(rst_n),
                .s_axis_tdata(s_axis_weight_tdata[i]),
                .s_axis_tvalid(s_axis_weight_tvalid[i]),
                .s_axis_tready(s_axis_weight_tready[i]),
                .m_axis_tdata(weight_fifo_data[i]),
                .m_axis_tvalid(weight_fifo_valid[i]),
                .m_axis_tready(weight_fifo_ready[i])
            );
        end
    endgenerate

    // Dispatcher
    logic [`INT4_WIDTH-1:0] unpacked_weights [0:`ARRAY_SIZE_H-1];
    logic       weights_ready_for_array;

    // Using Port 0 for now. Needs mapping logic for all ports depending on layout.
    TO_stlc_weight_dispatcher u_weight_unpacker (
        .clk(clk), .rst_n(rst_n),
        .fifo_data(weight_fifo_data[0]),
        .fifo_valid(weight_fifo_valid[0]),
        .fifo_ready(weight_fifo_ready[0]),
        .weight_out(unpacked_weights),
        .weight_valid(weights_ready_for_array)
    );


    // 3. Systolic Array Core (The Engine)
    
    logic [`DSP_RESULT_SIZE-1:0] raw_res_seq [0:`ARRAY_SIZE_H-1];
    logic [`DSP_RESULT_SIZE-1:0] raw_res_sum [0:`ARRAY_SIZE_H-1];

    stlc_NxN_array #(
        .ARRAY_HORIZONTAL(`ARRAY_SIZE_H),
        .ARRAY_VERTICAL(`ARRAY_SIZE_V)
    ) u_compute_core (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(npu_clear), 
        .i_weight_valid(global_weight_valid), 
        
        // Horizontal: Weights
        .H_in(unpacked_weights), 
        
        // Vertical: Feature Map Broadcast & Instructions (Staggered)
        .V_in(staggered_fmap), 
        .in_valid(staggered_fmap_valid),
        .inst_in(staggered_inst),
        .inst_valid_in(staggered_inst_valid),
        
        .V_out(raw_res_seq),
        .V_ACC_out(raw_res_sum)
    );

    // 4. Output Pipeline (Result Normalization -> Result Packer -> FIFO)

    // e_max Delay Line (Matches Array Latency)
    localparam TOTAL_LATENCY = `SYSTOLIC_TOTAL_LATENCY;
    logic [`BF16_EXP_WIDTH-1:0] emax_pipe [0:`ARRAY_SIZE_H-1][0:TOTAL_LATENCY-1];
    logic [`BF16_EXP_WIDTH-1:0] delayed_emax_32 [0:`ARRAY_SIZE_H-1];

    always_ff @(posedge clk) begin
        if (rst_n) begin
            for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
                emax_pipe[c][0] <= active_emax[c]; // From FMap extraction logic
                for (int d = 1; d < TOTAL_LATENCY; d++) begin
                    emax_pipe[c][d] <= emax_pipe[c][d-1];
                end
            end
        end
    end

    always_comb begin
        for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
            delayed_emax_32[c] = emax_pipe[c][TOTAL_LATENCY-1];
        end
    end

    // Normalizers
    logic [`BF16_WIDTH-1:0] norm_res_seq       [0:`ARRAY_SIZE_H-1];
    logic                   norm_res_seq_valid [0:`ARRAY_SIZE_H-1];

    genvar n;
    generate
        for (n = 0; n < `ARRAY_SIZE_H; n++) begin : gen_norm
            stlc_result_normalizer u_norm_seq (
                .clk(clk), .rst_n(rst_n),
                .data_in(raw_res_sum[n]),           
                .e_max(delayed_emax_32[n]),         
                .valid_in(1'b1), // Replace with actual valid signal from array 
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
        .clk(clk), .rst_n(rst_n),
        .row_res(norm_res_seq),             
        .row_res_valid(norm_res_seq_valid), 
        .packed_data(packed_res_data),
        .packed_valid(packed_res_valid),
        .packed_ready(packed_res_ready)
    );

    // Output FIFO
    xpm_fifo_axis #(
        .FIFO_DEPTH(`XPM_FIFO_DEPTH),
        .TDATA_WIDTH(`AXI_DATA_WIDTH),
        .FIFO_MEMORY_TYPE("block"),
        .CLOCKING_MODE("common_clock")
    ) u_output_fifo (
        .s_aclk(clk),  .m_aclk(clk),
        .s_aresetn(rst_n),
        .s_axis_tdata(packed_res_data),
        .s_axis_tvalid(packed_res_valid),
        .s_axis_tready(packed_res_ready),
        .m_axis_tdata(m_axis_result_tdata),
        .m_axis_tvalid(m_axis_result_tvalid),
        .m_axis_tready(m_axis_result_tready) 
    );

    // Status Assignment
    //assign mmio_npu_stat[0] = 1'b0; // Done flag (TODO)
    assign mmio_npu_stat[1] = 1'b0; // FMap Ready flag (TODO)
    assign mmio_npu_stat[31:2] = 30'd0;

endmodule