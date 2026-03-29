`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "mem_IO.svh"
`include "npu_interfaces.svh"

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
    input  logic clk,
    input  logic rst_n,

    // Control Plane (MMIO)
    input  logic [31:0] mmio_npu_cmd,  
    output logic [31:0] mmio_npu_stat, 

    // AXI4-Stream Interfaces (Clean & Modern)
    axis_if.slave  S_AXIS_FMAP,   // Feature Map Input 0 (128-bit, HPC0)
    axis_if.slave  S_AXI_WEIGHT,  // Weight Matrix Input (512-bit)
    axis_if.master M_AXIS_RESULT, // Final Result Output (128-bit)
    
    // Auxiliary Streaming Ports
    axis_if.slave  S_AXIS_HP1,
    axis_if.slave  S_AXIS_HP2,
    axis_if.slave  S_AXIS_HP3,
    axis_if.slave  S_AXIS_HPC1,   // Feature Map Input 1 (128-bit, HPC1)
    axis_if.master M_AXIS_ACP,

    // Secondary MMIO Interface
    input  logic [31:0] s_axi_hpm1_vliw,
    output logic [31:0] s_axi_hpm1_stat
);

    // ===| Bridge & Alignment: 256-bit Feature Map |=======
    // Combine HPC0 (S_AXIS_FMAP) and HPC1 (S_AXIS_HPC1) into a single 256-bit bus
    logic [255:0] s_axis_fmap_combined_tdata;
    logic         s_axis_fmap_combined_tvalid;
    logic         s_axis_fmap_combined_tready;
    
    // Concatenating {HPC1, HPC0} so HPC0 is the lower 128-bits
    assign s_axis_fmap_combined_tdata  = {S_AXIS_HPC1.tdata, S_AXIS_FMAP.tdata}; 
    
    // Valid when BOTH HPC ports have valid data (Lock-step assumption)
    assign s_axis_fmap_combined_tvalid = S_AXIS_FMAP.tvalid & S_AXIS_HPC1.tvalid;
    
    // Assert ready to BOTH ports simultaneously when the 256-bit FIFO is ready
    assign S_AXIS_FMAP.tready = s_axis_fmap_combined_tready & S_AXIS_HPC1.tvalid;
    assign S_AXIS_HPC1.tready = s_axis_fmap_combined_tready & S_AXIS_FMAP.tvalid;

    // S_AXI_WEIGHT Bridge (Mapping to 512-bit FLAT signals)
    logic [511:0] s_axis_weight_tdata_FLAT  = S_AXI_WEIGHT.tdata;
    logic [3:0]   s_axis_weight_tvalid_FLAT = {3'b0, S_AXI_WEIGHT.tvalid}; 
    logic [3:0]   s_axis_weight_tlast_FLAT  = {3'b0, S_AXI_WEIGHT.tlast};
    assign S_AXI_WEIGHT.tready = s_axis_weight_tready_FLAT[0];
    logic [3:0]   s_axis_weight_tready_FLAT;

    // M_AXIS_RESULT Bridge
    assign M_AXIS_RESULT.tdata  = m_axis_result_tdata;
    assign M_AXIS_RESULT.tvalid = m_axis_result_tvalid;
    assign M_AXIS_RESULT.tlast  = m_axis_result_tlast;
    logic [127:0] m_axis_result_tdata;
    logic         m_axis_result_tvalid;
    logic         m_axis_result_tlast;
    logic         m_axis_result_tready = M_AXIS_RESULT.tready;

    // Unused/Mirror Ports
    assign S_AXIS_HP1.tready  = 1'b1;
    assign S_AXIS_HP2.tready  = 1'b1;
    assign S_AXIS_HP3.tready  = 1'b1;
    
    // Result duplication to ACP
    assign M_AXIS_ACP.tdata  = m_axis_result_tdata;
    assign M_AXIS_ACP.tvalid = m_axis_result_tvalid;
    assign M_AXIS_ACP.tlast  = m_axis_result_tlast;
    
    assign s_axi_hpm1_stat = 32'hCAFE_BABE;

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
    logic packer_busy_status; // Connect to Packer

    stlc_global_fsm u_brain (
        .clk(clk), .rst_n(rst_n),
        .npu_start(npu_start),
        .npu_done(mmio_npu_stat[0]),
        
        .packer_busy(packer_busy_status), 

        .i_weight_valid(global_weight_valid),
        .sram_rd_start(global_sram_rd_start),
        .inst_out(global_inst),
        .inst_valid_out(global_inst_valid)
    );

    // 1. Feature Map Pipeline (HPC0 & HPC1 -> SRAM Cache)
    // 256-bit FIFO for FMap
    logic [255:0] fmap_fifo_data;
    logic         fmap_fifo_valid;
    logic         fmap_fifo_ready;

    xpm_fifo_axis #(    
        .FIFO_DEPTH(`XPM_FIFO_DEPTH),
        .TDATA_WIDTH(256), // Increased from 128 to 256
        .FIFO_MEMORY_TYPE("block"),
        .CLOCKING_MODE("common_clock")
    ) u_fmap_fifo (
        .s_aclk(clk), .m_aclk(clk),
        .s_aresetn(rst_n),
        .s_axis_tdata(s_axis_fmap_combined_tdata),
        .s_axis_tvalid(s_axis_fmap_combined_tvalid),
        .s_axis_tready(s_axis_fmap_combined_tready),
        .m_axis_tdata(fmap_fifo_data),
        .m_axis_tvalid(fmap_fifo_valid),
        .m_axis_tready(fmap_fifo_ready)
    );

    // ===| e_max parsing & cache logic               |=======
    // ===| 256-bit에서 16개씩 2사이클에 걸쳐 32개 추출     |=======
    logic [`BF16_EXP_WIDTH-1:0] active_emax [0:`ARRAY_SIZE_H-1];
    logic fmap_word_toggle;
    logic emax_group_valid; // 32개가 다 모였을 때 1이 됨

    always_ff @(posedge clk) begin
        if (!rst_n || npu_clear) begin
            fmap_word_toggle <= 1'b0;
            emax_group_valid <= 1'b0;
        end else if (fmap_fifo_valid && fmap_fifo_ready) begin
            fmap_word_toggle <= ~fmap_word_toggle;
            
            // 256-bit 버스에서 16개의 BF16 지수(Exponent) 동시 추출
            for (int k = 0; k < 16; k++) begin
                if (fmap_word_toggle == 1'b0) begin
                    // 첫 번째 클럭: [0~15] 채우기 (하위 16개)
                    active_emax[k] <= fmap_fifo_data[(k*16) + 7 +: 8]; 
                end else begin
                    // 두 번째 클럭: [16~31] 채우기 (상위 16개)
                    active_emax[k+16] <= fmap_fifo_data[(k*16) + 7 +: 8];
                end
            end
            
            // 토글이 1에서 0으로 넘어갈 때 (두 번째 클럭 처리가 끝날 때) 32개 완성
            emax_group_valid <= (fmap_word_toggle == 1'b1);
        end else begin
            emax_group_valid <= 1'b0;
        end
    end

    // e_max 전용 캐시 (SRAM/LUTRAM)
    // FMap 2048개 들어갈 때, 32개 묶음이므로 주소는 더 작게 됨. (여기선 단순 매핑)
    logic [`BF16_EXP_WIDTH-1:0] emax_cache_mem [0:1023][0:`ARRAY_SIZE_H-1]; 
    logic [9:0] emax_wr_addr;
    logic [9:0] emax_rd_addr;

    always_ff @(posedge clk) begin
        if (!rst_n || npu_clear) begin
            emax_wr_addr <= 0;
        end else if (emax_group_valid) begin
            for (int i = 0; i < `ARRAY_SIZE_H; i++) begin
                emax_cache_mem[emax_wr_addr][i] <= active_emax[i];
            end
            emax_wr_addr <= emax_wr_addr + 1;
        end
    end

    // 읽기 로직: global_sram_rd_start 신호에 맞춰서 e_max도 읽어냄
    logic [`BF16_EXP_WIDTH-1:0] cached_emax_out [0:`ARRAY_SIZE_H-1];
    
    always_ff @(posedge clk) begin
        if (!rst_n || npu_clear) begin
            emax_rd_addr <= 0;
        end else if (global_sram_rd_start) begin // SRAM 읽기 시작할 때 주소 리셋
            emax_rd_addr <= 0;
        end else if (fmap_broadcast_valid) begin // FMap이 브로드캐스트 될 때마다 다음 e_max 읽기
            // 필요에 따라 주소 증가 로직 수정 가능 (현재는 1씩 증가)
            // emax_rd_addr <= emax_rd_addr + 1;
        end
        
        // 캐시에서 읽은 e_max 출력
        for (int i = 0; i < `ARRAY_SIZE_H; i++) begin
            cached_emax_out[i] <= emax_cache_mem[emax_rd_addr][i];
        end
    end

    // Shifter (16-Lane Parallel, 256-bit in -> 432-bit out)
    logic [431:0] fixed_fmap;
    logic         fixed_fmap_valid;
    logic         fmap_shifter_ready;
    
    stlc_bf16_fixed_pipeline u_fmap_shifter (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(fmap_fifo_data), // Full 256-bit
        .s_axis_tvalid(fmap_fifo_valid),
        .s_axis_tready(fmap_shifter_ready),
        .m_axis_tdata(fixed_fmap),     // Full 432-bit (16 x 27-bit)
        .m_axis_tvalid(fixed_fmap_valid),
        .m_axis_tready(1'b1) 
    );
    assign fmap_fifo_ready = fmap_shifter_ready;

    // SRAM Cache
    logic [`FIXED_MANT_WIDTH-1:0] fmap_broadcast [0:`ARRAY_SIZE_H-1];
    logic        fmap_broadcast_valid;
    logic [6:0]  sram_wr_addr; // log2(2048/16) = 7 bits

    always_ff @(posedge clk) begin
        if (!rst_n || npu_clear) sram_wr_addr <= 0;
        else if (fixed_fmap_valid) sram_wr_addr <= sram_wr_addr + 1;
    end

    stlc_fmap_cache #(
        .DATA_WIDTH(`FIXED_MANT_WIDTH),
        .WRITE_LANES(16),
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
    logic                        raw_res_sum_valid [0:`ARRAY_SIZE_H-1];

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
        .V_ACC_out(raw_res_sum),
        .V_ACC_valid(raw_res_sum_valid)
    );

    // 4. Output Pipeline (Result Normalization -> Result Packer -> FIFO)
    localparam TOTAL_LATENCY = `SYSTOLIC_TOTAL_LATENCY;
    logic [`BF16_EXP_WIDTH-1:0] emax_pipe [0:`ARRAY_SIZE_H-1][0:TOTAL_LATENCY-1];
    logic [`BF16_EXP_WIDTH-1:0] delayed_emax_32 [0:`ARRAY_SIZE_H-1];

    always_ff @(posedge clk) begin
        if (rst_n) begin
            for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
                // FMap 데이터가 캐시에서 나와서 시스톨릭에 들어갈 때, 
                // e_max도 캐시에서 꺼내어 딜레이 라인에 태운다!
                emax_pipe[c][0] <= cached_emax_out[c]; 
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
        .clk(clk), .rst_n(rst_n),
        .row_res(norm_res_seq),             
        .row_res_valid(norm_res_seq_valid), 
        .packed_data(packed_res_data),
        .packed_valid(packed_res_valid),
        .packed_ready(packed_res_ready),
        .o_busy(packer_busy_status) 
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
    assign mmio_npu_stat[1] = 1'b0; 
    assign mmio_npu_stat[31:2] = 30'd0;

endmodule
