`timescale 1 ns / 1 ps

module gemma_npu_axi_slave_lite_v1_0_S00_AXI #
(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 5
)
(
    // ---------------------------------------------------------
    // [NEW] AXI-Stream interface (DMA and high-speed communication)
    // ---------------------------------------------------------
    // RX: DMA -> NPU (Token / Weight reception)
    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,
    
    // TX: NPU -> DMA (Result transmission)
    output wire [31:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast,

    // ---------------------------------------------------------
    // AXI-Lite basic interface
    // ---------------------------------------------------------
    input wire  S_AXI_ACLK,
    input wire  S_AXI_ARESETN,
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
    input wire [2 : 0] S_AXI_AWPROT,
    input wire  S_AXI_AWVALID,
    output wire  S_AXI_AWREADY,
    input wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
    input wire  S_AXI_WVALID,
    output wire  S_AXI_WREADY,
    output wire [1 : 0] S_AXI_BRESP,
    output wire  S_AXI_BVALID,
    input wire  S_AXI_BREADY,
    input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
    input wire [2 : 0] S_AXI_ARPROT,
    input wire  S_AXI_ARVALID,
    output wire  S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
    output wire [1 : 0] S_AXI_RRESP,
    output wire  S_AXI_RVALID,
    input wire  S_AXI_RREADY
);

    // AXI4LITE signals
    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_awaddr;
    reg axi_awready;
    reg axi_wready;
    reg [1 : 0] axi_bresp;
    reg axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1 : 0] axi_araddr;
    reg axi_arready;
    reg [1 : 0] axi_rresp;
    reg axi_rvalid;

    localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
    localparam integer OPT_MEM_ADDR_BITS = 2;

    // Registers
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg0;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg1;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg2;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg3;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg4;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg5;
    wire slv_reg_wren;
    integer byte_index;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;
    assign slv_reg_wren  = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

    // AXI Write Logic (Auto-clear Pulse added)
    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            axi_bvalid  <= 1'b0;
            slv_reg0 <= 0; slv_reg1 <= 0; slv_reg2 <= 0;
            slv_reg3 <= 0; slv_reg4 <= 0; slv_reg5 <= 0;
        end else begin
            // ... (Omit basic AXI Write Ready control - operates the same as existing logic)
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID) begin
                axi_awready <= 1'b1;
                axi_wready  <= 1'b1;
                axi_awaddr  <= S_AXI_AWADDR;
            end else begin
                axi_awready <= 1'b0;
                axi_wready  <= 1'b0;
            end

            if (axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID) begin
                axi_bvalid <= 1'b1;
            end else if (S_AXI_BREADY && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end

            if (slv_reg_wren) begin
                case (axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB])
                    3'h0: for (byte_index = 0; byte_index <= 3; byte_index = byte_index+1)
                            if (S_AXI_WSTRB[byte_index]) slv_reg0[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                    3'h1: for (byte_index = 0; byte_index <= 3; byte_index = byte_index+1)
                            if (S_AXI_WSTRB[byte_index]) slv_reg1[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                    3'h2: for (byte_index = 0; byte_index <= 3; byte_index = byte_index+1)
                            if (S_AXI_WSTRB[byte_index]) slv_reg2[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                    3'h3: for (byte_index = 0; byte_index <= 3; byte_index = byte_index+1)
                            if (S_AXI_WSTRB[byte_index]) slv_reg3[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                    3'h4: for (byte_index = 0; byte_index <= 3; byte_index = byte_index+1)
                            if (S_AXI_WSTRB[byte_index]) slv_reg4[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                    3'h5: for (byte_index = 0; byte_index <= 3; byte_index = byte_index+1)
                            if (S_AXI_WSTRB[byte_index]) slv_reg5[(byte_index*8) +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
                endcase
            end else begin
                // Auto-clear logic: Unconditionally lowers to 0 in cycles without register writing (pulse generation)
                slv_reg0[0] <= 1'b0; // i_npu_start
                slv_reg0[1] <= 1'b0; // i_acc_clear
            end
        end
    end

    // AXI Read Logic
    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
        end else begin
            if (~axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1'b1;
                axi_araddr  <= S_AXI_ARADDR;
            end else begin
                axi_arready <= 1'b0;
            end
            if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
                axi_rvalid <= 1'b1;
            end else if (axi_rvalid && S_AXI_RREADY) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    wire [15:0] w_final_result;
    wire        w_npu_done;
    wire [1023:0] w_npu_result_all;

    // Assign w_npu_done (Bit 16) and w_final_result (Bit 15:0) to address 0x10.
    assign S_AXI_RDATA = (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h0) ? slv_reg0 : 
                         (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h1) ? slv_reg1 :
                         (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h2) ? slv_reg2 : 
                         (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h3) ? slv_reg3 : 
                         (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h4) ? {15'd0, w_npu_done, w_final_result} : 
                         (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] == 3'h5) ? slv_reg5 : 0; 

    // =========================================================================
    // [RX] AXI-Stream -> BRAM conversion (Packing FSM)
    // =========================================================================
    reg [511:0] rx_pack_reg;
    reg [3:0]   rx_beat_cnt;
    reg [8:0]   rx_bram_addr;
    
    reg         r_dma_we_token;
    reg         r_dma_we_weight;
    reg [511:0] r_dma_wdata;

    assign s_axis_tready = 1'b1;

    always @(posedge S_AXI_ACLK) begin
        if (~S_AXI_ARESETN) begin
            rx_beat_cnt <= 0;
            rx_bram_addr <= 0;
            r_dma_we_token <= 0;
            r_dma_we_weight <= 0;
        end else begin
            r_dma_we_token <= 0;
            r_dma_we_weight <= 0;

            if (s_axis_tvalid && s_axis_tready) begin
                rx_pack_reg <= {s_axis_tdata, rx_pack_reg[511:32]}; // Little Endian Shift

                if (rx_beat_cnt == 15) begin
                    rx_beat_cnt <= 0;
                    r_dma_wdata <= {s_axis_tdata, rx_pack_reg[511:32]};
                    
                    // If slv_reg5[0] is 0, it is used as Token BRAM, and if it is 1, it is written as Weight BRAM.
                    if (slv_reg5[0] == 1'b0) r_dma_we_token  <= 1'b1;
                    else                     r_dma_we_weight <= 1'b1;
                    
                    rx_bram_addr <= rx_bram_addr + 1;
                end else begin
                    rx_beat_cnt <= rx_beat_cnt + 1;
                end
            end

            if (s_axis_tlast) begin
                rx_bram_addr <= 0; // When the transfer is completed, the BRAM pointer is reset to 0.
            end
        end
    end

    // =========================================================================
    // [TX] NPU Result -> AXI-Stream conversion (Unpacking FSM)
    // =========================================================================
    wire [511:0] packed_results;
    genvar i;
    generate
        // Among the 32-channel MAC 32-bit results, only the lower 16 bits are extracted and packed into 512 bits (since Python receives them as int16).
        for (i=0; i<32; i=i+1) begin : pack_loop
            assign packed_results[i*16 +: 16] = w_npu_result_all[i*32 +: 16];
        end
    endgenerate

    reg [511:0] tx_shift_reg;
    reg [4:0]   tx_beat_cnt;
    reg         tx_active;

    assign m_axis_tdata  = tx_shift_reg[31:0];
    assign m_axis_tvalid = tx_active;
    assign m_axis_tlast  = (tx_beat_cnt == 15);

    always @(posedge S_AXI_ACLK) begin
        if (~S_AXI_ARESETN) begin
            tx_active <= 0;
            tx_beat_cnt <= 0;
        end else begin
            // When the NPU operation completion pulse (w_npu_done) occurs, DMA transfer begins.
            if (w_npu_done && !tx_active) begin
                tx_active <= 1'b1;
                tx_beat_cnt <= 0;
                tx_shift_reg <= packed_results;
            end else if (tx_active && m_axis_tready) begin
                if (tx_beat_cnt == 15) begin
                    tx_active <= 1'b0; // Transmission ends after shooting 16 times (512 bits)
                end else begin
                    tx_beat_cnt <= tx_beat_cnt + 1;
                    tx_shift_reg <= {32'd0, tx_shift_reg[511:32]};
                end
            end
        end
    end

    // =========================================================================
    // NPU Top module instance (gemma_layer_top)
    // =========================================================================
    gemma_layer_top u_npu_core (
        .clk                (S_AXI_ACLK),
        .rst_n              (S_AXI_ARESETN),
        .i_npu_start        (slv_reg0[0]),     
        .i_acc_clear        (slv_reg0[1]),     
        .i_rms_mean_sq      (slv_reg2),        
        .i_ping_pong_sel    (slv_reg3[0]),     
        .i_result_sel       (slv_reg3[12:8]),  
        .i_gelu_en          (slv_reg4[0]),     
        .i_softmax_en       (slv_reg4[1]),     
        .o_npu_done         (w_npu_done),      
        .o_logic_anchor     (),  

        // Connect the FSM output created above to NPU BRAM
        .i_dma_we_token     (r_dma_we_token),          
        .i_dma_addr_token   (rx_bram_addr[7:0]),        
        .i_dma_wdata_token  (r_dma_wdata),      
        
        .i_dma_we_weight    (r_dma_we_weight),          
        .i_dma_addr_weight  (rx_bram_addr[7:0]),        
        .i_dma_wdata_weight (r_dma_wdata),      
        
        .o_npu_result_all   (w_npu_result_all),   
        .o_final_result     (w_final_result)
    );

endmodule