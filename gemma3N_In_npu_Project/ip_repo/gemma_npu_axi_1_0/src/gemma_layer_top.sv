`timescale 1ns / 1ps

module gemma_layer_top #(
    parameter SYSTOLIC_ARRAY_SIZE = 32 // 32x32 architecture
    parameter INT4_SYSTOLIC_ARRAY_SIZE = 4 *; //32 x 32 (4bit)
    parameter SYSTOLIC_RESULT_SIZE = 48
    parameter SYSTOLIC_INPUT_SIZE = 16;
)(
    input  logic               clk,
    input  logic               rst_n,

    // [AXI4-Lite MMIO control signals] (0x00 ~ 0x10)
    input  logic               i_npu_start,     // 0x00 [Bit 0] (Kernel Launch!)
    input  logic               i_acc_clear,     // 0x00 [Bit 1] (Accumulator reset)
    input  logic [31:0]        i_rms_mean_sq,   // 0x08 (RMSNorm denominator)
    input  logic               i_ping_pong_sel, // 0x0C (DMA NPU switch)
    input  logic               i_gelu_en,       // 0x10 [Bit 0] (GeLU enabled)
    input  logic               i_softmax_en,    // 0x10 [Bit 1] (Softmax enabled)
    output logic               o_npu_done,      // 0x04 [Bit 0] (Operation completion flag)

    // [AXI DMA streaming interface (example)]
    // In reality, the AXI-Stream (TVALID, TDATA, etc.) standard will be used, but it is expressed conceptually.
    input  logic               i_dma_we_token,
    input  logic [7:0]         i_dma_addr_token,
    input  logic [511:0]       i_dma_wdata_token,

    input  logic               i_dma_we_weight,
    input  logic [7:0]         i_dma_addr_weight,
    input  logic [511:0]       i_dma_wdata_weight, // One row of 32x8bit tiles

    input  logic [4:0]         i_result_sel,       // [Added] For selection of 32 channels
    output logic [1023:0]      o_npu_result_all,   // [Added] Full bus for DMA
    output logic               o_logic_anchor,     // [Special Move] Anchor to preserve all PE.
    output logic [15:0]        o_final_result      // Final result to return to CPU via DMA
);

    // 1. FSM (Warp Scheduler): Kernel life cycle control
    typedef enum logic [1:0] {
        ST_IDLE     = 2'd0,  // atmosphere
        ST_WAIT_RMS = 2'd1,  // Wait until mean_sq coming in at 0x08 is converted to inverse square root.
        ST_RUN      = 2'd2,  // Ping Pong BRAM data bombardment with a 32x32 array.
        ST_WAIT_MAC = 2'd3   // Systolic pipeline (Wavefront) Waiting for remaining operations to complete
    } state_t;

    state_t state, next_state;
    logic [5:0] feed_counter; 
    logic       npu_running;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            feed_counter <= 6'd0;
        end else begin
            state <= next_state;
            if (state == ST_RUN) feed_counter <= feed_counter + 1;
            else                 feed_counter <= 6'd0;
        end
    end

    logic rms_valid_out, mac_valid_out;

    always_comb begin
        next_state  = state;
        o_npu_done  = 1'b0;
        npu_running = 1'b0;

        case (state)
            ST_IDLE: begin
                if (i_npu_start) next_state = ST_WAIT_RMS; 
            end
            ST_WAIT_RMS: begin
                if (rms_valid_out) next_state = ST_RUN; 
            end
            ST_RUN: begin
                npu_running = 1'b1;
                if (feed_counter == 6'd31) next_state = ST_WAIT_MAC; // All data for number 32 has been entered.
            end
            ST_WAIT_MAC: begin
                if (mac_valid_out) begin 
                    o_npu_done = 1'b1;       // Raise the 0x04 flag to the CPU.
                    next_state = ST_IDLE;
                end
            end
        endcase
    end

    // 2. [Stage 1] Pre-Norm (RMSNorm Calculator)
    logic [15:0] rms_inv_sqrt_val;

    rmsnorm_inv_sqrt u_rmsnorm (
        .clk(clk), .rst_n(rst_n),
        .valid_in(i_npu_start),       // When the CPU hits Start, calculation begins.
        .i_mean_sq(i_rms_mean_sq),    // Scalar from 0x08
        .valid_out(rms_valid_out),    // After calculation, transfer FSM to RUN
        .o_inv_sqrt(rms_inv_sqrt_val)
    );

    // 3. Ping-Pong BRAM & Vector Scaling (On-the-fly)
    // Original data read from the ping-pong buffer (32 8-bit arrays)
    logic [7:0] raw_token_data  [0:31]; 
    logic [7:0] sys_weight_data [0:31]; 

    // [Edit] Receive from BRAM as a 512-bit flat vector and unpack it into an array
    logic [511:0] flat_token_data;
    logic [511:0] flat_weight_data;

    always_comb begin
        for (int i = 0; i < 32; i++) begin
            raw_token_data[i]  = flat_token_data[i*8 +: 8];
            sys_weight_data[i] = flat_weight_data[i*8 +: 8];
        end
    end

    // [Fix] Perfect port name matching and format casting
    ping_pong_bram #(
        .DATA_WIDTH(512),
        .ADDR_WIDTH(9)
    ) u_bram_token (
        .clk(clk), .rst_n(rst_n),
        .switch_buffer(i_ping_pong_sel),
        .dma_we(i_dma_we_token),
        .dma_addr({1'b0, i_dma_addr_token}),          // 8bit -> 9bit expansion
        .dma_write_data(i_dma_wdata_token), 
        .npu_addr_b(9'd0),                            // I don't use it
        .npu_read_data_a(flat_token_data),            // 512bit output
        .npu_read_data_b()                            // I don't use it
    );

    ping_pong_bram #(
        .DATA_WIDTH(512),
        .ADDR_WIDTH(9)
    ) u_bram_weight (
        .clk(clk), .rst_n(rst_n),
        .switch_buffer(i_ping_pong_sel),
        .dma_we(i_dma_we_weight),
        .dma_addr({1'b0, i_dma_addr_weight}),
        .dma_write_data(i_dma_wdata_weight), // {256'd0, ...} Erase and connect directly!        .npu_addr_a({3'd0, feed_counter});
        .npu_addr_b(9'd0),
        .npu_read_data_a(flat_weight_data),
        .npu_read_data_b()
    );

    // As soon as it comes out of BRAM, multiply it by the inverse square root of RMSNorm and push it to MAC.
    logic [7:0] scaled_token_data [0:31];
    genvar i;
    generate
        for (i = 0; i < 32; i++) begin : gen_scaling
            assign scaled_token_data[i] = (raw_token_data[i] * $signed({1'b0, rms_inv_sqrt_val})) >> 15;
        end
    endgenerate

    // 4. [Stage 2] 32x32 Systolic MAC Array (main body)
    logic [32*32*32-1:0] mac_out_acc_flat; // 1-dimensional flat bus instead of 2-dimensional array (connectivity guaranteed!)

    systolic_NxN #(
        .ARRAY_SIZE(32)
        .OUT_SIZE(48)
        .INPUT_SIZE(16)
    ) u_mac_engine (
        .clk(clk), .rst_n(rst_n),
        .i_clear(i_acc_clear),           
        .in_a(scaled_token_data),        
        .in_b(sys_weight_data),          
        .in_valid(npu_running),
        .out_acc_flat(mac_out_acc_flat)  // Connect to flat port
    );

    // Pipeline latency (Wavefront) tracking shift register
    logic [63:0] shift_reg_valid; 
    always_ff @(posedge clk) begin
        if (!rst_n) shift_reg_valid <= 0;
        else        shift_reg_valid <= {shift_reg_valid[62:0], npu_running};
    end
    assign mac_valid_out = shift_reg_valid[63]; // aproximately 64clocks after PE completed at bottom right

    // 5. [Stage 3] Output Buffering & Activation MUX
    
    // Array to store 48bit from 32 pe units
    logic [SYSTOLIC_RESULT_SIZE-1:0] mac_result_row [0:SYSTOLIC_ARRAY_SIZE-1];
    always_comb begin
        for (int j = 0; j < SYSTOLIC_ARRAY_SIZE; j++) begin
            // Index calculation: ((last line number) * array size + current column) * data width
            // Save all 32(31)th row values ​​-> 31(30) -> 30(29) -> ... -> 1(0)
            mac_result_row[j] = mac_out_acc_flat[((SYSTOLIC_ARRAY_SIZE-1)*SYSTOLIC_ARRAY_SIZE + j) * SYSTOLIC_RESULT_SIZE +: SYSTOLIC_RESULT_SIZE];
        end
    end

    // result address counter
    logic [4:0] result_addr_cnt;
    always_ff @(posedge clk) begin
        if (!rst_n || i_acc_clear) result_addr_cnt <= 5'd0;
        else if (mac_valid_out)    result_addr_cnt <= result_addr_cnt + 1;
    end

    //DMA output BUS size : 32(column) * 48bit(value) = 1536bit [1535:0]
    logic [(SYSTOLIC_ARRAY_SIZE * SYSTOLIC_RESULT_SIZE) - 1 : 0] dma_result_bus;

    result_ping_pong_bram u_out_buffer (
        .clk(clk), .rst_n(rst_n),
        .switch_buffer(i_ping_pong_sel), 
        .npu_we(mac_valid_out),          
        .npu_addr(result_addr_cnt),      
        .npu_data_in(mac_result_row),
        .dma_addr(5'd0),                 
        .dma_data_out(dma_result_bus)    
    );

    // [Pure method] Exposing the entire bus outside (for DMA preparation)
    assign o_npu_result_all = dma_result_bus;

    // [Special Move] Logical 'Anchor' to forcibly keep all PEs (1024) alive.
    // This value changes even if just one of the 1024 bits changes, so Vivado cannot delete the PE.
    assign o_logic_anchor = ^dma_result_bus; 

    // MUX for selecting the channel that the CPU will read into MMIO (Key to prevent DSP pruning!)
    logic [31:0] selected_channel_data;
    assign selected_channel_data = dma_result_bus[i_result_sel * 32 +: 32];

    // Existing debug variable names remain the same, but selected channel values ​​are burned.
    logic signed [15:0] mac_attn_score;
    assign mac_attn_score = selected_channel_data[15:0]; 

    logic [15:0] softmax_prob;
    softmax_exp_unit u_softmax (
        .clk(clk), .rst_n(rst_n), .valid_in(mac_valid_out),
        .i_x(mac_attn_score), .valid_out(), .o_exp(softmax_prob)
    );

    // Plug in GeLU hardware (LUT)
    logic signed [7:0] gelu_out_8bit;
    logic [15:0]       gelu_out;

    gelu_lut u_gelu_inst (
        .clk(clk),
        .valid_in(mac_valid_out),
        .data_in(mac_attn_score),      
        .valid_out(),
        .data_out(gelu_out_8bit)       
    );

    assign gelu_out = $signed(gelu_out_8bit);

    // Final output routing
    always_comb begin
        if (i_softmax_en)       o_final_result = softmax_prob;   
        else if (i_gelu_en)     o_final_result = gelu_out;       
        else                    o_final_result = mac_attn_score; 
    end

endmodule
