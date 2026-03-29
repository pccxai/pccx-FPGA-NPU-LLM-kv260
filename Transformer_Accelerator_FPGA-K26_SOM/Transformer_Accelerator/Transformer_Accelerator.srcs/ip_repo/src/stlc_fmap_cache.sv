`timescale 1ns / 1ps
`include "stlc_Array.svh"

/**
 * Module: stlc_fmap_cache
 * Description: 
 *   SRAM-based Feature Map Cache for Gemma 3N Decode Phase (GEMV).
 *   - Stores a 1x2048 Feature Map (BF16, converted to 27-bit Mantissas).
 *   - Write Interface: 432-bit (16 x 27-bit) to support high-bandwidth.
 *   - Read Interface: 27-bit (1 word) broadcast to 32 Vertical lanes.
 */
module stlc_fmap_cache #(
    parameter DATA_WIDTH = 27, // Fixed-point Mantissa width
    parameter WRITE_LANES = 16, // 16 words per write
    parameter CACHE_DEPTH = 2048, // Accommodates 1x2048 vector
    parameter LANES = 32       // Number of vertical lanes to feed
)(
    input  logic clk,
    input  logic rst_n,

    // ===| Write Interface (From BF16-to-Fixed Shifter) |=======
    input  logic [(DATA_WIDTH*WRITE_LANES)-1:0] wr_data,
    input  logic                  wr_valid,
    input  logic [6:0]            wr_addr, // log2(2048/16) = 7 bits
    input  logic                  wr_en,

    // ===| Read Interface (To Staggered Delay Line) |=======
    input  logic                  rd_start, // Trigger to start broadcasting
    output logic [DATA_WIDTH-1:0] rd_data_broadcast [0:LANES-1], // 32 identical copies
    output logic                  rd_valid
);

    logic [DATA_WIDTH-1:0] sram_rd_data;
    logic [10:0]           rd_addr;
    logic                  is_reading;

    // True Dual Port RAM inference / XPM macro with Asymmetric Ports
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(7),                // Write: 128 depth
        .ADDR_WIDTH_B(11),               // Read: 2048 depth
        .AUTO_SLEEP_TIME(0),            
        .BYTE_WRITE_WIDTH_A(DATA_WIDTH*WRITE_LANES), // Full word write
        .CLOCKING_MODE("common_clock"), 
        .MEMORY_INIT_FILE("none"),      
        .MEMORY_INIT_PARAM("0"),        
        .MEMORY_OPTIMIZATION("true"),   
        .MEMORY_PRIMITIVE("block"),      // Force BRAM usage
        .MEMORY_SIZE(DATA_WIDTH * CACHE_DEPTH), // 27 * 2048 = 55296 bits
        .MESSAGE_CONTROL(0),            
        .READ_DATA_WIDTH_B(DATA_WIDTH),  // Read: 27-bit
        .READ_LATENCY_B(2),              // 2-cycle latency for 400MHz
        .USE_EMBEDDED_CONSTRAINT(0),    
        .USE_MEM_INIT(1),               
        .WAKEUP_TIME("disable_sleep"),  
        .WRITE_DATA_WIDTH_A(DATA_WIDTH*WRITE_LANES), // Write: 432-bit
        .WRITE_MODE_B("read_first")      
    ) u_fmap_bram (
        .clka(clk),
        .ena(1'b1),
        .wea(wr_en & wr_valid),
        .addra(wr_addr),
        .dina(wr_data),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),

        .clkb(clk),
        .enb(is_reading),
        .addrb(rd_addr),
        .doutb(sram_rd_data),
        .sbiterrb(),
        .dbiterrb(),
        .sleep(1'b0),
        .rstb(~rst_n),
        .regceb(1'b1)
    );

    // ===| FSM / Read Controller |=======
    // Controls the rd_addr to sweep through the cached 2048 elements
    logic rd_valid_pipe_1, rd_valid_pipe_2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            is_reading <= 1'b0;
            rd_addr <= 11'd0;
            rd_valid_pipe_1 <= 1'b0;
            rd_valid_pipe_2 <= 1'b0;
            rd_valid <= 1'b0;
        end else begin
            // Start reading when triggered
            if (rd_start) begin
                is_reading <= 1'b1;
                rd_addr <= 11'd0;
            end 
            // Increment address while reading
            else if (is_reading) begin
                if (rd_addr == CACHE_DEPTH - 1) begin
                    is_reading <= 1'b0; // Stop after 2048 words
                end else begin
                    rd_addr <= rd_addr + 1;
                end
            end

            // Pipeline the valid signal to match BRAM latency
            rd_valid_pipe_1 <= is_reading;
            rd_valid_pipe_2 <= rd_valid_pipe_1;
            rd_valid        <= rd_valid_pipe_2;

            // Broadcast the read data to all 32 lanes
            if (rd_valid_pipe_2) begin
                for (int i = 0; i < LANES; i++) begin
                    rd_data_broadcast[i] <= sram_rd_data;
                end
            end
        end
    end

endmodule
