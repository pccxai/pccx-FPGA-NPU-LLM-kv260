`timescale 1ns / 1ps
`include "stlc_Array.svh"

/**
 * Module: stlc_global_fsm
 * Description: 
 *   Central controller for the NPU. Orchestrates the Zero-Bubble pipeline.
 *   Manages Weight Loading, SRAM reading, and VLIW instruction dispatching.
 */
module stlc_global_fsm (
    input  logic clk,
    input  logic rst_n,

    // ===| MMIO Control |===
    input  logic       npu_start,
    output logic       npu_done,

    // ===| Status from Packer |===
    input  logic       packer_busy,

    // ===| Datapath Control Signals |===
    // To Weight Dispatcher / Array
    output logic       i_weight_valid, // Enables horizontal weight shifting
    
    // To FMap SRAM Cache
    output logic       sram_rd_start,  // Triggers the SRAM to shoot 32 FMaps
    
    // To Systolic Array (VLIW Instruction)
    output logic [2:0] inst_out,       // [2]:Flush, [1]:GEMM/GEMV, [0]:Calc/Idle
    output logic       inst_valid_out  // Latches the instruction into the top row
);

    // ===| FSM States |===
    typedef enum logic [2:0] {
        IDLE,           // Wait for CPU start
        INIT_LOAD,      // Initial weight load for Tile 0 (32 cycles)
        COMPUTE_LOAD,   // Compute current tile (32c) while loading NEXT tile weights (32c)
        FLUSH_DRAIN,    // Final tile computation + Flush command
        DONE            // Finish and interrupt CPU
    } state_t;

    state_t state, next_state;

    // ===| Counters |===
    logic [5:0] tile_cnt;       // Counts up to 64 tiles (64 * 32 = 2048)
    logic [5:0] cycle_cnt;      // Counts 32 cycles for weight load / compute

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state     <= IDLE;
            tile_cnt  <= 6'd0;
            cycle_cnt <= 6'd0;
        end else begin
            state <= next_state;
            
            // Counter Logic
            if (state == IDLE) begin
                tile_cnt  <= 6'd0;
                cycle_cnt <= 6'd0;
            end else if (state == INIT_LOAD || state == COMPUTE_LOAD) begin
                if (cycle_cnt == 6'd31) begin
                    cycle_cnt <= 6'd0;
                    if (state == COMPUTE_LOAD) tile_cnt <= tile_cnt + 1;
                end else begin
                    cycle_cnt <= cycle_cnt + 1;
                end
            end
        end
    end

    // ===| Next State Logic |===
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (npu_start) next_state = INIT_LOAD;
            end
            
            INIT_LOAD: begin
                // Initial 32 cycles to just fill the array with the first weight tile
                if (cycle_cnt == 6'd31) next_state = COMPUTE_LOAD;
            end

            COMPUTE_LOAD: begin
                // Compute while loading. Repeat until the 63rd tile.
                // The 64th tile (tile_cnt == 63) needs to trigger a flush at the end.
                if (cycle_cnt == 6'd31 && tile_cnt == 6'd62) 
                    next_state = FLUSH_DRAIN;
            end

            FLUSH_DRAIN: begin
                // Wait for the final flush sequence to finish draining (approx 64 cycles)
                // AND wait for the packer to send all results over AXI-Stream.
                if (cycle_cnt >= 6'd63 && !packer_busy) 
                    next_state = DONE;
            end

            DONE: begin
                if (!npu_start) next_state = IDLE; // Wait for handshake clearance
            end
        endcase
    end

    // ===| Output Control Logic (The Brain) |===
    always_comb begin
        // Defaults
        i_weight_valid = 1'b0;
        sram_rd_start  = 1'b0;
        inst_out       = 3'b000;
        inst_valid_out = 1'b0;
        npu_done       = 1'b0;

        case (state)
            INIT_LOAD: begin
                

                i_weight_valid = 1'b1; // Shift weights horizontally
                // Instruction: IDLE(0), GEMV(0), No_Flush(0)
                if (cycle_cnt == 0) begin
                    inst_out = 3'b000;
                    inst_valid_out = 1'b1;
                end
            end

            COMPUTE_LOAD: begin
                
                i_weight_valid = 1'b1; // Keep shifting NEXT weights horizontally
                
                if (cycle_cnt == 0) begin
                    // Fire SRAM to start feeding 32 FMap elements
                    sram_rd_start = 1'b1; 
                    
                    // Instruction: CALC(1), GEMV(0), No_Flush(0)
                    // This command triggers `w_load` internally via the transition, 
                    // or we might need a specific flush bit. 
                    // Actually, in our DSP, w_load is tied to the flush_sequence!
                    // Wait, if w_load is tied to flush_sequence[3], we NEED to trigger Inst[2]=1 
                    // to load the weights! Let's adapt this.
                    inst_out = 3'b101; // CALC(1), GEMV(0), FLUSH/LOAD(1)
                    inst_valid_out = 1'b1;
                end
            end

            FLUSH_DRAIN: begin
                i_weight_valid = 1'b0; // No more weights to load
                
                if (cycle_cnt == 0) begin
                    sram_rd_start = 1'b1; // Fire the last 32 FMaps
                    
                    // Instruction: CALC(1), GEMV(0), FLUSH_AND_DRAIN(1)
                    inst_out = 3'b101; 
                    inst_valid_out = 1'b1;
                end
            end

            DONE: begin
                npu_done = 1'b1;
            end
        endcase
    end

endmodule
