`timescale 1ns / 1ps

`include "vdotm_Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

module vdotm_reduction(
    parameter fmap_cache_out_size = `FMAP_CACHE_OUT_SIZE,
    parameter weight_type = `INT4_RANGE,
    parameter line_cnt = 32
)(
    input logic clk,
    input logic rst_n,
    input logic [`FIXED_MANT_WIDTH+2:0] IN_fmap_LUT[0:fmap_cache_out_size-1][0:weight_type-1];

    input logic IN_valid,

    input logic [in_weight_size - 1:0] IN_weight[0:line_cnt -1],

    output logic [`FIXED_MANT_WIDTH+2:0] OUT_reduction_result;
);
    //2^5
    logic [`FIXED_MANT_WIDTH+2:0] reduction_32_fmap_wire[0:31];
    //2^4
    logic [`FIXED_MANT_WIDTH+2:0] reduction_16_fmap_wire[0:15];
    //2^3
    logic [`FIXED_MANT_WIDTH+2:0] reduction_8_fmap_wire[0:7];
    //2^2
    logic [`FIXED_MANT_WIDTH+2:0] reduction_4_fmap_wire[0:3];
    //2^1
    logic [`FIXED_MANT_WIDTH+2:0] reduction_2_fmap_wire[0:1];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for(int lane=0; lane<fmap_cache_out_size; lane++) begin
                //stage1_emax_q1[i] <= 0
            end
        end else begin
            if(IN_valid)begin
                for(int lane=0; lane<fmap_cache_out_size; lane++) begin
                    reduction_32_fmap_wire[lane] <= IN_fmap_LUT[lane][IN_weight[lane]];
                end
            end
        end
    end

    // ===| Stage 1: Reduction 32 -> 16 |==========================================
    // Instantiates 16 DSP48E2 slices to add adjacent pairs of the 32 input wires
    // ============================================================================
    generate
        genvar i;
        for (i = 0; i < 16; i++) begin : gen_dsp_reduce_32_to_16

            // --- Internal 48-bit wires for DSP port matching ---
            logic [47:0] dsp_in_ab;
            logic [47:0] dsp_in_c;
            logic [47:0] dsp_out_p;

            // Map inputs to 48-bit width (Zero or Sign extension depending on your data)
            // Operand 1: Even index (2*i) -> Routed to A:B ports
            // Operand 2: Odd index (2*i+1) -> Routed to C port
            assign dsp_in_ab = 48'(reduction_32_fmap_wire[2*i]);
            assign dsp_in_c  = 48'(reduction_32_fmap_wire[2*i+1]);

            DSP48E2 #(
                // [IMPORTANT] Changed from "TWO24" to "ONE48" for standard addition.
                // "TWO24" breaks the carry chain at bit 24. Use "ONE48" for full precision.
                .USE_SIMD("ONE48"),

                // --- Register Control (0 = Comb, 1 = Registered) ---
                .AREG(0), .BREG(0), .CREG(0),
                .PREG(1)                  // Enable P register (1 clock delay for sum)
            ) u_dsp (
                // --- Clock and Reset ---
                .CLK(clk),
                .RSTP(~rst_n),            // Reset for P register (Active High inside DSP)

                // --- Operation Mode (Fixed for A:B + C) ---
                .ALUMODE(4'b0000),        // 0000 = ADD
                .INMODE(5'b00000),        // Default A and B routing
                .OPMODE(9'b000_00_11_11), // Z=0, W=0, Y=C, X=A:B  =>  P = A:B + C

                // --- Clock Enables (Tie to high for continuous pipeline) ---
                .CEP(1'b1),               // Enable P register updates

                // --- Data Inputs ---
                .A(dsp_in_ab[47:18]),     // Upper 30 bits go to A port
                .B(dsp_in_ab[17:0]),      // Lower 18 bits go to B port
                .C(dsp_in_c),             // 48 bits go to C port

                // --- Data Output ---
                .P(dsp_out_p)             // 48-bit Result
            );

            // --- Truncate 48-bit result back to parameterized wire width ---
            assign reduction_16_fmap_wire[i] = dsp_out_p[`FIXED_MANT_WIDTH+2:0];
        end
    endgenerate
    // ============================================================================
    // ===| REDUCTION TREE: LUT-based Pipelined Adders (Optimized for 400MHz) |====
    // UltraScale+ CARRY8 primitives combined with immediate FDRE (Registers)
    // provide better routing and timing than forcing DSPs for simple additions.
    // ============================================================================

    // ===| Stage 2: Reduction 16 -> 8 |===========================================
    // Latency: 1 Clock Cycle
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 8; i++) begin
                reduction_8_fmap_wire[i] <= '0;
            end
        end else begin
            for (int i = 0; i < 8; i++) begin
                // Simple addition. Vivado will infer CARRY8 + FF in the same slice.
                reduction_8_fmap_wire[i] <= reduction_16_fmap_wire[2*i] + reduction_16_fmap_wire[2*i+1];
            end
        end
    end

    // ===| Stage 3: Reduction 8 -> 4 |============================================
    // Latency: 1 Clock Cycle
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin
                reduction_4_fmap_wire[i] <= '0;
            end
        end else begin
            for (int i = 0; i < 4; i++) begin
                reduction_4_fmap_wire[i] <= reduction_8_fmap_wire[2*i] + reduction_8_fmap_wire[2*i+1];
            end
        end
    end

    // ===| Stage 4: Reduction 4 -> 2 |============================================
    // Latency: 1 Clock Cycle
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 2; i++) begin
                reduction_2_fmap_wire[i] <= '0;
            end
        end else begin
            for (int i = 0; i < 2; i++) begin
                reduction_2_fmap_wire[i] <= reduction_4_fmap_wire[2*i] + reduction_4_fmap_wire[2*i+1];
            end
        end
    end

    // ===| Stage 5: Reduction 2 -> 1 (Final Sum) |================================
    // Latency: 1 Clock Cycle
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            OUT_reduction_result <= '0;
        end else begin
            OUT_reduction_result <= reduction_2_fmap_wire[0] + reduction_2_fmap_wire[1];
        end
    end

    // ============================================================================
    // Total Pipeline Latency for Reduction Tree:
    // Stage 1 (DSP: 32->16) : 1 Cycle (or more depending on DSP PREG/MREG configs)
    // Stage 2 (LUT: 16->8)  : 1 Cycle
    // Stage 3 (LUT: 8->4)   : 1 Cycle
    // Stage 4 (LUT: 4->2)   : 1 Cycle
    // Stage 5 (LUT: 2->1)   : 1 Cycle
    // ----------------------------------------
    // Total latency after FMap/Weight input: 5 Cycles
    // ============================================================================
endmodule
