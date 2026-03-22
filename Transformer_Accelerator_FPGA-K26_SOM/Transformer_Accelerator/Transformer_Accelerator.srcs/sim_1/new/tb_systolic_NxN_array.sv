/*
`timescale 1ns / 1ps

`include "stlc_Array.svh"

module tb_systolic_NxN_array;
    // Parameters
    localparam ARRAY_H = `ARRAY_SIZE_H;
    localparam ARRAY_V = `ARRAY_SIZE_V;
    localparam DATA_WIDTH_H = `STLC_MAC_UNIT_IN_H;
    localparam DATA_WIDTH_V = `STLC_MAC_UNIT_IN_V;

    // Signals
    logic   clk;
    logic   rst_n;
    logic   i_clear;
    logic   [DATA_WIDTH_H-1:0] H_in [0:ARRAY_H-1];
    logic   [DATA_WIDTH_V-1:0] V_in [0:ARRAY_V-1];
    logic   in_valid;

    logic   is_last_op;

    wire    [`DSP_RESULT_SIZE-1:0] H_out [0:ARRAY_H-1];
    //wire    [DATA_WIDTH_V-1:0] V_out [0:ARRAY_V-1];

    // DUT Instantiation
    systolic_NxN_array #(
        .ARRAY_HORIZONTAL(ARRAY_H),
        .ARRAY_VERTICAL(ARRAY_V)
    ) uut (
        .clk        (clk),
        .rst_n      (rst_n),
        .i_clear    (i_clear),
        .H_in       (H_in),
        .V_in       (V_in),
        .i_valid   (in_valid),
        .is_last_op(is_last_op),
        .H_out      (H_out)
        //.V_out      (V_out)
    );

    // Clock Generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz

    // Simulation Task: Initialize inputs
    task init_inputs();
        for (int i = 0; i < ARRAY_H; i++) H_in[i] = 0;
        for (int j = 0; j < ARRAY_V; j++) V_in[j] = 0;
        is_last_op = 0;
        in_valid = 0;
        i_clear = 0;
    endtask

    // Simulation Task: Apply Reset
    task reset_uut();
        rst_n = 0;
        #100;
        rst_n = 1;
        #20;
    endtask

    // Simulation Task: Clear DSP Units
    task clear_dsp();
        @(posedge clk);
        i_clear <= 1;
        @(posedge clk);
        i_clear <= 0;
        $display("[TB] DSP Accumulators cleared.");
    endtask

    // Main Test Sequence
    initial begin
        // 1. Initialize & Reset
        init_inputs();
        reset_uut();
        
        // 2. Clear Accumulators
        clear_dsp();

        // 3. Drive Data (Staggered Pattern for Systolic Array)
        // Systolic array requires inputs to be shifted in time across rows/cols
        $display("[TB] Starting Data Injection...");
        
        for (int cycle = 0; cycle < (ARRAY_H + ARRAY_V + 10); cycle++) begin
            @(posedge clk);
            in_valid <= 1; // Assuming RTL uses this to trigger internal logic
            
            // Feed Weight (H_in) and Activation (V_in)
            for (int i = 0; i < ARRAY_H; i++) begin
                // Only feed data if the cycle is right for the i-th row/column
                if (cycle >= i && cycle < (i + 10)) begin
                    H_in[i] <= 16'h3C00; // BF16 for 1.0 (approx)
                    V_in[i] <= 4'h1;    // INT4 for 1
                end else begin
                    H_in[i] <= 0;
                    V_in[i] <= 0;
                end
            end
        end

        

        // 4. Wait for output propagation
        is_last_op <= 1;
        in_valid <= 0;

        repeat (ARRAY_H + ARRAY_V + 5) @(posedge clk);
        
        is_last_op <= 0;

        repeat (10) @(posedge clk);

        

        $display("[TB] Simulation Finished.");
        $finish;
    end

    // Monitor (Optional)
    initial begin
        $monitor("Time: %0t | H_in[0]: %h | V_in[0]: %h | H_out[0]: %h", $time, H_in[0], V_in[0], H_out[0]);
    end

endmodule

*/