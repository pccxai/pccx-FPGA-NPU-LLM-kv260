`timescale 1ns / 1ps

module systolic_NxN #(
    parameter ARRAY_SIZE = 32 // 32x32 architecture.
)(
    input  logic clk,
    input  logic rst_n,
    input  logic i_clear, // Global clear signal received from outside (Top FSM)

    // 64 data pouring ‘simultaneously’ from npu_core_top
    input  logic [7:0] in_a [0:ARRAY_SIZE-1], 
    input  logic [7:0] in_b [0:ARRAY_SIZE-1],
    input  logic       in_valid,

    output logic [ARRAY_SIZE*ARRAY_SIZE*32-1:0] out_acc_flat
);

    // Two-dimensional array for internal operations
    logic [31:0] out_acc [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // 2D -> 1D Flat conversion
    genvar r, c;
    generate
        for (r = 0; r < ARRAY_SIZE; r++) begin
            for (c = 0; c < ARRAY_SIZE; c++) begin
                assign out_acc_flat[(r*ARRAY_SIZE+c)*32 +: 32] = out_acc[r][c];
            end
        end
    endgenerate

    // Internal wires to connect between each PE
    logic [7:0] wire_a [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    logic [7:0] wire_b [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    logic       wire_v [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]; 

    // -----------------------------------------------------------------
    // Input data cascade delay using Delay Line (Wavefront Skewing)
    // -----------------------------------------------------------------
    genvar i;
    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : delay_skewing
            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_a (
                .clk(clk), .rst_n(rst_n),
                .in_data(in_a[i]), .out_data(wire_a[i][0])
            );

            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_b (
                .clk(clk), .rst_n(rst_n),
                .in_data(in_b[i]), .out_data(wire_b[0][i])
            );
        end 
    endgenerate

    // -----------------------------------------------------------------
    // Create 2D PE Array (automatically copy 4,096 mac_units)
    // -----------------------------------------------------------------
    genvar row, col;
    generate
        for (row = 0; row < ARRAY_SIZE; row++) begin : row_loop
            for (col = 0; col < ARRAY_SIZE; col++) begin : col_loop
                
                // Reflecting Claude's point: 100% composition compatibility is secured by removing variable declarations outside of if.
                logic current_i_valid;
                
                if (row == 0 && col == 0) begin
                    assign current_i_valid = in_valid;
                end else if (col > 0) begin
                    assign current_i_valid = wire_v[row][col-1]; 
                end else begin //col = 0, then upper row 
                    assign current_i_valid = wire_v[row-1][col]; 
                end

                pe_unit u_pe (
                    .clk(clk), .rst_n(rst_n),
                    .i_clear(i_clear),                // Simultaneous clear wiring to 4096 PEs.
                    .i_valid(current_i_valid),
                    .i_a(wire_a[row][col]),
                    .i_b(wire_b[row][col]),
                    .o_a(wire_a[row][col+1]),
                    .o_b(wire_b[row+1][col]),
                    .o_valid(wire_v[row][col]),
                    .o_acc(out_acc[row][col])
                );
            end
        end
    endgenerate

endmodule