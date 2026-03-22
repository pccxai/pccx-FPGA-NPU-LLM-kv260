`timescale 1ns / 1ps
`include "stlc_Array.svh"

module stlc_fmap_staggered_delay #(
    // Fixed-point width after shifter
    parameter DATA_WIDTH = 27, 
    parameter ARRAY_SIZE = 32
)(
    input  logic clk,
    input  logic rst_n,

    // =| Input from FMap Cache Broadcast |=
    input  logic [DATA_WIDTH-1:0] fmap_in        [0:ARRAY_SIZE-1],
    input  logic                  fmap_valid,

    // =| Global Instruction from FSM |=
    input  logic [2:0]            global_inst,
    input  logic                  global_inst_valid,

    // =| 32 Staggered Outputs to Systolic Array Vertical Lanes |=
    output logic [DATA_WIDTH-1:0] row_data       [0:ARRAY_SIZE-1],
    output logic                  row_valid      [0:ARRAY_SIZE-1],
    output logic [2:0]            row_inst       [0:ARRAY_SIZE-1],
    output logic                  row_inst_valid [0:ARRAY_SIZE-1]
);

    // ===| Delay Line Implementation |=======
    // We use a shift register chain to delay data, valid, and instructions.
    // Col[0] = 0 delay, Col[1] = 1 delay, ..., Col[31] = 31 delay.

    genvar c;
    generate
        for (c = 0; c < ARRAY_SIZE; c++) begin : col_gen
            
            // =| Delay Logic for each Column |=
            if (c == 0) begin : no_delay
                // Col 0 has 0 additional delay (only 1-stage for timing)
                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        row_valid[c]      <= 1'b0;
                        row_inst_valid[c] <= 1'b0;
                        row_data[c]       <= '0;
                        row_inst[c]       <= 3'b000;
                    end else begin
                        row_data[c]       <= fmap_in[c];
                        row_valid[c]      <= fmap_valid;
                        row_inst[c]       <= global_inst;
                        row_inst_valid[c] <= global_inst_valid;
                    end
                end
            end else begin : shift_delay
                // Col[c] uses a shift register of length 'c'
                logic [DATA_WIDTH-1:0] shift_data       [0:c];
                logic                  shift_valid      [0:c];
                logic [2:0]            shift_inst       [0:c];
                logic                  shift_inst_valid [0:c];

                always_ff @(posedge clk) begin
                    if (!rst_n) begin
                        for (int i = 0; i <= c; i++) begin
                            shift_valid[i]      <= 1'b0;
                            shift_inst_valid[i] <= 1'b0;
                            shift_data[i]       <= '0;
                            shift_inst[i]       <= 3'b000;
                        end
                    end else begin
                        // Input to shift register
                        shift_data[0]       <= fmap_in[c];
                        shift_valid[0]      <= fmap_valid;
                        shift_inst[0]       <= global_inst;
                        shift_inst_valid[0] <= global_inst_valid;
                        
                        // =| Chain the registers |=
                        for (int i = 1; i <= c; i++) begin
                            shift_data[i]       <= shift_data[i-1];
                            shift_valid[i]      <= shift_valid[i-1];
                            shift_inst[i]       <= shift_inst[i-1];
                            shift_inst_valid[i] <= shift_inst_valid[i-1];
                        end
                    end
                end
                
                assign row_data[c]       = shift_data[c];
                assign row_valid[c]      = shift_valid[c];
                assign row_inst[c]       = shift_inst[c];
                assign row_inst_valid[c] = shift_inst_valid[c];
            end
        end
    endgenerate
endmodule

