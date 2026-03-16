`timescale 1ns / 1ps

module delay_line #(
    parameter WIDTH = 8,
    parameter DELAY = 1 
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [WIDTH-1:0] in_data,
    output logic [WIDTH-1:0] out_data
);

    // [Vivado bug bypass] Trick to prevent array index from becoming [-1] when DELAY is 0
    localparam SAFE_DELAY = (DELAY == 0) ? 1 : DELAY;
    logic [WIDTH-1:0] shift_reg [0:SAFE_DELAY-1];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < SAFE_DELAY; i++) shift_reg[i] <= 0;
        end else begin
            shift_reg[0] <= in_data;
            for (int i = 1; i < SAFE_DELAY; i++) begin
                shift_reg[i] <= shift_reg[i-1];
            end
        end
    end

    // If DELAY is 0, wire (direct) connection, otherwise delayed data (end) output.
    assign out_data = (DELAY == 0) ? in_data : shift_reg[DELAY-1];

endmodule