`timescale 1ns / 1ps

/**
 * Module: stlc_weight_dispatcher
 * Description: 
 *   Unpacks 128-bit wide data into 32 individual 4-bit INT4 weights.
 *   Provides registered outputs to maintain 400MHz timing.
 */

module TO_stlc_weight_dispatcher(
    input  logic clk,
    input  logic rst_n,

// ===| 128-bit Input from FIFO |============================
    input  logic [127:0] fifo_data,
    input  logic         fifo_valid,
    output logic         fifo_ready,

// ===| 32 x 4-bit Outputs to Systolic Array (V_in) |========
    output logic [3:0]   weight_out [0:31],
    output logic         weight_valid
);

// ===| Flow Control: Always ready if not stalled by downstream |=====================
    assign fifo_ready = 1'b1; 

// ===| Unpacking Logic with Pipeline Registers |=====================================
// ===| This ensures that the massive fan-out (1 to 32) doesn't break timing. |=======
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight_valid <= 1'b0;
            for (int i=0; i<32; i++) weight_out[i] <= 4'd0;
        end else begin
            weight_valid <= fifo_valid;
            
// ===| Unpack 128-bit into 32 x 4-bit |==============================================
            for (int i=0; i<32; i++) begin
                weight_out[i] <= fifo_data[(i*4) +: 4];
            end
        end
    end
endmodule