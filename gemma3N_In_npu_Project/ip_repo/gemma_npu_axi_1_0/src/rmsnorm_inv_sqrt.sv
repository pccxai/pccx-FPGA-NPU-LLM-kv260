`timescale 1ns / 1ps

module rmsnorm_inv_sqrt (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               valid_in,
    input  logic [31:0]        i_mean_sq,   // 32-bit input (mean of squares)
    
    output logic               valid_out,
    output logic [15:0]        o_inv_sqrt   // 16-bit output (Q1.15 format)
);
    // 1. Separate section index and decimal point residue
    logic [9:0]  segment_idx;
    logic [21:0] fractional_x;

    assign segment_idx  = i_mean_sq[31:22]; // Top 10 bits (BRAM address)
    assign fractional_x = i_mean_sq[21:0];  // Lower 22 bits (x for interpolation)

    // 2. 1024-segment ultra-precision BRAM (slope & intercept)
    (* rom_style = "block" *) logic signed [15:0] lut_slope [0:1023];
    (* rom_style = "block" *) logic signed [31:0] lut_inter [0:1023]; // Expanded to 32 bit.

    initial begin
        // Load two 1024-split cheat sheets made with Python.
        $readmemh("rmsnorm_slope.mem", lut_slope);
        $readmemh("rmsnorm_inter.mem", lut_inter);
    end

    // 3. Pipeline stage 1 (BRAM Read & data delay)
    logic signed [15:0] reg_a;          // Slope register (16bit)
    logic signed [31:0] reg_b;          // Section register (32bit modified!)
    logic        [21:0] reg_frac_x;     // Register for decimal point delay (22bit)
    logic               reg_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            reg_a <= 0;
            reg_b <= 0;
            reg_frac_x <= 0;
            reg_valid <= 0;
        end else if (valid_in) begin
            reg_a <= lut_slope[segment_idx]; // BRAM read (takes 1 clock)
            reg_b <= lut_inter[segment_idx]; // Read BRAM
            reg_frac_x <= fractional_x;      // Delay x degree to match timing while reading BRAM.
            reg_valid <= 1'b1;
        end else begin
            reg_valid <= 1'b0;
        end
    end

    // 4. Pipeline stage 2 (DSP48E2 operation: y = ax + b)
    // 
    // DSP48E2 supports 27x18 multiplication. a(16bit) * x(23bit sign extension)
    // Instant calculation of multiplication and addition using ‘wire’ regardless of clock.
    logic signed [47:0] dsp_mult_comb;
    assign dsp_mult_comb = (reg_a * $signed({1'b0, reg_frac_x})) + reg_b; 
    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            o_inv_sqrt <= 0;
            valid_out  <= 0;
        end else if (reg_valid) begin
            // Safely grab the value of the wire (comb) that has already been calculated.
            o_inv_sqrt <= dsp_mult_comb[30:15]; 
            valid_out  <= 1'b1;
        end else begin
            valid_out  <= 1'b0;
        end
    end

endmodule