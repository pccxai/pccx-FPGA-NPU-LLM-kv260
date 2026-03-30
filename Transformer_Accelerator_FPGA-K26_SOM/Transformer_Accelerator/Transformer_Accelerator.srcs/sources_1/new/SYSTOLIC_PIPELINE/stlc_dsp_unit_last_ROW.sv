`timescale 1ns / 1ps
`include "stlc_Array.svh"

module stlc_dsp_unit_last_ROW #(
    parameter IS_TOP_ROW = 0 // By definition, last row is 0, but added for consistency if needed
)(
    input   logic clk,
    input   logic rst_n,

    input   logic i_clear,
    input   logic i_valid,
    input   logic inst_valid_in_V,
    input   logic i_weight_valid,
    output  logic o_valid,

    // [Horizontal] int4 (4-bit) -> DSP B port
    input   logic [`STLC_MAC_UNIT_IN_H - 1:0] in_H,
    output  logic [`STLC_MAC_UNIT_IN_H - 1:0] out_H,

    // [Vertical] 30-bit -> DSP ACIN port
    input   logic [29:0] ACIN_in,
    output  logic [29:0] ACOUT_out,

    // [3-Bit VLIW Instruction]
    input   logic [2:0] instruction_in_V,
    output  logic [2:0] instruction_out_V,
    output  logic       inst_valid_out_V,

    // vertical shift port
    // pass value to Accumulator
    input   logic [47:0] V_result_in,   // PCIN from upper DSP's PCOUT
    output  logic [47:0] V_result_out,  // PCOUT to lower DSP's PCIN (if any)

    // Pass value to barrelshifter
    output  logic [47:0] stlc_unit_results
);

    // ===| [Instruction Latch (Event-Driven)] |============================
    logic [2:0] current_inst;

    always_ff @(posedge clk) begin
        if (!rst_n || i_clear) begin
            current_inst <= 3'b000;
        end else if (inst_valid_in_V) begin
            current_inst <= instruction_in_V;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            instruction_out_V <= 3'b000;
            inst_valid_out_V  <= 1'b0;
        end else begin
            instruction_out_V <= instruction_in_V;
            inst_valid_out_V  <= inst_valid_in_V;
        end
    end

    // ===| [The "Flush & Load" Sequencer] |================================
    logic [3:0] flush_sequence;

    always_ff @(posedge clk) begin
        if (!rst_n || i_clear) begin
            flush_sequence <= 4'd0;
        end else begin
            flush_sequence <= {flush_sequence[2:0], 1'b0};
            if (inst_valid_in_V && instruction_in_V[2] == 1'b1) begin
                flush_sequence[0] <= 1'b1;
            end
        end
    end

    // ===| [Hardware Mapping (VLIW Decoding)] |============================
    logic [8:0] dynamic_opmode;
    logic [3:0] dynamic_alumode;

    logic is_flushing;
    assign is_flushing = flush_sequence[1] | flush_sequence[2];

    always_comb begin
        if (is_flushing) begin
            dynamic_opmode  = 9'b00_001_00_00;
            dynamic_alumode = 4'b0000;
        end else if (current_inst[0] == 1'b1) begin
            dynamic_opmode  = 9'b00_010_01_01;
            dynamic_alumode = 4'b0000;
        end else begin
            dynamic_opmode  = 9'b00_010_00_00;
            dynamic_alumode = 4'b0000;
        end
    end

    logic dsp_ce_p;
    assign dsp_ce_p = current_inst[0] | is_flushing;

    // ===| [Fabric FF & Weight Pipeline] |=================================
    always_ff @(posedge clk) begin
        if(!rst_n || i_clear) begin
            out_H <= 0;
        end else begin
            if (i_weight_valid) begin
                out_H <= in_H;
            end
        end
    end

    // ===| [Dual B-Register Control] |================
    logic dsp_ce_b1;
    logic dsp_ce_b2;
    logic load_trigger;

    assign load_trigger = flush_sequence[3];

    always_comb begin
        if (current_inst[1] == 1'b1) begin
            dsp_ce_b1 = i_valid;
            dsp_ce_b2 = i_valid;
        end else begin
            dsp_ce_b1 = load_trigger | i_weight_valid;
            dsp_ce_b2 = load_trigger;
        end
    end

    logic valid_delay;
    always_ff @(posedge clk) begin
        if (!rst_n) valid_delay <= 1'b0;
        else        valid_delay <= i_valid;
    end
    assign o_valid = valid_delay;

    // <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    // [DSP48E2 primitive instantiation] <><><><><><><><><><><><><><><><><>
    logic [17:0] in_H_padded;
    assign in_H_padded = {{14{in_H[`STLC_MAC_UNIT_IN_H - 1]}}, in_H};

    DSP48E2 #(
        .A_INPUT(IS_TOP_ROW ? "DIRECT" : "CASCADE"),
        .B_INPUT("DIRECT"),
        .AREG(1), .BREG(2), .CREG(0), .MREG(1), .PREG(1),
        .OPMODEREG(1),
        .ALUMODEREG(1),
        .USE_MULT("MULTIPLY")
    ) DSP_HARD_BLOCK (
        .CLK(clk),
        .RSTA(i_clear), .RSTB(i_clear), .RSTM(i_clear), .RSTP(i_clear),
        .RSTCTRL(i_clear), .RSTALLCARRYIN(i_clear), .RSTALUMODE(i_clear), .RSTC(i_clear),

        .CEA1(i_valid), .CEA2(i_valid),
        .CEB1(dsp_ce_b1), .CEB2(dsp_ce_b2),
        .CEM(i_valid),
        .CEP(dsp_ce_p),
        .CECTRL(1'b1),
        .CEALUMODE(1'b1),

        .A(30'd0),
        .ACIN(ACIN_in),
        .ACOUT(ACOUT_out),

        .B(in_H_padded),
        .C(48'd0),

        .PCIN(V_result_in),
        .PCOUT(V_result_out),

        .OPMODE(dynamic_opmode),
        .ALUMODE(dynamic_alumode),

        // All result will send to Barrelshifter(FF)
        .P(stlc_unit_results)
    );
    // [DSP48E2 primitive instantiation] <><><><><><><><><><><><><><><><><>
    // <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

endmodule