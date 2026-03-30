`timescale 1ns / 1ps
`include "stlc_Array.svh"

module stlc_dsp_unit #(
    parameter IS_TOP_ROW = 0,
    parameter BREAK_CASCADE = 0 // If 1, break the vertical cascade chain here
)(
    input   logic clk,
    input   logic rst_n,

    input   logic i_clear,
    input   logic i_valid,        // Feature Map Data Valid
    input   logic i_weight_valid, // Background Weight Shift Enable
    output  logic o_valid,

    // [Horizontal] int4 (4-bit) -> external FF -> DSP B port
    input   logic [`STLC_MAC_UNIT_IN_H - 1:0] in_H,
    output  logic [`STLC_MAC_UNIT_IN_H - 1:0] out_H,

    // [Vertical] 30-bit -> DSP A/ACIN port
    input   logic [29:0] in_V, // Used if IS_TOP_ROW == 1 or BREAK_CASCADE == 1
    input   logic [29:0] ACIN_in, // Used if IS_TOP_ROW == 0 and BREAK_CASCADE == 0
    output  logic [29:0] ACOUT_out,

    // [3-Bit VLIW Instruction]
    input   logic [2:0] instruction_in_V,
    output  logic [2:0] instruction_out_V,
    input   logic       inst_valid_in_V,   // Cascaded Instruction Valid
    output  logic       inst_valid_out_V,

    // vertical shift port
    input   logic [47:0] V_result_in,   // PCIN (or Fabric C) from upper DSP
    output  logic [47:0] V_result_out,  // PCOUT to lower DSP's PCIN
    output  logic [47:0] P_fabric_out   // P to lower DSP's Fabric C if broken
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

    // Pass instruction and its valid signal down to the next PE
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

    // OPMODE Selection
    // W(2), Z(3), Y(2), X(2)
    // If BREAK_CASCADE == 1, we must take the previous result from the C port instead of PCIN.
    // Z-mux: 001 is PCIN, 011 is C.
    localparam logic [2:0] Z_MUX = BREAK_CASCADE ? 3'b011 : 3'b001;

    always_comb begin
        if (is_flushing) begin
            // Flush: P = 0 + 0 + 0 (Clear accumulator)
            dynamic_opmode  = 9'b00_000_00_00;
            dynamic_alumode = 4'b0000;
        end else if (current_inst[0] == 1'b1) begin
            // Calc: P = P_prev + A*B
            dynamic_opmode  = {2'b00, Z_MUX, 2'b01, 2'b01};
            dynamic_alumode = 4'b0000;
        end else begin
            // Idle: P = P_prev (Pass through)
            dynamic_opmode  = {2'b00, Z_MUX, 2'b00, 2'b00};
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

    // If TOP_ROW or BREAK_CASCADE, we get A from Fabric (in_V). Otherwise from ACIN.
    logic [29:0] dsp_a_input;
    assign dsp_a_input = (IS_TOP_ROW || BREAK_CASCADE) ? in_V : 30'd0;

    logic [29:0] dsp_acin_input;
    assign dsp_acin_input = (IS_TOP_ROW || BREAK_CASCADE) ? 30'd0 : ACIN_in;

    // If BREAK_CASCADE, we receive the accumulated result from Fabric C (V_result_in)
    logic [47:0] dsp_c_input;
    assign dsp_c_input = BREAK_CASCADE ? V_result_in : 48'd0;

    logic [47:0] dsp_pcin_input;
    assign dsp_pcin_input = BREAK_CASCADE ? 48'd0 : V_result_in;

    logic [47:0] p_internal;

    DSP48E2 #(
        .A_INPUT((IS_TOP_ROW || BREAK_CASCADE) ? "DIRECT" : "CASCADE"),
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
        .CEC(1'b1), // Enable C register if breaking cascade

        .A(dsp_a_input),
        .ACIN(dsp_acin_input),
        .ACOUT(ACOUT_out),

        .B(in_H_padded),
        .C(dsp_c_input),

        .PCIN(dsp_pcin_input),
        .PCOUT(V_result_out),

        .OPMODE(dynamic_opmode),
        .ALUMODE(dynamic_alumode),
        .P(p_internal)
    );

    // We must expose the internal P so it can be routed via fabric to the next DSP
    assign P_fabric_out = p_internal;

endmodule