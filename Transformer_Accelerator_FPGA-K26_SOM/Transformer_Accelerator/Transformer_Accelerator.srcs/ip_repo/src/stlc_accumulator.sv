`timescale 1ns / 1ps
`include "stlc_Array.svh"

module stlc_accumulator (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        i_clear,   
    input  logic        i_valid,   
    
    
    input  logic [47:0] PCIN,     
    
    // final output -> to P port
    output logic [47:0] stlc_ACC_result  
);

    // OPMODE: W=00, Z=PCIN(001), Y=0(00), X=P(10) -> P = P + PCIN
    wire [8:0] static_opmode  = 9'b00_001_00_10;
    wire [3:0] static_alumode = 4'b0000; 


    DSP48E2 #(
        
        .AREG(0), .BREG(0), .CREG(0), .MREG(0), .PREG(1),
        .ACASCREG(0), .BCASCREG(0),
        .OPMODEREG(0), .ALUMODEREG(0),        
        // Disable multiplier
        .USE_MULT("NONE") 
    ) DSP_ACC (
        .CLK(clk),
        
        .RSTP(i_clear || ~rst_n), 
        .RSTA(1'b0), .RSTB(1'b0), .RSTM(1'b0), .RSTCTRL(1'b0), 
        .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTC(1'b0),


        .CEP(i_valid), 
        .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), 
        .CEM(1'b0), .CECTRL(1'b0), .CEALUMODE(1'b0), .CEC(1'b0),


        .A(30'd0), .B(18'd0), .C(48'd0), 
        
        .PCIN(PCIN),
        .PCOUT(), 
        .ACOUT(),

        .OPMODE(static_opmode),
        .ALUMODE(static_alumode),


        .P(stlc_ACC_result)  
    );
endmodule