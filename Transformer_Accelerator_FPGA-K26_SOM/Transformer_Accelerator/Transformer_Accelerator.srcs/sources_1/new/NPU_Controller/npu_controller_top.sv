`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"

// AXIL_STAT_OUT
// AXI4-Lite Read path : NPU → CPU
// Upper module pushes status into FIFO continuously.
// Drains FIFO to CPU when AXI4-Lite read handshake happens.

module npu_controller_top #(

) (
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // AXI4-Lite Slave : PS ↔ NPU control plane
    axil_if.slave S_AXIL_CTRL,

    //output instruction_t OUT_inst,

    output logic OUT_memcpy_uop_x64_valid;
    output memory_uop_x64_t OUT_memcpy_uop_x64;

    output logic OUT_vdotm_uop_x64_valid;
    output vdotm_uop_x64_t  OUT_vdotm_uop_x64;

    output logic OUT_mdotm_uop_x64_valid;
    output mdotm_uop_x64_t  OUT_mdotm_uop_x64;
);

    logic [`ISA_WIDTH-1:0] instruction_valid;
    logic [`ISA_WIDTH-1:0] instruction;

    ctrl_npu_frontend #()
    u_npu_frontend (
        .clk(clk),
        .rst_n(rst_n),
        .IN_clear(i_clear),

        // AXI4-Lite Slave : PS ↔ NPU control plane
        // Write channels : CPU → NPU (commands)
        .axil_if.slave(S_AXIL_CTRL),

        // Control from Brain
        //.IN_rd_start(),

        // Decoded command out → Dispatcher / FSM
        .OUT_RAW_instruction(raw_instruction),
        .OUT_kick(raw_instruction_pop_valid),

        // Status in ← Encoder / FSM
        .IN_enc_stat(),
        .IN_enc_valid(),

        .IN_fetch_ready(fetch_PC_ready)
    );


    logic fetch_PC_ready;
    logic [`ISA_WIDTH-1:0] raw_instruction;
    logic raw_instruction_pop_valid;
    instruction_t inst;
    logic valid_inst;

    memory_uop_x64_t memcpy_uop_x64;
    vdotm_uop_x64_t vdotm_uop_x64;
    mdotm_uop_x64_t mdotm_uop_x64;

    cu_npu_decoder u_decoder(
        .clk(clk),
        .rst_n(rst_n),
        .IN_RAW_instruction(raw_instruction),
        .raw_instruction_pop_valid(raw_instruction_pop_valid),
        .OUT_fetch_PC_ready(fetch_PC_ready),

        .OUT_memcpy_uop_x64_valid(memcpy_uop_x64_valid),
        .OUT_memcpy_uop_x64(memcpy_uop_x64),

        .OUT_vdotm_uop_x64_valid(vdotm_uop_x64_valid),
        .OUT_vdotm_uop_x64(vdotm_uop_x64),

        .OUT_mdotm_uop_x64_valid(mdotm_uop_x64_valid),
        .OUT_mdotm_uop_x64(mdotm_uop_x64)
    );

endmodule

