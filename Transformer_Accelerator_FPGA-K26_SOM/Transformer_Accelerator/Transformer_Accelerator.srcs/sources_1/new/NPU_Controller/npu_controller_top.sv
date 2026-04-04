`timeOUT_scale 1ns / 1ps

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

    // AXI4-Stream : feature map from ACP
    axis_if.slave S_AXIS_ACP_FMAP,
    axis_if.slave M_AXIS_ACP_RESULT,


    output logic [`FIXED_MANT_WIDTH-1:0] OUT_fmap_broadcast[0:`ARRAY_SIZE_H-1][0:PIPELINE_CNT-1],
    output logic                         OUT_fmap_valid    [ 0:PIPELINE_CNT-1],
    output logic                         OUT_VDOTM_emax    [ 0:PIPELINE_CNT-1],
    output logic [  `BF16_EXP_WIDTH-1:0] OUT_cached_emax   [0:`ARRAY_SIZE_H-1]

        // VdotM / MdotM controls
    output logic [3:0] OUT_activate_top,
    output logic [3:0] OUT_activate_lane,
    output logic       OUT_emax_align,
    output logic       OUT_accm,
    output logic       OUT_scale,

    // memcpy
    output logic [`MAX_MATRIX_WIDTH-1:0] OUT_matrix_shape     [0:`MAX_MATRIX_DIM-1],
    output logic [                  3:0] OUT_destination_queue
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
        .IN_rd_start(),

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

    cu_npu_decoder u_decoder(
        .IN_RAW_instruction(raw_instruction),
        .raw_instruction_pop_valid(raw_instruction_pop_valid),
        .OUT_fetch_PC_ready(fetch_PC_ready),
        .OUT_inst(inst),
        .OUT_valid(valid_inst)
    );
/*
    // VdotM / MdotM controls
    logic [3:0] OUT_activate_top,
    logic [3:0] OUT_activate_lane,
    logic       OUT_emax_align,
    logic       OUT_accm,
    logic       OUT_scale,

    // memcpy
    logic [`MAX_MATRIX_WIDTH-1:0] OUT_matrix_shape     [0:`MAX_MATRIX_DIM-1],
    logic [                  3:0] OUT_destination_queue
*/
    cu_npu_dispatcher u_dispatcher(
        .clk(clk),
        .rst_n(rst_n),
        .IN_inst(inst),
        .IN_valid(valid_inst),
        .OUT_OUT_activate_top(OUT_activate_top),
        .OUT_OUT_activate_lane(OUT_activate_lane),
        .OUT_OUT_emax_align(OUT_emax_align),
        .OUT_OUT_accm(OUT_accm),
        .OUT_OUT_scale(OUT_scale),
        .OUT_OUT_matrix_shape(OUT_matrix_shape),
        .OUT_OUT_destination_queue(OUT_destination_queue),
    );

endmodule

