`define MAX_CMD_CHAIN 4

// NPU Architecture
//`define ISA_WIDTH 32

`define DIM_X 1'b00
`define DIM_Y 1'b01
`define DIM_Z 2'b10
`define DIM_W 3'b11

`define MAX_MATRIX_DIM 4
`define MAX_MATRIX_WIDTH 32

//`define VDOTM_LANE_0 4'b0000
`define VDOTM_LANE_1 4'b0001
`define VDOTM_LANE_2 4'b0010
`define VDOTM_LANE_3 4'b0100
`define VDOTM_LANE_4 4'b1000

`define MASKING_WEIGHT 2'b00
`define BUFFER_WEIGHT_A1 4'b0000
`define BUFFER_WEIGHT_A2 4'b0001
`define BUFFER_WEIGHT_A3 4'b0010
`define BUFFER_WEIGHT_A4 4'b0011

`define MASKING_SCALE 2'b01
`define BUFFER_SCALE 4'b0100
`define CACHE_SCALE 4'b0101

`define MASKING_FMAP 2'b10
`define BUFFER_FMAP_C 4'b1000
`define CACHE_FMAP_C1 4'b1001
`define CACHE_FMAP_C2 4'b1010
