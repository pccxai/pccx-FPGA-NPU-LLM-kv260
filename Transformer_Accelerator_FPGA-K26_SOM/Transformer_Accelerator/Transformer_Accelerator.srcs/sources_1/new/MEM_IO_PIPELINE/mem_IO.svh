`define AXI_FMAP_PORT_CNT 1
`define AXI_WEIGHT_PORT_CNT 4
`define AXI_RESULT_PORT_CNT 1

// Mappings for MMIO commands
`define CMD_START_BIT 0
`define CMD_CLEAR_BIT 1
`define CMD_INST_START_BIT 2
`define CMD_INST_END_BIT 4

// Ports Size
`define FMAP_HPC_IN 256
`define RESULT_ACP_OUT 128
`define WEIGHT_HP 512
// max 128 * 2
`define INSTRUCTION_HPM 128