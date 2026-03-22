`define TRUE 1'b1
`define FALSE 1'b0

`define PREG_SIZE 48
`define MREG_SIZE 48

`define ARRAY_SIZE_H 32
`define ARRAY_SIZE_V 32

`define stlc_instruction_dispatcher_CLOCK_CONSUMPTION 1

`define KV260_AXIDMA_FULL_BANDWIDTH 128

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// DSP INSTRUCTION
`define DSP_INSTRUCTION_CNT 4

`define DSP_IDLE_MOD 2'b00
`define DSP_SYSTOLIC_MOD_P 2'b01
`define DSP_GEMV_STATIONARY_MOD 2'b10 // Used for Weight-Stationary GEMV
`define DSP_SHIFT_RESULT_MOD 2'b11

/*
`define DSP_SUB_MOD = 2'b11
`define DSP_INV_DIV_MOD 
*/
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

`define DSP48E2_MAXIN_H 18
`define DSP48E2_MAXIN_V 30
`define DSP48E2_MAXOUT 48

// INT4 - DSP48E2_MAXIN_H
`define STLC_MAC_UNIT_IN_H 4

// BFLOAT16 - DSP48E2_MAXIN_V
// aligned Mantissa size 27
`define STLC_MAC_UNIT_IN_V 27

// DSP48E2_MAXOUT
`define DSP_RESULT_SIZE 48


// ===| SYSTEM-WIDE ARCHITECTURAL CONSTANTS |==============

// [AXI-Stream & DMA]
`define AXI_DATA_WIDTH 128
`define AXI_PORT_CNT 4

// [Elastic Buffers (FIFOs)]
`define XPM_FIFO_DEPTH 512

// [Feature Map Cache (SRAM)]
`define FMAP_CACHE_DEPTH 2048
`define FMAP_ADDR_WIDTH 11 // log2(2048)

// [Pipelining & Latency Hiding]
// Calculated as Array H (32) + Array V (32) + Pipeline Overheads
`define SYSTOLIC_TOTAL_LATENCY 64

// [BF16 Data Formats]
`define BF16_WIDTH 16
`define BF16_EXP_WIDTH 8
`define BF16_MANT_WIDTH 7
`define FIXED_MANT_WIDTH 27

// ========================================================

// systolic delay line 
`define MINIMUM_DELAY_LINE_LENGTH 1

// systolic delay line V | TYPE:INT4
`define INT4_WIDTH 4

// systolic delay line H | TYPE: BFLOAT 16
`define BFLOAT_WIDTH 16