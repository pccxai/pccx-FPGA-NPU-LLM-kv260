`define TRUE 1'b1
`define FALSE 1'b0


// NPU Architecture
`define ISA_WIDTH 32

// activate_top
`define TOP_STLC 0
`define TOP_VDOTM 1

// ===| instruction MOD |========================
`define MOD_M_DOT_M 10
`define MOD_V_DOT_M 11

`define PIPELINE_CNT 2
`define PIPELINE_M_DOT_M 0
`define PIPELINE_V_DOT_M 1

// ===| KV260's DSP48E2 |========================

`define DSP48E2_MAXIN_H 18
`define DSP48E2_MAXIN_V 30
`define DSP48E2_MAXOUT 48
`define PREG_SIZE 48
`define MREG_SIZE 48

// ===| KV260's DSP48E2 - END |==================

`define BF16 16
`define BF16_EXP 8
`define BF16_MANTISSA 7
`define INT4 4

`define DATA_WIDTH 27

`define TRUE 1
`define FALSE 0





// ===| BF16 Data Formats |==========================
`define BF16_WIDTH 16
`define BF16_EXP_WIDTH 8
`define BF16_MANT_WIDTH 7
`define FIXED_MANT_WIDTH 27
// ===| BF16 Data Formats - END |====================


// ===| FP32 Data Formats |==========================
`define FMAP_CACHE_OUT_SIZE 32
`define FP32 32
// ===| FP32 Data Formats - END |====================

// ===| INT4 Data Formats |==========================
`define INT4_TO_IDX(val) ((val) + 8)
`define INT4_MAX_VAL 7
`define INT4_MIN_VAL -8
`define INT4_RANGE 16
// ===| INT4 Data Formats - END |====================



// ===| SYSTEM-WIDE ARCHITECTURAL CONSTANTS |==============

// [AXI-Stream & DMA]
`define HP_PORT_MAX_WIDTH 512
`define HP_PORT_SINGLE_WIDTH 128
`define HP_PORT_CNT 4
`define HP_WEIGHT_CNT(P_size, W_size) (P_size / W_size)

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


// ========================================================

// DSP48E2_MAXOUT
`define DSP48E2_POUT_SIZE 48
`define DSP48E2_AB_WIDTH 48
`define DSP48E2_C_WIDTH 48
`define DSP48E2_A_WIDTH 30
`define DSP48E2_B_WIDTH 18

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



// INT4 - DSP48E2_MAXIN_H
`define STLC_MAC_UNIT_IN_H 4

// BFLOAT16 - DSP48E2_MAXIN_V
// aligned Mantissa size 27
`define STLC_MAC_UNIT_IN_V 27
