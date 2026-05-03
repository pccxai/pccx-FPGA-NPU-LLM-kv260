# =============================================================================
# pccx_timing.xdc — timing-only constraints for the pccx v002 NPU core.
#
# Pin / IO constraints are NOT here. This core is a PL-side slave sitting
# behind the Zynq PS, so pin-level placement is delegated to the BD that
# instantiates it. This file just tells Vivado about the two clock
# domains and how they cross.
#
# Used under out-of-context synthesis (synth_design -mode out_of_context)
# and then re-validated in the BD.
# =============================================================================

# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------
# AXI / control-plane clock — AXI-Lite MMIO, CDC FIFO drain sides, etc.
create_clock -period 4.000 -name axi_clk   -waveform {0.000 2.000} [get_ports clk_axi]

# Core compute clock — DSP48E2 array, GEMV lanes, CVO SFU.
create_clock -period 2.500 -name core_clk  -waveform {0.000 1.250} [get_ports clk_core]

# The two domains are genuinely asynchronous; every path across them is
# expected to use a CDC FIFO or a properly-timed synchroniser. Mark them
# as async so Vivado does not waste effort timing single-flop crossings.
set_clock_groups -asynchronous \
    -group [get_clocks axi_clk] \
    -group [get_clocks core_clk]

# ---------------------------------------------------------------------------
# Reset synchronisers
# ---------------------------------------------------------------------------
# Any path into the first flop of a reset bridge is inherently async.
# Mark those as false paths so they don't inflate the WNS report.
set_false_path -quiet \
    -to [get_cells -quiet -hier -filter {NAME =~ */u_reset_sync*/sync_reg_reg[0]}]

# ---------------------------------------------------------------------------
# Asynchronous FIFO flag paths
# ---------------------------------------------------------------------------
# Xilinx XPM_FIFO_ASYNC instances handle their own meta-stability; mark
# the gray-coded pointer crossings as async.
set_false_path -quiet \
    -from [get_cells -quiet -hier -filter {NAME =~ */wr_pntr_gray_reg*}] \
    -to   [get_cells -quiet -hier -filter {NAME =~ */wr_pntr_gray_sync_reg*}]
set_false_path -quiet \
    -from [get_cells -quiet -hier -filter {NAME =~ */rd_pntr_gray_reg*}] \
    -to   [get_cells -quiet -hier -filter {NAME =~ */rd_pntr_gray_sync_reg*}]

# ---------------------------------------------------------------------------
# Multi-cycle paths on the accumulator drain
# ---------------------------------------------------------------------------
# The GEMM systolic array accumulates for up to 1024 cycles before the
# controller issues a flush (§2.2 of the Phase A audit). The drain path
# from P-register to the result packer can tolerate multiple cycles
# because the controller stalls new MACs during flush.
set_multicycle_path -quiet -setup 2 \
    -from [get_cells -quiet -hier -filter {NAME =~ */GEMM_dsp_unit*/DSP_HARD_BLOCK*}] \
    -to   [get_cells -quiet -hier -filter {NAME =~ */u_mat_result_normalizer*}]
set_multicycle_path -quiet -hold  1 \
    -from [get_cells -quiet -hier -filter {NAME =~ */GEMM_dsp_unit*/DSP_HARD_BLOCK*}] \
    -to   [get_cells -quiet -hier -filter {NAME =~ */u_mat_result_normalizer*}]
