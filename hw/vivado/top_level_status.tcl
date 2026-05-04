# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
# =============================================================================
# top_level_status.tcl — explicit full KV260 top-level / BD bitstream gate.
#
# This script does not synthesize or route the OOC NPU core. It records whether
# the repo has the ingredients needed for a full board bitstream flow and fails
# fast when the flow is incomplete. The point is to keep OOC timing evidence
# separate from a real KV260 top-level implementation.
#
# Usage:
#   vivado -mode batch -source vivado/top_level_status.tcl -tclargs status
#   vivado -mode batch -source vivado/top_level_status.tcl -tclargs bitstream
# =============================================================================

set HW_ROOT  [file normalize [file dirname [info script]]/..]
set REPORTS  $HW_ROOT/build/reports
file mkdir $REPORTS

set MODE "status"
if {[llength $argv] > 0 && [lindex $argv 0] ne ""} {
    set MODE [lindex $argv 0]
}
if {$MODE ne "status" && $MODE ne "bitstream"} {
    puts "\[pccx\] invalid top-level mode '$MODE'; expected status or bitstream."
    exit 2
}

set TARGET_PART  xck26-sfvc784-2LV-c
set TARGET_BOARD xilinx.com:kv260_som:part0:1.4
set WRAPPER      [file normalize $HW_ROOT/vivado/npu_core_wrapper.sv]
set FILELIST     [file normalize $HW_ROOT/vivado/filelist.f]
set CONSTRAINTS  [file normalize $HW_ROOT/constraints/pccx_timing.xdc]
set DEFAULT_BD   [file normalize $HW_ROOT/vivado/system_bd.tcl]
set BD_SCRIPT    $DEFAULT_BD
if {[info exists ::env(PCCX_TOP_BD_TCL)] && $::env(PCCX_TOP_BD_TCL) ne ""} {
    set BD_SCRIPT [file normalize $::env(PCCX_TOP_BD_TCL)]
}

set STATUS_FILE [file normalize $REPORTS/top_level_bitstream_status.txt]
set board_part_status "MISSING"
if {[llength [get_board_parts -quiet $TARGET_BOARD]] > 0} {
    set board_part_status "AVAILABLE"
}

set wrapper_status "MISSING"
if {[file exists $WRAPPER]} {
    set wrapper_status "PRESENT"
}

set filelist_status "MISSING"
if {[file exists $FILELIST]} {
    set filelist_status "PRESENT"
}

set constraints_status "MISSING"
if {[file exists $CONSTRAINTS]} {
    set constraints_status "PRESENT"
}

set bd_status "MISSING"
if {[file exists $BD_SCRIPT]} {
    set bd_status "PRESENT"
}

set full_top_status "FULL_TOP_IMPL_NOT_RUN"
set bitstream_status "BITSTREAM_NOT_RUN_TOP_FLOW_INCOMPLETE"
set blocker "missing_full_top_level_block_design"
if {$bd_status eq "PRESENT"} {
    set full_top_status "FULL_TOP_FLOW_SCAFFOLDED"
    set bitstream_status "BITSTREAM_NOT_REQUESTED"
    set blocker "full_top_level_bitstream_not_launched_by_status_mode"
}

if {$MODE eq "bitstream"} {
    if {$bd_status ne "PRESENT"} {
        set full_top_status "FULL_TOP_IMPL_NOT_RUN"
        set bitstream_status "BITSTREAM_NOT_RUN_TOP_FLOW_INCOMPLETE"
        set blocker "missing $BD_SCRIPT"
    } else {
        set full_top_status "FULL_TOP_FLOW_SCAFFOLDED"
        set bitstream_status "BITSTREAM_BLOCKED_PENDING_RUN_SCRIPT"
        set blocker "system_bd.tcl exists but automated full-top implementation is not enabled in this guard script"
    }
}

set fp [open $STATUS_FILE w]
puts $fp "implementation_scope=FULL_TOP_LEVEL"
puts $fp "mode=$MODE"
puts $fp "target_part=$TARGET_PART"
puts $fp "target_board=$TARGET_BOARD"
puts $fp "board_part_status=$board_part_status"
puts $fp "wrapper_source=$WRAPPER"
puts $fp "wrapper_status=$wrapper_status"
puts $fp "filelist=$FILELIST"
puts $fp "filelist_status=$filelist_status"
puts $fp "constraints=$CONSTRAINTS"
puts $fp "constraints_status=$constraints_status"
puts $fp "bd_script=$BD_SCRIPT"
puts $fp "bd_status=$bd_status"
puts $fp "full_top_level_flow=$full_top_status"
puts $fp "bitstream_status=$bitstream_status"
puts $fp "blocker=$blocker"
puts $fp "required_next_step=author hw/vivado/system_bd.tcl or set PCCX_TOP_BD_TCL to a board-design Tcl that packages npu_core_wrapper, instantiates Zynq MPSoC, binds clocks/resets, connects AXI, generates the HDL wrapper, runs full implementation, and writes bitstream only after timing is clean."
close $fp

puts "\[pccx\] top-level status written to $STATUS_FILE"
puts "\[pccx\] full_top_level_flow = $full_top_status"
puts "\[pccx\] bitstream_status = $bitstream_status"

if {$MODE eq "bitstream"} {
    exit 3
}
