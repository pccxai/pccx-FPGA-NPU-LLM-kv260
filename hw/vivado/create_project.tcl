# PCCX(TM) — reusable AI accelerator project.
# SPDX-FileCopyrightText: 2026 Hyun Woo Kim
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# create_project.tcl — builds the pccx v002 Vivado project for KV260.
#
# Usage
# -----
#   vivado -mode batch -source vivado/create_project.tcl
#
# The project is created at $PROJ_DIR (default: build/pccx_v002_kv260). Delete
# that directory between re-runs; the script will not force-overwrite.
#
# Target
# ------
#   KV260 SOM (xck26-sfvc784-2LV-c), Zynq UltraScale+ MPSoC ZU5EV.
# =============================================================================

set HW_ROOT             [file normalize [file dirname [info script]]/..]
set REPO_ROOT           [file normalize $HW_ROOT/..]
set PCCX_V002_RTL_ROOT  [file normalize $REPO_ROOT/third_party/pccx-v002]
set PROJ_DIR            [file normalize $HW_ROOT/build/pccx_v002_kv260]
set PROJ_NAME           pccx_v002_kv260
set TARGET_PART         xck26-sfvc784-2LV-c
set TARGET_BOARD        xilinx.com:kv260_som:part0:1.4
set TOP_MODULE          pccx_npu_top

puts "\[pccx\] HW_ROOT    = $HW_ROOT"
puts "\[pccx\] PROJ_DIR   = $PROJ_DIR"
puts "\[pccx\] part       = $TARGET_PART"
puts "\[pccx\] top        = $TOP_MODULE"

file mkdir [file dirname $PROJ_DIR]

# ----------------------------------------------------------------------------
# Project shell
# ----------------------------------------------------------------------------
create_project -force $PROJ_NAME $PROJ_DIR -part $TARGET_PART
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# The board file is optional; fall through silently if it is not installed.
if {[llength [get_board_parts -quiet $TARGET_BOARD]] > 0} {
    set_property board_part $TARGET_BOARD [current_project]
    puts "\[pccx\] board_part = $TARGET_BOARD"
} else {
    puts "\[pccx\] board_part $TARGET_BOARD not in repo — skipping."
}

# ----------------------------------------------------------------------------
# Source fileset — driven by hw/vivado/filelist.v002.f
# ----------------------------------------------------------------------------
set sv_files {}

proc append_filelist_entries {filelist_path base_dir sv_files_var} {
    upvar $sv_files_var sv_files
    set fp [open $filelist_path r]
    while {[gets $fp line] >= 0} {
        set line [string trim $line]
        if {$line eq "" || [string index $line 0] eq "#"} continue

        if {[regexp {^-f[ \t]+(.+)$} $line -> nested_rel]} {
            set nested_path [file normalize $base_dir/$nested_rel]
            set nested_base [file normalize [file dirname $nested_path]/../..]
            append_filelist_entries $nested_path $nested_base sv_files
        } else {
            lappend sv_files [file normalize $base_dir/$line]
        }
    }
    close $fp
}

set filelist_path $HW_ROOT/vivado/filelist.v002.f
append_filelist_entries $filelist_path $HW_ROOT sv_files

add_files -fileset sources_1 $sv_files
set_property file_type SystemVerilog [get_files -of [get_filesets sources_1]]
set_property top $TOP_MODULE [current_fileset]

# Vivado's file manager requires included SystemVerilog headers to be present
# in the project as Verilog Header files, not only discoverable via include_dirs.
proc collect_svh_files {root} {
    set files {}
    foreach entry [glob -nocomplain -directory $root *] {
        if {[file isdirectory $entry]} {
            set files [concat $files [collect_svh_files $entry]]
        } elseif {[file extension $entry] eq ".svh"} {
            lappend files [file normalize $entry]
        }
    }
    return $files
}

set svh_files [lsort [collect_svh_files $HW_ROOT/rtl]]
if {[llength $svh_files] > 0} {
    add_files -fileset sources_1 $svh_files
    foreach svh_file $svh_files {
        set_property file_type "Verilog Header" [get_files $svh_file]
    }
    puts "\[pccx\] registered [llength $svh_files] SystemVerilog headers"
}

# ----------------------------------------------------------------------------
# Header / include search paths
# ----------------------------------------------------------------------------
set include_dirs [list \
    $PCCX_V002_RTL_ROOT/common/rtl/packages/legacy \
    $PCCX_V002_RTL_ROOT/common/rtl/packages \
    $PCCX_V002_RTL_ROOT/common/rtl/interfaces \
    $PCCX_V002_RTL_ROOT/LLM/rtl/packages/isa \
    $PCCX_V002_RTL_ROOT/LLM/rtl/packages/controller \
    $PCCX_V002_RTL_ROOT/LLM/rtl/core/mat \
    $PCCX_V002_RTL_ROOT/LLM/rtl/core/vec \
    $PCCX_V002_RTL_ROOT/LLM/rtl/interfaces \
]
set_property include_dirs $include_dirs [get_filesets sources_1]
set_property include_dirs $include_dirs [get_filesets sim_1]

# ----------------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------------
set xdc_dir $HW_ROOT/constraints
if {[file exists $xdc_dir]} {
    foreach xdc [glob -nocomplain -directory $xdc_dir *.xdc] {
        add_files -fileset constrs_1 [file normalize $xdc]
    }
}

puts "\[pccx\] create_project done. Run synth.tcl next."
