# =============================================================================
# impl.tcl — implementation + bitstream for the pccx v002 NPU core.
#
# This is NOT auto-run by any other script. It exists as a deliberate
# gate — implementation + bitstream is an hour-scale job; launch it only
# after synth is clean and the design has a wrapper that binds the SV
# interface ports.
#
# Usage
# -----
#   vivado -mode batch -source vivado/impl.tcl
# =============================================================================

set HW_ROOT  [file normalize [file dirname [info script]]/..]
set PROJ_DIR [file normalize $HW_ROOT/build/pccx_v002_kv260]
set REPORTS  $HW_ROOT/build/reports
file mkdir $REPORTS

open_project $PROJ_DIR/pccx_v002_kv260.xpr

# Confirm the synth run is clean before committing to impl.
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    puts "\[pccx\] synth_1 incomplete. Run synth.tcl first."
    exit 1
}

# Reset and kick off implementation.
foreach r [get_runs -quiet impl_1] {
    reset_run $r
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    puts "\[pccx\] impl_1 did not finish (progress=[get_property PROGRESS [get_runs impl_1]])."
    exit 1
}

open_run impl_1

report_utilization -hierarchical -file $REPORTS/utilization_post_impl.rpt
report_clock_interaction -file $REPORTS/clock_interaction_post_impl.rpt
report_timing_summary -delay_type min_max -report_unconstrained \
                      -check_timing_verbose -max_paths 20 \
                      -file                $REPORTS/timing_summary_post_impl.rpt
report_drc            -file                $REPORTS/drc_post_impl.rpt
report_power          -file                $REPORTS/power_post_impl.rpt

set bit_file [file normalize $PROJ_DIR/pccx_v002_kv260.runs/impl_1/NPU_top.bit]
if {[file exists $bit_file]} {
    file copy -force $bit_file $HW_ROOT/build/pccx_v002_kv260.bit
    puts "\[pccx\] bitstream copied to build/pccx_v002_kv260.bit"
} else {
    puts "\[pccx\] warning: expected bitstream not found at $bit_file"
}

close_project
