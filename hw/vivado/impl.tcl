# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
# =============================================================================
# impl.tcl — routed implementation evidence for the pccx v002 NPU core.
#
# This is NOT auto-run by any other script. It exists as a deliberate
# gate — implementation + bitstream is an hour-scale job; launch it only
# after synth is clean and the design has a wrapper that binds the SV
# interface ports.
#
# Usage
# -----
#   vivado -mode batch -source vivado/impl.tcl -tclargs route
#   vivado -mode batch -source vivado/impl.tcl -tclargs bitstream
# =============================================================================

set HW_ROOT  [file normalize [file dirname [info script]]/..]
set PROJ_DIR [file normalize $HW_ROOT/build/pccx_v002_kv260]
set REPORTS  $HW_ROOT/build/reports
file mkdir $REPORTS

set IMPL_MODE "route"
if {[llength $argv] > 0 && [lindex $argv 0] ne ""} {
    set IMPL_MODE [lindex $argv 0]
}
if {$IMPL_MODE ne "route" && $IMPL_MODE ne "bitstream"} {
    puts "\[pccx\] invalid impl mode '$IMPL_MODE'; expected route or bitstream."
    exit 2
}

set IMPL_JOBS 4
if {[info exists ::env(PCCX_VIVADO_JOBS)] && $::env(PCCX_VIVADO_JOBS) ne ""} {
    set IMPL_JOBS $::env(PCCX_VIVADO_JOBS)
}
if {![string is integer -strict $IMPL_JOBS] || $IMPL_JOBS < 1} {
    puts "\[pccx\] invalid PCCX_VIVADO_JOBS=$IMPL_JOBS; expected positive integer."
    exit 2
}
if {[catch {set_param general.maxThreads $IMPL_JOBS} msg]} {
    puts "\[pccx\] warning: could not set general.maxThreads=$IMPL_JOBS: $msg"
}
puts "\[pccx\] impl jobs/threads = $IMPL_JOBS"
puts "\[pccx\] impl mode = $IMPL_MODE"

set IMPL_STRATEGY ""
if {[info exists ::env(PCCX_IMPL_STRATEGY)] && $::env(PCCX_IMPL_STRATEGY) ne ""} {
    set IMPL_STRATEGY $::env(PCCX_IMPL_STRATEGY)
}

open_project $PROJ_DIR/pccx_v002_kv260.xpr

# Confirm the synth run is clean before committing to impl.
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    puts "\[pccx\] synth_1 incomplete. Run synth.tcl first."
    exit 1
}

# The current v002 project intentionally synthesizes NPU_top out of context.
# Vivado rejects bitstream generation for OOC modules (DRC HDOOC-3). Keep the
# bitstream step explicit so implementation/timing evidence can still complete
# cleanly while bitstream remains blocked until a full top-level/BD flow exists.
if {$IMPL_MODE eq "bitstream"} {
    set status_file [file normalize $REPORTS/bitstream_status.txt]
    set fp [open $status_file w]
    puts $fp "bitstream_status=BITSTREAM_BLOCKED_OOC"
    puts $fp "reason=Vivado DRC HDOOC-3: bitstream generation is not allowed for out-of-context module implementations."
    puts $fp "required_next_step=add a full KV260 top-level or block-design wrapper flow, then run write_bitstream there."
    puts $fp "project=$PROJ_DIR/pccx_v002_kv260.xpr"
    close $fp
    puts "\[pccx\] bitstream blocked for OOC module flow. See $status_file"
    close_project
    exit 3
}

# Reset and kick off routed implementation.
foreach r [get_runs -quiet impl_1] {
    reset_run $r
}

if {$IMPL_STRATEGY ne ""} {
    if {[catch {set_property strategy $IMPL_STRATEGY [get_runs impl_1]} msg]} {
        puts "\[pccx\] invalid PCCX_IMPL_STRATEGY=$IMPL_STRATEGY: $msg"
        close_project
        exit 2
    }
}
puts "\[pccx\] impl strategy = [get_property strategy [get_runs impl_1]]"

launch_runs impl_1 -to_step route_design -jobs $IMPL_JOBS
wait_on_run impl_1

set impl_progress [get_property PROGRESS [get_runs impl_1]]
set impl_status   [get_property STATUS [get_runs impl_1]]
puts "\[pccx\] impl progress = $impl_progress"
puts "\[pccx\] impl status = $impl_status"
if {$impl_progress ne "100%" && [string first "route_design Complete" $impl_status] < 0} {
    puts "\[pccx\] impl_1 did not finish."
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
    set status_file [file normalize $REPORTS/bitstream_status.txt]
    set fp [open $status_file w]
    puts $fp "bitstream_status=BITSTREAM_NOT_REQUESTED"
    puts $fp "reason=impl mode stops at route_design for OOC timing evidence."
    puts $fp "required_next_step=run ./vivado/build.sh bitstream after a full top-level/BD bitstream flow exists."
    close $fp
    puts "\[pccx\] bitstream not requested in route mode. See $status_file"
}

close_project
