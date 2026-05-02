# =============================================================================
# synth.tcl — out-of-context synthesis pass for the pccx v002 NPU_top core.
#
# Usage
# -----
#   vivado -mode batch -source vivado/create_project.tcl \
#          -source vivado/synth.tcl
#
# What this does
# --------------
#   * open the project created by create_project.tcl
#   * run synth_design -mode out_of_context, leaving interface ports unbound
#   * emit resource, timing, and clocking reports into build/reports/
#   * does NOT run opt_design, place_design, route_design, or write_bitstream
#
# The out-of-context flag is important: NPU_top uses SV interface ports
# (axil_if / axis_if) which need a block-design wrapper to terminate. OOC
# synth lets us synthesise the core in isolation and catch RTL bugs early.
# =============================================================================

set HW_ROOT  [file normalize [file dirname [info script]]/..]
set PROJ_DIR [file normalize $HW_ROOT/build/pccx_v002_kv260]
set REPORTS  $HW_ROOT/build/reports
file mkdir $REPORTS

set SYNTH_JOBS 4
if {[info exists ::env(PCCX_VIVADO_JOBS)] && $::env(PCCX_VIVADO_JOBS) ne ""} {
    set SYNTH_JOBS $::env(PCCX_VIVADO_JOBS)
}
if {![string is integer -strict $SYNTH_JOBS] || $SYNTH_JOBS < 1} {
    puts "\[pccx\] invalid PCCX_VIVADO_JOBS=$SYNTH_JOBS; expected positive integer."
    exit 2
}
if {[catch {set_param general.maxThreads $SYNTH_JOBS} msg]} {
    puts "\[pccx\] warning: could not set general.maxThreads=$SYNTH_JOBS: $msg"
}
puts "\[pccx\] synth jobs/threads = $SYNTH_JOBS"

# Open the project only if create_project.tcl wasn't sourced first.
if {[llength [current_project -quiet]] == 0} {
    open_project $PROJ_DIR/pccx_v002_kv260.xpr
}
set_property top NPU_top [current_fileset]

# Remove any stale synth runs so the flow is idempotent.
foreach r [get_runs -quiet synth_1] {
    reset_run $r
}

# Out-of-context synthesis — needed for the interface-port top.
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
             -value {-mode out_of_context -flatten_hierarchy rebuilt} \
             -objects [get_runs synth_1]

# Synthesize.
launch_runs synth_1 -jobs $SYNTH_JOBS
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    puts "\[pccx\] synth_1 did not finish. See project log for root cause."
    exit 1
}

open_run synth_1 -name synth_1

# ----------------------------------------------------------------------------
# Post-synth reports
# ----------------------------------------------------------------------------
report_utilization -hierarchical -file $REPORTS/utilization_post_synth.rpt
report_clocks       -file                 $REPORTS/clocks_post_synth.rpt
report_clock_interaction -file            $REPORTS/clock_interaction_post_synth.rpt
report_timing_summary -delay_type min_max -report_unconstrained \
                      -check_timing_verbose -max_paths 10 \
                      -input_pins -routable_nets \
                      -file                $REPORTS/timing_summary_post_synth.rpt
report_drc          -file                 $REPORTS/drc_post_synth.rpt

puts "\[pccx\] synth complete. Reports in $REPORTS."
