# =============================================================================
# record_synth_baseline.tcl -- emit a machine-readable post-synth baseline.
#
# Source this after synth_1 has completed and the synthesized design is open.
# The normal ./vivado/build.sh synth flow sources this automatically after
# vivado/synth.tcl.
# =============================================================================

set SCRIPT_DIR [file normalize [file dirname [info script]]]
set HW_ROOT    [file normalize $SCRIPT_DIR/../..]
set REPO_ROOT  [file normalize $HW_ROOT/..]
set REPORTS    [file normalize $HW_ROOT/build/reports]
set JSON_PATH  [file normalize $REPORTS/synth_baseline.json]

if {[llength [info commands current_design]] == 0} {
    puts "\[pccx\] record_synth_baseline.tcl requires Vivado Tcl after synthesis; skipping outside Vivado."
    return
}

proc pccx_read_file {path} {
    if {![file exists $path]} {
        return ""
    }
    set fp [open $path r]
    set text [read $fp]
    close $fp
    return $text
}

proc pccx_write_file {path text} {
    file mkdir [file dirname $path]
    set fp [open $path w]
    puts -nonewline $fp $text
    close $fp
}

proc pccx_relpath {base path} {
    set base_norm [file normalize $base]
    set path_norm [file normalize $path]
    set prefix "$base_norm/"
    if {[string first $prefix $path_norm] == 0} {
        return [string range $path_norm [string length $prefix] end]
    }
    return $path_norm
}

proc pccx_optional_exec {args} {
    if {[catch {exec {*}$args} value]} {
        return ""
    }
    return [string trim $value]
}

proc pccx_get_prop {prop obj} {
    if {$obj eq ""} {
        return ""
    }
    if {[catch {get_property $prop $obj} value]} {
        return ""
    }
    return [string trim $value]
}

proc pccx_json_quote {value} {
    set mapped [string map [list "\\" "\\\\" "\"" "\\\"" "\b" "\\b" "\f" "\\f" "\n" "\\n" "\r" "\\r" "\t" "\\t"] $value]
    return "\"$mapped\""
}

proc pccx_json_number_or_null {value} {
    set value [string trim $value]
    if {$value eq "" || [string equal -nocase $value "NA"] || [string equal -nocase $value "N/A"] || $value eq "--"} {
        return "null"
    }
    if {[string is double -strict $value] || [string is integer -strict $value]} {
        return $value
    }
    return [pccx_json_quote $value]
}

proc pccx_json_string_array {items} {
    set out "\["
    set first 1
    foreach item $items {
        if {!$first} {
            append out ", "
        }
        append out [pccx_json_quote $item]
        set first 0
    }
    append out "\]"
    return $out
}

proc pccx_ensure_synth_design {} {
    if {[catch {current_design} design] == 0 && [string trim $design] ne ""} {
        return [string trim $design]
    }

    set runs {}
    if {[llength [info commands get_runs]] > 0} {
        catch {set runs [get_runs -quiet synth_1]}
    }
    if {[llength $runs] > 0} {
        catch {open_run synth_1 -name synth_1}
    }

    if {[catch {current_design} design] == 0 && [string trim $design] ne ""} {
        return [string trim $design]
    }
    error "\[pccx\] no synthesized design is open; run synth_1 before sourcing record_synth_baseline.tcl"
}

proc pccx_report_text {path report_cmd} {
    set text [pccx_read_file $path]
    if {$text ne ""} {
        return $text
    }

    set generated ""
    if {![catch {{*}$report_cmd -return_string} generated] && $generated ne ""} {
        pccx_write_file $path $generated
        return $generated
    }

    catch {{*}$report_cmd -file $path}
    return [pccx_read_file $path]
}

proc pccx_parse_timing_summary {text} {
    array set timing {
        wns_ns ""
        tns_ns ""
        whs_ns ""
        ths_ns ""
    }

    set seen_header 0
    foreach line [split $text "\n"] {
        if {[regexp {WNS\(ns\).*TNS\(ns\).*WHS\(ns\).*THS\(ns\)} $line]} {
            set seen_header 1
            continue
        }
        if {!$seen_header} {
            continue
        }

        set fields [regexp -all -inline {[-+]?[0-9]+[.][0-9]*|[-+]?[.][0-9]+|[-+]?[0-9]+|N/A|NA|--} $line]
        if {[llength $fields] >= 6 && [lindex $fields 0] ne "--"} {
            set timing(wns_ns) [lindex $fields 0]
            set timing(tns_ns) [lindex $fields 1]
            set timing(whs_ns) [lindex $fields 4]
            set timing(ths_ns) [lindex $fields 5]
            break
        }
    }

    return [array get timing]
}

proc pccx_cell_utilization {} {
    array set counts {
        dsp 0
        bram36 0
        bram18 0
        bram_36k_equiv 0
        uram 0
        lut 0
        ff 0
    }

    set cells {}
    catch {set cells [get_cells -hier -quiet]}
    foreach cell $cells {
        set ref [pccx_get_prop REF_NAME $cell]
        if {[string match "DSP48*" $ref]} {
            incr counts(dsp)
        } elseif {[string match "RAMB36*" $ref]} {
            incr counts(bram36)
        } elseif {[string match "RAMB18*" $ref]} {
            incr counts(bram18)
        } elseif {[string match "URAM*" $ref]} {
            incr counts(uram)
        } elseif {[regexp {^LUT[1-6](_2)?$} $ref]} {
            incr counts(lut)
        } elseif {[regexp {^FD[A-Z0-9_]*$} $ref]} {
            incr counts(ff)
        }
    }

    set counts(bram_36k_equiv) [expr {$counts(bram36) + ($counts(bram18) * 0.5)}]
    return [array get counts]
}

proc pccx_parse_utilization_report {text} {
    array set util {
        dsp ""
        bram_36k_equiv ""
        uram ""
        lut ""
        ff ""
    }

    foreach line [split $text "\n"] {
        if {![regexp {^\|\s*([^|]+?)\s*\|\s*([0-9,.]+)\s*\|} $line -> raw_name used]} {
            continue
        }
        set name [string trim $raw_name]
        set used [string map {"," ""} $used]
        switch -exact -- $name {
            "Slice LUTs" -
            "CLB LUTs" {
                if {$util(lut) eq ""} { set util(lut) $used }
            }
            "Slice Registers" -
            "CLB Registers" {
                if {$util(ff) eq ""} { set util(ff) $used }
            }
            "DSPs" {
                if {$util(dsp) eq ""} { set util(dsp) $used }
            }
            "Block RAM Tile" {
                if {$util(bram_36k_equiv) eq ""} { set util(bram_36k_equiv) $used }
            }
            "URAM" {
                if {$util(uram) eq ""} { set util(uram) $used }
            }
        }
    }

    return [array get util]
}

proc pccx_collect_timing_paths {delay_type max_paths} {
    set paths {}
    if {[catch {set paths [get_timing_paths -delay_type $delay_type -max_paths $max_paths -nworst 1 -quiet]}]} {
        return {}
    }

    set rows {}
    set rank 1
    foreach path $paths {
        set source [pccx_get_prop STARTPOINT_PIN $path]
        if {$source eq ""} {
            set source [pccx_get_prop STARTPOINT $path]
        }
        set destination [pccx_get_prop ENDPOINT_PIN $path]
        if {$destination eq ""} {
            set destination [pccx_get_prop ENDPOINT $path]
        }
        set group [pccx_get_prop PATH_GROUP $path]
        if {$group eq ""} {
            set group [pccx_get_prop GROUP $path]
        }

        lappend rows [dict create \
            rank $rank \
            delay_type $delay_type \
            slack_ns [pccx_get_prop SLACK $path] \
            requirement_ns [pccx_get_prop REQUIREMENT $path] \
            data_path_delay_ns [pccx_get_prop DATAPATH_DELAY $path] \
            logic_levels [pccx_get_prop LOGIC_LEVELS $path] \
            path_group $group \
            source $source \
            destination $destination \
            source_clock [pccx_get_prop STARTPOINT_CLOCK $path] \
            destination_clock [pccx_get_prop ENDPOINT_CLOCK $path]]
        incr rank
    }
    return $rows
}

proc pccx_filelist_paths {hw_root} {
    set filelist [file normalize $hw_root/vivado/filelist.f]
    set paths {}
    set fp [open $filelist r]
    while {[gets $fp line] >= 0} {
        set line [string trim $line]
        if {$line eq "" || [string index $line 0] eq "#"} {
            continue
        }
        lappend paths [file normalize $hw_root/$line]
    }
    close $fp
    return $paths
}

proc pccx_package_name_from_file {path} {
    foreach line [split [pccx_read_file $path] "\n"] {
        if {[regexp {^\s*package\s+([A-Za-z_][A-Za-z0-9_$]*)} $line -> name]} {
            return $name
        }
    }
    return [file rootname [file tail $path]]
}

proc pccx_count_pkg_parameters {path} {
    set total 0
    foreach line [split [pccx_read_file $path] "\n"] {
        set line [regsub {//.*$} $line ""]
        if {[regexp {^\s*(localparam|parameter)\y} $line]} {
            incr total
        }
    }
    return $total
}

proc pccx_modules_referencing_pkg {source_files pkg_name} {
    set modules {}
    foreach path $source_files {
        if {![file exists $path] || [file extension $path] ne ".sv"} {
            continue
        }
        set text [pccx_read_file $path]
        set scoped_re [format {(^|[^A-Za-z0-9_])%s::} $pkg_name]
        set import_re [format {import[ \t\r\n]+%s::\*} $pkg_name]
        set references_pkg [expr {[regexp $scoped_re $text] || [regexp $import_re $text]}]
        if {!$references_pkg} {
            continue
        }
        foreach line [split $text "\n"] {
            if {[regexp {^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)} $line -> module_name]} {
                lappend modules $module_name
            }
        }
    }
    return [lsort -unique $modules]
}

proc pccx_count_module_instances {module_name top_name} {
    set total 0
    set cells {}
    catch {set cells [get_cells -hier -quiet]}
    foreach cell $cells {
        set ref [pccx_get_prop REF_NAME $cell]
        set orig_ref [pccx_get_prop ORIG_REF_NAME $cell]
        if {$ref eq $module_name || $orig_ref eq $module_name} {
            incr total
        }
    }
    if {$module_name eq $top_name} {
        incr total
    }
    return $total
}

proc pccx_collect_package_groups {hw_root repo_root top_name} {
    set source_files [pccx_filelist_paths $hw_root]
    set groups {}
    foreach path $source_files {
        if {![string match "*_pkg.sv" [file tail $path]]} {
            continue
        }
        set pkg_name [pccx_package_name_from_file $path]
        set modules [pccx_modules_referencing_pkg $source_files $pkg_name]
        set instance_total 0
        foreach module_name $modules {
            incr instance_total [pccx_count_module_instances $module_name $top_name]
        }
        lappend groups [dict create \
            name $pkg_name \
            path [pccx_relpath $repo_root $path] \
            parameter_total [pccx_count_pkg_parameters $path] \
            instance_total $instance_total \
            referencing_modules $modules \
            referencing_instance_total $instance_total]
    }
    return [lsort -dictionary -index 1 $groups]
}

proc pccx_json_path_row {row indent is_last} {
    set pad [string repeat " " $indent]
    set pad2 [string repeat " " [expr {$indent + 2}]]
    set suffix [expr {$is_last ? "" : ","}]
    set text "$pad\{\n"
    append text "$pad2\"rank\": [dict get $row rank],\n"
    append text "$pad2\"delay_type\": [pccx_json_quote [dict get $row delay_type]],\n"
    append text "$pad2\"slack_ns\": [pccx_json_number_or_null [dict get $row slack_ns]],\n"
    append text "$pad2\"requirement_ns\": [pccx_json_number_or_null [dict get $row requirement_ns]],\n"
    append text "$pad2\"data_path_delay_ns\": [pccx_json_number_or_null [dict get $row data_path_delay_ns]],\n"
    append text "$pad2\"logic_levels\": [pccx_json_number_or_null [dict get $row logic_levels]],\n"
    append text "$pad2\"path_group\": [pccx_json_quote [dict get $row path_group]],\n"
    append text "$pad2\"source\": [pccx_json_quote [dict get $row source]],\n"
    append text "$pad2\"destination\": [pccx_json_quote [dict get $row destination]],\n"
    append text "$pad2\"source_clock\": [pccx_json_quote [dict get $row source_clock]],\n"
    append text "$pad2\"destination_clock\": [pccx_json_quote [dict get $row destination_clock]]\n"
    append text "$pad\}$suffix\n"
    return $text
}

proc pccx_json_package_group {group indent is_last} {
    set pad [string repeat " " $indent]
    set pad2 [string repeat " " [expr {$indent + 2}]]
    set suffix [expr {$is_last ? "" : ","}]
    set text "$pad[pccx_json_quote [dict get $group name]]: \{\n"
    append text "$pad2\"path\": [pccx_json_quote [dict get $group path]],\n"
    append text "$pad2\"parameter_total\": [dict get $group parameter_total],\n"
    append text "$pad2\"instance_total\": [dict get $group instance_total],\n"
    append text "$pad2\"referencing_modules\": [pccx_json_string_array [dict get $group referencing_modules]],\n"
    append text "$pad2\"referencing_instance_total\": [dict get $group referencing_instance_total]\n"
    append text "$pad\}$suffix\n"
    return $text
}

set design_name [pccx_ensure_synth_design]
file mkdir $REPORTS

set timing_report_path [file normalize $REPORTS/timing_summary_post_synth.rpt]
set util_report_path   [file normalize $REPORTS/utilization_post_synth.rpt]

set timing_text [pccx_report_text $timing_report_path [list report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets]]
array set timing [pccx_parse_timing_summary $timing_text]

set max_paths [pccx_collect_timing_paths max 5]
set min_paths [pccx_collect_timing_paths min 1]
if {$timing(wns_ns) eq "" && [llength $max_paths] > 0} {
    set timing(wns_ns) [dict get [lindex $max_paths 0] slack_ns]
}
if {$timing(whs_ns) eq "" && [llength $min_paths] > 0} {
    set timing(whs_ns) [dict get [lindex $min_paths 0] slack_ns]
}
set critical_paths [concat $max_paths $min_paths]

array set util [pccx_cell_utilization]
set util_text [pccx_report_text $util_report_path [list report_utilization -hierarchical]]
array set util_report [pccx_parse_utilization_report $util_text]
foreach key {dsp bram_36k_equiv uram lut ff} {
    if {$util_report($key) ne ""} {
        set util($key) $util_report($key)
    }
}

set top_name [pccx_get_prop NAME [current_design]]
if {$top_name eq ""} {
    set top_name $design_name
}
set package_groups [pccx_collect_package_groups $HW_ROOT $REPO_ROOT $top_name]

set vivado_version ""
catch {set vivado_version [string trim [version -short]]}
set part ""
catch {set part [get_property PART [current_project]]}
set board_part ""
catch {set board_part [get_property BOARD_PART [current_project]]}

set commit [pccx_optional_exec git -C $REPO_ROOT rev-parse HEAD]
set branch [pccx_optional_exec git -C $REPO_ROOT rev-parse --abbrev-ref HEAD]

set json "\{\n"
append json "  \"schema\": \"pccx.synth_baseline.v1\",\n"
append json "  \"generated_at_utc\": [pccx_json_quote [clock format [clock seconds] -gmt true -format {%Y-%m-%dT%H:%M:%SZ}]],\n"
append json "  \"claim_guard\": \"report_capture_only_no_closure_claim\",\n"
append json "  \"design\": \{\n"
append json "    \"top\": [pccx_json_quote $top_name],\n"
append json "    \"part\": [pccx_json_quote $part],\n"
append json "    \"board_part\": [pccx_json_quote $board_part],\n"
append json "    \"vivado_version\": [pccx_json_quote $vivado_version],\n"
append json "    \"git_commit\": [pccx_json_quote $commit],\n"
append json "    \"git_branch\": [pccx_json_quote $branch]\n"
append json "  \},\n"
append json "  \"source_reports\": \{\n"
append json "    \"timing_summary_post_synth\": [pccx_json_quote [pccx_relpath $REPO_ROOT $timing_report_path]],\n"
append json "    \"utilization_post_synth\": [pccx_json_quote [pccx_relpath $REPO_ROOT $util_report_path]]\n"
append json "  \},\n"
append json "  \"timing\": \{\n"
append json "    \"wns_ns\": [pccx_json_number_or_null $timing(wns_ns)],\n"
append json "    \"tns_ns\": [pccx_json_number_or_null $timing(tns_ns)],\n"
append json "    \"whs_ns\": [pccx_json_number_or_null $timing(whs_ns)],\n"
append json "    \"ths_ns\": [pccx_json_number_or_null $timing(ths_ns)]\n"
append json "  \},\n"
append json "  \"utilization\": \{\n"
append json "    \"dsp\": [pccx_json_number_or_null $util(dsp)],\n"
append json "    \"bram_36k_equiv\": [pccx_json_number_or_null $util(bram_36k_equiv)],\n"
append json "    \"uram\": [pccx_json_number_or_null $util(uram)],\n"
append json "    \"lut\": [pccx_json_number_or_null $util(lut)],\n"
append json "    \"ff\": [pccx_json_number_or_null $util(ff)]\n"
append json "  \},\n"
append json "  \"critical_paths\": \[\n"
for {set i 0} {$i < [llength $critical_paths]} {incr i} {
    append json [pccx_json_path_row [lindex $critical_paths $i] 4 [expr {$i == ([llength $critical_paths] - 1)}]]
}
append json "  \],\n"
append json "  \"package_parameter_groups\": \{\n"
for {set i 0} {$i < [llength $package_groups]} {incr i} {
    append json [pccx_json_package_group [lindex $package_groups $i] 4 [expr {$i == ([llength $package_groups] - 1)}]]
}
append json "  \}\n"
append json "\}\n"

pccx_write_file $JSON_PATH $json
puts "\[pccx\] synth baseline JSON written to $JSON_PATH"
