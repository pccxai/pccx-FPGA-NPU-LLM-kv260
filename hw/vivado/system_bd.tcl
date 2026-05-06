# =============================================================================
# system_bd.tcl -- KV260 full-top block-design scaffold for pccx v002.1.
#
# This is a pre-flight build scaffold, not a timing or board-readiness claim.
# It creates a Zynq UltraScale+ MPSoC BD around the existing plain-signal
# npu_core_wrapper module, which instantiates the public RTL top NPU_top.
#
# Usage from hw/:
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs prepare
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs synth
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs impl
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs bitstream
# =============================================================================

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set HW_ROOT    [file normalize [file join $SCRIPT_DIR ..]]
set BUILD_DIR  [file normalize [file join $HW_ROOT build]]
set PROJ_DIR   [file normalize [file join $BUILD_DIR pccx_v002_kv260_bd]]
set PROJ_NAME  pccx_v002_kv260_bd
set BD_NAME    system
set REPORTS    [file normalize [file join $BUILD_DIR reports]]
set DCP_DIR    [file normalize [file join $BUILD_DIR checkpoints]]

set TARGET_PART  xczu5ev-sfvc784-1-i
set TARGET_BOARD xilinx.com:kv260_som:part0:1.4

set TOP_RTL_MODULE     NPU_top
set NPU_BD_MODULE_REF  npu_core_wrapper

set CLK_AXI_MHZ 200.0
set CLK_NPU_MHZ 250.0
set AXIL_BASE   0xA0000000

set ACTION prepare
if {[llength $argv] > 0} {
    set ACTION [lindex $argv 0]
}
if {$ACTION ni {clean prepare synth impl bitstream}} {
    puts "error: unsupported action '$ACTION'"
    puts "usage: system_bd.tcl {clean|prepare|synth|impl|bitstream}"
    exit 2
}

proc pccx_msg {msg} {
    puts "\[pccx\] $msg"
}

set VIVADO_JOBS 4
if {[info exists ::env(PCCX_VIVADO_JOBS)] && $::env(PCCX_VIVADO_JOBS) ne ""} {
    set VIVADO_JOBS $::env(PCCX_VIVADO_JOBS)
}
if {![string is integer -strict $VIVADO_JOBS] || $VIVADO_JOBS < 1} {
    puts "error: PCCX_VIVADO_JOBS must be a positive integer"
    exit 2
}
if {[catch {set_param general.maxThreads $VIVADO_JOBS} msg]} {
    pccx_msg "warning: could not set general.maxThreads=$VIVADO_JOBS: $msg"
}
pccx_msg "Vivado jobs/threads = $VIVADO_JOBS"

proc pccx_filelist {} {
    global HW_ROOT
    set filelist_path [file join $HW_ROOT vivado filelist.f]
    set fp [open $filelist_path r]
    set sv_files {}
    while {[gets $fp line] >= 0} {
        set line [string trim $line]
        if {$line eq "" || [string index $line 0] eq "#"} {
            continue
        }
        lappend sv_files [file normalize [file join $HW_ROOT $line]]
    }
    close $fp
    return $sv_files
}

proc pccx_include_dirs {} {
    global HW_ROOT
    return [list \
        [file join $HW_ROOT rtl] \
        [file join $HW_ROOT rtl Constants compilePriority_Order A_const_svh] \
        [file join $HW_ROOT rtl MAT_CORE] \
        [file join $HW_ROOT rtl MEM_control IO] \
        [file join $HW_ROOT rtl NPU_Controller] \
        [file join $HW_ROOT rtl NPU_Controller NPU_Control_Unit] \
        [file join $HW_ROOT rtl NPU_Controller NPU_Control_Unit ISA_PACKAGE] \
        [file join $HW_ROOT rtl VEC_CORE] \
    ]
}

proc pccx_set_props {obj props} {
    foreach {key value} $props {
        if {[catch {set_property $key $value $obj} msg]} {
            pccx_msg "warning: could not set $key on $obj: $msg"
        }
    }
}

proc pccx_bd_intf {cell candidates purpose} {
    foreach name $candidates {
        set pins [get_bd_intf_pins -quiet "$cell/$name"]
        if {[llength $pins] > 0} {
            return [lindex $pins 0]
        }
    }
    set available [get_bd_intf_pins -quiet -of_objects [get_bd_cells $cell]]
    error "missing $purpose interface on $cell. Tried: $candidates. Available: $available"
}

proc pccx_bd_pin {path purpose} {
    set pins [get_bd_pins -quiet $path]
    if {[llength $pins] == 0} {
        error "missing $purpose pin: $path"
    }
    return [lindex $pins 0]
}

proc pccx_connect_pin_if_present {src dst} {
    set sp [get_bd_pins -quiet $src]
    set dp [get_bd_pins -quiet $dst]
    if {[llength $sp] > 0 && [llength $dp] > 0} {
        connect_bd_net [lindex $sp 0] [lindex $dp 0]
    }
}

proc pccx_create_project_shell {} {
    global PROJ_NAME PROJ_DIR TARGET_PART TARGET_BOARD

    file mkdir [file dirname $PROJ_DIR]
    create_project -force $PROJ_NAME $PROJ_DIR -part $TARGET_PART
    set_property target_language Verilog [current_project]
    set_property simulator_language Mixed [current_project]

    if {[llength [get_board_parts -quiet $TARGET_BOARD]] > 0} {
        set_property board_part $TARGET_BOARD [current_project]
        pccx_msg "board_part = $TARGET_BOARD"
    } else {
        pccx_msg "board_part $TARGET_BOARD not installed; continuing with part $TARGET_PART"
    }
}

proc pccx_add_sources {} {
    global HW_ROOT

    set sv_files [pccx_filelist]
    add_files -fileset sources_1 $sv_files
    set_property file_type SystemVerilog [get_files -of [get_filesets sources_1]]
    set_property include_dirs [pccx_include_dirs] [get_filesets sources_1]
    set_property include_dirs [pccx_include_dirs] [get_filesets sim_1]

    set xdc_dir [file join $HW_ROOT constraints]
    foreach xdc [glob -nocomplain -directory $xdc_dir *.xdc] {
        add_files -fileset constrs_1 [file normalize $xdc]
    }

    update_compile_order -fileset sources_1
}

proc pccx_dma {name mode} {
    set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 $name]
    set props [list \
        CONFIG.c_include_sg 0 \
        CONFIG.c_addr_width 40 \
        CONFIG.c_m_axis_mm2s_tdata_width 128 \
        CONFIG.c_s_axis_s2mm_tdata_width 128 \
    ]
    if {$mode eq "mm2s"} {
        lappend props CONFIG.c_include_mm2s 1 CONFIG.c_include_s2mm 0
    } elseif {$mode eq "s2mm"} {
        lappend props CONFIG.c_include_mm2s 0 CONFIG.c_include_s2mm 1
    } elseif {$mode eq "both"} {
        lappend props CONFIG.c_include_mm2s 1 CONFIG.c_include_s2mm 1
    } else {
        error "bad DMA mode $mode"
    }
    pccx_set_props $dma $props
    return $dma
}

proc pccx_connect_dma_clocking {dma clk rstn} {
    foreach pin {s_axi_lite_aclk m_axi_mm2s_aclk m_axis_mm2s_aclk m_axi_s2mm_aclk s_axis_s2mm_aclk} {
        pccx_connect_pin_if_present $clk "$dma/$pin"
    }
    pccx_connect_pin_if_present $rstn "$dma/axi_resetn"
}

proc pccx_prepare_bd {} {
    global BD_NAME TOP_RTL_MODULE NPU_BD_MODULE_REF CLK_AXI_MHZ CLK_NPU_MHZ AXIL_BASE REPORTS

    set existing [get_bd_designs -quiet $BD_NAME]
    if {[llength $existing] > 0} {
        current_bd_design $BD_NAME
        return
    }

    create_bd_design $BD_NAME
    current_bd_design $BD_NAME

    set axi_hz [expr {int($CLK_AXI_MHZ * 1000000.0)}]
    set npu_hz [expr {int($CLK_NPU_MHZ * 1000000.0)}]

    set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 ps_0]
    if {[catch {
        apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
            -config {apply_board_preset "1"} $ps
    } msg]} {
        pccx_msg "warning: board automation skipped: $msg"
    }
    pccx_set_props $ps [list \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $CLK_AXI_MHZ \
        CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ $CLK_NPU_MHZ \
        CONFIG.PSU__USE__M_AXI_GP0 1 \
        CONFIG.PSU__USE__S_AXI_GP0 1 \
        CONFIG.PSU__USE__S_AXI_GP1 1 \
        CONFIG.PSU__USE__S_AXI_GP2 1 \
        CONFIG.PSU__USE__S_AXI_ACP 1 \
    ]

    set npu [create_bd_cell -type module -reference $NPU_BD_MODULE_REF npu_0]

    set rst_axi [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_axi]
    set rst_npu [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_npu]
    pccx_set_props $rst_axi [list CONFIG.C_EXT_RESET_HIGH 0]
    pccx_set_props $rst_npu [list CONFIG.C_EXT_RESET_HIGH 0]

    set clear_const [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 soft_clear_zero]
    pccx_set_props $clear_const [list CONFIG.CONST_WIDTH 1 CONFIG.CONST_VAL 0]

    set ctrl_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_ctrl_interconnect]
    set hp_ic   [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_hp_interconnect]
    set acp_ic  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_acp_interconnect]
    pccx_set_props $ctrl_ic [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 6]
    pccx_set_props $hp_ic   [list CONFIG.NUM_SI 4 CONFIG.NUM_MI 2]
    pccx_set_props $acp_ic  [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1]

    set dma_hp0 [pccx_dma dma_hp0_weight_mm2s mm2s]
    set dma_hp1 [pccx_dma dma_hp1_weight_mm2s mm2s]
    set dma_hp2 [pccx_dma dma_hp2_weight_mm2s mm2s]
    set dma_hp3 [pccx_dma dma_hp3_weight_mm2s mm2s]
    set dma_acp [pccx_dma dma_acp_fmap_result both]

    connect_bd_net [pccx_bd_pin "$ps/pl_clk0" "PS PL0 clock"] \
                   [pccx_bd_pin "$npu/clk_axi" "NPU AXI clock"]
    connect_bd_net [pccx_bd_pin "$ps/pl_clk0" "PS PL0 clock"] \
                   [pccx_bd_pin "$rst_axi/slowest_sync_clk" "AXI reset clock"]
    connect_bd_net [pccx_bd_pin "$ps/pl_clk1" "PS PL1 clock"] \
                   [pccx_bd_pin "$npu/clk_core" "NPU core clock"]
    connect_bd_net [pccx_bd_pin "$ps/pl_clk1" "PS PL1 clock"] \
                   [pccx_bd_pin "$rst_npu/slowest_sync_clk" "NPU reset clock"]
    connect_bd_net [pccx_bd_pin "$ps/pl_resetn0" "PS PL reset"] \
                   [pccx_bd_pin "$rst_axi/ext_reset_in" "AXI external reset"]
    connect_bd_net [pccx_bd_pin "$ps/pl_resetn0" "PS PL reset"] \
                   [pccx_bd_pin "$rst_npu/ext_reset_in" "NPU external reset"]
    connect_bd_net [pccx_bd_pin "$rst_axi/peripheral_aresetn" "AXI resetn"] \
                   [pccx_bd_pin "$npu/rst_axi_n" "NPU AXI resetn"]
    connect_bd_net [pccx_bd_pin "$rst_npu/peripheral_aresetn" "NPU resetn"] \
                   [pccx_bd_pin "$npu/rst_n_core" "NPU core resetn"]
    connect_bd_net [pccx_bd_pin "$clear_const/dout" "clear constant"] \
                   [pccx_bd_pin "$npu/i_clear" "NPU clear"]

    foreach cell [list $ctrl_ic $hp_ic $acp_ic] {
        pccx_connect_pin_if_present "$ps/pl_clk0" "$cell/aclk"
        pccx_connect_pin_if_present "$rst_axi/peripheral_aresetn" "$cell/aresetn"
    }
    foreach dma [list $dma_hp0 $dma_hp1 $dma_hp2 $dma_hp3 $dma_acp] {
        pccx_connect_dma_clocking $dma "$ps/pl_clk0" "$rst_axi/peripheral_aresetn"
    }

    set ps_hpm0 [pccx_bd_intf $ps {M_AXI_HPM0_FPD M_AXI_HPM0_LPD M_AXI_GP0 M_AXI_HPM0} "PS AXI-Lite master"]
    set ps_hp0  [pccx_bd_intf $ps {S_AXI_HP0_FPD S_AXI_HP0 S_AXI_HP0_FPD} "PS HP0 slave"]
    set ps_hp1  [pccx_bd_intf $ps {S_AXI_HP1_FPD S_AXI_HP1 S_AXI_HP1_FPD} "PS HP1 slave"]
    set ps_acp  [pccx_bd_intf $ps {S_AXI_ACP_FPD S_AXI_HPC0_FPD S_AXI_HPC1_FPD S_AXI_ACP} "PS coherent ACP/HPC slave"]

    set npu_axil [pccx_bd_intf $npu {S_AXIL_CTRL S_AXIL s_axil S_AXI_CTRL S_AXI} "NPU AXI-Lite slave"]
    set npu_hp0  [pccx_bd_intf $npu {S_AXI_HP0_WEIGHT S_AXIS_HP0_WEIGHT S_AXIS_HP0 s_axis_hp0} "NPU HP0 stream"]
    set npu_hp1  [pccx_bd_intf $npu {S_AXI_HP1_WEIGHT S_AXIS_HP1_WEIGHT S_AXIS_HP1 s_axis_hp1} "NPU HP1 stream"]
    set npu_hp2  [pccx_bd_intf $npu {S_AXI_HP2_WEIGHT S_AXIS_HP2_WEIGHT S_AXIS_HP2 s_axis_hp2} "NPU HP2 stream"]
    set npu_hp3  [pccx_bd_intf $npu {S_AXI_HP3_WEIGHT S_AXIS_HP3_WEIGHT S_AXIS_HP3 s_axis_hp3} "NPU HP3 stream"]
    set npu_fmap [pccx_bd_intf $npu {S_AXIS_ACP_FMAP S_AXI_ACP_FMAP s_axis_acp_fmap} "NPU ACP fmap stream"]
    set npu_res  [pccx_bd_intf $npu {M_AXIS_ACP_RESULT M_AXI_ACP_RESULT m_axis_acp_result} "NPU ACP result stream"]

    # S_AXIL_CTRL reaches AXIL_CMD_IN and AXIL_STAT_OUT inside NPU_top.
    connect_bd_intf_net $ps_hpm0 [pccx_bd_intf $ctrl_ic {S00_AXI} "control interconnect S00"]
    connect_bd_intf_net [pccx_bd_intf $ctrl_ic {M00_AXI} "control interconnect M00"] $npu_axil
    connect_bd_intf_net [pccx_bd_intf $ctrl_ic {M01_AXI} "control interconnect M01"] [pccx_bd_intf $dma_hp0 {S_AXI_LITE} "HP0 DMA control"]
    connect_bd_intf_net [pccx_bd_intf $ctrl_ic {M02_AXI} "control interconnect M02"] [pccx_bd_intf $dma_hp1 {S_AXI_LITE} "HP1 DMA control"]
    connect_bd_intf_net [pccx_bd_intf $ctrl_ic {M03_AXI} "control interconnect M03"] [pccx_bd_intf $dma_hp2 {S_AXI_LITE} "HP2 DMA control"]
    connect_bd_intf_net [pccx_bd_intf $ctrl_ic {M04_AXI} "control interconnect M04"] [pccx_bd_intf $dma_hp3 {S_AXI_LITE} "HP3 DMA control"]
    connect_bd_intf_net [pccx_bd_intf $ctrl_ic {M05_AXI} "control interconnect M05"] [pccx_bd_intf $dma_acp {S_AXI_LITE} "ACP DMA control"]

    connect_bd_intf_net [pccx_bd_intf $dma_hp0 {M_AXI_MM2S} "HP0 DMA MM2S master"] [pccx_bd_intf $hp_ic {S00_AXI} "HP interconnect S00"]
    connect_bd_intf_net [pccx_bd_intf $dma_hp1 {M_AXI_MM2S} "HP1 DMA MM2S master"] [pccx_bd_intf $hp_ic {S01_AXI} "HP interconnect S01"]
    connect_bd_intf_net [pccx_bd_intf $dma_hp2 {M_AXI_MM2S} "HP2 DMA MM2S master"] [pccx_bd_intf $hp_ic {S02_AXI} "HP interconnect S02"]
    connect_bd_intf_net [pccx_bd_intf $dma_hp3 {M_AXI_MM2S} "HP3 DMA MM2S master"] [pccx_bd_intf $hp_ic {S03_AXI} "HP interconnect S03"]
    connect_bd_intf_net [pccx_bd_intf $hp_ic {M00_AXI} "HP interconnect M00"] $ps_hp0
    connect_bd_intf_net [pccx_bd_intf $hp_ic {M01_AXI} "HP interconnect M01"] $ps_hp1

    connect_bd_intf_net [pccx_bd_intf $dma_hp0 {M_AXIS_MM2S} "HP0 DMA stream"] $npu_hp0
    connect_bd_intf_net [pccx_bd_intf $dma_hp1 {M_AXIS_MM2S} "HP1 DMA stream"] $npu_hp1
    connect_bd_intf_net [pccx_bd_intf $dma_hp2 {M_AXIS_MM2S} "HP2 DMA stream"] $npu_hp2
    connect_bd_intf_net [pccx_bd_intf $dma_hp3 {M_AXIS_MM2S} "HP3 DMA stream"] $npu_hp3

    connect_bd_intf_net [pccx_bd_intf $dma_acp {M_AXI_MM2S} "ACP DMA MM2S master"] [pccx_bd_intf $acp_ic {S00_AXI} "ACP interconnect S00"]
    connect_bd_intf_net [pccx_bd_intf $dma_acp {M_AXI_S2MM} "ACP DMA S2MM master"] [pccx_bd_intf $acp_ic {S01_AXI} "ACP interconnect S01"]
    connect_bd_intf_net [pccx_bd_intf $acp_ic {M00_AXI} "ACP interconnect M00"] $ps_acp
    connect_bd_intf_net [pccx_bd_intf $dma_acp {M_AXIS_MM2S} "ACP fmap stream"] $npu_fmap
    connect_bd_intf_net $npu_res [pccx_bd_intf $dma_acp {S_AXIS_S2MM} "ACP result stream"]

    assign_bd_address
    if {[catch {
        set npu_seg [get_bd_addr_segs -quiet -of_objects $npu_axil]
        if {[llength $npu_seg] > 0} {
            set npu_addr [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces $ps_hpm0] -filter "NAME =~ */[get_property NAME [lindex $npu_seg 0]]"]
            if {[llength $npu_addr] > 0} {
                set_property offset $AXIL_BASE [lindex $npu_addr 0]
                set_property range 64K [lindex $npu_addr 0]
            }
        }
    } msg]} {
        pccx_msg "warning: NPU AXI-Lite base address left to Vivado: $msg"
    }

    validate_bd_design
    save_bd_design
    report_bd_address -file [file join $REPORTS address_map.rpt]

    pccx_msg "BD $BD_NAME prepared around $NPU_BD_MODULE_REF, which instantiates $TOP_RTL_MODULE"
    pccx_msg "clk_axi placeholder = ${CLK_AXI_MHZ} MHz (${axi_hz} Hz)"
    pccx_msg "clk_npu placeholder = ${CLK_NPU_MHZ} MHz (${npu_hz} Hz)"
}

proc pccx_open_or_prepare {} {
    global PROJ_DIR PROJ_NAME BD_NAME

    set xpr [file join $PROJ_DIR ${PROJ_NAME}.xpr]
    if {[file exists $xpr]} {
        open_project $xpr
    } else {
        pccx_create_project_shell
        pccx_add_sources
    }
    pccx_prepare_bd
}

proc pccx_import_bd_wrapper {} {
    global BD_NAME

    set bd_file [get_files -quiet */${BD_NAME}.bd]
    if {[llength $bd_file] == 0} {
        set bd_file [get_files -quiet ${BD_NAME}.bd]
    }
    if {[llength $bd_file] == 0} {
        error "BD file for $BD_NAME not found"
    }

    generate_target all [lindex $bd_file 0]
    set wrapper [make_wrapper -files [lindex $bd_file 0] -top]
    if {[llength $wrapper] > 0} {
        add_files -norecurse $wrapper
    }
    update_compile_order -fileset sources_1
}

proc pccx_synth {} {
    global TARGET_PART BD_NAME DCP_DIR REPORTS

    file mkdir $DCP_DIR
    file mkdir $REPORTS
    pccx_open_or_prepare
    pccx_import_bd_wrapper

    synth_design -top ${BD_NAME}_wrapper -part $TARGET_PART -flatten_hierarchy rebuilt
    write_checkpoint -force [file join $DCP_DIR ${BD_NAME}_post_synth.dcp]

    report_utilization -hierarchical -file [file join $REPORTS utilization_full_top_post_synth.rpt]
    report_clocks -file [file join $REPORTS clocks_full_top_post_synth.rpt]
    report_clock_interaction -file [file join $REPORTS clock_interaction_full_top_post_synth.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained \
        -check_timing_verbose -max_paths 20 \
        -file [file join $REPORTS timing_summary_full_top_post_synth.rpt]
    report_drc -file [file join $REPORTS drc_full_top_post_synth.rpt]
}

proc pccx_impl {} {
    global DCP_DIR REPORTS

    set synth_dcp [file join $DCP_DIR system_post_synth.dcp]
    if {![file exists $synth_dcp]} {
        error "missing $synth_dcp; run 'synth' first"
    }

    open_checkpoint $synth_dcp
    opt_design
    place_design
    route_design
    write_checkpoint -force [file join $DCP_DIR system_routed.dcp]

    report_utilization -hierarchical -file [file join $REPORTS utilization_full_top_post_impl.rpt]
    report_clock_interaction -file [file join $REPORTS clock_interaction_full_top_post_impl.rpt]
    report_route_status -file [file join $REPORTS route_status_full_top_post_impl.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained \
        -check_timing_verbose -max_paths 50 \
        -file [file join $REPORTS timing_summary_full_top_post_impl.rpt]
    report_drc -file [file join $REPORTS drc_full_top_post_impl.rpt]
    report_methodology -file [file join $REPORTS methodology_full_top_post_impl.rpt]
    report_power -file [file join $REPORTS power_full_top_post_impl.rpt]
}

proc pccx_bitstream {} {
    global DCP_DIR BUILD_DIR REPORTS

    set routed_dcp [file join $DCP_DIR system_routed.dcp]
    if {![file exists $routed_dcp]} {
        error "missing $routed_dcp; run 'impl' first"
    }

    open_checkpoint $routed_dcp
    write_bitstream -force [file join $BUILD_DIR pccx_v002_kv260.bit]
    report_timing_summary -delay_type min_max -report_unconstrained \
        -check_timing_verbose -max_paths 50 \
        -file [file join $REPORTS timing_summary_full_top_bitstream_checkpoint.rpt]
}

file mkdir $BUILD_DIR
file mkdir $REPORTS
file mkdir $DCP_DIR

if {$ACTION eq "clean"} {
    if {[file exists $PROJ_DIR]} {
        file delete -force $PROJ_DIR
    }
    foreach f [list \
        [file join $DCP_DIR system_post_synth.dcp] \
        [file join $DCP_DIR system_routed.dcp] \
        [file join $BUILD_DIR pccx_v002_kv260.bit] \
    ] {
        if {[file exists $f]} {
            file delete -force $f
        }
    }
    pccx_msg "cleaned BD project/checkpoint outputs"
} elseif {$ACTION eq "prepare"} {
    pccx_open_or_prepare
} elseif {$ACTION eq "synth"} {
    pccx_synth
} elseif {$ACTION eq "impl"} {
    pccx_impl
} elseif {$ACTION eq "bitstream"} {
    pccx_bitstream
}

pccx_msg "system_bd.tcl action '$ACTION' complete"
