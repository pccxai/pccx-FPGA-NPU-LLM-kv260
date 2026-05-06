`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===============================================================================
// Testbench: tb_global_scheduler_hazard_chaining
// Phase : pccx v002 - Global Scheduler hazard/chaining policy
//
// Purpose
// -------
//   Exercises the current Global_Scheduler translation boundary with a TB-local
//   hazard/fence golden model. The architectural docs describe address/resource
//   hazards and async completion fences in the Global Scheduler, but the current
//   RTL module has no busy, done, ready, or status-error ports. Therefore this TB
//   treats those timing events as deterministic testbench inputs:
//
//     * front-end issue is withheld while the SV golden model reports RAW/WAW/WAR,
//       resource occupancy, or FIFO backpressure;
//     * completion is an explicit TB event on the cycle named in each test case;
//     * accepted instructions are checked against SV-function uop goldens.
//
//   This makes the review boundary visible: the xsim PASS proves deterministic
//   uop translation and the externally enforced policy assumptions. It does not
//   claim that the current RTL contains internal interlock/status machinery.
// ===============================================================================

module tb_global_scheduler_hazard_chaining;

    typedef enum logic [2:0] {
        KIND_GEMV,
        KIND_GEMM,
        KIND_MEMCPY,
        KIND_MEMSET,
        KIND_CVO
    } op_kind_e;

    typedef enum logic [2:0] {
        RES_NONE,
        RES_GEMM,
        RES_GEMV,
        RES_SFU,
        RES_MEMCPY
    } resource_e;

    typedef struct packed {
        op_kind_e   kind;
        resource_e  resource;
        logic       reads_l2;
        addr_t      read_addr;
        logic       writes_l2;
        addr_t      write_addr;
        logic       is_async;
    } instr_meta_t;

    localparam int PendingSlots = 8;

    // ===| Clock + reset |=======================================================
    logic clk;
    logic rst_n;
    initial clk = 1'b0;
    always #2 clk = ~clk;

    // ===| Global_Scheduler DUT IO |============================================
    logic IN_GEMV_op_x64_valid;
    logic IN_GEMM_op_x64_valid;
    logic IN_memcpy_op_x64_valid;
    logic IN_memset_op_x64_valid;
    logic IN_cvo_op_x64_valid;
    instruction_op_x64_t instruction;

    gemm_control_uop_t   OUT_GEMM_uop;
    GEMV_control_uop_t   OUT_GEMV_uop;
    memory_control_uop_t OUT_LOAD_uop;
    memory_control_uop_t OUT_STORE_uop;
    memory_set_uop_t     OUT_mem_set_uop;
    cvo_control_uop_t    OUT_CVO_uop;
    logic                OUT_sram_rd_start;

    Global_Scheduler u_dut (
        .clk_core               (clk),
        .rst_n_core             (rst_n),
        .IN_GEMV_op_x64_valid   (IN_GEMV_op_x64_valid),
        .IN_GEMM_op_x64_valid   (IN_GEMM_op_x64_valid),
        .IN_memcpy_op_x64_valid (IN_memcpy_op_x64_valid),
        .IN_memset_op_x64_valid (IN_memset_op_x64_valid),
        .IN_cvo_op_x64_valid    (IN_cvo_op_x64_valid),
        .instruction            (instruction),
        .OUT_GEMM_uop           (OUT_GEMM_uop),
        .OUT_GEMV_uop           (OUT_GEMV_uop),
        .OUT_LOAD_uop           (OUT_LOAD_uop),
        .OUT_STORE_uop          (OUT_STORE_uop),
        .OUT_mem_set_uop        (OUT_mem_set_uop),
        .OUT_CVO_uop            (OUT_CVO_uop),
        .OUT_sram_rd_start      (OUT_sram_rd_start)
    );

    // ===| FIFO backpressure harness |===========================================
    logic     IN_acp_rdy;
    acp_uop_t IN_acp_cmd;
    acp_uop_t OUT_acp_cmd;
    logic     OUT_acp_cmd_valid;
    logic     OUT_acp_cmd_fifo_full;
    logic     IN_acp_is_busy;

    logic     IN_npu_rdy;
    npu_uop_t IN_npu_cmd;
    npu_uop_t OUT_npu_cmd;
    logic     OUT_npu_cmd_valid;
    logic     OUT_npu_cmd_fifo_full;
    logic     IN_npu_is_busy;

    mem_u_operation_queue #(
        .EnablePerfCounters(1'b1)
    ) u_queue (
        .clk_core              (clk),
        .rst_n_core            (rst_n),
        .IN_acp_rdy            (IN_acp_rdy),
        .IN_acp_cmd            (IN_acp_cmd),
        .OUT_acp_cmd           (OUT_acp_cmd),
        .OUT_acp_cmd_valid     (OUT_acp_cmd_valid),
        .OUT_acp_cmd_fifo_full (OUT_acp_cmd_fifo_full),
        .IN_acp_is_busy        (IN_acp_is_busy),
        .IN_npu_rdy            (IN_npu_rdy),
        .IN_npu_cmd            (IN_npu_cmd),
        .OUT_npu_cmd           (OUT_npu_cmd),
        .OUT_npu_cmd_valid     (OUT_npu_cmd_valid),
        .OUT_npu_cmd_fifo_full (OUT_npu_cmd_fifo_full),
        .IN_npu_is_busy        (IN_npu_is_busy)
    );

    // ===| Scoreboard |==========================================================
    int errors;
    int checks;
    int issued_cases;
    int blocked_cases;
    int completed_cases;
    int fifo_pushes;
    int cycle_count;

    logic      pending_write_valid [0:PendingSlots-1];
    addr_t     pending_write_addr  [0:PendingSlots-1];
    logic      pending_read_valid  [0:PendingSlots-1];
    addr_t     pending_read_addr   [0:PendingSlots-1];
    logic      resource_busy       [0:4];

    memory_control_uop_t last_load;
    memory_control_uop_t last_store;
    memory_set_uop_t     last_memset;
    gemm_control_uop_t   last_gemm;
    GEMV_control_uop_t   last_gemv;
    cvo_control_uop_t    last_cvo;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
        end
    end

    // ===| ISA packing helpers |=================================================
    function automatic flags_t make_flags(
        input logic findemax,
        input logic accm,
        input logic w_scale
    );
        make_flags = '{
            findemax : findemax,
            accm     : accm,
            w_scale  : w_scale,
            reserved : 3'b000
        };
    endfunction

    function automatic cvo_flags_t make_cvo_flags(
        input logic sub_emax,
        input logic recip_scale,
        input logic accm
    );
        make_cvo_flags = '{
            sub_emax    : sub_emax,
            recip_scale : recip_scale,
            accm        : accm,
            reserved    : 2'b00
        };
    endfunction

    function automatic instruction_op_x64_t make_gemv_inst(
        input dest_addr_t dest,
        input src_addr_t src,
        input flags_t flags,
        input ptr_addr_t size_ptr,
        input ptr_addr_t shape_ptr,
        input parallel_lane_t lanes
    );
        GEMV_op_x64_t body;
        body = '{
            dest_reg        : dest,
            src_addr        : src,
            flags           : flags,
            size_ptr_addr   : size_ptr,
            shape_ptr_addr  : shape_ptr,
            parallel_lane   : lanes,
            reserved        : 3'b000
        };
        make_gemv_inst.instruction = VLIW_instruction_x64'(body);
    endfunction

    function automatic instruction_op_x64_t make_gemm_inst(
        input dest_addr_t dest,
        input src_addr_t src,
        input flags_t flags,
        input ptr_addr_t size_ptr,
        input ptr_addr_t shape_ptr,
        input parallel_lane_t lanes
    );
        GEMM_op_x64_t body;
        body = '{
            dest_reg        : dest,
            src_addr        : src,
            flags           : flags,
            size_ptr_addr   : size_ptr,
            shape_ptr_addr  : shape_ptr,
            parallel_lane   : lanes,
            reserved        : 3'b000
        };
        make_gemm_inst.instruction = VLIW_instruction_x64'(body);
    endfunction

    function automatic instruction_op_x64_t make_memcpy_inst(
        input from_device_e from_device,
        input to_device_e to_device,
        input dest_addr_t dest,
        input src_addr_t src,
        input addr_t aux,
        input ptr_addr_t shape_ptr,
        input async_e async_mode
    );
        memcpy_op_x64_t body;
        body = '{
            from_device    : from_device,
            to_device      : to_device,
            dest_addr      : dest,
            src_addr       : src,
            aux_addr       : aux,
            shape_ptr_addr : shape_ptr,
            async          : async_mode
        };
        make_memcpy_inst.instruction = VLIW_instruction_x64'(body);
    endfunction

    function automatic instruction_op_x64_t make_memset_inst(
        input dest_cache_e dest_cache,
        input ptr_addr_t dest_addr,
        input a_value_t a_value,
        input b_value_t b_value,
        input c_value_t c_value
    );
        memset_op_x64_t body;
        body = '{
            dest_cache : dest_cache,
            dest_addr  : dest_addr,
            a_value    : a_value,
            b_value    : b_value,
            c_value    : c_value,
            reserved   : 4'h0
        };
        make_memset_inst.instruction = VLIW_instruction_x64'(body);
    endfunction

    function automatic instruction_op_x64_t make_cvo_inst(
        input cvo_func_e func,
        input src_addr_t src,
        input addr_t dst,
        input length_t length,
        input cvo_flags_t flags,
        input async_e async_mode
    );
        cvo_op_x64_t body;
        body = '{
            cvo_func : func,
            src_addr : src,
            dst_addr : dst,
            length   : length,
            flags    : flags,
            async    : async_mode
        };
        make_cvo_inst.instruction = VLIW_instruction_x64'(body);
    endfunction

    // ===| Golden uop helpers |==================================================
    function automatic data_route_e golden_memcpy_route(input memcpy_op_x64_t op);
        if (op.from_device == FROM_HOST && op.to_device == TO_NPU) begin
            golden_memcpy_route = from_host_to_L2;
        end else begin
            golden_memcpy_route = from_L2_to_host;
        end
    endfunction

    function automatic memory_control_uop_t golden_load(
        input op_kind_e kind,
        input instruction_op_x64_t inst
    );
        GEMV_op_x64_t   gemv;
        GEMM_op_x64_t   gemm;
        memcpy_op_x64_t memcpy;
        cvo_op_x64_t    cvo;

        gemv   = GEMV_op_x64_t'(inst.instruction);
        gemm   = GEMM_op_x64_t'(inst.instruction);
        memcpy = memcpy_op_x64_t'(inst.instruction);
        cvo    = cvo_op_x64_t'(inst.instruction);
        golden_load = '0;

        case (kind)
            KIND_GEMV: begin
                golden_load = '{
                    data_dest      : from_L2_to_L1_GEMV,
                    dest_addr      : '0,
                    src_addr       : gemv.src_addr,
                    shape_ptr_addr : gemv.shape_ptr_addr,
                    async          : SYNC_OP
                };
            end
            KIND_GEMM: begin
                golden_load = '{
                    data_dest      : from_L2_to_L1_GEMM,
                    dest_addr      : '0,
                    src_addr       : gemm.src_addr,
                    shape_ptr_addr : gemm.shape_ptr_addr,
                    async          : SYNC_OP
                };
            end
            KIND_MEMCPY: begin
                golden_load = '{
                    data_dest      : golden_memcpy_route(memcpy),
                    dest_addr      : memcpy.dest_addr,
                    src_addr       : memcpy.src_addr,
                    shape_ptr_addr : memcpy.shape_ptr_addr,
                    async          : memcpy.async
                };
            end
            KIND_CVO: begin
                golden_load = '{
                    data_dest      : from_L2_to_CVO,
                    dest_addr      : '0,
                    src_addr       : cvo.src_addr,
                    shape_ptr_addr : '0,
                    async          : cvo.async
                };
            end
            default: golden_load = '0;
        endcase
    endfunction

    function automatic memory_control_uop_t golden_store(
        input op_kind_e kind,
        input instruction_op_x64_t inst
    );
        GEMV_op_x64_t gemv;
        GEMM_op_x64_t gemm;
        cvo_op_x64_t  cvo;

        gemv = GEMV_op_x64_t'(inst.instruction);
        gemm = GEMM_op_x64_t'(inst.instruction);
        cvo  = cvo_op_x64_t'(inst.instruction);
        golden_store = '0;

        case (kind)
            KIND_GEMV: begin
                golden_store = '{
                    data_dest      : from_GEMV_res_to_L2,
                    dest_addr      : gemv.dest_reg,
                    src_addr       : '0,
                    shape_ptr_addr : gemv.shape_ptr_addr,
                    async          : SYNC_OP
                };
            end
            KIND_GEMM: begin
                golden_store = '{
                    data_dest      : from_GEMM_res_to_L2,
                    dest_addr      : gemm.dest_reg,
                    src_addr       : '0,
                    shape_ptr_addr : gemm.shape_ptr_addr,
                    async          : SYNC_OP
                };
            end
            KIND_CVO: begin
                golden_store = '{
                    data_dest      : from_CVO_res_to_L2,
                    dest_addr      : cvo.dst_addr,
                    src_addr       : '0,
                    shape_ptr_addr : '0,
                    async          : cvo.async
                };
            end
            default: golden_store = '0;
        endcase
    endfunction

    function automatic instr_meta_t meta_gemv(
        input instruction_op_x64_t inst,
        input logic async_hint
    );
        GEMV_op_x64_t op;
        op = GEMV_op_x64_t'(inst.instruction);
        meta_gemv = '{
            kind       : KIND_GEMV,
            resource   : RES_GEMV,
            reads_l2   : 1'b1,
            read_addr  : op.src_addr,
            writes_l2  : 1'b1,
            write_addr : op.dest_reg,
            is_async   : async_hint
        };
    endfunction

    function automatic instr_meta_t meta_gemm(
        input instruction_op_x64_t inst,
        input logic async_hint
    );
        GEMM_op_x64_t op;
        op = GEMM_op_x64_t'(inst.instruction);
        meta_gemm = '{
            kind       : KIND_GEMM,
            resource   : RES_GEMM,
            reads_l2   : 1'b1,
            read_addr  : op.src_addr,
            writes_l2  : 1'b1,
            write_addr : op.dest_reg,
            is_async   : async_hint
        };
    endfunction

    function automatic instr_meta_t meta_memcpy(
        input instruction_op_x64_t inst
    );
        memcpy_op_x64_t op;
        op = memcpy_op_x64_t'(inst.instruction);
        meta_memcpy = '{
            kind       : KIND_MEMCPY,
            resource   : RES_MEMCPY,
            reads_l2   : (op.from_device == FROM_NPU),
            read_addr  : op.src_addr,
            writes_l2  : (op.to_device == TO_NPU),
            write_addr : op.dest_addr,
            is_async   : op.async == ASYNC_OP
        };
    endfunction

    function automatic instr_meta_t meta_memset(input instruction_op_x64_t inst);
        meta_memset = '{
            kind       : KIND_MEMSET,
            resource   : RES_NONE,
            reads_l2   : 1'b0,
            read_addr  : '0,
            writes_l2  : 1'b0,
            write_addr : '0,
            is_async   : 1'b0
        };
    endfunction

    function automatic instr_meta_t meta_cvo(input instruction_op_x64_t inst);
        cvo_op_x64_t op;
        op = cvo_op_x64_t'(inst.instruction);
        meta_cvo = '{
            kind       : KIND_CVO,
            resource   : RES_SFU,
            reads_l2   : 1'b1,
            read_addr  : op.src_addr,
            writes_l2  : 1'b1,
            write_addr : op.dst_addr,
            is_async   : op.async == ASYNC_OP
        };
    endfunction

    function automatic logic golden_raw(input instr_meta_t meta);
        golden_raw = 1'b0;
        if (meta.reads_l2) begin
            for (int i = 0; i < PendingSlots; i++) begin
                if (pending_write_valid[i] && pending_write_addr[i] == meta.read_addr) begin
                    golden_raw = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic golden_waw(input instr_meta_t meta);
        golden_waw = 1'b0;
        if (meta.writes_l2) begin
            for (int i = 0; i < PendingSlots; i++) begin
                if (pending_write_valid[i] && pending_write_addr[i] == meta.write_addr) begin
                    golden_waw = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic golden_war(input instr_meta_t meta);
        golden_war = 1'b0;
        if (meta.writes_l2) begin
            for (int i = 0; i < PendingSlots; i++) begin
                if (pending_read_valid[i] && pending_read_addr[i] == meta.write_addr) begin
                    golden_war = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic golden_resource_busy(input instr_meta_t meta);
        golden_resource_busy = 1'b0;
        if (meta.resource != RES_NONE) begin
            golden_resource_busy = resource_busy[int'(meta.resource)];
        end
    endfunction

    function automatic logic golden_can_issue(
        input instr_meta_t meta,
        input logic fifo_block
    );
        golden_can_issue = !fifo_block &&
                           !golden_raw(meta) &&
                           !golden_waw(meta) &&
                           !golden_war(meta) &&
                           !golden_resource_busy(meta);
    endfunction

    // ===| Utility tasks |=======================================================
    task automatic clear_inputs;
        begin
            IN_GEMV_op_x64_valid   = 1'b0;
            IN_GEMM_op_x64_valid   = 1'b0;
            IN_memcpy_op_x64_valid = 1'b0;
            IN_memset_op_x64_valid = 1'b0;
            IN_cvo_op_x64_valid    = 1'b0;
            instruction            = '0;
        end
    endtask

    task automatic clear_queue_inputs;
        begin
            IN_acp_rdy = 1'b0;
            IN_acp_cmd = '0;
            IN_npu_rdy = 1'b0;
            IN_npu_cmd = '0;
        end
    endtask

    task automatic reset_golden;
        begin
            for (int i = 0; i < PendingSlots; i++) begin
                pending_write_valid[i] = 1'b0;
                pending_write_addr[i]  = '0;
                pending_read_valid[i]  = 1'b0;
                pending_read_addr[i]   = '0;
            end
            for (int i = 0; i < 5; i++) begin
                resource_busy[i] = 1'b0;
            end
        end
    endtask

    task automatic fail_msg(input string msg);
        begin
            errors++;
            $display("[%0t] FAIL: %s", $time, msg);
        end
    endtask

    task automatic expect_bit(input string tag, input logic got, input logic exp);
        begin
            checks++;
            if (got !== exp) begin
                errors++;
                $display("[%0t] mismatch %s: got=%0b exp=%0b", $time, tag, got, exp);
            end
        end
    endtask

    task automatic expect_load(
        input string tag,
        input memory_control_uop_t got,
        input memory_control_uop_t exp
    );
        begin
            checks++;
            if (got.data_dest      !== exp.data_dest ||
                got.dest_addr      !== exp.dest_addr ||
                got.src_addr       !== exp.src_addr ||
                got.shape_ptr_addr !== exp.shape_ptr_addr ||
                got.async          !== exp.async) begin
                errors++;
                $display("[%0t] mismatch %s.load: got route=%0h dest=%0d src=%0d shape=%0d async=%0d exp route=%0h dest=%0d src=%0d shape=%0d async=%0d",
                         $time, tag,
                         got.data_dest, got.dest_addr, got.src_addr, got.shape_ptr_addr, got.async,
                         exp.data_dest, exp.dest_addr, exp.src_addr, exp.shape_ptr_addr, exp.async);
            end
        end
    endtask

    task automatic expect_store(
        input string tag,
        input memory_control_uop_t got,
        input memory_control_uop_t exp
    );
        begin
            checks++;
            if (got.data_dest      !== exp.data_dest ||
                got.dest_addr      !== exp.dest_addr ||
                got.src_addr       !== exp.src_addr ||
                got.shape_ptr_addr !== exp.shape_ptr_addr ||
                got.async          !== exp.async) begin
                errors++;
                $display("[%0t] mismatch %s.store: got route=%0h dest=%0d src=%0d shape=%0d async=%0d exp route=%0h dest=%0d src=%0d shape=%0d async=%0d",
                         $time, tag,
                         got.data_dest, got.dest_addr, got.src_addr, got.shape_ptr_addr, got.async,
                         exp.data_dest, exp.dest_addr, exp.src_addr, exp.shape_ptr_addr, exp.async);
            end
        end
    endtask

    task automatic expect_memset(
        input string tag,
        input memory_set_uop_t got,
        input memory_set_uop_t exp
    );
        begin
            checks++;
            if (got.dest_cache !== exp.dest_cache ||
                got.dest_addr  !== exp.dest_addr ||
                got.a_value    !== exp.a_value ||
                got.b_value    !== exp.b_value ||
                got.c_value    !== exp.c_value) begin
                errors++;
                $display("[%0t] mismatch %s.memset: got cache=%0d addr=%0d a=%0d b=%0d c=%0d exp cache=%0d addr=%0d a=%0d b=%0d c=%0d",
                         $time, tag,
                         got.dest_cache, got.dest_addr, got.a_value, got.b_value, got.c_value,
                         exp.dest_cache, exp.dest_addr, exp.a_value, exp.b_value, exp.c_value);
            end
        end
    endtask

    task automatic expect_gemm(
        input string tag,
        input gemm_control_uop_t got,
        input gemm_control_uop_t exp
    );
        begin
            checks++;
            if (got.flags         !== exp.flags ||
                got.size_ptr_addr !== exp.size_ptr_addr ||
                got.parallel_lane !== exp.parallel_lane) begin
                errors++;
                $display("[%0t] mismatch %s.gemm: got flags=%0h size=%0d lanes=%0d exp flags=%0h size=%0d lanes=%0d",
                         $time, tag,
                         got.flags, got.size_ptr_addr, got.parallel_lane,
                         exp.flags, exp.size_ptr_addr, exp.parallel_lane);
            end
        end
    endtask

    task automatic expect_gemv(
        input string tag,
        input GEMV_control_uop_t got,
        input GEMV_control_uop_t exp
    );
        begin
            checks++;
            if (got.flags         !== exp.flags ||
                got.size_ptr_addr !== exp.size_ptr_addr ||
                got.parallel_lane !== exp.parallel_lane) begin
                errors++;
                $display("[%0t] mismatch %s.gemv: got flags=%0h size=%0d lanes=%0d exp flags=%0h size=%0d lanes=%0d",
                         $time, tag,
                         got.flags, got.size_ptr_addr, got.parallel_lane,
                         exp.flags, exp.size_ptr_addr, exp.parallel_lane);
            end
        end
    endtask

    task automatic expect_cvo(
        input string tag,
        input cvo_control_uop_t got,
        input cvo_control_uop_t exp
    );
        begin
            checks++;
            if (got.cvo_func !== exp.cvo_func ||
                got.src_addr !== exp.src_addr ||
                got.dst_addr !== exp.dst_addr ||
                got.length   !== exp.length ||
                got.flags    !== exp.flags ||
                got.async    !== exp.async) begin
                errors++;
                $display("[%0t] mismatch %s.cvo: got func=%0d src=%0d dst=%0d len=%0d flags=%0h async=%0d exp func=%0d src=%0d dst=%0d len=%0d flags=%0h async=%0d",
                         $time, tag,
                         got.cvo_func, got.src_addr, got.dst_addr, got.length, got.flags, got.async,
                         exp.cvo_func, exp.src_addr, exp.dst_addr, exp.length, exp.flags, exp.async);
            end
        end
    endtask

    task automatic expect_outputs_stable(input string tag);
        begin
            expect_load({tag, ".held_load"}, OUT_LOAD_uop, last_load);
            expect_store({tag, ".held_store"}, OUT_STORE_uop, last_store);
            expect_memset({tag, ".held_memset"}, OUT_mem_set_uop, last_memset);
            expect_gemm({tag, ".held_gemm"}, OUT_GEMM_uop, last_gemm);
            expect_gemv({tag, ".held_gemv"}, OUT_GEMV_uop, last_gemv);
            expect_cvo({tag, ".held_cvo"}, OUT_CVO_uop, last_cvo);
            expect_bit({tag, ".sram_idle"}, OUT_sram_rd_start, 1'b0);
        end
    endtask

    task automatic snapshot_outputs;
        begin
            last_load   = OUT_LOAD_uop;
            last_store  = OUT_STORE_uop;
            last_memset = OUT_mem_set_uop;
            last_gemm   = OUT_GEMM_uop;
            last_gemv   = OUT_GEMV_uop;
            last_cvo    = OUT_CVO_uop;
        end
    endtask

    task automatic drive_valid(input op_kind_e kind, input instruction_op_x64_t inst);
        begin
            clear_inputs();
            instruction = inst;
            case (kind)
                KIND_GEMV:   IN_GEMV_op_x64_valid   = 1'b1;
                KIND_GEMM:   IN_GEMM_op_x64_valid   = 1'b1;
                KIND_MEMCPY: IN_memcpy_op_x64_valid = 1'b1;
                KIND_MEMSET: IN_memset_op_x64_valid = 1'b1;
                KIND_CVO:    IN_cvo_op_x64_valid    = 1'b1;
                default: ;
            endcase
        end
    endtask

    task automatic add_pending_write(input addr_t addr);
        bit inserted;
        begin
            inserted = 1'b0;
            for (int i = 0; i < PendingSlots; i++) begin
                if (!pending_write_valid[i] && !inserted) begin
                    pending_write_valid[i] = 1'b1;
                    pending_write_addr[i]  = addr;
                    inserted = 1'b1;
                end
            end
            if (!inserted) fail_msg("pending write scoreboard overflow");
        end
    endtask

    task automatic add_pending_read(input addr_t addr);
        bit inserted;
        begin
            inserted = 1'b0;
            for (int i = 0; i < PendingSlots; i++) begin
                if (!pending_read_valid[i] && !inserted) begin
                    pending_read_valid[i] = 1'b1;
                    pending_read_addr[i]  = addr;
                    inserted = 1'b1;
                end
            end
            if (!inserted) fail_msg("pending read scoreboard overflow");
        end
    endtask

    task automatic clear_pending_write(input addr_t addr);
        begin
            for (int i = 0; i < PendingSlots; i++) begin
                if (pending_write_valid[i] && pending_write_addr[i] == addr) begin
                    pending_write_valid[i] = 1'b0;
                end
            end
        end
    endtask

    task automatic clear_pending_read(input addr_t addr);
        begin
            for (int i = 0; i < PendingSlots; i++) begin
                if (pending_read_valid[i] && pending_read_addr[i] == addr) begin
                    pending_read_valid[i] = 1'b0;
                end
            end
        end
    endtask

    task automatic golden_accept(input instr_meta_t meta);
        begin
            if (meta.resource != RES_NONE) resource_busy[int'(meta.resource)] = 1'b1;
            if (meta.writes_l2) add_pending_write(meta.write_addr);
            if (meta.reads_l2) add_pending_read(meta.read_addr);
        end
    endtask

    task automatic golden_complete(input instr_meta_t meta, input string tag);
        begin
            if (meta.resource != RES_NONE) resource_busy[int'(meta.resource)] = 1'b0;
            if (meta.writes_l2) clear_pending_write(meta.write_addr);
            if (meta.reads_l2) clear_pending_read(meta.read_addr);
            completed_cases++;
            $display("[%0t] COMPLETE: %s", $time, tag);
        end
    endtask

    task automatic check_accepted_issue(
        input string tag,
        input instr_meta_t meta,
        input instruction_op_x64_t inst
    );
        GEMV_op_x64_t   gemv;
        GEMM_op_x64_t   gemm;
        memset_op_x64_t memset;
        cvo_op_x64_t    cvo;
        begin
            gemv   = GEMV_op_x64_t'(inst.instruction);
            gemm   = GEMM_op_x64_t'(inst.instruction);
            memset = memset_op_x64_t'(inst.instruction);
            cvo    = cvo_op_x64_t'(inst.instruction);

            case (meta.kind)
                KIND_GEMV: begin
                    expect_load(tag, OUT_LOAD_uop, golden_load(meta.kind, inst));
                    expect_store(tag, OUT_STORE_uop, golden_store(meta.kind, inst));
                    expect_gemv(tag, OUT_GEMV_uop, '{
                        flags         : gemv.flags,
                        size_ptr_addr : gemv.size_ptr_addr,
                        parallel_lane : gemv.parallel_lane
                    });
                    expect_bit({tag, ".sram_start"}, OUT_sram_rd_start, 1'b1);
                end
                KIND_GEMM: begin
                    expect_load(tag, OUT_LOAD_uop, golden_load(meta.kind, inst));
                    expect_store(tag, OUT_STORE_uop, golden_store(meta.kind, inst));
                    expect_gemm(tag, OUT_GEMM_uop, '{
                        flags         : gemm.flags,
                        size_ptr_addr : gemm.size_ptr_addr,
                        parallel_lane : gemm.parallel_lane
                    });
                    expect_bit({tag, ".sram_start"}, OUT_sram_rd_start, 1'b1);
                end
                KIND_MEMCPY: begin
                    expect_load(tag, OUT_LOAD_uop, golden_load(meta.kind, inst));
                    expect_bit({tag, ".sram_start"}, OUT_sram_rd_start, 1'b0);
                end
                KIND_MEMSET: begin
                    expect_memset(tag, OUT_mem_set_uop, '{
                        dest_cache : dest_cache_e'(memset.dest_cache),
                        dest_addr  : memset.dest_addr,
                        a_value    : memset.a_value,
                        b_value    : memset.b_value,
                        c_value    : memset.c_value
                    });
                    expect_bit({tag, ".sram_start"}, OUT_sram_rd_start, 1'b0);
                end
                KIND_CVO: begin
                    expect_load(tag, OUT_LOAD_uop, golden_load(meta.kind, inst));
                    expect_store(tag, OUT_STORE_uop, golden_store(meta.kind, inst));
                    expect_cvo(tag, OUT_CVO_uop, '{
                        cvo_func : cvo_func_e'(cvo.cvo_func),
                        src_addr : cvo.src_addr,
                        dst_addr : cvo.dst_addr,
                        length   : cvo.length,
                        flags    : cvo_flags_t'(cvo.flags),
                        async    : cvo.async
                    });
                    expect_bit({tag, ".sram_start"}, OUT_sram_rd_start, 1'b0);
                end
                default: fail_msg({tag, ": unsupported accepted issue kind"});
            endcase
        end
    endtask

    task automatic issue_and_check(
        input string tag,
        input instr_meta_t meta,
        input instruction_op_x64_t inst
    );
        begin
            if (!golden_can_issue(meta, 1'b0)) begin
                fail_msg({tag, ": golden attempted to issue a blocked instruction"});
            end

            drive_valid(meta.kind, inst);
            @(posedge clk);
            #1;
            check_accepted_issue(tag, meta, inst);
            golden_accept(meta);
            issued_cases++;

            clear_inputs();
            @(posedge clk);
            #1;
            expect_bit({tag, ".sram_one_cycle"}, OUT_sram_rd_start, 1'b0);
            snapshot_outputs();
        end
    endtask

    task automatic block_and_check(
        input string tag,
        input instr_meta_t meta,
        input int cycles,
        input string reason,
        input logic fifo_block
    );
        begin
            if (golden_can_issue(meta, fifo_block)) begin
                fail_msg({tag, ": expected block was absent for ", reason});
            end
            for (int i = 0; i < cycles; i++) begin
                clear_inputs();
                @(posedge clk);
                #1;
                expect_outputs_stable($sformatf("%s.block_%0d", tag, i));
                blocked_cases++;
            end
        end
    endtask

    task automatic push_npu_from_load(input string tag, input memory_control_uop_t load_uop);
        begin
            IN_npu_cmd = '{
                write_en  : 1'b0,
                base_addr : load_uop.src_addr,
                end_addr  : load_uop.src_addr + 17'd1
            };
            IN_npu_rdy = 1'b1;
            @(posedge clk);
            #1;
            if (OUT_npu_cmd_fifo_full) begin
                checks++;
            end
            IN_npu_rdy = 1'b0;
            IN_npu_cmd = '0;
            fifo_pushes++;
            snapshot_outputs();
        end
    endtask

    task automatic issue_for_fifo(input int index);
        instruction_op_x64_t inst;
        instr_meta_t meta;
        flags_t flags;
        begin
            flags = make_flags(1'b0, 1'b0, 1'b0);
            inst  = make_gemv_inst(17'(17'h3000 + index),
                                   17'(17'h2000 + index),
                                   flags,
                                   6'd4,
                                   6'd5,
                                   5'd4);
            meta  = meta_gemv(inst, 1'b0);

            drive_valid(meta.kind, inst);
            @(posedge clk);
            #1;
            check_accepted_issue($sformatf("fifo_gemv_%0d", index), meta, inst);

            clear_inputs();
            push_npu_from_load($sformatf("fifo_push_%0d", index), OUT_LOAD_uop);
            @(posedge clk);
            #1;
            expect_bit($sformatf("fifo_gemv_%0d.sram_one_cycle", index),
                       OUT_sram_rd_start, 1'b0);
            issued_cases++;
        end
    endtask

    task automatic reserved_opcode_silent_drop_check;
        instruction_op_x64_t inst;
        begin
            inst.instruction = 60'hDEAD_BEEF_0123_456;
            clear_inputs();
            instruction = inst;
            @(posedge clk);
            #1;
            expect_outputs_stable("reserved_opcode_no_scheduler_valid");
            blocked_cases++;
        end
    endtask

    // ===| Test cases |==========================================================
    task automatic run_hazard_cases;
        instruction_op_x64_t inst_a;
        instruction_op_x64_t inst_b;
        instruction_op_x64_t inst_c;
        instr_meta_t meta_a;
        instr_meta_t meta_b;
        instr_meta_t meta_c;
        flags_t flags_base;
        flags_t flags_accm;
        cvo_flags_t cvo_flags;
        begin
            flags_base = make_flags(1'b0, 1'b0, 1'b1);
            flags_accm = make_flags(1'b0, 1'b1, 1'b1);
            cvo_flags  = make_cvo_flags(1'b1, 1'b0, 1'b0);

            // RAW + GEMV -> SFU chain: CVO reads the GEMV destination and is
            // held until the TB completion event releases the pending write.
            inst_a = make_gemv_inst(17'h0120, 17'h0040, flags_base, 6'd2, 6'd3, 5'd4);
            meta_a = meta_gemv(inst_a, 1'b0);
            issue_and_check("raw_chain_gemv", meta_a, inst_a);

            inst_b = make_cvo_inst(CVO_GELU, 17'h0120, 17'h0180, 16'd64, cvo_flags, SYNC_OP);
            meta_b = meta_cvo(inst_b);
            block_and_check("raw_chain_cvo_wait", meta_b, 2, "RAW pending GEMV result", 1'b0);
            golden_complete(meta_a, "GEMV result 0x0120 available for SFU");
            issue_and_check("raw_chain_cvo_issue", meta_b, inst_b);
            golden_complete(meta_b, "SFU result 0x0180 complete");

            // WAW: second writer to the same L2 destination waits for first
            // writeback/drain completion. This models K-split accumulation too.
            inst_a = make_gemm_inst(17'h0200, 17'h0800, flags_base, 6'd4, 6'd8, 5'd8);
            meta_a = meta_gemm(inst_a, 1'b0);
            issue_and_check("waw_ksplit_gemm_first", meta_a, inst_a);

            inst_b = make_gemm_inst(17'h0200, 17'h0900, flags_accm, 6'd4, 6'd8, 5'd8);
            meta_b = meta_gemm(inst_b, 1'b0);
            block_and_check("waw_ksplit_gemm_accm_wait", meta_b, 2, "WAW K-split drain", 1'b0);
            golden_complete(meta_a, "GEMM K-split drain 0 complete");
            issue_and_check("waw_ksplit_gemm_accm_issue", meta_b, inst_b);
            golden_complete(meta_b, "GEMM K-split drain 1 complete");

            // WAR: older SFU read keeps the source address protected until the
            // SFU completion event, so a later GEMV write to that address waits.
            inst_a = make_cvo_inst(CVO_REDUCE_SUM, 17'h0300, 17'h0340, 16'd32,
                                   make_cvo_flags(1'b0, 1'b0, 1'b0), ASYNC_OP);
            meta_a = meta_cvo(inst_a);
            issue_and_check("war_cvo_read_async", meta_a, inst_a);

            inst_b = make_gemv_inst(17'h0300, 17'h0400, flags_base, 6'd5, 6'd6, 5'd4);
            meta_b = meta_gemv(inst_b, 1'b0);
            block_and_check("war_gemv_write_wait", meta_b, 2, "WAR pending SFU read", 1'b0);
            golden_complete(meta_a, "SFU async read 0x0300 complete");
            issue_and_check("war_gemv_write_issue", meta_b, inst_b);
            golden_complete(meta_b, "GEMV write 0x0300 complete");

            // Resource hazards: one busy check per engine/resource class. The
            // blocked candidates use disjoint addresses so only resource_busy
            // can be responsible for the wait.
            inst_a = make_gemm_inst(17'h1000, 17'h1100, flags_base, 6'd1, 6'd2, 5'd4);
            meta_a = meta_gemm(inst_a, 1'b0);
            issue_and_check("resource_gemm_busy_first", meta_a, inst_a);
            inst_b = make_gemm_inst(17'h1200, 17'h1300, flags_base, 6'd1, 6'd2, 5'd4);
            meta_b = meta_gemm(inst_b, 1'b0);
            block_and_check("resource_gemm_busy_second", meta_b, 1, "GEMM busy", 1'b0);
            golden_complete(meta_a, "GEMM resource free");
            issue_and_check("resource_gemm_after_free", meta_b, inst_b);
            golden_complete(meta_b, "GEMM second complete");

            inst_a = make_gemv_inst(17'h1400, 17'h1500, flags_base, 6'd1, 6'd2, 5'd4);
            meta_a = meta_gemv(inst_a, 1'b0);
            issue_and_check("resource_gemv_busy_first", meta_a, inst_a);
            inst_b = make_gemv_inst(17'h1600, 17'h1700, flags_base, 6'd1, 6'd2, 5'd4);
            meta_b = meta_gemv(inst_b, 1'b0);
            block_and_check("resource_gemv_busy_second", meta_b, 1, "GEMV busy", 1'b0);
            golden_complete(meta_a, "GEMV resource free");
            issue_and_check("resource_gemv_after_free", meta_b, inst_b);
            golden_complete(meta_b, "GEMV second complete");

            inst_a = make_cvo_inst(CVO_SCALE, 17'h1800, 17'h1900, 16'd16,
                                   make_cvo_flags(1'b0, 1'b0, 1'b0), SYNC_OP);
            meta_a = meta_cvo(inst_a);
            issue_and_check("resource_sfu_busy_first", meta_a, inst_a);
            inst_b = make_cvo_inst(CVO_RECIP, 17'h1A00, 17'h1B00, 16'd16,
                                   make_cvo_flags(1'b0, 1'b0, 1'b0), SYNC_OP);
            meta_b = meta_cvo(inst_b);
            block_and_check("resource_sfu_busy_second", meta_b, 1, "SFU busy", 1'b0);
            golden_complete(meta_a, "SFU resource free");
            issue_and_check("resource_sfu_after_free", meta_b, inst_b);
            golden_complete(meta_b, "SFU second complete");

            inst_a = make_memcpy_inst(FROM_HOST, TO_NPU, 17'h1C00, 17'h0000,
                                      17'h0000, 6'd7, ASYNC_OP);
            meta_a = meta_memcpy(inst_a);
            issue_and_check("resource_memcpy_busy_first", meta_a, inst_a);
            inst_b = make_memcpy_inst(FROM_HOST, TO_NPU, 17'h1D00, 17'h0100,
                                      17'h0000, 6'd7, ASYNC_OP);
            meta_b = meta_memcpy(inst_b);
            block_and_check("resource_memcpy_busy_second", meta_b, 1, "MEMCPY busy", 1'b0);
            golden_complete(meta_a, "MEMCPY resource free");
            issue_and_check("resource_memcpy_after_free", meta_b, inst_b);
            golden_complete(meta_b, "MEMCPY second complete");

            // Async fence: async MEMCPY permits an independent MEMSET to issue,
            // but a dependent CVO RAW is held until completion.
            inst_a = make_memcpy_inst(FROM_HOST, TO_NPU, 17'h0500, 17'h0000,
                                      17'h0000, 6'd9, ASYNC_OP);
            meta_a = meta_memcpy(inst_a);
            issue_and_check("async_memcpy_issue", meta_a, inst_a);

            inst_b = make_memset_inst(data_to_fmap_shape, 6'd9, 16'd2, 16'd2, 16'd8);
            meta_b = meta_memset(inst_b);
            issue_and_check("async_independent_memset", meta_b, inst_b);

            inst_c = make_cvo_inst(CVO_EXP, 17'h0500, 17'h0550, 16'd8,
                                   make_cvo_flags(1'b0, 1'b0, 1'b0), SYNC_OP);
            meta_c = meta_cvo(inst_c);
            block_and_check("async_dependent_cvo_wait", meta_c, 2, "async completion fence", 1'b0);
            golden_complete(meta_a, "async MEMCPY 0x0500 complete");
            issue_and_check("async_dependent_cvo_issue", meta_c, inst_c);
            golden_complete(meta_c, "dependent SFU complete");

            reserved_opcode_silent_drop_check();
        end
    endtask

    task automatic run_fifo_backpressure_case;
        instr_meta_t fifo_meta;
        instruction_op_x64_t fifo_candidate;
        flags_t flags;
        begin
            IN_npu_is_busy = 1'b1;
            fifo_pushes    = 0;

            for (int i = 0; i < 140; i++) begin
                if (!OUT_npu_cmd_fifo_full) begin
                    issue_for_fifo(i);
                end
            end

            if (!OUT_npu_cmd_fifo_full) begin
                fail_msg("NPU FIFO did not assert backpressure during fill");
            end

            flags = make_flags(1'b0, 1'b0, 1'b0);
            fifo_candidate = make_gemv_inst(17'h3F00, 17'h3E00, flags, 6'd4, 6'd5, 5'd4);
            fifo_meta = meta_gemv(fifo_candidate, 1'b0);
            block_and_check("fifo_backpressure_issue_wait", fifo_meta, 3,
                            "NPU op FIFO full", OUT_npu_cmd_fifo_full);

            IN_npu_is_busy = 1'b0;
            repeat (900) @(posedge clk);
            #1;

            checks++;
            if (u_queue.npu_perf.in_count < 32'(fifo_pushes - 8)) begin
                errors++;
                $display("[%0t] mismatch fifo.in_count: got=%0d exp>=%0d",
                         $time, u_queue.npu_perf.in_count, fifo_pushes - 8);
            end
            checks++;
            if (u_queue.npu_perf.out_count < 32'(fifo_pushes - 12)) begin
                errors++;
                $display("[%0t] mismatch fifo.out_count: got=%0d exp>=%0d",
                         $time, u_queue.npu_perf.out_count, fifo_pushes - 12);
            end
            expect_bit("fifo_backpressure_deassert", OUT_npu_cmd_fifo_full, 1'b0);
            expect_bit("fifo_backpressure_no_forced_drop",
                       u_queue.npu_perf.stall_cycles != 32'd0, 1'b0);
            snapshot_outputs();
        end
    endtask

    // ===| Stimulus |============================================================
    initial begin
        errors          = 0;
        checks          = 0;
        issued_cases    = 0;
        blocked_cases   = 0;
        completed_cases = 0;
        fifo_pushes     = 0;

        rst_n          = 1'b0;
        IN_acp_is_busy = 1'b1;
        IN_npu_is_busy = 1'b1;
        clear_inputs();
        clear_queue_inputs();
        reset_golden();
        snapshot_outputs();

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        #1;
        snapshot_outputs();

        run_hazard_cases();
        run_fifo_backpressure_case();

        if (errors == 0) begin
            $display("PASS: %0d cycles, global scheduler hazard/chaining cases=%0d blocked=%0d completions=%0d fifo_pushes=%0d checks=%0d golden=sv_functions assumptions=tb_local_completion_no_internal_interlock",
                     cycle_count, issued_cases, blocked_cases, completed_cases, fifo_pushes, checks);
        end else begin
            $display("FAIL: global scheduler hazard/chaining mismatches=%0d checks=%0d cases=%0d blocked=%0d fifo_pushes=%0d",
                     errors, checks, issued_cases, blocked_cases, fifo_pushes);
        end
        $finish;
    end

    // ===| Watchdog |============================================================
    initial begin
        #200000 $display("FAIL: global scheduler hazard/chaining timeout"); $finish;
    end

endmodule
