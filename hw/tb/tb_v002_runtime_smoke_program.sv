`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===============================================================================
// Testbench: tb_v002_runtime_smoke_program
// Phase : pccx v002 - generated runtime program handoff
//
// Purpose
// -------
//   Consumes the .memh artifact emitted by tools/v002/generate_smoke_program.py
//   and checks that the generated ISA words cross the decoder + scheduler
//   boundary with the expected one-hot opcodes and micro-op fields.
// ===============================================================================

module tb_v002_runtime_smoke_program;

  localparam int ProgramLen = 7;

  // ===| Clock + reset |=========================================================
  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;

  // ===| Program artifact |======================================================
  string program_path;
  logic [`ISA_WIDTH-1:0] program_words [0:ProgramLen-1];

  // ===| Decoder IO |============================================================
  logic [`ISA_WIDTH-1:0] IN_raw_instruction;
  logic                  raw_instruction_pop_valid;
  logic                  OUT_fetch_PC_ready;
  logic                  OUT_GEMV_op_x64_valid;
  logic                  OUT_GEMM_op_x64_valid;
  logic                  OUT_memcpy_op_x64_valid;
  logic                  OUT_memset_op_x64_valid;
  logic                  OUT_cvo_op_x64_valid;
  instruction_op_x64_t   OUT_op_x64;

  ctrl_npu_decoder u_decoder (
      .clk                      (clk),
      .rst_n                    (rst_n),
      .IN_raw_instruction       (IN_raw_instruction),
      .raw_instruction_pop_valid(raw_instruction_pop_valid),
      .OUT_fetch_PC_ready       (OUT_fetch_PC_ready),
      .OUT_GEMV_op_x64_valid    (OUT_GEMV_op_x64_valid),
      .OUT_GEMM_op_x64_valid    (OUT_GEMM_op_x64_valid),
      .OUT_memcpy_op_x64_valid  (OUT_memcpy_op_x64_valid),
      .OUT_memset_op_x64_valid  (OUT_memset_op_x64_valid),
      .OUT_cvo_op_x64_valid     (OUT_cvo_op_x64_valid),
      .OUT_op_x64               (OUT_op_x64)
  );

  // ===| Scheduler IO |==========================================================
  gemm_control_uop_t   OUT_GEMM_uop;
  GEMV_control_uop_t   OUT_GEMV_uop;
  memory_control_uop_t OUT_LOAD_uop;
  memory_control_uop_t OUT_STORE_uop;
  logic                OUT_STORE_uop_valid;
  memory_set_uop_t     OUT_mem_set_uop;
  cvo_control_uop_t    OUT_CVO_uop;
  logic                OUT_sram_rd_start;

  Global_Scheduler u_scheduler (
      .clk_core               (clk),
      .rst_n_core             (rst_n),
      .IN_GEMV_op_x64_valid   (OUT_GEMV_op_x64_valid),
      .IN_GEMM_op_x64_valid   (OUT_GEMM_op_x64_valid),
      .IN_memcpy_op_x64_valid (OUT_memcpy_op_x64_valid),
      .IN_memset_op_x64_valid (OUT_memset_op_x64_valid),
      .IN_cvo_op_x64_valid    (OUT_cvo_op_x64_valid),
      .instruction            (OUT_op_x64),
      .OUT_GEMM_uop           (OUT_GEMM_uop),
      .OUT_GEMV_uop           (OUT_GEMV_uop),
      .OUT_LOAD_uop           (OUT_LOAD_uop),
      .OUT_STORE_uop          (OUT_STORE_uop),
      .OUT_STORE_uop_valid    (OUT_STORE_uop_valid),
      .OUT_mem_set_uop        (OUT_mem_set_uop),
      .OUT_CVO_uop            (OUT_CVO_uop),
      .OUT_sram_rd_start      (OUT_sram_rd_start)
  );

  // ===| Scoreboard |============================================================
  int errors = 0;
  int checks = 0;

  task automatic expect_equal64(input string tag, input logic [63:0] got, input logic [63:0] exp);
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%h exp=%h", $time, tag, got, exp);
      end
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

  task automatic expect_decoder(input int index, input logic [3:0] opcode);
    logic exp_gemv;
    logic exp_gemm;
    logic exp_memcpy;
    logic exp_memset;
    logic exp_cvo;
    begin
      exp_gemv   = opcode == OP_GEMV;
      exp_gemm   = opcode == OP_GEMM;
      exp_memcpy = opcode == OP_MEMCPY;
      exp_memset = opcode == OP_MEMSET;
      exp_cvo    = opcode == OP_CVO;

      expect_bit($sformatf("decode[%0d].gemv", index), OUT_GEMV_op_x64_valid, exp_gemv);
      expect_bit($sformatf("decode[%0d].gemm", index), OUT_GEMM_op_x64_valid, exp_gemm);
      expect_bit($sformatf("decode[%0d].memcpy", index), OUT_memcpy_op_x64_valid, exp_memcpy);
      expect_bit($sformatf("decode[%0d].memset", index), OUT_memset_op_x64_valid, exp_memset);
      expect_bit($sformatf("decode[%0d].cvo", index), OUT_cvo_op_x64_valid, exp_cvo);
      expect_equal64($sformatf("decode[%0d].body", index),
                     {4'h0, OUT_op_x64.instruction},
                     {4'h0, program_words[index][`ISA_BODY_WIDTH-1:0]});
    end
  endtask

  task automatic expect_memset(
      input int index,
      input dest_cache_e exp_cache,
      input ptr_addr_t exp_addr,
      input a_value_t exp_x,
      input b_value_t exp_y,
      input c_value_t exp_z
  );
    begin
      checks++;
      if (OUT_mem_set_uop.dest_cache !== exp_cache ||
          OUT_mem_set_uop.dest_addr  !== exp_addr ||
          OUT_mem_set_uop.a_value    !== exp_x ||
          OUT_mem_set_uop.b_value    !== exp_y ||
          OUT_mem_set_uop.c_value    !== exp_z) begin
        errors++;
        $display("[%0t] mismatch scheduler[%0d].memset: got cache=%0d addr=%0d x=%0d y=%0d z=%0d exp cache=%0d addr=%0d x=%0d y=%0d z=%0d",
                 $time, index,
                 OUT_mem_set_uop.dest_cache, OUT_mem_set_uop.dest_addr,
                 OUT_mem_set_uop.a_value, OUT_mem_set_uop.b_value, OUT_mem_set_uop.c_value,
                 exp_cache, exp_addr, exp_x, exp_y, exp_z);
      end
    end
  endtask

  task automatic expect_load(
      input int index,
      input data_route_e exp_route,
      input dest_addr_t exp_dest,
      input src_addr_t exp_src,
      input ptr_addr_t exp_shape_ptr,
      input async_e exp_async
  );
    begin
      checks++;
      if (OUT_LOAD_uop.data_dest      !== exp_route ||
          OUT_LOAD_uop.dest_addr      !== exp_dest ||
          OUT_LOAD_uop.src_addr       !== exp_src ||
          OUT_LOAD_uop.shape_ptr_addr !== exp_shape_ptr ||
          OUT_LOAD_uop.async          !== exp_async) begin
        errors++;
        $display("[%0t] mismatch scheduler[%0d].load: got route=%0h dest=%0d src=%0d shape=%0d async=%0d exp route=%0h dest=%0d src=%0d shape=%0d async=%0d",
                 $time, index,
                 OUT_LOAD_uop.data_dest, OUT_LOAD_uop.dest_addr, OUT_LOAD_uop.src_addr,
                 OUT_LOAD_uop.shape_ptr_addr, OUT_LOAD_uop.async,
                 exp_route, exp_dest, exp_src, exp_shape_ptr, exp_async);
      end
    end
  endtask

  task automatic expect_store(
      input int index,
      input data_route_e exp_route,
      input dest_addr_t exp_dest,
      input ptr_addr_t exp_shape_ptr,
      input logic exp_valid
  );
    begin
      expect_bit($sformatf("scheduler[%0d].store_valid", index), OUT_STORE_uop_valid, exp_valid);
      checks++;
      if (OUT_STORE_uop.data_dest      !== exp_route ||
          OUT_STORE_uop.dest_addr      !== exp_dest ||
          OUT_STORE_uop.shape_ptr_addr !== exp_shape_ptr) begin
        errors++;
        $display("[%0t] mismatch scheduler[%0d].store: got route=%0h dest=%0d shape=%0d exp route=%0h dest=%0d shape=%0d",
                 $time, index,
                 OUT_STORE_uop.data_dest, OUT_STORE_uop.dest_addr,
                 OUT_STORE_uop.shape_ptr_addr,
                 exp_route, exp_dest, exp_shape_ptr);
      end
    end
  endtask

  task automatic expect_scheduler(input int index);
    begin
      case (index)
        0: begin
          expect_memset(index, data_to_fmap_shape, 6'd3, 16'd1, 16'd1, 16'd9);
          expect_bit("scheduler[0].store_valid", OUT_STORE_uop_valid, 1'b0);
          expect_bit("scheduler[0].sram_start", OUT_sram_rd_start, 1'b0);
        end
        1: begin
          expect_memset(index, data_to_fmap_shape, 6'd9, 16'd4, 16'd4, 16'd4);
          expect_bit("scheduler[1].store_valid", OUT_STORE_uop_valid, 1'b0);
          expect_bit("scheduler[1].sram_start", OUT_sram_rd_start, 1'b0);
        end
        2: begin
          expect_load(index, from_host_to_L2, 17'd100, 17'd0, 6'd3, SYNC_OP);
          expect_bit("scheduler[2].store_valid", OUT_STORE_uop_valid, 1'b0);
          expect_bit("scheduler[2].sram_start", OUT_sram_rd_start, 1'b0);
        end
        3: begin
          expect_load(index, from_L2_to_L1_GEMM, 17'd0, 17'd300, 6'd3, SYNC_OP);
          expect_store(index, from_GEMM_res_to_L2, 17'd512, 6'd3, 1'b1);
          expect_bit("scheduler[3].gemm.w_scale", OUT_GEMM_uop.flags.w_scale, 1'b1);
          expect_equal64("scheduler[3].gemm.size_lane",
                         {53'd0, OUT_GEMM_uop.size_ptr_addr, OUT_GEMM_uop.parallel_lane},
                         {53'd0, 6'd1, 5'd4});
          expect_bit("scheduler[3].sram_start", OUT_sram_rd_start, 1'b1);
        end
        4: begin
          expect_load(index, from_L2_to_L1_GEMV, 17'd0, 17'd400, 6'd9, SYNC_OP);
          expect_store(index, from_GEMV_res_to_L2, 17'd768, 6'd9, 1'b1);
          expect_bit("scheduler[4].gemv.accm", OUT_GEMV_uop.flags.accm, 1'b1);
          expect_equal64("scheduler[4].gemv.size_lane",
                         {53'd0, OUT_GEMV_uop.size_ptr_addr, OUT_GEMV_uop.parallel_lane},
                         {53'd0, 6'd2, 5'd4});
          expect_bit("scheduler[4].sram_start", OUT_sram_rd_start, 1'b1);
        end
        5: begin
          expect_load(index, from_L2_to_CVO, 17'd0, 17'd768, 6'd0, SYNC_OP);
          expect_store(index, from_CVO_res_to_L2, 17'd896, 6'd0, 1'b1);
          checks++;
          if (OUT_CVO_uop.cvo_func !== CVO_REDUCE_SUM ||
              OUT_CVO_uop.src_addr !== 17'd768 ||
              OUT_CVO_uop.dst_addr !== 17'd896 ||
              OUT_CVO_uop.length   !== 16'd64 ||
              OUT_CVO_uop.async    !== SYNC_OP) begin
            errors++;
            $display("[%0t] mismatch scheduler[5].cvo: got func=%0d src=%0d dst=%0d len=%0d async=%0d",
                     $time,
                     OUT_CVO_uop.cvo_func, OUT_CVO_uop.src_addr, OUT_CVO_uop.dst_addr,
                     OUT_CVO_uop.length, OUT_CVO_uop.async);
          end
          expect_bit("scheduler[5].sram_start", OUT_sram_rd_start, 1'b0);
        end
        6: begin
          expect_load(index, from_L2_to_host, 17'd0, 17'd896, 6'd3, SYNC_OP);
          expect_bit("scheduler[6].store_valid", OUT_STORE_uop_valid, 1'b0);
          expect_bit("scheduler[6].sram_start", OUT_sram_rd_start, 1'b0);
        end
        default: begin
          errors++;
          $display("[%0t] unexpected scheduler index %0d", $time, index);
        end
      endcase
    end
  endtask

  task automatic issue_instruction(input int index);
    logic [3:0] opcode;
    begin
      opcode = program_words[index][`ISA_WIDTH-1:`ISA_WIDTH-`ISA_OPCODE_WIDTH];
      IN_raw_instruction        = program_words[index];
      raw_instruction_pop_valid = 1'b1;
      @(posedge clk);
      #1;
      expect_decoder(index, opcode);

      raw_instruction_pop_valid = 1'b0;
      IN_raw_instruction        = '0;
      @(posedge clk);
      #1;
      expect_scheduler(index);
    end
  endtask

  // ===| Stimulus |=============================================================
  initial begin
    rst_n                     = 1'b0;
    IN_raw_instruction        = '0;
    raw_instruction_pop_valid = 1'b0;

    if (!$value$plusargs("PROGRAM_MEMH=%s", program_path)) begin
      program_path = "v002_runtime_smoke.memh";
    end
    $readmemh(program_path, program_words);

    for (int i = 0; i < ProgramLen; i++) begin
      if (^program_words[i] === 1'bx) begin
        errors++;
        $display("[%0t] program word %0d is unknown after reading %s", $time, i, program_path);
      end
    end

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);
    #1;

    for (int i = 0; i < ProgramLen; i++) begin
      issue_instruction(i);
      @(posedge clk);
      #1;
    end

    if (errors == 0) begin
      $display("PASS: %0d generated runtime instructions decoded and scheduled.", ProgramLen);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #100000 $display("FAIL: v002 runtime smoke program timeout"); $finish;
  end

endmodule
