`timescale 1ns / 1ps

// ==============================================================================
// Testbench: tb_gemm_w4a8_dsp_dual_mac
// Phase : pccx v002, GEMM W4A8 DSP lane check
//
// Purpose
// -------
//   Deterministic verification of the W4A8 dual-MAC datapath:
//     * signed INT4 upper/lower weights
//     * signed INT8 activation sign extension
//     * packed A/B multiplication
//     * lower/upper sign recovery
//     * hold, clear, flush, and last-row cascade behavior
// ==============================================================================

`include "npu_arch.svh"
`include "kv260_device.svh"

import dtype_pkg::*;

module tb_gemm_w4a8_dsp_dual_mac;

  localparam int Int4Bits      = dtype_pkg::Int4Width;
  localparam int Int8Bits      = dtype_pkg::Int8Width;
  localparam int APortW        = `DEVICE_DSP_A_WIDTH;
  localparam int BPortW        = `DEVICE_DSP_B_WIDTH;
  localparam int PPortW        = dtype_pkg::DspPWidth;
  localparam int UpperShift    = 21;
  localparam int LowerW        = UpperShift;
  localparam int UpperW        = 21;
  localparam int ClockHalfNs   = 2;
  localparam int VecCaseCount  = 16;
  localparam int LenCaseCount  = 6;
  localparam int UnitCaseCount = 3;
  localparam int TotalCases    = VecCaseCount + LenCaseCount + UnitCaseCount;

  logic clk;
  initial clk = 1'b0;
  always #ClockHalfNs clk = ~clk;

  typedef struct packed {
    logic signed [Int4Bits-1:0] lower;
    logic signed [Int4Bits-1:0] upper;
    logic signed [Int8Bits-1:0] act;
  } vec_case_t;

  vec_case_t vec_cases [0:VecCaseCount-1];
  initial begin
    vec_cases[0]  = '{lower:  4'sd3, upper:  4'sd2, act:    8'sd5};
    vec_cases[1]  = '{lower: -4'sd3, upper: -4'sd2, act:    8'sd5};
    vec_cases[2]  = '{lower:  4'sd3, upper:  4'sd2, act:   -8'sd5};
    vec_cases[3]  = '{lower: -4'sd3, upper: -4'sd2, act:   -8'sd5};
    vec_cases[4]  = '{lower:  4'sd0, upper:  4'sd7, act:   8'sd13};
    vec_cases[5]  = '{lower:  4'sd7, upper:  4'sd0, act:  -8'sd13};
    vec_cases[6]  = '{lower:  4'sd1, upper: -4'sd1, act:    8'sd0};
    vec_cases[7]  = '{lower:  4'sh8, upper:  4'sd7, act:  8'sd127};
    vec_cases[8]  = '{lower:  4'sd7, upper:  4'sh8, act:  8'sh80};
    vec_cases[9]  = '{lower:  4'sh8, upper:  4'sh8, act:  8'sh80};
    vec_cases[10] = '{lower:  4'sd7, upper:  4'sd7, act:  8'sh80};
    vec_cases[11] = '{lower: -4'sd1, upper:  4'sd1, act:   -8'sd1};
    vec_cases[12] = '{lower:  4'sd1, upper: -4'sd1, act:    8'sd1};
    vec_cases[13] = '{lower:  4'sd4, upper: -4'sd4, act:   8'sd64};
    vec_cases[14] = '{lower: -4'sd4, upper:  4'sd4, act:  -8'sd64};
    vec_cases[15] = '{lower:  4'sd6, upper: -4'sd7, act:   -8'sd3};
  end

  int length_cases [0:LenCaseCount-1];
  initial begin
    length_cases[0] = 1;
    length_cases[1] = 2;
    length_cases[2] = 15;
    length_cases[3] = 127;
    length_cases[4] = 1024;
    length_cases[5] = 4096;
  end

  logic signed [Int4Bits-1:0] tb_w_lower;
  logic signed [Int4Bits-1:0] tb_w_upper;
  logic signed [Int8Bits-1:0] tb_act;
  logic signed [APortW-1:0]   tb_a_packed;
  logic signed [BPortW-1:0]   tb_b_extended;
  logic signed [PPortW-1:0]   tb_p_accum;
  logic signed [LowerW-1:0]   tb_rec_lower;
  logic signed [UpperW-1:0]   tb_rec_upper;

  GEMM_dsp_packer #(
    .INT4_BITS   (Int4Bits),
    .INT8_BITS   (Int8Bits),
    .A_PORT_W    (APortW),
    .B_PORT_W    (BPortW),
    .UPPER_SHIFT (UpperShift)
  ) u_packer (
    .in_w_lower     (tb_w_lower),
    .in_w_upper     (tb_w_upper),
    .in_act         (tb_act),
    .out_a_packed   (tb_a_packed),
    .out_b_extended (tb_b_extended)
  );

  GEMM_sign_recovery #(
    .P_PORT_W    (PPortW),
    .UPPER_SHIFT (UpperShift),
    .LOWER_W     (LowerW),
    .UPPER_W     (UpperW)
  ) u_recovery (
    .in_p_accum    (tb_p_accum),
    .out_lower_sum (tb_rec_lower),
    .out_upper_sum (tb_rec_upper)
  );

  logic rst_n;

  logic last_clear;
  logic last_valid;
  logic last_weight_valid;
  logic last_inst_valid;
  logic [2:0] last_inst;
  logic [Int4Bits-1:0] last_h_upper_in;
  logic [Int4Bits-1:0] last_h_lower_in;
  logic [Int4Bits-1:0] last_h_upper_out;
  logic [Int4Bits-1:0] last_h_lower_out;
  logic [Int8Bits-1:0] last_v_in;
  logic [BPortW-1:0] last_bcin;
  logic [BPortW-1:0] last_bcout;
  logic [2:0] last_inst_out;
  logic last_inst_valid_out;
  logic [PPortW-1:0] last_pcin;
  logic [PPortW-1:0] last_pcout;
  logic [PPortW-1:0] last_result;
  logic last_o_valid;
  logic signed [LowerW-1:0] last_rec_lower;
  logic signed [UpperW-1:0] last_rec_upper;

  GEMM_dsp_unit_last_ROW #(
    .IS_TOP_ROW  (1),
    .INT4_BITS   (Int4Bits),
    .INT8_BITS   (Int8Bits),
    .A_PORT_W    (APortW),
    .B_PORT_W    (BPortW),
    .P_PORT_W    (PPortW),
    .UPPER_SHIFT (UpperShift)
  ) u_last_direct (
    .clk                (clk),
    .rst_n              (rst_n),
    .i_clear            (last_clear),
    .i_valid            (last_valid),
    .i_weight_valid     (last_weight_valid),
    .inst_valid_in_V    (last_inst_valid),
    .o_valid            (last_o_valid),
    .in_H_upper         (last_h_upper_in),
    .out_H_upper        (last_h_upper_out),
    .in_H_lower         (last_h_lower_in),
    .out_H_lower        (last_h_lower_out),
    .in_V               (last_v_in),
    .BCIN_in            (last_bcin),
    .BCOUT_out          (last_bcout),
    .instruction_in_V   (last_inst),
    .instruction_out_V  (last_inst_out),
    .inst_valid_out_V   (last_inst_valid_out),
    .V_result_in        (last_pcin),
    .V_result_out       (last_pcout),
    .gemm_unit_results  (last_result)
  );

  GEMM_sign_recovery #(
    .P_PORT_W    (PPortW),
    .UPPER_SHIFT (UpperShift),
    .LOWER_W     (LowerW),
    .UPPER_W     (UpperW)
  ) u_last_recovery (
    .in_p_accum    (last_result),
    .out_lower_sum (last_rec_lower),
    .out_upper_sum (last_rec_upper)
  );

  logic cas_clear;
  logic cas_valid;
  logic cas_weight_valid;
  logic cas_inst_valid;
  logic [2:0] cas_inst;
  logic [Int4Bits-1:0] cas_h_upper_in;
  logic [Int4Bits-1:0] cas_h_lower_in;
  logic [Int4Bits-1:0] cas_h_upper_out;
  logic [Int4Bits-1:0] cas_h_lower_out;
  logic [Int8Bits-1:0] cas_v_in;
  logic [BPortW-1:0] cas_bcin;
  logic [BPortW-1:0] cas_bcout;
  logic [2:0] cas_inst_out;
  logic cas_inst_valid_out;
  logic [PPortW-1:0] cas_pcin;
  logic [PPortW-1:0] cas_pcout;
  logic [PPortW-1:0] cas_result;
  logic cas_o_valid;
  logic signed [LowerW-1:0] cas_rec_lower;
  logic signed [UpperW-1:0] cas_rec_upper;

  GEMM_dsp_unit_last_ROW #(
    .IS_TOP_ROW  (0),
    .INT4_BITS   (Int4Bits),
    .INT8_BITS   (Int8Bits),
    .A_PORT_W    (APortW),
    .B_PORT_W    (BPortW),
    .P_PORT_W    (PPortW),
    .UPPER_SHIFT (UpperShift)
  ) u_last_cascade (
    .clk                (clk),
    .rst_n              (rst_n),
    .i_clear            (cas_clear),
    .i_valid            (cas_valid),
    .i_weight_valid     (cas_weight_valid),
    .inst_valid_in_V    (cas_inst_valid),
    .o_valid            (cas_o_valid),
    .in_H_upper         (cas_h_upper_in),
    .out_H_upper        (cas_h_upper_out),
    .in_H_lower         (cas_h_lower_in),
    .out_H_lower        (cas_h_lower_out),
    .in_V               (cas_v_in),
    .BCIN_in            (cas_bcin),
    .BCOUT_out          (cas_bcout),
    .instruction_in_V   (cas_inst),
    .instruction_out_V  (cas_inst_out),
    .inst_valid_out_V   (cas_inst_valid_out),
    .V_result_in        (cas_pcin),
    .V_result_out       (cas_pcout),
    .gemm_unit_results  (cas_result)
  );

  GEMM_sign_recovery #(
    .P_PORT_W    (PPortW),
    .UPPER_SHIFT (UpperShift),
    .LOWER_W     (LowerW),
    .UPPER_W     (UpperW)
  ) u_cas_recovery (
    .in_p_accum    (cas_result),
    .out_lower_sum (cas_rec_lower),
    .out_upper_sum (cas_rec_upper)
  );

  logic unit_clear;
  logic unit_valid;
  logic unit_weight_valid;
  logic unit_inst_valid;
  logic [2:0] unit_inst;
  logic [Int4Bits-1:0] unit_h_upper_in;
  logic [Int4Bits-1:0] unit_h_lower_in;
  logic [Int4Bits-1:0] unit_h_upper_out;
  logic [Int4Bits-1:0] unit_h_lower_out;
  logic [Int8Bits-1:0] unit_v_in;
  logic [BPortW-1:0] unit_bcin;
  logic [BPortW-1:0] unit_bcout;
  logic [2:0] unit_inst_out;
  logic unit_inst_valid_out;
  logic [PPortW-1:0] unit_pcin;
  logic [PPortW-1:0] unit_pcout;
  logic [PPortW-1:0] unit_fabric_p;
  logic unit_o_valid;
  logic signed [LowerW-1:0] unit_rec_lower;
  logic signed [UpperW-1:0] unit_rec_upper;

  GEMM_dsp_unit #(
    .IS_TOP_ROW    (1),
    .BREAK_CASCADE (0),
    .INT4_BITS     (Int4Bits),
    .INT8_BITS     (Int8Bits),
    .A_PORT_W      (APortW),
    .B_PORT_W      (BPortW),
    .P_PORT_W      (PPortW),
    .UPPER_SHIFT   (UpperShift)
  ) u_unit_direct (
    .clk                (clk),
    .rst_n              (rst_n),
    .i_clear            (unit_clear),
    .i_valid            (unit_valid),
    .i_weight_valid     (unit_weight_valid),
    .o_valid            (unit_o_valid),
    .in_H_upper         (unit_h_upper_in),
    .out_H_upper        (unit_h_upper_out),
    .in_H_lower         (unit_h_lower_in),
    .out_H_lower        (unit_h_lower_out),
    .in_V               (unit_v_in),
    .BCIN_in            (unit_bcin),
    .BCOUT_out          (unit_bcout),
    .instruction_in_V   (unit_inst),
    .instruction_out_V  (unit_inst_out),
    .inst_valid_in_V    (unit_inst_valid),
    .inst_valid_out_V   (unit_inst_valid_out),
    .V_result_in        (unit_pcin),
    .V_result_out       (unit_pcout),
    .P_fabric_out       (unit_fabric_p)
  );

  GEMM_sign_recovery #(
    .P_PORT_W    (PPortW),
    .UPPER_SHIFT (UpperShift),
    .LOWER_W     (LowerW),
    .UPPER_W     (UpperW)
  ) u_unit_recovery (
    .in_p_accum    (unit_fabric_p),
    .out_lower_sum (unit_rec_lower),
    .out_upper_sum (unit_rec_upper)
  );

  int errors;
  int first_fail_case;
  int case_count;

  function automatic int signed i4_to_int(input logic signed [Int4Bits-1:0] value);
    return int'($signed(value));
  endfunction

  function automatic int signed i8_to_int(input logic signed [Int8Bits-1:0] value);
    return int'($signed(value));
  endfunction

  function automatic logic signed [APortW-1:0] golden_a(
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper
  );
    logic signed [APortW-1:0] lower_ext;
    logic signed [APortW-1:0] upper_ext;
    begin
      lower_ext = APortW'(signed'(lower));
      upper_ext = APortW'(signed'(upper)) <<< UpperShift;
      return upper_ext + lower_ext;
    end
  endfunction

  function automatic logic signed [BPortW-1:0] golden_b(
    input logic signed [Int8Bits-1:0] value
  );
    return BPortW'(signed'(value));
  endfunction

  function automatic logic signed [PPortW-1:0] golden_p(
    input int signed lower_sum,
    input int signed upper_sum
  );
    logic signed [PPortW-1:0] lower_ext;
    logic signed [PPortW-1:0] upper_ext;
    begin
      lower_ext = PPortW'(lower_sum);
      upper_ext = PPortW'(upper_sum) <<< UpperShift;
      return upper_ext + lower_ext;
    end
  endfunction

  function automatic logic signed [Int4Bits-1:0] seq_lower(input int idx);
    case (idx % 10)
      0: return -4'sd2;
      1: return -4'sd1;
      2: return  4'sd0;
      3: return  4'sd1;
      4: return  4'sd2;
      5: return  4'sd1;
      6: return -4'sd2;
      7: return  4'sd0;
      8: return  4'sd2;
      default: return -4'sd1;
    endcase
  endfunction

  function automatic logic signed [Int4Bits-1:0] seq_upper(input int idx);
    case (idx % 8)
      0: return  4'sd2;
      1: return  4'sd0;
      2: return -4'sd2;
      3: return  4'sd1;
      4: return -4'sd1;
      5: return  4'sd2;
      6: return -4'sd2;
      default: return 4'sd1;
    endcase
  endfunction

  function automatic logic signed [Int8Bits-1:0] seq_act(input int idx);
    case (idx % 12)
      0: return -8'sd7;
      1: return -8'sd3;
      2: return  8'sd0;
      3: return  8'sd5;
      4: return  8'sd7;
      5: return  8'sd1;
      6: return -8'sd1;
      7: return  8'sd2;
      8: return -8'sd2;
      9: return  8'sd3;
      10: return -8'sd5;
      default: return 8'sd6;
    endcase
  endfunction

  task automatic note_error(
    input int case_idx,
    input string label,
    input string field,
    input logic signed [PPortW-1:0] got,
    input logic signed [PPortW-1:0] expected
  );
    begin
      if (errors == 0) begin
        first_fail_case = case_idx;
      end
      errors++;
      $display("FAIL: TB_FAILED case=%0d %s %s got=%0d expected=%0d got_hex=%h expected_hex=%h",
               case_idx, label, field, got, expected, got, expected);
    end
  endtask

  task automatic check_recovered(
    input int case_idx,
    input string label,
    input logic signed [LowerW-1:0] got_lower,
    input logic signed [UpperW-1:0] got_upper,
    input int signed expected_lower_int,
    input int signed expected_upper_int
  );
    logic signed [LowerW-1:0] expected_lower;
    logic signed [UpperW-1:0] expected_upper;
    begin
      expected_lower = LowerW'(expected_lower_int);
      expected_upper = UpperW'(expected_upper_int);
      if (got_lower !== expected_lower) begin
        note_error(case_idx, label, "lower",
                   PPortW'(got_lower), PPortW'(expected_lower));
      end
      if (got_upper !== expected_upper) begin
        note_error(case_idx, label, "upper",
                   PPortW'(got_upper), PPortW'(expected_upper));
      end
    end
  endtask

  task automatic check_direct_step(
    input int case_idx,
    input string label,
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper,
    input logic signed [Int8Bits-1:0] act,
    inout int signed expected_lower,
    inout int signed expected_upper
  );
    logic signed [APortW-1:0] expected_a;
    logic signed [BPortW-1:0] expected_b;
    begin
      tb_w_lower = lower;
      tb_w_upper = upper;
      tb_act     = act;
      #1;

      expected_a = golden_a(lower, upper);
      expected_b = golden_b(act);
      if (tb_a_packed !== expected_a) begin
        note_error(case_idx, label, "a_pack",
                   PPortW'(tb_a_packed), PPortW'(expected_a));
      end
      if (tb_a_packed[Int4Bits-1:0] !== lower) begin
        note_error(case_idx, label, "a_lower_bits",
                   PPortW'(tb_a_packed[Int4Bits-1:0]), PPortW'(lower));
      end
      if (tb_b_extended !== expected_b) begin
        note_error(case_idx, label, "b_extend",
                   PPortW'(tb_b_extended), PPortW'(expected_b));
      end
      if (tb_b_extended[Int8Bits-1:0] !== act) begin
        note_error(case_idx, label, "b_payload",
                   PPortW'(tb_b_extended[Int8Bits-1:0]), PPortW'(act));
      end
      if (tb_b_extended[BPortW-1:Int8Bits] !== {(BPortW-Int8Bits){act[Int8Bits-1]}}) begin
        note_error(case_idx, label, "b_sign",
                   PPortW'(tb_b_extended[BPortW-1:Int8Bits]),
                   PPortW'({(BPortW-Int8Bits){act[Int8Bits-1]}}));
      end

      tb_p_accum = tb_p_accum + ($signed(tb_a_packed) * $signed(tb_b_extended));
      expected_lower += i4_to_int(lower) * i8_to_int(act);
      expected_upper += i4_to_int(upper) * i8_to_int(act);
      #1;
      check_recovered(case_idx, label, tb_rec_lower, tb_rec_upper,
                      expected_lower, expected_upper);
    end
  endtask

  task automatic run_single_vec(input int case_idx, input vec_case_t item);
    int signed expected_lower;
    int signed expected_upper;
    begin
      tb_p_accum = '0;
      expected_lower = 0;
      expected_upper = 0;
      check_direct_step(case_idx, "vec", item.lower, item.upper, item.act,
                        expected_lower, expected_upper);
      case_count++;
    end
  endtask

  task automatic run_length_case(input int case_idx, input int length);
    int signed expected_lower;
    int signed expected_upper;
    int idx;
    begin
      tb_p_accum = '0;
      expected_lower = 0;
      expected_upper = 0;
      for (idx = 0; idx < length; idx++) begin
        check_direct_step(case_idx, "length", seq_lower(idx), seq_upper(idx), seq_act(idx),
                          expected_lower, expected_upper);
      end
      case_count++;
    end
  endtask

  task automatic clear_all_unit_inputs;
    begin
      last_clear = 1'b0;
      last_valid = 1'b0;
      last_weight_valid = 1'b0;
      last_inst_valid = 1'b0;
      last_inst = 3'b000;
      last_h_upper_in = '0;
      last_h_lower_in = '0;
      last_v_in = '0;
      last_bcin = '0;
      last_pcin = '0;

      cas_clear = 1'b0;
      cas_valid = 1'b0;
      cas_weight_valid = 1'b0;
      cas_inst_valid = 1'b0;
      cas_inst = 3'b000;
      cas_h_upper_in = '0;
      cas_h_lower_in = '0;
      cas_v_in = '0;
      cas_bcin = '0;
      cas_pcin = '0;

      unit_clear = 1'b0;
      unit_valid = 1'b0;
      unit_weight_valid = 1'b0;
      unit_inst_valid = 1'b0;
      unit_inst = 3'b000;
      unit_h_upper_in = '0;
      unit_h_lower_in = '0;
      unit_v_in = '0;
      unit_bcin = '0;
      unit_pcin = '0;
    end
  endtask

  task automatic reset_units;
    begin
      clear_all_unit_inputs();
      rst_n = 1'b0;
      last_clear = 1'b1;
      cas_clear = 1'b1;
      unit_clear = 1'b1;
      repeat (3) @(posedge clk);
      rst_n = 1'b1;
      @(negedge clk);
      last_clear = 1'b0;
      cas_clear = 1'b0;
      unit_clear = 1'b0;
      repeat (1) @(posedge clk);
    end
  endtask

  task automatic load_last_direct(
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper
  );
    begin
      @(negedge clk);
      last_h_lower_in = lower;
      last_h_upper_in = upper;
      last_weight_valid = 1'b1;
      last_inst = 3'b001;
      last_inst_valid = 1'b1;
      @(negedge clk);
      last_weight_valid = 1'b0;
      last_inst_valid = 1'b0;
    end
  endtask

  task automatic mac_last_direct(
    input logic signed [Int8Bits-1:0] act,
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper,
    inout int signed expected_lower,
    inout int signed expected_upper
  );
    begin
      @(negedge clk);
      last_v_in = act;
      last_valid = 1'b1;
      @(posedge clk);
      expected_lower += i4_to_int(lower) * i8_to_int(act);
      expected_upper += i4_to_int(upper) * i8_to_int(act);
      #1;
      @(negedge clk);
      last_valid = 1'b0;
      last_v_in = '0;
    end
  endtask

  task automatic run_last_direct_case(input int case_idx);
    int signed expected_lower;
    int signed expected_upper;
    logic signed [Int4Bits-1:0] lower;
    logic signed [Int4Bits-1:0] upper;
    begin
      reset_units();
      lower = -4'sd3;
      upper =  4'sd2;
      expected_lower = 0;
      expected_upper = 0;
      load_last_direct(lower, upper);
      mac_last_direct(8'sd5, lower, upper, expected_lower, expected_upper);
      mac_last_direct(-8'sd7, lower, upper, expected_lower, expected_upper);
      check_recovered(case_idx, "last_direct_accum", last_rec_lower, last_rec_upper,
                      expected_lower, expected_upper);

      @(negedge clk);
      last_inst = 3'b000;
      last_inst_valid = 1'b1;
      @(negedge clk);
      last_inst_valid = 1'b0;
      last_v_in = 8'sd55;
      last_valid = 1'b1;
      @(posedge clk);
      #1;
      @(negedge clk);
      last_valid = 1'b0;
      check_recovered(case_idx, "last_direct_hold", last_rec_lower, last_rec_upper,
                      expected_lower, expected_upper);

      @(negedge clk);
      last_clear = 1'b1;
      @(posedge clk);
      #1;
      if (last_result !== '0) begin
        note_error(case_idx, "last_direct_clear", "p",
                   PPortW'(last_result), '0);
      end
      if (last_h_lower_out !== '0) begin
        note_error(case_idx, "last_direct_clear", "h_lower",
                   PPortW'(last_h_lower_out), '0);
      end
      if (last_h_upper_out !== '0) begin
        note_error(case_idx, "last_direct_clear", "h_upper",
                   PPortW'(last_h_upper_out), '0);
      end
      @(negedge clk);
      last_clear = 1'b0;
      case_count++;
    end
  endtask

  task automatic run_last_cascade_case(input int case_idx);
    int signed expected_lower;
    int signed expected_upper;
    logic signed [Int4Bits-1:0] lower;
    logic signed [Int4Bits-1:0] upper;
    logic signed [Int8Bits-1:0] act;
    begin
      reset_units();
      lower =  4'sd4;
      upper = -4'sd5;
      act = -8'sd9;
      expected_lower = 0;
      expected_upper = 0;

      @(negedge clk);
      cas_h_lower_in = lower;
      cas_h_upper_in = upper;
      cas_weight_valid = 1'b1;
      cas_inst = 3'b001;
      cas_inst_valid = 1'b1;
      @(negedge clk);
      cas_weight_valid = 1'b0;
      cas_inst_valid = 1'b0;
      cas_v_in = 8'sd42;
      cas_bcin = golden_b(act);
      cas_valid = 1'b1;
      @(posedge clk);
      expected_lower += i4_to_int(lower) * i8_to_int(act);
      expected_upper += i4_to_int(upper) * i8_to_int(act);
      #1;
      @(negedge clk);
      cas_valid = 1'b0;
      check_recovered(case_idx, "last_row_bcascade", cas_rec_lower, cas_rec_upper,
                      expected_lower, expected_upper);
      case_count++;
    end
  endtask

  task automatic run_unit_flush_case(input int case_idx);
    int signed expected_lower;
    int signed expected_upper;
    logic signed [Int4Bits-1:0] lower;
    logic signed [Int4Bits-1:0] upper;
    logic signed [Int8Bits-1:0] act;
    logic signed [PPortW-1:0] drain_p;
    begin
      reset_units();
      lower = -4'sd6;
      upper =  4'sd3;
      act = 8'sd11;
      expected_lower = 0;
      expected_upper = 0;

      @(negedge clk);
      unit_h_lower_in = lower;
      unit_h_upper_in = upper;
      unit_weight_valid = 1'b1;
      unit_inst = 3'b001;
      unit_inst_valid = 1'b1;
      @(negedge clk);
      unit_weight_valid = 1'b0;
      unit_inst_valid = 1'b0;
      unit_v_in = act;
      unit_valid = 1'b1;
      @(posedge clk);
      expected_lower += i4_to_int(lower) * i8_to_int(act);
      expected_upper += i4_to_int(upper) * i8_to_int(act);
      #1;
      @(negedge clk);
      unit_valid = 1'b0;
      check_recovered(case_idx, "unit_mac", unit_rec_lower, unit_rec_upper,
                      expected_lower, expected_upper);

      @(negedge clk);
      unit_inst = 3'b100;
      unit_inst_valid = 1'b1;
      @(negedge clk);
      unit_inst_valid = 1'b0;
      repeat (4) @(posedge clk);
      #1;
      if (unit_fabric_p !== '0) begin
        note_error(case_idx, "unit_flush", "p",
                   PPortW'(unit_fabric_p), '0);
      end

      drain_p = golden_p(-19, 23);
      @(negedge clk);
      last_pcin = drain_p;
      last_inst = 3'b100;
      last_inst_valid = 1'b1;
      @(negedge clk);
      last_inst_valid = 1'b0;
      repeat (4) @(posedge clk);
      #1;
      if (last_result !== drain_p) begin
        note_error(case_idx, "last_row_flush", "pcin",
                   PPortW'(last_result), PPortW'(drain_p));
      end
      check_recovered(case_idx, "last_row_flush", last_rec_lower, last_rec_upper,
                      -19, 23);
      case_count++;
    end
  endtask

  int idx;
  initial begin
    errors = 0;
    first_fail_case = -1;
    case_count = 0;
    tb_w_lower = '0;
    tb_w_upper = '0;
    tb_act = '0;
    tb_p_accum = '0;
    rst_n = 1'b1;
    clear_all_unit_inputs();

    for (idx = 0; idx < VecCaseCount; idx++) begin
      run_single_vec(idx, vec_cases[idx]);
    end

    for (idx = 0; idx < LenCaseCount; idx++) begin
      run_length_case(VecCaseCount + idx, length_cases[idx]);
    end

    run_last_direct_case(VecCaseCount + LenCaseCount);
    run_last_cascade_case(VecCaseCount + LenCaseCount + 1);
    run_unit_flush_case(VecCaseCount + LenCaseCount + 2);

    if (case_count !== TotalCases) begin
      note_error(TotalCases, "case_count", "count",
                 PPortW'(case_count), PPortW'(TotalCases));
    end

    if (errors == 0) begin
      $display("PASS: %0d cycles, TB_PASSED cases=%0d vec=%0d lengths=%0d unit=%0d",
               case_count, case_count, VecCaseCount, LenCaseCount, UnitCaseCount);
    end else begin
      $display("FAIL: TB_FAILED case=%0d mismatches=%0d cases=%0d",
               first_fail_case, errors, case_count);
    end
    $finish;
  end

  initial begin
    #1000000;
    $display("FAIL: TB_FAILED case=-1 timeout");
    $finish;
  end

endmodule

module DSP48E2 #(
  parameter A_INPUT     = "DIRECT",
  parameter B_INPUT     = "DIRECT",
  parameter AREG        = 1,
  parameter BREG        = 2,
  parameter CREG        = 0,
  parameter MREG        = 1,
  parameter PREG        = 1,
  parameter OPMODEREG   = 1,
  parameter ALUMODEREG  = 1,
  parameter USE_MULT    = "MULTIPLY"
) (
  input  logic CLK,
  input  logic RSTA,
  input  logic RSTB,
  input  logic RSTM,
  input  logic RSTP,
  input  logic RSTCTRL,
  input  logic RSTALLCARRYIN,
  input  logic RSTALUMODE,
  input  logic RSTC,
  input  logic CEA1,
  input  logic CEA2,
  input  logic CEB1,
  input  logic CEB2,
  input  logic CEM,
  input  logic CEP,
  input  logic CECTRL,
  input  logic CEALUMODE,
  input  logic CEC,
  input  logic signed [29:0] A,
  input  logic signed [29:0] ACIN,
  output logic signed [29:0] ACOUT,
  input  logic signed [17:0] B,
  input  logic signed [17:0] BCIN,
  output logic signed [17:0] BCOUT,
  input  logic signed [47:0] C,
  input  logic signed [47:0] PCIN,
  output logic signed [47:0] PCOUT,
  input  logic [8:0] OPMODE,
  input  logic [3:0] ALUMODE,
  output logic signed [47:0] P
);

  logic signed [17:0] selected_b;
  logic signed [47:0] product;

  always_comb begin
    selected_b = (B_INPUT == "CASCADE") ? BCIN : B;
    product = $signed(A) * $signed(selected_b);
  end

  always_ff @(posedge CLK) begin
    if (RSTA) begin
      ACOUT <= '0;
    end else if (CEA2) begin
      ACOUT <= A;
    end

    if (RSTB) begin
      BCOUT <= '0;
    end else if (CEB2) begin
      BCOUT <= selected_b;
    end

    if (RSTP || RSTM || RSTCTRL || RSTALLCARRYIN || RSTALUMODE || RSTC) begin
      P <= '0;
    end else if (CEP) begin
      unique case (OPMODE)
        9'b00_000_00_00: P <= '0;
        9'b00_001_00_00: P <= PCIN;
        9'b00_001_01_01: P <= CEM ? (PCIN + product) : PCIN;
        9'b00_010_01_01: P <= CEM ? (P + product) : P;
        9'b00_011_01_01: P <= CEM ? (C + product) : C;
        default:          P <= P;
      endcase
    end

    PCOUT <= P;
  end

endmodule
