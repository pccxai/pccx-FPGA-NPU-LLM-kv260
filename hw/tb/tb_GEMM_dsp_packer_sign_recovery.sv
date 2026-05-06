`timescale 1ns / 1ps

// ==============================================================================
// Testbench: tb_GEMM_dsp_packer_sign_recovery
// Phase : pccx v002, Phase A §2.2
//
// Purpose
// -------
//   Directed and deterministic-random verification of the standalone
//   GEMM_dsp_packer + GEMM_sign_recovery pair.
//
//   The harness drives W4A8 operand pairs through the RTL packer, accumulates
//   the packed A*B product in a 48-bit DSP-P stand-in, and compares both the
//   compressed accumulator and recovered lane outputs against SV golden
//   functions. The field map follows issue #35:
//
//     lower_sign = acc[22]
//     upper      = acc[47:23] + acc[22]
//     lower      = acc[22:0]
//
//   No RTL is modified by this TB.
// ==============================================================================

`include "GLOBAL_CONST.svh"

module tb_GEMM_dsp_packer_sign_recovery;

  // ===| Parameters |=============================================================
  localparam int Int4Bits     = `INT4_WIDTH;
  localparam int Int8Bits     = 8;
  localparam int APortW       = `DEVICE_DSP_A_WIDTH;
  localparam int BPortW       = `DEVICE_DSP_B_WIDTH;
  localparam int PPortW       = `DEVICE_DSP_P_WIDTH;
  localparam int UpperShift   = 23;
  localparam int LowerW       = UpperShift;
  localparam int UpperW       = PPortW - UpperShift;
  localparam int GuardBits    = UpperShift - Int4Bits;
  localparam int RandomCycles = 1024;
  localparam int ClockHalfNs  = 2;

  localparam int signed LowerMin = -(1 <<< (LowerW - 1));
  localparam int signed LowerMax =  (1 <<< (LowerW - 1)) - 1;
  localparam int signed UpperMin = -(1 <<< (UpperW - 1));
  localparam int signed UpperMax =  (1 <<< (UpperW - 1)) - 1;

  // ===| Clock |==================================================================
  logic clk;
  initial clk = 1'b0;
  always #ClockHalfNs clk = ~clk;

  // ===| DUT: packer |============================================================
  logic signed [Int4Bits-1:0] w_lower;
  logic signed [Int4Bits-1:0] w_upper;
  logic signed [Int8Bits-1:0] act;
  logic signed [APortW-1:0]   a_packed;
  logic signed [BPortW-1:0]   b_extended;

  GEMM_dsp_packer #(
    .INT4_BITS   (Int4Bits),
    .INT8_BITS   (Int8Bits),
    .A_PORT_W    (APortW),
    .B_PORT_W    (BPortW),
    .UPPER_SHIFT (UpperShift)
  ) u_packer (
    .in_w_lower     (w_lower),
    .in_w_upper     (w_upper),
    .in_act         (act),
    .out_a_packed   (a_packed),
    .out_b_extended (b_extended)
  );

  // ===| DSP-P stand-in + recovery DUT |=========================================
  logic signed [PPortW-1:0] p_accum;
  logic signed [LowerW-1:0] rec_lower;
  logic signed [UpperW-1:0] rec_upper;

  GEMM_sign_recovery #(
    .P_PORT_W    (PPortW),
    .UPPER_SHIFT (UpperShift),
    .LOWER_W     (LowerW),
    .UPPER_W     (UpperW)
  ) u_recovery (
    .in_p_accum    (p_accum),
    .out_lower_sum (rec_lower),
    .out_upper_sum (rec_upper)
  );

  // ===| Scoreboard state |=======================================================
  int errors;
  int case_count;
  int directed_cases;
  int random_cases;
  int manual_cases;
  int overflow_cases;
  int pack_checks;
  int recovery_checks;
  int boundary_checks;
  int overflow_checks;
  int logical_cycles;
  int signed expected_lower_sum;
  int signed expected_upper_sum;

  // ===| Golden model |===========================================================
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

  function automatic logic signed [PPortW-1:0] golden_packed_p(
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

  function automatic logic signed [LowerW-1:0] golden_lower_from_p(
    input logic signed [PPortW-1:0] value
  );
    return value[LowerW-1:0];
  endfunction

  function automatic logic signed [UpperW-1:0] golden_upper_raw_from_p(
    input logic signed [PPortW-1:0] value
  );
    return value[PPortW-1:UpperShift];
  endfunction

  function automatic logic signed [UpperW-1:0] golden_upper_from_p(
    input logic signed [PPortW-1:0] value
  );
    logic signed [UpperW-1:0] upper_raw;
    begin
      upper_raw = golden_upper_raw_from_p(value);
      return value[LowerW-1] ? (upper_raw + UpperW'(1)) : upper_raw;
    end
  endfunction

  function automatic logic signed [UpperW-1:0] expected_upper_raw(
    input int signed upper_sum,
    input logic lower_negative
  );
    return UpperW'(upper_sum - (lower_negative ? 1 : 0));
  endfunction

  function automatic logic signed [PPortW-1:0] dsp_product(
    input logic signed [APortW-1:0] packed_a,
    input logic signed [BPortW-1:0] packed_b
  );
    return $signed(packed_a) * $signed(packed_b);
  endfunction

  function automatic logic sum_fits_width(input int signed value, input int width);
    int signed min_value;
    int signed max_value;
    begin
      min_value = -(1 <<< (width - 1));
      max_value =  (1 <<< (width - 1)) - 1;
      return (value >= min_value) && (value <= max_value);
    end
  endfunction

  function automatic int unsigned lcg_next(input int unsigned state);
    return (state * 32'd1664525) + 32'd1013904223;
  endfunction

  // ===| Check helpers |==========================================================
  task automatic note_error(
    input string label,
    input string field,
    input logic signed [PPortW-1:0] got,
    input logic signed [PPortW-1:0] expected
  );
    begin
      errors++;
      $display("FAIL: TB_FAILED %s %s got=%0d expected=%0d got_hex=%h expected_hex=%h",
               label, field, got, expected, got, expected);
    end
  endtask

  task automatic check_packer(
    input string label,
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper,
    input logic signed [Int8Bits-1:0] value
  );
    logic signed [APortW-1:0] expected_a;
    logic signed [BPortW-1:0] expected_b;
    logic [GuardBits-1:0] expected_guard;
    begin
      expected_a = golden_a(lower, upper);
      expected_b = golden_b(value);
      expected_guard = {GuardBits{lower[Int4Bits-1]}};
      pack_checks++;

      if (a_packed !== expected_a) begin
        note_error(label, "a_packed", PPortW'(a_packed), PPortW'(expected_a));
      end
      if (b_extended !== expected_b) begin
        note_error(label, "b_extended", PPortW'(b_extended), PPortW'(expected_b));
      end
      if (a_packed[Int4Bits-1:0] !== lower) begin
        note_error(label, "a_lower_bits",
                   PPortW'(a_packed[Int4Bits-1:0]), PPortW'(lower));
      end
      if (a_packed[UpperShift-1:Int4Bits] !== expected_guard) begin
        note_error(label, "a_guard_sign",
                   PPortW'(a_packed[UpperShift-1:Int4Bits]), PPortW'(expected_guard));
      end
      if (b_extended[Int8Bits-1:0] !== value) begin
        note_error(label, "b_payload",
                   PPortW'(b_extended[Int8Bits-1:0]), PPortW'(value));
      end
      if (b_extended[BPortW-1:Int8Bits] !== {(BPortW-Int8Bits){value[Int8Bits-1]}}) begin
        note_error(label, "b_sign_extension",
                   PPortW'(b_extended[BPortW-1:Int8Bits]),
                   PPortW'({(BPortW-Int8Bits){value[Int8Bits-1]}}));
      end
    end
  endtask

  task automatic check_recovery(input string label);
    logic signed [PPortW-1:0] expected_p;
    logic signed [LowerW-1:0] expected_lower;
    logic signed [UpperW-1:0] expected_upper;
    logic signed [LowerW-1:0] expected_lower_from_bits;
    logic signed [UpperW-1:0] expected_upper_from_bits;
    logic signed [UpperW-1:0] upper_raw;
    logic signed [UpperW-1:0] raw_expected;
    logic lower_negative;
    begin
      expected_p = golden_packed_p(expected_lower_sum, expected_upper_sum);
      expected_lower = LowerW'(expected_lower_sum);
      expected_upper = UpperW'(expected_upper_sum);
      expected_lower_from_bits = golden_lower_from_p(expected_p);
      expected_upper_from_bits = golden_upper_from_p(expected_p);
      upper_raw = golden_upper_raw_from_p(p_accum);
      lower_negative = expected_p[LowerW-1];
      raw_expected = expected_upper_raw(expected_upper_sum, lower_negative);
      recovery_checks++;

      if (p_accum !== expected_p) begin
        note_error(label, "p_accum", p_accum, expected_p);
      end
      if (rec_lower !== expected_lower) begin
        note_error(label, "rec_lower_math", PPortW'(rec_lower), PPortW'(expected_lower));
      end
      if (rec_upper !== expected_upper) begin
        note_error(label, "rec_upper_math", PPortW'(rec_upper), PPortW'(expected_upper));
      end
      if (rec_lower !== expected_lower_from_bits) begin
        note_error(label, "rec_lower_bits",
                   PPortW'(rec_lower), PPortW'(expected_lower_from_bits));
      end
      if (rec_upper !== expected_upper_from_bits) begin
        note_error(label, "rec_upper_bits",
                   PPortW'(rec_upper), PPortW'(expected_upper_from_bits));
      end
      if (upper_raw !== raw_expected) begin
        note_error(label, "upper_raw_boundary",
                   PPortW'(upper_raw), PPortW'(raw_expected));
      end
      if (lower_negative) begin
        boundary_checks++;
        if ((upper_raw + UpperW'(1)) !== expected_upper) begin
          note_error(label, "lower_sign_plus_one",
                     PPortW'(upper_raw + UpperW'(1)), PPortW'(expected_upper));
        end
      end
    end
  endtask

  task automatic reset_scoreboard;
    begin
      p_accum = '0;
      expected_lower_sum = 0;
      expected_upper_sum = 0;
      w_lower = '0;
      w_upper = '0;
      act = '0;
      #1;
    end
  endtask

  task automatic apply_step(
    input string label,
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper,
    input logic signed [Int8Bits-1:0] value
  );
    begin
      w_lower = lower;
      w_upper = upper;
      act = value;
      #1;
      check_packer(label, lower, upper, value);

      p_accum = p_accum + dsp_product(a_packed, b_extended);
      expected_lower_sum += i4_to_int(lower) * i8_to_int(value);
      expected_upper_sum += i4_to_int(upper) * i8_to_int(value);
      logical_cycles++;
      #1;
      check_recovery(label);
    end
  endtask

  task automatic run_directed_case(
    input string label,
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper,
    input logic signed [Int8Bits-1:0] value
  );
    begin
      reset_scoreboard();
      apply_step(label, lower, upper, value);
      directed_cases++;
      case_count++;
    end
  endtask

  task automatic run_repeat_case(
    input string label,
    input logic signed [Int4Bits-1:0] lower,
    input logic signed [Int4Bits-1:0] upper,
    input logic signed [Int8Bits-1:0] value,
    input int count
  );
    begin
      reset_scoreboard();
      for (int idx = 0; idx < count; idx++) begin
        apply_step(label, lower, upper, value);
      end
      directed_cases++;
      case_count++;
    end
  endtask

  task automatic run_random_case(input string label, input int count, input int unsigned seed);
    int unsigned state;
    logic signed [Int4Bits-1:0] lower;
    logic signed [Int4Bits-1:0] upper;
    logic signed [Int8Bits-1:0] value;
    begin
      reset_scoreboard();
      state = seed;
      for (int idx = 0; idx < count; idx++) begin
        state = lcg_next(state);
        lower = state[Int4Bits-1:0];
        state = lcg_next(state);
        upper = state[Int4Bits-1:0];
        state = lcg_next(state);
        value = state[Int8Bits-1:0];
        apply_step(label, lower, upper, value);
      end
      random_cases++;
      case_count++;
    end
  endtask

  task automatic run_manual_compressed_case(
    input string label,
    input int signed lower_sum,
    input int signed upper_sum,
    input logic expect_lower_sign
  );
    logic signed [PPortW-1:0] expected_p;
    logic signed [UpperW-1:0] raw_expected;
    begin
      reset_scoreboard();
      expected_lower_sum = lower_sum;
      expected_upper_sum = upper_sum;
      expected_p = golden_packed_p(lower_sum, upper_sum);
      raw_expected = expected_upper_raw(upper_sum, expect_lower_sign);
      p_accum = expected_p;
      #1;

      if (p_accum[LowerW-1] !== expect_lower_sign) begin
        note_error(label, "manual_lower_sign",
                   PPortW'(p_accum[LowerW-1]), PPortW'(expect_lower_sign));
      end
      if (golden_upper_raw_from_p(p_accum) !== raw_expected) begin
        note_error(label, "manual_upper_raw",
                   PPortW'(golden_upper_raw_from_p(p_accum)), PPortW'(raw_expected));
      end
      check_recovery(label);
      manual_cases++;
      case_count++;
    end
  endtask

  task automatic run_overflow_guard_case;
    begin
      overflow_checks++;
      if (!sum_fits_width(LowerMax, LowerW)) begin
        note_error("guardband_overflow", "lower_max_fit", PPortW'(0), PPortW'(1));
      end
      if (!sum_fits_width(LowerMin, LowerW)) begin
        note_error("guardband_overflow", "lower_min_fit", PPortW'(0), PPortW'(1));
      end
      if (sum_fits_width(LowerMax + 1, LowerW)) begin
        note_error("guardband_overflow", "lower_positive_overflow_detect",
                   PPortW'(0), PPortW'(1));
      end
      if (sum_fits_width(LowerMin - 1, LowerW)) begin
        note_error("guardband_overflow", "lower_negative_overflow_detect",
                   PPortW'(0), PPortW'(1));
      end
      if (sum_fits_width(UpperMax + 1, UpperW)) begin
        note_error("guardband_overflow", "upper_positive_overflow_detect",
                   PPortW'(0), PPortW'(1));
      end
      if (sum_fits_width(UpperMin - 1, UpperW)) begin
        note_error("guardband_overflow", "upper_negative_overflow_detect",
                   PPortW'(0), PPortW'(1));
      end
      overflow_cases++;
      case_count++;
    end
  endtask

  // ===| Main stimulus |==========================================================
  initial begin
    errors = 0;
    case_count = 0;
    directed_cases = 0;
    random_cases = 0;
    manual_cases = 0;
    overflow_cases = 0;
    pack_checks = 0;
    recovery_checks = 0;
    boundary_checks = 0;
    overflow_checks = 0;
    logical_cycles = 0;
    reset_scoreboard();

    // Positive x positive: proves the literal zero guard field case.
    run_directed_case("pos_pos_zero_guard", 4'sd3, 4'sd2, 8'sd5);

    // Sign-asymmetric one-cycle cases.
    run_directed_case("lower_neg_upper_pos", -4'sd1, 4'sd3, 8'sd1);
    run_directed_case("lower_pos_upper_neg", 4'sd2, -4'sd3, 8'sd7);
    run_directed_case("act_neg_asym", -4'sd4, 4'sd5, -8'sd9);
    run_directed_case("int4_min_int8_min", 4'sh8, 4'sd7, 8'sh80);

    // Carry/borrow through the lower-to-upper pack boundary over many MACs.
    run_repeat_case("carry_through_pack_boundary", -4'sd1, 4'sd3, 8'sd1, 17);

    // Deterministic pseudo-random drain-window coverage with an SV integer model.
    run_random_case("deterministic_random_1024", RandomCycles, 32'h35c0ffee);

    // Direct compressed-encoding boundary probes for acc[22] sign recovery.
    run_manual_compressed_case("compressed_lower_msb_negative",
                               LowerMin, 17, 1'b1);
    run_manual_compressed_case("compressed_lower_msb_positive",
                               LowerMax, -9, 1'b0);

    // Negative test for the TB guard-band detector. The RTL has no overflow port.
    run_overflow_guard_case();

    if (errors == 0) begin
      $display("PASS: %0d cycles, TB_PASSED cases=%0d directed=%0d random=%0d manual=%0d overflow=%0d pack_checks=%0d recovery_checks=%0d boundary_checks=%0d golden=sv-function",
               logical_cycles, case_count, directed_cases, random_cases, manual_cases,
               overflow_cases, pack_checks, recovery_checks, boundary_checks);
    end else begin
      $display("FAIL: TB_FAILED mismatches=%0d cases=%0d pack_checks=%0d recovery_checks=%0d overflow_checks=%0d",
               errors, case_count, pack_checks, recovery_checks, overflow_checks);
    end
    $finish;
  end

  initial begin
    #1000000;
    $display("FAIL: TB_FAILED timeout");
    $finish;
  end

endmodule
