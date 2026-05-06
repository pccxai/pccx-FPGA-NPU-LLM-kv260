`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_preprocess_bf16_to_int8_golden
// Phase : pccx v002.1 - preprocess activation quantization gate
//
// Purpose
// -------
//   Drives BF16 blocks through the existing 32-element preprocess path and
//   compares its e_max-aligned 27-bit output against an SV golden model.
//   The same golden also projects each aligned lane to the reviewed
//   ACT_SCALE_EMAX_BFP INT8 activation value used by the follow-on #32 RTL
//   migration.
//
//   Claim guard: current RTL still emits 16 x 27-bit aligned mantissas. The
//   INT8 value checked here is a TB-local projection of that RTL-visible
//   stream, not a claim that PREPROCESS RTL has an INT8 output port.
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module tb_preprocess_bf16_to_int8_golden;

  localparam int LanesPerBeat    = 16;
  localparam int LanesPerBlock   = 32;
  localparam int DirectedBlocks  = 7;
  localparam int RandomBlocks    = 64;
  localparam int TotalBlocks     = DirectedBlocks + RandomBlocks;
  localparam int ExpectedBeats   = TotalBlocks * 2;
  localparam int Int8FracShift   = 13;
  localparam int MaxDrainCycles  = 256;

  // ===| Clock + reset |=========================================================
  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;

  // ===| DUT IO |================================================================
  logic [255:0] s_axis_tdata;
  logic         s_axis_tvalid;
  logic         s_axis_tready;
  logic [431:0] m_axis_tdata;
  logic         m_axis_tvalid;
  logic         m_axis_tready;

  preprocess_bf16_fixed_pipeline u_dut (
      .clk          (clk),
      .rst_n        (rst_n),
      .s_axis_tdata (s_axis_tdata),
      .s_axis_tvalid(s_axis_tvalid),
      .s_axis_tready(s_axis_tready),
      .m_axis_tdata (m_axis_tdata),
      .m_axis_tvalid(m_axis_tvalid),
      .m_axis_tready(m_axis_tready)
  );

  // ===| Stimulus and expected data storage |====================================
  logic [15:0] block_words     [0:TotalBlocks-1][0:LanesPerBlock-1];
  logic [ 7:0] expected_emax   [0:TotalBlocks-1];
  logic [431:0] expected_fixed [0:ExpectedBeats-1];
  logic [127:0] expected_int8  [0:ExpectedBeats-1];

  // ===| Golden model |==========================================================
  function automatic logic [7:0] bf16_exp(input logic [15:0] bf);
    return bf[14:7];
  endfunction

  function automatic logic [7:0] golden_emax_block(input int block_idx);
    logic [7:0] max_exp;

    max_exp = 8'd0;
    for (int i = 0; i < LanesPerBlock; i++) begin
      if (bf16_exp(block_words[block_idx][i]) > max_exp) begin
        max_exp = bf16_exp(block_words[block_idx][i]);
      end
    end

    return max_exp;
  endfunction

  function automatic logic [26:0] golden_fixed_lane(
      input logic [15:0] bf,
      input logic [ 7:0] emax
  );
    logic        sign;
    logic [ 7:0] exp;
    logic [ 6:0] mantissa;
    logic [26:0] base_mant;
    logic [ 7:0] delta_e;
    logic [26:0] shifted_mant;

    sign     = bf[15];
    exp      = bf[14:7];
    mantissa = bf[6:0];

    // Bit-identical to preprocess_bf16_fixed_pipeline:
    //   normal   = {8'h01, mantissa, 12'b0}
    //   denormal = {8'h00, mantissa, 12'b0}
    base_mant = (exp == 8'd0) ? {8'h00, mantissa, 12'b0}
                              : {8'h01, mantissa, 12'b0};
    delta_e = emax - exp;

    if (delta_e >= 8'd27) begin
      shifted_mant = 27'd0;
    end else begin
      shifted_mant = base_mant >> delta_e;
    end

    return sign ? (~shifted_mant + 27'd1) : shifted_mant;
  endfunction

  function automatic logic signed [7:0] fixed_to_int8_bfp(input logic [26:0] fixed);
    logic signed [26:0] fixed_s;
    logic [26:0]        abs_fixed;
    int unsigned        magnitude;
    int signed          quantized;

    fixed_s   = $signed(fixed);
    abs_fixed = fixed_s[26] ? 27'd0 : fixed;
    if (fixed_s[26]) begin
      abs_fixed = ~fixed + 27'd1;
    end

    magnitude = abs_fixed >> Int8FracShift;
    if (magnitude > 127) begin
      magnitude = 127;
    end

    quantized = fixed_s[26] ? -int'(magnitude) : int'(magnitude);
    return $signed(quantized[7:0]);
  endfunction

  function automatic logic signed [7:0] golden_int8_lane(
      input logic [15:0] bf,
      input logic [ 7:0] emax
  );
    return fixed_to_int8_bfp(golden_fixed_lane(bf, emax));
  endfunction

  function automatic logic [255:0] pack_half(input int block_idx, input int half);
    logic [255:0] word;
    int           base;

    word = '0;
    base   = half * LanesPerBeat;
    for (int lane = 0; lane < LanesPerBeat; lane++) begin
      word[lane*16 +: 16] = block_words[block_idx][base + lane];
    end

    return word;
  endfunction

  function automatic logic [431:0] golden_fixed_half(
      input int block_idx,
      input int half,
      input logic [7:0] emax
  );
    logic [431:0] word;
    int           base;

    word = '0;
    base   = half * LanesPerBeat;
    for (int lane = 0; lane < LanesPerBeat; lane++) begin
      word[lane*27 +: 27] = golden_fixed_lane(block_words[block_idx][base + lane], emax);
    end

    return word;
  endfunction

  function automatic logic [127:0] golden_int8_half(
      input int block_idx,
      input int half,
      input logic [7:0] emax
  );
    logic [127:0] word;
    int           base;

    word = '0;
    base   = half * LanesPerBeat;
    for (int lane = 0; lane < LanesPerBeat; lane++) begin
      word[lane*8 +: 8] = golden_int8_lane(block_words[block_idx][base + lane], emax);
    end

    return word;
  endfunction

  function automatic logic [127:0] dut_int8_from_fixed(input logic [431:0] fixed_word);
    logic [127:0] word;

    word = '0;
    for (int lane = 0; lane < LanesPerBeat; lane++) begin
      word[lane*8 +: 8] = fixed_to_int8_bfp(fixed_word[lane*27 +: 27]);
    end

    return word;
  endfunction

  task automatic set_block(input int block_idx, input logic [15:0] value);
    for (int lane = 0; lane < LanesPerBlock; lane++) begin
      block_words[block_idx][lane] = value;
    end
  endtask

  task automatic build_vectors;
    int unsigned seed;

    // 0: zero block.
    set_block(0, 16'h0000);

    // 1: mixed signs, same exponent ties for e_max.
    for (int lane = 0; lane < LanesPerBlock; lane++) begin
      block_words[1][lane] = {lane[0], 8'h80, lane[6:0]};
    end

    // 2: INT8 rail values at the selected BFP exponent.
    for (int lane = 0; lane < LanesPerBlock; lane++) begin
      block_words[2][lane] = {lane[0], 8'h8A, 7'h7F};
    end

    // 3: denormal-only block, including signed denormals.
    for (int lane = 0; lane < LanesPerBlock; lane++) begin
      block_words[3][lane] = {lane[0], 8'h00, lane[6:0] ^ 7'h55};
    end

    // 4: NaN/Inf policy block. Current preprocess treats exp=255 as a raw
    // BF16 exponent with no special-case payload handling.
    for (int lane = 0; lane < LanesPerBlock; lane++) begin
      case (lane % 4)
        0: block_words[4][lane] = 16'h7F80;  // +Inf encoding
        1: block_words[4][lane] = 16'hFF80;  // -Inf encoding
        2: block_words[4][lane] = 16'h7FC1;  // qNaN-like payload
        default: block_words[4][lane] = 16'hFFC2;
      endcase
    end

    // 5: one large exponent forces small normal lanes to zero after alignment.
    set_block(5, 16'h3F80);
    block_words[5][0]  = 16'h5F7F;
    block_words[5][17] = 16'hDF7F;

    // 6: phase-order sentinel, unique lane payloads across both 16-lane beats.
    for (int lane = 0; lane < LanesPerBlock; lane++) begin
      block_words[6][lane] = {lane[4], 8'(8'h70 + lane[3:0]), lane[6:0]};
    end

    seed = 32'h33C0FFEE;
    for (int block = DirectedBlocks; block < TotalBlocks; block++) begin
      for (int lane = 0; lane < LanesPerBlock; lane++) begin
        seed = (seed * 32'd1664525) + 32'd1013904223;
        block_words[block][lane][15]   = seed[31];
        block_words[block][lane][14:7] = seed[23:16];
        block_words[block][lane][6:0]  = seed[14:8];
      end
    end
  endtask

  task automatic build_expected;
    logic [7:0] emax;

    for (int block = 0; block < TotalBlocks; block++) begin
      emax = golden_emax_block(block);
      expected_emax[block] = emax;
      expected_fixed[(block * 2) + 0] = golden_fixed_half(block, 0, emax);
      expected_fixed[(block * 2) + 1] = golden_fixed_half(block, 1, emax);
      expected_int8[(block * 2) + 0]  = golden_int8_half(block, 0, emax);
      expected_int8[(block * 2) + 1]  = golden_int8_half(block, 1, emax);
    end
  endtask

  // ===| Scoreboard |============================================================
  int errors     = 0;
  int emax_seen  = 0;
  int beats_seen = 0;
  int cycles     = 0;

  always @(posedge clk) begin
    if (rst_n) begin
      cycles++;
    end

    #1;

    if (rst_n && u_dut.block_valid) begin
      if (emax_seen >= TotalBlocks) begin
        errors++;
        $display("[%0t] unexpected e_max block: got=%h", $time, u_dut.global_emax);
      end else if (u_dut.global_emax !== expected_emax[emax_seen]) begin
        errors++;
        if (errors <= 10) begin
          $display("[%0t] e_max mismatch block=%0d got=%h exp=%h",
                   $time, emax_seen, u_dut.global_emax, expected_emax[emax_seen]);
        end
      end
      emax_seen++;
    end

    if (rst_n && m_axis_tvalid) begin
      if (beats_seen >= ExpectedBeats) begin
        errors++;
        $display("[%0t] unexpected output beat: fixed=%h", $time, m_axis_tdata);
      end else begin
        logic [127:0] got_int8;
        got_int8 = dut_int8_from_fixed(m_axis_tdata);

        if (m_axis_tdata !== expected_fixed[beats_seen]) begin
          errors++;
          if (errors <= 10) begin
            $display("[%0t] fixed mismatch beat=%0d block=%0d half=%0d\n got=%h\n exp=%h",
                     $time, beats_seen, beats_seen / 2, beats_seen % 2,
                     m_axis_tdata, expected_fixed[beats_seen]);
          end
        end

        if (got_int8 !== expected_int8[beats_seen]) begin
          errors++;
          if (errors <= 10) begin
            $display("[%0t] int8 mismatch beat=%0d block=%0d half=%0d got=%h exp=%h",
                     $time, beats_seen, beats_seen / 2, beats_seen % 2,
                     got_int8, expected_int8[beats_seen]);
          end
        end
      end
      beats_seen++;
    end
  end

  // ===| Stimulus |==============================================================
  initial begin
    build_vectors();
    build_expected();

    rst_n          = 1'b0;
    s_axis_tdata   = '0;
    s_axis_tvalid  = 1'b0;
    m_axis_tready  = 1'b1;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    for (int block = 0; block < TotalBlocks; block++) begin
      s_axis_tdata  <= pack_half(block, 0);
      s_axis_tvalid <= 1'b1;
      @(posedge clk);

      s_axis_tdata  <= pack_half(block, 1);
      s_axis_tvalid <= 1'b1;
      @(posedge clk);

      // Exercise phase retention through an input bubble after directed
      // checkpoints while keeping random blocks at one 16-lane beat/cycle.
      if (block == DirectedBlocks - 1) begin
        s_axis_tvalid <= 1'b0;
        s_axis_tdata  <= '0;
        @(posedge clk);
      end
    end

    s_axis_tvalid <= 1'b0;
    s_axis_tdata  <= '0;

    for (int drain = 0; drain < MaxDrainCycles; drain++) begin
      if ((beats_seen >= ExpectedBeats) && (emax_seen >= TotalBlocks)) begin
        break;
      end
      @(posedge clk);
    end

    if (emax_seen != TotalBlocks) begin
      errors++;
      $display("e_max count mismatch: got=%0d exp=%0d", emax_seen, TotalBlocks);
    end
    if (beats_seen != ExpectedBeats) begin
      errors++;
      $display("output beat count mismatch: got=%0d exp=%0d", beats_seen, ExpectedBeats);
    end

    if (errors == 0) begin
      $display("PASS: %0d cycles, BF16->INT8 e_max golden matched; cases=%0d directed=%0d random=%0d golden=sv-function.",
               cycles, TotalBlocks, DirectedBlocks, RandomBlocks);
    end else begin
      $display("FAIL: %0d mismatches over %0d cycles; cases=%0d.",
               errors, cycles, TotalBlocks);
    end
    $finish;
  end

  initial begin
    #100000 $display("TIMEOUT"); $finish;
  end

endmodule
