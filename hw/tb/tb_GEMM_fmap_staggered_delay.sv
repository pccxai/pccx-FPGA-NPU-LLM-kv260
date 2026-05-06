`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

// ===============================================================================
// Testbench: tb_GEMM_fmap_staggered_delay
//
// Purpose
// -------
//   Smoke-validates the W4A8 GEMM_fmap_staggered_dispatch contract:
//   column c emits its own signed INT8 fmap/data-valid/instruction stream c
//   clocks after column 0. The scale sideband is represented by the v002.1
//   ACT_SCALE_EMAX_BFP policy chosen in PR #80: the activation bytes are
//   already quantized INT8, while e_max travels in the GEMM_systolic_top
//   SYSTOLIC_TOTAL_LATENCY pipe for later normalization.
//
// Latency table
// -------------
//   Signal group                  RTL/TB-visible source at sample cycle n
//   row_data/row_valid            source cycle n - column
//   row_inst/row_inst_valid       source cycle n - column
//   e_max sideband                source cycle n - (SYSTOLIC_TOTAL_LATENCY - 1)
//
//   Column 0 is registered once internally but has zero relative stagger in the
//   cycle-indexed golden above. Columns 1..N are checked against the same
//   relative diagonal wave-front seen by the systolic array.
// ===============================================================================

module tb_GEMM_fmap_staggered_delay;

  localparam int FmapWidth             = 8;
  localparam int FmapOutWidth          = 8;
  localparam int ArraySize             = 5;
  localparam int SystolicTotalLatency  = `SYSTOLIC_TOTAL_LATENCY;
  localparam int SampleCount           = SystolicTotalLatency + ArraySize + 8;

  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;

  logic [FmapWidth-1:0] fmap_in[0:ArraySize-1];
  logic                 fmap_valid;
  logic [          2:0] global_inst;
  logic                 global_inst_valid;

  logic [FmapOutWidth-1:0] row_data      [0:ArraySize-1];
  logic                    row_valid     [0:ArraySize-1];
  logic [             2:0] row_inst      [0:ArraySize-1];
  logic                    row_inst_valid[0:ArraySize-1];

  logic [`BF16_EXP_WIDTH-1:0] emax_in     [0:ArraySize-1];
  logic [`BF16_EXP_WIDTH-1:0] emax_pipe   [0:ArraySize-1][0:SystolicTotalLatency-1];
  logic [`BF16_EXP_WIDTH-1:0] delayed_emax[0:ArraySize-1];

  GEMM_fmap_staggered_dispatch #(
      .fmap_width    (FmapWidth),
      .array_size    (ArraySize),
      .fmap_out_width(FmapOutWidth)
  ) u_dut (
      .clk              (clk),
      .rst_n            (rst_n),
      .fmap_in          (fmap_in),
      .fmap_valid       (fmap_valid),
      .global_inst      (global_inst),
      .global_inst_valid(global_inst_valid),
      .row_data         (row_data),
      .row_valid        (row_valid),
      .row_inst         (row_inst),
      .row_inst_valid   (row_inst_valid)
  );

  int errors = 0;
  int checks = 0;

  function automatic logic [FmapOutWidth-1:0] sample_data(input int cycle, input int col);
    int sel;
    if (cycle < 0) begin
      sample_data = '0;
    end else begin
      sel = (cycle + (col * 3)) % 16;
      case (sel)
        0:  sample_data = 8'h80; // -128
        1:  sample_data = 8'h81; // -127
        2:  sample_data = 8'hc0; //  -64
        3:  sample_data = 8'hef; //  -17
        4:  sample_data = 8'hff; //   -1
        5:  sample_data = 8'h00;
        6:  sample_data = 8'h01;
        7:  sample_data = 8'h07;
        8:  sample_data = 8'h0f;
        9:  sample_data = 8'h1f;
        10: sample_data = 8'h3f;
        11: sample_data = 8'h40;
        12: sample_data = 8'h55;
        13: sample_data = 8'h6a;
        14: sample_data = 8'h7e;
        default: sample_data = 8'h7f;
      endcase
    end
  endfunction

  function automatic logic sample_valid(input int cycle);
    if (cycle < 0) begin
      sample_valid = 1'b0;
    end else begin
      sample_valid = (cycle % 11 != 3) && (cycle % 17 != 8);
    end
  endfunction

  function automatic logic [2:0] sample_inst(input int cycle);
    if (cycle < 0) begin
      sample_inst = 3'b000;
    end else begin
      sample_inst = cycle[2:0] ^ 3'b101;
    end
  endfunction

  function automatic logic sample_inst_valid(input int cycle);
    if (cycle < 0) begin
      sample_inst_valid = 1'b0;
    end else begin
      sample_inst_valid = (cycle % 13 != 4) && (cycle % 19 != 9);
    end
  endfunction

  function automatic logic [`BF16_EXP_WIDTH-1:0] sample_emax(input int cycle, input int col);
    if (cycle < 0) begin
      sample_emax = '0;
    end else begin
      sample_emax = 8'd118 + ((cycle + (col * 5)) % 10);
    end
  endfunction

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int col = 0; col < ArraySize; col++) begin
        for (int d = 0; d < SystolicTotalLatency; d++) begin
          emax_pipe[col][d] <= '0;
        end
      end
    end else begin
      for (int col = 0; col < ArraySize; col++) begin
        emax_pipe[col][0] <= emax_in[col];
        for (int d = 1; d < SystolicTotalLatency; d++) begin
          emax_pipe[col][d] <= emax_pipe[col][d-1];
        end
      end
    end
  end

  always_comb begin
    for (int col = 0; col < ArraySize; col++) begin
      delayed_emax[col] = emax_pipe[col][SystolicTotalLatency-1];
    end
  end

  task automatic drive_sample(input int cycle);
    begin
      for (int col = 0; col < ArraySize; col++) begin
        fmap_in[col] = sample_data(cycle, col);
        emax_in[col] = sample_emax(cycle, col);
      end
      fmap_valid        = sample_valid(cycle);
      global_inst       = sample_inst(cycle);
      global_inst_valid = sample_inst_valid(cycle);
    end
  endtask

  task automatic expect_col(input int cycle, input int col);
    int src_cycle;
    begin
      src_cycle = cycle - col;
      checks++;

      if (row_data[col] !== sample_data(src_cycle, col)) begin
        errors++;
        $display("[%0t] data mismatch cycle=%0d col=%0d got_raw=0x%02h got_s=%0d exp_raw=0x%02h exp_s=%0d",
                 $time, cycle, col, row_data[col], $signed(row_data[col]),
                 sample_data(src_cycle, col), $signed(sample_data(src_cycle, col)));
      end

      if (row_valid[col] !== sample_valid(src_cycle)) begin
        errors++;
        $display("[%0t] valid mismatch cycle=%0d col=%0d got=%0b exp=%0b",
                 $time, cycle, col, row_valid[col], sample_valid(src_cycle));
      end

      if (row_inst[col] !== sample_inst(src_cycle)) begin
        errors++;
        $display("[%0t] inst mismatch cycle=%0d col=%0d got=%0b exp=%0b",
                 $time, cycle, col, row_inst[col], sample_inst(src_cycle));
      end

      if (row_inst_valid[col] !== sample_inst_valid(src_cycle)) begin
        errors++;
        $display("[%0t] inst_valid mismatch cycle=%0d col=%0d got=%0b exp=%0b",
                 $time, cycle, col, row_inst_valid[col], sample_inst_valid(src_cycle));
      end
    end
  endtask

  task automatic expect_emax(input int cycle, input int col);
    int src_cycle;
    begin
      src_cycle = cycle - (SystolicTotalLatency - 1);
      checks++;

      if (delayed_emax[col] !== sample_emax(src_cycle, col)) begin
        errors++;
        $display("[%0t] emax mismatch cycle=%0d col=%0d got=%0d exp=%0d latency=%0d",
                 $time, cycle, col, delayed_emax[col],
                 sample_emax(src_cycle, col), SystolicTotalLatency);
      end
    end
  endtask

  task automatic expect_reset_zero(input string tag);
    begin
      for (int col = 0; col < ArraySize; col++) begin
        checks++;
        if (row_data[col] !== '0 ||
            row_valid[col] !== 1'b0 ||
            row_inst[col] !== 3'b000 ||
            row_inst_valid[col] !== 1'b0 ||
            delayed_emax[col] !== '0) begin
          errors++;
          $display("[%0t] reset mismatch %s col=%0d data=%0d valid=%0b inst=%0b inst_valid=%0b emax=%0d",
                   $time, tag, col, row_data[col], row_valid[col],
                   row_inst[col], row_inst_valid[col], delayed_emax[col]);
        end
      end
    end
  endtask

  initial begin
    rst_n             = 1'b0;
    fmap_valid        = 1'b0;
    global_inst       = 3'b000;
    global_inst_valid = 1'b0;
    for (int col = 0; col < ArraySize; col++) begin
      fmap_in[col] = '0;
      emax_in[col] = '0;
    end

    repeat (3) @(posedge clk);
    #1;
    expect_reset_zero("initial");

    rst_n = 1'b1;

    for (int cycle = 0; cycle < SampleCount; cycle++) begin
      drive_sample(cycle);
      @(posedge clk);
      #1;
      for (int col = 0; col < ArraySize; col++) begin
        expect_col(cycle, col);
        expect_emax(cycle, col);
      end
    end

    rst_n = 1'b0;
    @(posedge clk);
    #1;
    expect_reset_zero("second");

    if (errors == 0) begin
      $display("PASS: %0d cycles, TB_PASSED columns=%0d checks=%0d golden=sv_functions act_scale_policy=ACT_SCALE_EMAX_BFP systolic_total_latency=%0d.",
               SampleCount, ArraySize, checks, SystolicTotalLatency);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #100000 $display("FAIL: GEMM_fmap_staggered_delay timeout"); $finish;
  end

endmodule
