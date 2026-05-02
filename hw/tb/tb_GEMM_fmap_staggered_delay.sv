`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_GEMM_fmap_staggered_delay
//
// Purpose
// -------
//   Smoke-validates the current GEMM_fmap_staggered_dispatch contract:
//   column c emits its own fmap/data-valid/instruction stream c clocks after
//   column 0. This is a focused xsim harness for the active RTL interface; it
//   does not change arithmetic behavior or claim full systolic integration.
// ===============================================================================

module tb_GEMM_fmap_staggered_delay;

  localparam int FmapWidth    = 13;
  localparam int FmapOutWidth = 13;
  localparam int ArraySize    = 5;
  localparam int SampleCount  = ArraySize + 6;

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
    if (cycle < 0) begin
      sample_data = '0;
    end else begin
      sample_data = ((cycle + 1) * 16) + col;
    end
  endfunction

  function automatic logic sample_valid(input int cycle);
    if (cycle < 0) begin
      sample_valid = 1'b0;
    end else begin
      sample_valid = (cycle != 2) && (cycle != 7);
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
      sample_inst_valid = (cycle != 3) && (cycle != 8);
    end
  endfunction

  task automatic drive_sample(input int cycle);
    begin
      for (int col = 0; col < ArraySize; col++) begin
        fmap_in[col] = sample_data(cycle, col);
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
        $display("[%0t] data mismatch cycle=%0d col=%0d got=%0d exp=%0d",
                 $time, cycle, col, row_data[col], sample_data(src_cycle, col));
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

  task automatic expect_reset_zero(input string tag);
    begin
      for (int col = 0; col < ArraySize; col++) begin
        checks++;
        if (row_data[col] !== '0 ||
            row_valid[col] !== 1'b0 ||
            row_inst[col] !== 3'b000 ||
            row_inst_valid[col] !== 1'b0) begin
          errors++;
          $display("[%0t] reset mismatch %s col=%0d data=%0d valid=%0b inst=%0b inst_valid=%0b",
                   $time, tag, col, row_data[col], row_valid[col],
                   row_inst[col], row_inst_valid[col]);
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
      end
    end

    rst_n = 1'b0;
    @(posedge clk);
    #1;
    expect_reset_zero("second");

    if (errors == 0) begin
      $display("PASS: %0d checks, fmap staggered delay matches golden.", checks);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #100000 $display("FAIL: GEMM_fmap_staggered_delay timeout"); $finish;
  end

endmodule
