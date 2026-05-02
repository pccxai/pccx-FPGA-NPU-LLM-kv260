`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_shape_const_ram
// Phase : pccx v002 — MEM_control shape descriptor RAM
//
// Purpose
// -------
//   Smoke-validates the parameterised shape_const_ram used by mem_dispatcher
//   for MEMSET shape descriptors. The TB checks the hardware contract that
//   matters for the dispatcher migration:
//
//     1. Reset clears every sampled entry to zero.
//     2. Writes commit on the rising clock edge, not before it.
//     3. Reads are combinational after the write clock edge.
//     4. wr_en=0 preserves existing contents.
//     5. A later reset clears previously written entries.
//
//   This does not claim board execution; it is an xsim validation candidate
//   for the shape-RAM bring-up path.
// ===============================================================================

module tb_shape_const_ram;

  import isa_pkg::*;

  // ===| Clock + reset |=========================================================
  logic clk;
  logic rst_n;
  initial clk = 1'b0;
  always #2 clk = ~clk;

  // ===| DUT IO |================================================================
  logic       wr_en;
  logic [5:0] wr_addr;
  shape_xyz_t wr_xyz;

  ptr_addr_t  rd_addr;
  shape_xyz_t rd_xyz;

  shape_const_ram u_dut (
      .clk    (clk),
      .rst_n  (rst_n),
      .wr_en  (wr_en),
      .wr_addr(wr_addr),
      .wr_xyz (wr_xyz),
      .rd_addr(rd_addr),
      .rd_xyz (rd_xyz)
  );

  // ===| Scoreboard |============================================================
  int errors = 0;
  int checks = 0;

  task automatic set_read_addr(input ptr_addr_t addr);
    begin
      rd_addr = addr;
      #1;
    end
  endtask

  task automatic expect_shape(
      input string tag,
      input shape_dim_t exp_x,
      input shape_dim_t exp_y,
      input shape_dim_t exp_z
  );
    begin
      checks++;
      if (rd_xyz.x !== exp_x || rd_xyz.y !== exp_y || rd_xyz.z !== exp_z) begin
        errors++;
        $display("[%0t] mismatch %s: got x=%0d y=%0d z=%0d exp x=%0d y=%0d z=%0d",
                 $time, tag,
                 rd_xyz.x, rd_xyz.y, rd_xyz.z,
                 exp_x, exp_y, exp_z);
      end
    end
  endtask

  task automatic write_shape(
      input ptr_addr_t addr,
      input shape_dim_t x_val,
      input shape_dim_t y_val,
      input shape_dim_t z_val
  );
    begin
      wr_en   = 1'b1;
      wr_addr = addr;
      wr_xyz  = '{z: z_val, y: y_val, x: x_val};
      @(posedge clk);
      #1;
      wr_en = 1'b0;
    end
  endtask

  // ===| Stimulus |==============================================================
  initial begin
    rst_n   = 1'b0;
    wr_en   = 1'b0;
    wr_addr = '0;
    wr_xyz  = '0;
    rd_addr = '0;

    repeat (4) @(posedge clk);
    #1;

    set_read_addr(6'd0);
    expect_shape("reset_addr0", 17'd0, 17'd0, 17'd0);
    set_read_addr(6'd63);
    expect_shape("reset_addr63", 17'd0, 17'd0, 17'd0);

    rst_n = 1'b1;
    @(posedge clk);
    #1;

    set_read_addr(6'd8);
    wr_en   = 1'b1;
    wr_addr = 6'd8;
    wr_xyz  = '{z: 17'd33, y: 17'd22, x: 17'd11};
    #1;
    expect_shape("pre_clock_write_addr8", 17'd0, 17'd0, 17'd0);
    @(posedge clk);
    #1;
    wr_en = 1'b0;
    expect_shape("post_clock_write_addr8", 17'd11, 17'd22, 17'd33);

    wr_addr = 6'd8;
    wr_xyz  = '{z: 17'd333, y: 17'd222, x: 17'd111};
    @(posedge clk);
    #1;
    set_read_addr(6'd8);
    expect_shape("write_disabled_hold_addr8", 17'd11, 17'd22, 17'd33);

    write_shape(6'd0, 17'd1, 17'd2, 17'd3);
    set_read_addr(6'd0);
    expect_shape("write_addr0", 17'd1, 17'd2, 17'd3);

    write_shape(6'd63, 17'd131071, 17'd65535, 17'd4096);
    set_read_addr(6'd63);
    expect_shape("write_addr63", 17'd131071, 17'd65535, 17'd4096);

    write_shape(6'd0, 17'd17, 17'd34, 17'd51);
    set_read_addr(6'd0);
    expect_shape("overwrite_addr0", 17'd17, 17'd34, 17'd51);
    set_read_addr(6'd63);
    expect_shape("addr63_preserved", 17'd131071, 17'd65535, 17'd4096);

    set_read_addr(6'd17);
    expect_shape("unwritten_addr17", 17'd0, 17'd0, 17'd0);

    // Simultaneous read of an existing entry while writing a different entry.
    set_read_addr(6'd63);
    wr_en   = 1'b1;
    wr_addr = 6'd17;
    wr_xyz  = '{z: 17'd777, y: 17'd666, x: 17'd555};
    @(posedge clk);
    #1;
    wr_en = 1'b0;
    expect_shape("read_during_other_write", 17'd131071, 17'd65535, 17'd4096);
    set_read_addr(6'd17);
    expect_shape("write_addr17", 17'd555, 17'd666, 17'd777);

    rst_n = 1'b0;
    repeat (2) @(posedge clk);
    #1;
    set_read_addr(6'd0);
    expect_shape("second_reset_addr0", 17'd0, 17'd0, 17'd0);
    set_read_addr(6'd17);
    expect_shape("second_reset_addr17", 17'd0, 17'd0, 17'd0);
    set_read_addr(6'd63);
    expect_shape("second_reset_addr63", 17'd0, 17'd0, 17'd0);

    if (errors == 0) begin
      $display("PASS: %0d cycles, shape_const_ram matches golden.", checks);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #100000 $display("FAIL: shape_const_ram timeout"); $finish;
  end

endmodule
