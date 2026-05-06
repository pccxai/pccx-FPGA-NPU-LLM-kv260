`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_mem_l2_uram_mapping_arbitration
// Phase : pccx v002 — MEM_control L2 URAM mapping / arbitration
//
// Purpose
// -------
//   Verifies the RTL-visible contract of `mem_L2_cache_fmap`, the central
//   114,688 x 128-bit L2 scratchpad used by ACP DMA on port A and NPU compute
//   on port B.
//
// Timing assumptions made explicit:
//   1. Port A and port B share one common core clock.
//   2. `xpm_memory_tdpram` READ_LATENCY_A/B is 3 clocks. The TB samples data
//      only after three rising edges from the address drive point.
//   3. WRITE_MODE_A/B is `no_change`. Same-port writes hold that port's
//      previously registered read data; the new contents are checked by a
//      later deterministic read.
//   4. Cross-port same-word collisions where either port writes are undefined
//      for this URAM contract. The TB treats those as forbidden arbitration
//      cases and verifies the guard classification, but does not sample the
//      XPM's device-specific data result for them.
//   5. URAM contents are undefined after reset, so every checked word is
//      explicitly written before it is read.
// ===============================================================================

module tb_mem_l2_uram_mapping_arbitration;

  localparam int L2Depth     = 114688;
  localparam int AddrWidth   = 17;
  localparam int WordBytes   = 16;
  localparam int WordBits    = 128;
  localparam int BankCount   = 8;
  localparam int ReadLatency = 3;
  localparam int UramBits    = 294912;

  localparam int L2Bytes       = L2Depth * WordBytes;
  localparam int L2Bits        = L2Depth * WordBits;
  localparam int ExpectedUrams = (L2Bits + UramBits - 1) / UramBits;

  // ===| Clock + reset |=========================================================
  logic clk_core;
  logic rst_n_core;

  initial clk_core = 1'b0;
  always #2 clk_core = ~clk_core;  // 250 MHz nominal TB clock; ordering only.

  // ===| DUT IO |================================================================
  logic         acp_we;
  logic [16:0]  acp_addr;
  logic [127:0] acp_wdata;
  logic [127:0] acp_rdata;

  logic         npu_we;
  logic [16:0]  npu_addr;
  logic [127:0] npu_wdata;
  logic [127:0] npu_rdata;

  mem_L2_cache_fmap #(
      .Depth(L2Depth)
  ) dut (
      .clk_core     (clk_core),
      .rst_n_core   (rst_n_core),
      .IN_acp_we    (acp_we),
      .IN_acp_addr  (acp_addr),
      .IN_acp_wdata (acp_wdata),
      .OUT_acp_rdata(acp_rdata),
      .IN_npu_we    (npu_we),
      .IN_npu_addr  (npu_addr),
      .IN_npu_wdata (npu_wdata),
      .OUT_npu_rdata(npu_rdata)
  );

  // ===| Scoreboard |============================================================
  int errors = 0;
  int checks = 0;
  int cases  = 0;
  int collision_guards = 0;

  function automatic logic [2:0] golden_bank(input logic [16:0] addr);
    golden_bank = addr[2:0];
  endfunction

  function automatic longint unsigned byte_lo(input logic [16:0] addr);
    byte_lo = longint'(addr) * WordBytes;
  endfunction

  function automatic longint unsigned byte_hi(input logic [16:0] addr);
    byte_hi = byte_lo(addr) + WordBytes - 1;
  endfunction

  function automatic logic [127:0] golden_word(
      input logic [16:0] addr,
      input logic [31:0] salt
  );
    logic [31:0] mix0;
    logic [31:0] mix1;
    logic [31:0] mix2;
    logic [31:0] mix3;
    begin
      mix0 = {15'd0, addr} ^ salt ^ 32'h1357_9bdf;
      mix1 = ({15'd0, addr} << 5) ^ (salt << 1) ^ 32'h2468_ace0;
      mix2 = ({15'd0, addr} * 32'd17) ^ (salt >> 1) ^ 32'h55aa_00ff;
      mix3 = {golden_bank(addr), addr[16:0], addr[4:0], 7'h52} ^ salt ^ 32'ha5a5_5a5a;
      golden_word = {mix3, mix2, mix1, mix0};
    end
  endfunction

  function automatic logic forbidden_cross_port_collision(
      input logic        a_we,
      input logic [16:0] a_addr,
      input logic        b_we,
      input logic [16:0] b_addr
  );
    forbidden_cross_port_collision = (a_addr == b_addr) && (a_we || b_we);
  endfunction

  function automatic logic [16:0] park_addr(input logic [16:0] active_addr);
    park_addr = active_addr ^ 17'd1;
  endfunction

  task automatic fail_msg(input string tag, input string detail);
    begin
      errors++;
      $display("[%0t] mismatch %s: %s", $time, tag, detail);
    end
  endtask

  task automatic expect_int(input string tag, input int got, input int exp);
    begin
      checks++;
      if (got != exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%0d exp=%0d", $time, tag, got, exp);
      end
    end
  endtask

  task automatic expect_longint(
      input string tag,
      input longint unsigned got,
      input longint unsigned exp
  );
    begin
      checks++;
      if (got != exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%0d exp=%0d", $time, tag, got, exp);
      end
    end
  endtask

  task automatic expect_word(
      input string tag,
      input logic [127:0] got,
      input logic [127:0] exp
  );
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%032h exp=%032h",
                 $time, tag, got, exp);
      end
    end
  endtask

  task automatic drive_idle;
    begin
      acp_we    = 1'b0;
      acp_addr  = '0;
      acp_wdata = '0;
      npu_we    = 1'b0;
      npu_addr  = '0;
      npu_wdata = '0;
    end
  endtask

  task automatic write_acp(input logic [16:0] addr, input logic [127:0] data);
    begin
      @(negedge clk_core);
      acp_addr  = addr;
      acp_wdata = data;
      acp_we    = 1'b1;
      npu_we    = 1'b0;
      npu_addr  = park_addr(addr);
      @(posedge clk_core);
      #1;
      acp_we = 1'b0;
      cases++;
    end
  endtask

  task automatic write_npu(input logic [16:0] addr, input logic [127:0] data);
    begin
      @(negedge clk_core);
      acp_we    = 1'b0;
      acp_addr  = park_addr(addr);
      npu_addr  = addr;
      npu_wdata = data;
      npu_we    = 1'b1;
      @(posedge clk_core);
      #1;
      npu_we = 1'b0;
      cases++;
    end
  endtask

  task automatic read_acp_expect(
      input string tag,
      input logic [16:0] addr,
      input logic [127:0] exp
  );
    begin
      @(negedge clk_core);
      acp_we   = 1'b0;
      acp_addr = addr;
      repeat (ReadLatency) @(posedge clk_core);
      #1;
      expect_word(tag, acp_rdata, exp);
      cases++;
    end
  endtask

  task automatic read_npu_expect(
      input string tag,
      input logic [16:0] addr,
      input logic [127:0] exp
  );
    begin
      @(negedge clk_core);
      npu_we   = 1'b0;
      npu_addr = addr;
      repeat (ReadLatency) @(posedge clk_core);
      #1;
      expect_word(tag, npu_rdata, exp);
      cases++;
    end
  endtask

  task automatic read_both_expect(
      input string tag,
      input logic [16:0] a_addr,
      input logic [127:0] a_exp,
      input logic [16:0] b_addr,
      input logic [127:0] b_exp
  );
    begin
      if (forbidden_cross_port_collision(1'b0, a_addr, 1'b0, b_addr)) begin
        fail_msg({tag, " guard"}, "unexpected read-only collision classification");
      end
      @(negedge clk_core);
      acp_we   = 1'b0;
      acp_addr = a_addr;
      npu_we   = 1'b0;
      npu_addr = b_addr;
      repeat (ReadLatency) @(posedge clk_core);
      #1;
      expect_word({tag, " portA"}, acp_rdata, a_exp);
      expect_word({tag, " portB"}, npu_rdata, b_exp);
      cases++;
    end
  endtask

  task automatic write_both_distinct(
      input string tag,
      input logic [16:0] a_addr,
      input logic [127:0] a_data,
      input logic [16:0] b_addr,
      input logic [127:0] b_data
  );
    begin
      if (forbidden_cross_port_collision(1'b1, a_addr, 1'b1, b_addr)) begin
        fail_msg({tag, " guard"}, "same-word write/write collision is forbidden");
      end
      @(negedge clk_core);
      acp_we    = 1'b1;
      acp_addr  = a_addr;
      acp_wdata = a_data;
      npu_we    = 1'b1;
      npu_addr  = b_addr;
      npu_wdata = b_data;
      @(posedge clk_core);
      #1;
      acp_we = 1'b0;
      npu_we = 1'b0;
      cases++;
    end
  endtask

  task automatic write_a_read_b_distinct(
      input string tag,
      input logic [16:0] a_addr,
      input logic [127:0] a_data,
      input logic [16:0] b_addr,
      input logic [127:0] b_exp
  );
    begin
      if (forbidden_cross_port_collision(1'b1, a_addr, 1'b0, b_addr)) begin
        fail_msg({tag, " guard"}, "same-word write/read collision is forbidden");
      end
      @(negedge clk_core);
      acp_we    = 1'b1;
      acp_addr  = a_addr;
      acp_wdata = a_data;
      npu_we    = 1'b0;
      npu_addr  = b_addr;
      @(posedge clk_core);
      #1;
      acp_we = 1'b0;
      repeat (ReadLatency - 1) @(posedge clk_core);
      #1;
      expect_word({tag, " portB read"}, npu_rdata, b_exp);
      cases++;
    end
  endtask

  task automatic check_static_mapping;
    begin
      expect_int("addr_width", AddrWidth, 17);
      expect_int("depth_words", L2Depth, 114688);
      expect_int("word_bits", WordBits, 128);
      expect_longint("capacity_bytes", L2Bytes, 64'd1835008);
      expect_int("expected_uram_blocks", ExpectedUrams, 50);
      expect_longint("last_word_lo_byte", byte_lo(17'(L2Depth - 1)), 64'd1834992);
      expect_longint("last_word_hi_byte", byte_hi(17'(L2Depth - 1)), 64'd1835007);

      for (int bank = 0; bank < BankCount; bank++) begin
        expect_int("logical_bank_decode", golden_bank(17'(bank)), bank);
      end
      cases++;
    end
  endtask

  task automatic check_directed_addresses;
    logic [16:0] addr_vec [0:15];
    logic [127:0] exp;
    begin
      addr_vec[0]  = 17'd0;
      addr_vec[1]  = 17'd1;
      addr_vec[2]  = 17'd7;
      addr_vec[3]  = 17'd8;
      addr_vec[4]  = 17'd9;
      addr_vec[5]  = 17'd1023;
      addr_vec[6]  = 17'd1024;
      addr_vec[7]  = 17'd8191;
      addr_vec[8]  = 17'd8192;
      addr_vec[9]  = 17'(L2Depth / 2);
      addr_vec[10] = 17'(L2Depth - 17);
      addr_vec[11] = 17'(L2Depth - 9);
      addr_vec[12] = 17'(L2Depth - 8);
      addr_vec[13] = 17'(L2Depth - 2);
      addr_vec[14] = 17'(L2Depth - 1);
      addr_vec[15] = 17'd16;

      for (int i = 0; i < 16; i++) begin
        write_acp(addr_vec[i], golden_word(addr_vec[i], 32'h1000_0000 + i));
      end

      for (int i = 0; i < 16; i++) begin
        exp = golden_word(addr_vec[i], 32'h1000_0000 + i);
        read_acp_expect("directed address readback portA", addr_vec[i], exp);
        read_npu_expect("directed address readback portB", addr_vec[i], exp);
      end
    end
  endtask

  task automatic check_dual_port_arbitration;
    logic [16:0] a0;
    logic [16:0] b0;
    logic [16:0] a1;
    logic [16:0] b1;
    logic [127:0] a0_data;
    logic [127:0] b0_data;
    logic [127:0] a1_data;
    logic [127:0] b1_data;
    begin
      a0 = 17'd64;
      b0 = 17'd65;
      a1 = 17'd4096;
      b1 = 17'd4103;
      a0_data = golden_word(a0, 32'ha0a0_0001);
      b0_data = golden_word(b0, 32'hb0b0_0002);
      a1_data = golden_word(a1, 32'ha1a1_0003);
      b1_data = golden_word(b1, 32'hb1b1_0004);

      write_both_distinct("dual write adjacent logical banks", a0, a0_data, b0, b0_data);
      read_both_expect("cross-read dual write result", b0, b0_data, a0, a0_data);

      write_acp(b1, b1_data);
      write_a_read_b_distinct("write A while read B", a1, a1_data, b1, b1_data);
      read_both_expect("post write/read arbitration", a1, a1_data, b1, b1_data);
    end
  endtask

  task automatic check_no_change_write_mode;
    logic [16:0] hold_addr_a;
    logic [16:0] write_addr_a;
    logic [16:0] hold_addr_b;
    logic [16:0] write_addr_b;
    logic [127:0] hold_data_a;
    logic [127:0] write_data_a;
    logic [127:0] hold_data_b;
    logic [127:0] write_data_b;
    begin
      hold_addr_a  = 17'd512;
      write_addr_a = 17'd520;
      hold_addr_b  = 17'd768;
      write_addr_b = 17'd776;

      hold_data_a  = golden_word(hold_addr_a, 32'hca00_0001);
      write_data_a = golden_word(write_addr_a, 32'hca00_0002);
      hold_data_b  = golden_word(hold_addr_b, 32'hcb00_0003);
      write_data_b = golden_word(write_addr_b, 32'hcb00_0004);

      write_acp(hold_addr_a, hold_data_a);
      write_npu(hold_addr_b, hold_data_b);

      read_acp_expect("prime portA no_change output", hold_addr_a, hold_data_a);
      @(negedge clk_core);
      acp_we    = 1'b1;
      acp_addr  = write_addr_a;
      acp_wdata = write_data_a;
      npu_we    = 1'b0;
      npu_addr  = park_addr(write_addr_a);
      @(posedge clk_core);
      #1;
      expect_word("portA no_change holds during write", acp_rdata, hold_data_a);
      acp_we = 1'b0;
      read_acp_expect("portA no_change write committed", write_addr_a, write_data_a);

      read_npu_expect("prime portB no_change output", hold_addr_b, hold_data_b);
      @(negedge clk_core);
      acp_we    = 1'b0;
      acp_addr  = park_addr(write_addr_b);
      npu_we    = 1'b1;
      npu_addr  = write_addr_b;
      npu_wdata = write_data_b;
      @(posedge clk_core);
      #1;
      expect_word("portB no_change holds during write", npu_rdata, hold_data_b);
      npu_we = 1'b0;
      read_npu_expect("portB no_change write committed", write_addr_b, write_data_b);
      cases++;
    end
  endtask

  task automatic check_collision_policy_guard;
    begin
      checks++;
      if (!forbidden_cross_port_collision(1'b1, 17'd900, 1'b0, 17'd900)) begin
        fail_msg("collision guard write/read", "same-word ACP write + NPU read was not forbidden");
      end else begin
        collision_guards++;
      end

      checks++;
      if (!forbidden_cross_port_collision(1'b0, 17'd901, 1'b1, 17'd901)) begin
        fail_msg("collision guard read/write", "same-word ACP read + NPU write was not forbidden");
      end else begin
        collision_guards++;
      end

      checks++;
      if (!forbidden_cross_port_collision(1'b1, 17'd902, 1'b1, 17'd902)) begin
        fail_msg("collision guard write/write", "same-word dual write was not forbidden");
      end else begin
        collision_guards++;
      end

      checks++;
      if (forbidden_cross_port_collision(1'b1, 17'd903, 1'b0, 17'd904)) begin
        fail_msg("collision guard distinct", "distinct-word write/read was incorrectly forbidden");
      end
      cases++;
    end
  endtask

  // ===| Stimulus |==============================================================
  initial begin
    rst_n_core = 1'b0;
    drive_idle();

    repeat (5) @(posedge clk_core);
    rst_n_core = 1'b1;
    repeat (4) @(posedge clk_core);

    check_static_mapping();
    check_directed_addresses();
    check_dual_port_arbitration();
    check_no_change_write_mode();
    check_collision_policy_guard();

    if (errors == 0) begin
      $display("PASS: %0d cases, %0d checks, golden=sv_functions, l2_depth=%0d, uram_expected=%0d, collision_guards=%0d.",
               cases, checks, L2Depth, ExpectedUrams, collision_guards);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #200000 $display("FAIL: mem_L2 URAM mapping/arbitration timeout"); $finish;
  end

endmodule
