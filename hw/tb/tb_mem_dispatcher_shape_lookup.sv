`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_mem_dispatcher_shape_lookup
// Phase : pccx v002 - MEM_control dispatcher MEMCPY + shape lookup
//
// Purpose
// -------
//   Validates the public mem_dispatcher MEMSET -> MEMCPY path with a
//   SystemVerilog golden model:
//
//     - shape_ptr_addr based length decode through shape_const_ram
//     - Host -> L2 ingress over S_AXIS_ACP_FMAP
//     - L2 -> Host egress over M_AXIS_ACP_RESULT
//     - multi-tile sequencing with split stream traffic
//
//   The key regression surfaces are bit-exact data preservation through L2 and
//   pointer selection: a LOAD must use its own shape_ptr_addr for the command
//   generated on that clock, not a stale shape address left behind by a previous
//   MEMSET or LOAD.
// ===============================================================================

module tb_mem_dispatcher_shape_lookup;

  import isa_pkg::*;

  // ===| Clocks + reset |=======================================================
  logic clk_core;
  logic clk_axi;
  logic rst_n_core;
  logic rst_axi_n;

  initial clk_core = 1'b0;
  initial clk_axi  = 1'b0;
  always #2 clk_core = ~clk_core;
  always #3 clk_axi  = ~clk_axi;

  // ===| DUT IO |===============================================================
  axis_if #(.DATA_WIDTH(128)) s_axis_acp_fmap ();
  axis_if #(.DATA_WIDTH(128)) m_axis_acp_result ();

  memory_control_uop_t load_uop;
  memory_set_uop_t     mem_set_uop;
  cvo_control_uop_t    cvo_uop;
  logic                cvo_uop_valid;

  logic [15:0] cvo_data;
  logic        cvo_valid;
  logic        cvo_data_ready;
  logic [15:0] cvo_result;
  logic        cvo_result_valid;
  logic        cvo_result_ready;
  logic        fifo_full;
  logic        cvo_busy;

  mem_dispatcher dut (
      .clk_core            (clk_core),
      .rst_n_core          (rst_n_core),
      .clk_axi             (clk_axi),
      .rst_axi_n           (rst_axi_n),
      .S_AXIS_ACP_FMAP     (s_axis_acp_fmap),
      .M_AXIS_ACP_RESULT   (m_axis_acp_result),
      .IN_LOAD_uop         (load_uop),
      .IN_mem_set_uop      (mem_set_uop),
      .IN_CVO_uop          (cvo_uop),
      .IN_cvo_uop_valid    (cvo_uop_valid),
      .OUT_cvo_data        (cvo_data),
      .OUT_cvo_valid       (cvo_valid),
      .IN_cvo_data_ready   (cvo_data_ready),
      .IN_cvo_result       (cvo_result),
      .IN_cvo_result_valid (cvo_result_valid),
      .OUT_cvo_result_ready(cvo_result_ready),
      .OUT_fifo_full       (fifo_full),
      .OUT_cvo_busy        (cvo_busy)
  );

  // ===| Test vectors |=========================================================
  localparam ptr_addr_t Tile0ShapePtr = 6'd3;
  localparam ptr_addr_t Tile1ShapePtr = 6'd9;
  localparam ptr_addr_t Tile2ShapePtr = 6'd12;

  localparam dest_addr_t Tile0Base = 17'd100;
  localparam dest_addr_t Tile1Base = 17'd220;
  localparam dest_addr_t Tile2Base = 17'd360;

  localparam int unsigned Tile0Id = 0;
  localparam int unsigned Tile1Id = 1;
  localparam int unsigned Tile2Id = 2;

  // ===| Scoreboard |===========================================================
  int errors = 0;
  int checks = 0;

  function automatic logic [16:0] golden_word_total(
      input int unsigned x_val,
      input int unsigned y_val,
      input int unsigned z_val
  );
    int unsigned elems;
    begin
      elems = x_val * y_val * z_val;
      golden_word_total = 17'((elems + 7) >> 3);
    end
  endfunction

  function automatic logic [127:0] golden_tile_word(
      input int unsigned tile_id,
      input int unsigned word_idx
  );
    logic [127:0] word;
    begin
      word[ 15:  0] = 16'(16'h1000 + (tile_id << 8) + word_idx);
      word[ 31: 16] = 16'(16'h2100 + (tile_id << 7) + (word_idx * 3));
      word[ 47: 32] = 16'(16'h3200 ^ (tile_id << 5) ^ word_idx);
      word[ 63: 48] = 16'(16'h4300 + (word_idx << 1));
      word[ 79: 64] = 16'(16'h5400 ^ (tile_id << 9) ^ (word_idx * 5));
      word[ 95: 80] = 16'(16'h6500 + tile_id + (word_idx << 2));
      word[111: 96] = 16'(16'h7600 ^ (tile_id << 3) ^ (word_idx * 7));
      word[127:112] = 16'(16'h8700 + (tile_id << 4) + word_idx);
      golden_tile_word = word;
    end
  endfunction

  function automatic logic [16:0] golden_acp_base(
      input data_route_e route,
      input dest_addr_t dest_addr,
      input src_addr_t src_addr
  );
    begin
      if (route == from_host_to_L2) begin
        golden_acp_base = dest_addr;
      end else begin
        golden_acp_base = src_addr;
      end
    end
  endfunction

  task automatic check_bit(
      input string tag,
      input logic got,
      input logic exp
  );
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%0b exp=%0b", $time, tag, got, exp);
      end
    end
  endtask

  task automatic check_17(
      input string tag,
      input logic [16:0] got,
      input logic [16:0] exp
  );
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=%0d exp=%0d", $time, tag, got, exp);
      end
    end
  endtask

  task automatic check_128(
      input string tag,
      input logic [127:0] got,
      input logic [127:0] exp
  );
    begin
      checks++;
      if (got !== exp) begin
        errors++;
        $display("[%0t] mismatch %s: got=0x%032h exp=0x%032h",
                 $time, tag, got, exp);
      end
    end
  endtask

  task automatic set_load_idle;
    begin
      load_uop = '{
          data_dest      : from_L2_to_CVO,
          dest_addr      : '0,
          src_addr       : '0,
          shape_ptr_addr : '0,
          async          : SYNC_OP
      };
    end
  endtask

  task automatic set_memset_idle;
    begin
      mem_set_uop = '{
          dest_cache : data_to_weight_shape,
          dest_addr  : 6'd63,
          a_value    : 16'd0,
          b_value    : 16'd0,
          c_value    : 16'd0
      };
    end
  endtask

  task automatic expect_fmap_shape(
      input string tag,
      input ptr_addr_t addr,
      input a_value_t exp_x,
      input b_value_t exp_y,
      input c_value_t exp_z
  );
    begin
      checks++;
      if (dut.u_fmap_shape.mem[addr].x !== 17'(exp_x) ||
          dut.u_fmap_shape.mem[addr].y !== 17'(exp_y) ||
          dut.u_fmap_shape.mem[addr].z !== 17'(exp_z)) begin
        errors++;
        $display("[%0t] mismatch %s: fmap[%0d] got x=%0d y=%0d z=%0d exp x=%0d y=%0d z=%0d",
                 $time, tag, addr,
                 dut.u_fmap_shape.mem[addr].x,
                 dut.u_fmap_shape.mem[addr].y,
                 dut.u_fmap_shape.mem[addr].z,
                 exp_x, exp_y, exp_z);
      end
    end
  endtask

  task automatic program_fmap_shape(
      input ptr_addr_t addr,
      input a_value_t x_val,
      input b_value_t y_val,
      input c_value_t z_val
  );
    begin
      set_load_idle();
      mem_set_uop = '{
          dest_cache : data_to_fmap_shape,
          dest_addr  : addr,
          a_value    : x_val,
          b_value    : y_val,
          c_value    : z_val
      };
      repeat (2) @(posedge clk_core);
      set_memset_idle();
      @(posedge clk_core);
      #1;
      expect_fmap_shape("MEMSET fmap write", addr, x_val, y_val, z_val);
    end
  endtask

  task automatic issue_acp_load(
      input string tag,
      input data_route_e route,
      input ptr_addr_t shape_ptr,
      input dest_addr_t dest_addr,
      input src_addr_t src_addr,
      input async_e async_mode,
      input logic exp_write_en,
      input logic [16:0] exp_words
  );
    logic [16:0] exp_base;
    logic [16:0] exp_end;
    begin
      exp_base = golden_acp_base(route, dest_addr, src_addr);
      exp_end  = exp_base + exp_words;
      set_memset_idle();
      load_uop = '{
          data_dest      : route,
          dest_addr      : dest_addr,
          src_addr       : src_addr,
          shape_ptr_addr : shape_ptr,
          async          : async_mode
      };
      @(posedge clk_core);
      #1;

      check_bit({tag, " acp_rx_start"}, dut.acp_rx_start, 1'b1);
      check_bit({tag, " acp_write_en"}, dut.acp_uop.write_en, exp_write_en);
      check_17 ({tag, " acp_base"},     dut.acp_uop.base_addr, exp_base);
      check_17 ({tag, " acp_end"},      dut.acp_uop.end_addr,  exp_end);

      set_load_idle();
    end
  endtask

  task automatic issue_npu_load(
      input string tag,
      input data_route_e route,
      input ptr_addr_t shape_ptr,
      input src_addr_t src_addr,
      input logic [16:0] exp_words
  );
    logic [16:0] exp_end;
    begin
      exp_end = src_addr + exp_words;
      set_memset_idle();
      load_uop = '{
          data_dest      : route,
          dest_addr      : '0,
          src_addr       : src_addr,
          shape_ptr_addr : shape_ptr,
          async          : SYNC_OP
      };
      @(posedge clk_core);
      #1;

      check_bit({tag, " npu_rx_start"}, dut.npu_rx_start, 1'b1);
      check_bit({tag, " npu_write_en"}, dut.npu_uop.write_en, 1'b0);
      check_17 ({tag, " npu_base"},     dut.npu_uop.base_addr, src_addr);
      check_17 ({tag, " npu_end"},      dut.npu_uop.end_addr,  exp_end);

      set_load_idle();
      @(posedge clk_core);
      #1;
      check_bit({tag, " npu_rx_start clears"}, dut.npu_rx_start, 1'b0);
    end
  endtask

  task automatic wait_acp_started(input string tag);
    int guard;
    begin
      guard = 0;
      while (dut.u_l2_cache.acp_is_busy !== 1'b1 && guard < 1000) begin
        @(posedge clk_core);
        guard++;
      end
      #1;
      checks++;
      if (dut.u_l2_cache.acp_is_busy !== 1'b1) begin
        errors++;
        $display("[%0t] mismatch %s: ACP transfer did not start", $time, tag);
      end
    end
  endtask

  task automatic wait_acp_done(input string tag);
    int guard;
    begin
      guard = 0;
      while (dut.u_l2_cache.acp_is_busy === 1'b1 && guard < 4000) begin
        @(posedge clk_core);
        guard++;
      end
      #1;
      checks++;
      if (dut.u_l2_cache.acp_is_busy !== 1'b0) begin
        errors++;
        $display("[%0t] mismatch %s: ACP transfer did not complete", $time, tag);
      end
    end
  endtask

  task automatic wait_acp_idle(input string tag);
    int guard;
    begin
      guard = 0;
      while (dut.u_l2_cache.acp_is_busy !== 1'b0 && guard < 4000) begin
        @(posedge clk_core);
        guard++;
      end
      #1;
      checks++;
      if (dut.u_l2_cache.acp_is_busy !== 1'b0) begin
        errors++;
        $display("[%0t] mismatch %s: ACP did not return idle", $time, tag);
      end
    end
  endtask

  task automatic wait_npu_done(input string tag);
    int guard;
    begin
      guard = 0;
      while (dut.u_l2_cache.npu_is_busy !== 1'b1 && guard < 1000) begin
        @(posedge clk_core);
        guard++;
      end
      checks++;
      if (dut.u_l2_cache.npu_is_busy !== 1'b1) begin
        errors++;
        $display("[%0t] mismatch %s: NPU transfer did not start", $time, tag);
      end

      guard = 0;
      while (dut.u_l2_cache.npu_is_busy === 1'b1 && guard < 4000) begin
        @(posedge clk_core);
        guard++;
      end
      checks++;
      if (dut.u_l2_cache.npu_is_busy !== 1'b0) begin
        errors++;
        $display("[%0t] mismatch %s: NPU transfer did not complete", $time, tag);
      end
    end
  endtask

  task automatic drive_host_tile(
      input string tag,
      input int unsigned tile_id,
      input int unsigned word_count,
      input int split_after
  );
    int unsigned sent;
    int guard;
    begin
      sent = 0;
      while (sent < word_count) begin
        @(negedge clk_axi);
        s_axis_acp_fmap.tdata  = golden_tile_word(tile_id, sent);
        s_axis_acp_fmap.tvalid = 1'b1;
        s_axis_acp_fmap.tlast  = (sent == word_count - 1);
        s_axis_acp_fmap.tkeep  = '1;

        guard = 0;
        do begin
          @(posedge clk_axi);
          guard++;
        end while (s_axis_acp_fmap.tready !== 1'b1 && guard < 1000);
        checks++;
        if (s_axis_acp_fmap.tready !== 1'b1) begin
          errors++;
          $display("[%0t] mismatch %s: host ingress tready timeout at word %0d",
                   $time, tag, sent);
        end

        sent++;

        if (split_after >= 0 && sent == int'(split_after)) begin
          @(negedge clk_axi);
          s_axis_acp_fmap.tvalid = 1'b0;
          s_axis_acp_fmap.tlast  = 1'b0;
          s_axis_acp_fmap.tdata  = '0;
          repeat (5) @(posedge clk_axi);
        end
      end

      @(negedge clk_axi);
      s_axis_acp_fmap.tvalid = 1'b0;
      s_axis_acp_fmap.tlast  = 1'b0;
      s_axis_acp_fmap.tdata  = '0;
      @(posedge clk_axi);
    end
  endtask

  task automatic capture_host_tile(
      input string tag,
      input int unsigned tile_id,
      input int unsigned word_count
  );
    int unsigned got;
    int guard;
    begin
      got = 0;
      guard = 0;
      m_axis_acp_result.tready = 1'b1;

      while (got < word_count && guard < 8000) begin
        @(posedge clk_axi);
        guard++;
        if (m_axis_acp_result.tvalid && m_axis_acp_result.tready) begin
          check_128({tag, " egress word"}, m_axis_acp_result.tdata,
                    golden_tile_word(tile_id, got));
          got++;
        end
      end

      m_axis_acp_result.tready = 1'b1;
      checks++;
      if (got != word_count) begin
        errors++;
        $display("[%0t] mismatch %s: captured %0d/%0d result words",
                 $time, tag, got, word_count);
      end

      guard = 0;
      while (m_axis_acp_result.tvalid === 1'b1 && guard < 1000) begin
        @(posedge clk_axi);
        guard++;
      end
    end
  endtask

  task automatic host_to_l2_tile(
      input string tag,
      input int unsigned tile_id,
      input ptr_addr_t shape_ptr,
      input dest_addr_t base_addr,
      input logic [16:0] word_count,
      input async_e async_mode,
      input int split_after
  );
    begin
      wait_acp_idle({tag, " pre"});
      fork
        begin
          issue_acp_load(tag, from_host_to_L2, shape_ptr, base_addr, '0,
                         async_mode, 1'b1, word_count);
          wait_acp_started({tag, " host_to_L2"});
          wait_acp_done({tag, " host_to_L2"});
        end
        begin
          drive_host_tile(tag, tile_id, word_count, split_after);
        end
      join
    end
  endtask

  task automatic l2_to_host_tile(
      input string tag,
      input int unsigned tile_id,
      input ptr_addr_t shape_ptr,
      input src_addr_t base_addr,
      input logic [16:0] word_count,
      input async_e async_mode
  );
    begin
      wait_acp_idle({tag, " pre"});
      fork
        begin
          issue_acp_load(tag, from_L2_to_host, shape_ptr, '0, base_addr,
                         async_mode, 1'b0, word_count);
          wait_acp_started({tag, " L2_to_host"});
          wait_acp_done({tag, " L2_to_host"});
        end
        begin
          capture_host_tile(tag, tile_id, word_count);
        end
      join
    end
  endtask

  // ===| Stimulus |=============================================================
  initial begin
    logic [16:0] tile0_words;
    logic [16:0] tile1_words;
    logic [16:0] tile2_words;

    rst_n_core = 1'b0;
    rst_axi_n  = 1'b0;

    set_load_idle();
    set_memset_idle();
    cvo_uop             = '0;
    cvo_uop_valid       = 1'b0;
    cvo_data_ready      = 1'b1;
    cvo_result          = '0;
    cvo_result_valid    = 1'b0;
    s_axis_acp_fmap.tdata  = '0;
    s_axis_acp_fmap.tvalid = 1'b0;
    s_axis_acp_fmap.tlast  = 1'b0;
    s_axis_acp_fmap.tkeep  = '1;
    m_axis_acp_result.tready = 1'b1;

    repeat (4) @(posedge clk_core);
    rst_n_core = 1'b1;
    rst_axi_n  = 1'b1;
    repeat (8) @(posedge clk_core);
    #1;

    // Shape golden:
    //   tile0: 75 BF16 elems -> 10 128-bit words
    //   tile1: 64 BF16 elems ->  8 128-bit words
    //   tile2: 30 BF16 elems ->  4 128-bit words (ceil edge / async route)
    tile0_words = golden_word_total(3, 5, 5);
    tile1_words = golden_word_total(4, 4, 4);
    tile2_words = golden_word_total(2, 3, 5);

    program_fmap_shape(Tile0ShapePtr, 16'd3, 16'd5, 16'd5);
    program_fmap_shape(Tile1ShapePtr, 16'd4, 16'd4, 16'd4);
    program_fmap_shape(Tile2ShapePtr, 16'd2, 16'd3, 16'd5);

    host_to_l2_tile("tile0 host_to_L2", Tile0Id, Tile0ShapePtr,
                    Tile0Base, tile0_words, SYNC_OP, 1);
    host_to_l2_tile("tile1 host_to_L2 split", Tile1Id, Tile1ShapePtr,
                    Tile1Base, tile1_words, SYNC_OP, 3);
    host_to_l2_tile("tile2 async host_to_L2 ceil", Tile2Id, Tile2ShapePtr,
                    Tile2Base, tile2_words, ASYNC_OP, 1);

    l2_to_host_tile("tile0 L2_to_host", Tile0Id, Tile0ShapePtr,
                    Tile0Base, tile0_words, SYNC_OP);
    l2_to_host_tile("tile1 L2_to_host", Tile1Id, Tile1ShapePtr,
                    Tile1Base, tile1_words, SYNC_OP);
    issue_npu_load("L2_to_L1_GEMM tile0 shape", from_L2_to_L1_GEMM,
                   Tile0ShapePtr, Tile0Base, tile0_words);
    wait_npu_done("L2_to_L1_GEMM tile0 shape");

    issue_npu_load("L2_to_L1_GEMV tile1 shape", from_L2_to_L1_GEMV,
                   Tile1ShapePtr, Tile1Base, tile1_words);
    wait_npu_done("L2_to_L1_GEMV tile1 shape");

    if (errors == 0) begin
      $display("PASS: %0d checks, mem_dispatcher MEMCPY paths match SV golden.", checks);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #200000 $display("FAIL: mem_dispatcher comprehensive timeout"); $finish;
  end

endmodule
