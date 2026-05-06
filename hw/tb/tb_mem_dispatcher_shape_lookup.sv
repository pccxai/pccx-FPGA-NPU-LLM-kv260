`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_mem_dispatcher_shape_lookup
// Phase : pccx v002 - MEM_control dispatcher shape lookup
//
// Purpose
// -------
//   Smoke-validates the higher-level MEMSET -> LOAD path that consumes
//   shape_const_ram inside mem_dispatcher. The TB writes fmap shape entries via
//   the public IN_mem_set_uop port, then issues ACP and NPU LOAD routes and
//   checks the generated command bounds against a small golden model.
//
//   The key regression surface is pointer selection: a LOAD must use its own
//   shape_ptr_addr for the command generated on that clock, not a stale
//   shape address left behind by a previous MEMSET or LOAD.
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
  memory_control_uop_t store_uop;
  logic                store_uop_valid;
  memory_set_uop_t     mem_set_uop;
  cvo_control_uop_t    cvo_uop;
  logic                cvo_uop_valid;

  logic [15:0] cvo_data;
  logic        cvo_valid;
  logic        cvo_data_ready;
  logic [15:0] cvo_result;
  logic        cvo_result_valid;
  logic        cvo_result_ready;
  logic [127:0] gemm_result_data;
  logic        gemm_result_valid;
  logic        gemm_result_ready;
  logic        fifo_full;
  logic        cvo_busy;
  logic        store_busy;
  logic        store_done;
  logic        tb_force_direct_en;
  logic        tb_force_direct_we;
  logic [16:0] tb_force_direct_addr;
  logic [127:0] tb_force_direct_wdata;

  mem_dispatcher dut (
      .clk_core            (clk_core),
      .rst_n_core          (rst_n_core),
      .clk_axi             (clk_axi),
      .rst_axi_n           (rst_axi_n),
      .S_AXIS_ACP_FMAP     (s_axis_acp_fmap),
      .M_AXIS_ACP_RESULT   (m_axis_acp_result),
      .IN_LOAD_uop         (load_uop),
      .IN_STORE_uop        (store_uop),
      .IN_store_uop_valid  (store_uop_valid),
      .IN_mem_set_uop      (mem_set_uop),
      .IN_CVO_uop          (cvo_uop),
      .IN_cvo_uop_valid    (cvo_uop_valid),
      .OUT_cvo_data        (cvo_data),
      .OUT_cvo_valid       (cvo_valid),
      .IN_cvo_data_ready   (cvo_data_ready),
      .IN_cvo_result       (cvo_result),
      .IN_cvo_result_valid (cvo_result_valid),
      .OUT_cvo_result_ready(cvo_result_ready),
      .IN_gemm_result_data (gemm_result_data),
      .IN_gemm_result_valid(gemm_result_valid),
      .OUT_gemm_result_ready(gemm_result_ready),
      .OUT_fifo_full       (fifo_full),
      .OUT_cvo_busy        (cvo_busy),
      .OUT_store_busy      (store_busy),
      .OUT_store_done      (store_done)
  );

  // ===| Scoreboard |===========================================================
  localparam int StoreBeats = 4;

  int errors = 0;
  int checks = 0;
  logic [127:0] store_expected [0:StoreBeats-1];

  function automatic logic [16:0] ceil_words(
      input int unsigned x_val,
      input int unsigned y_val,
      input int unsigned z_val
  );
    int unsigned elems;
    begin
      elems = x_val * y_val * z_val;
      ceil_words = 17'((elems + 7) >> 3);
    end
  endfunction

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

  task automatic set_store_idle;
    begin
      store_uop = '{
          data_dest      : from_L2_to_CVO,
          dest_addr      : '0,
          src_addr       : '0,
          shape_ptr_addr : '0,
          async          : SYNC_OP
      };
      store_uop_valid = 1'b0;
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
      input logic exp_write_en,
      input logic [16:0] exp_base,
      input logic [16:0] exp_words
  );
    logic [16:0] exp_end;
    begin
      exp_end = exp_base + exp_words;
      set_memset_idle();
      load_uop = '{
          data_dest      : route,
          dest_addr      : dest_addr,
          src_addr       : src_addr,
          shape_ptr_addr : shape_ptr,
          async          : SYNC_OP
      };
      @(posedge clk_core);
      #1;

      checks++;
      if (dut.acp_rx_start !== 1'b1) begin
        errors++;
        $display("[%0t] mismatch %s: acp_rx_start=%0b exp 1",
                 $time, tag, dut.acp_rx_start);
      end

      checks++;
      if (dut.acp_uop.write_en !== exp_write_en ||
          dut.acp_uop.base_addr !== exp_base ||
          dut.acp_uop.end_addr  !== exp_end) begin
        errors++;
        $display("[%0t] mismatch %s: acp cmd got we=%0b base=%0d end=%0d exp we=%0b base=%0d end=%0d",
                 $time, tag,
                 dut.acp_uop.write_en, dut.acp_uop.base_addr, dut.acp_uop.end_addr,
                 exp_write_en, exp_base, exp_end);
      end

      set_load_idle();
      @(posedge clk_core);
      #1;
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

      checks++;
      if (dut.npu_rx_start !== 1'b1) begin
        errors++;
        $display("[%0t] mismatch %s: npu_rx_start=%0b exp 1",
                 $time, tag, dut.npu_rx_start);
      end

      checks++;
      if (dut.npu_uop.write_en !== 1'b0 ||
          dut.npu_uop.base_addr !== src_addr ||
          dut.npu_uop.end_addr  !== exp_end) begin
        errors++;
        $display("[%0t] mismatch %s: npu cmd got we=%0b base=%0d end=%0d exp we=0 base=%0d end=%0d",
                 $time, tag,
                 dut.npu_uop.write_en, dut.npu_uop.base_addr, dut.npu_uop.end_addr,
                 src_addr, exp_end);
      end

      set_load_idle();
      @(posedge clk_core);
      #1;
    end
  endtask

  task automatic issue_gemm_store_writeback(
      input dest_addr_t dest_addr,
      input ptr_addr_t shape_ptr
  );
    bit done_seen;
    int wait_cycles;
    begin
      set_load_idle();
      set_memset_idle();
      store_uop = '{
          data_dest      : from_GEMM_res_to_L2,
          dest_addr      : dest_addr,
          src_addr       : '0,
          shape_ptr_addr : shape_ptr,
          async          : SYNC_OP
      };
      store_uop_valid = 1'b1;
      @(posedge clk_core);
      #1;
      store_uop_valid = 1'b0;

      checks++;
      if (store_busy !== 1'b1) begin
        errors++;
        $display("[%0t] mismatch STORE: store_busy=%0b exp 1", $time, store_busy);
      end
      checks++;
      if (dut.GemmStoreWords !== StoreBeats) begin
        errors++;
        $display("[%0t] mismatch STORE word count: got=%0d exp=%0d",
                 $time, dut.GemmStoreWords, StoreBeats);
      end

      done_seen = 1'b0;
      for (int beat = 0; beat < StoreBeats; beat++) begin
        bit accepted;
        accepted = 1'b0;
        wait_cycles = 0;
        gemm_result_data  = store_expected[beat];
        gemm_result_valid = 1'b1;

        while (gemm_result_ready !== 1'b1 && wait_cycles < 200) begin
          @(posedge clk_core);
          #1;
          wait_cycles++;
        end
        if (gemm_result_ready === 1'b1) begin
          @(posedge clk_core);
          #1;
          accepted = 1'b1;
          if (store_done === 1'b1) done_seen = 1'b1;
        end

        checks++;
        if (!accepted) begin
          errors++;
          $display("[%0t] timeout STORE beat %0d ready", $time, beat);
          $display("[%0t] STORE debug: active=%0b count=%0d words=%0d port_ready=%0b cvo_busy=%0b npu_busy=%0b done_pending=%0b l2_we=%0b",
                   $time,
                   dut.store_active,
                   dut.store_word_count,
                   dut.GemmStoreWords,
                   dut.store_port_ready,
                   dut.cvo_bridge_busy,
                   dut.npu_is_busy_wire,
                   dut.store_done_pending,
                   dut.store_l2_we);
        end else begin
          checks++;
          if (dut.final_npu_direct_en !== 1'b1 ||
              dut.final_npu_we        !== 1'b1 ||
              dut.final_npu_addr      !== dest_addr + 17'(beat) ||
              dut.final_npu_wdata     !== store_expected[beat] ||
              dut.u_l2_cache.uram_npu_we    !== 1'b1 ||
              dut.u_l2_cache.uram_npu_addr  !== dest_addr + 17'(beat) ||
              dut.u_l2_cache.uram_npu_wdata !== store_expected[beat]) begin
            errors++;
            $display("[%0t] mismatch STORE L2 beat %0d: en=%0b we=%0b addr=%0d data=%032h uram_we=%0b uram_addr=%0d uram_data=%032h exp_addr=%0d exp_data=%032h",
                     $time, beat,
                     dut.final_npu_direct_en,
                     dut.final_npu_we,
                     dut.final_npu_addr,
                     dut.final_npu_wdata,
                     dut.u_l2_cache.uram_npu_we,
                     dut.u_l2_cache.uram_npu_addr,
                     dut.u_l2_cache.uram_npu_wdata,
                     dest_addr + 17'(beat),
                     store_expected[beat]);
          end
        end
      end

      gemm_result_valid = 1'b0;
      gemm_result_data  = '0;

      wait_cycles = 0;
      while (!done_seen && wait_cycles < 20) begin
        @(posedge clk_core);
        #1;
        if (store_done === 1'b1) done_seen = 1'b1;
        wait_cycles++;
      end

      checks++;
      if (!done_seen) begin
        errors++;
        $display("[%0t] timeout STORE done pulse", $time);
      end

      checks++;
      if (store_busy !== 1'b0) begin
        errors++;
        $display("[%0t] mismatch STORE: store_busy=%0b exp 0 after done", $time, store_busy);
      end

      for (int beat = 0; beat < StoreBeats; beat++) begin
        tb_force_direct_en    = 1'b1;
        tb_force_direct_we    = 1'b0;
        tb_force_direct_addr  = dest_addr + 17'(beat);
        tb_force_direct_wdata = '0;
        force dut.u_l2_cache.IN_npu_direct_en    = tb_force_direct_en;
        force dut.u_l2_cache.IN_npu_direct_we    = tb_force_direct_we;
        force dut.u_l2_cache.IN_npu_direct_addr  = tb_force_direct_addr;
        force dut.u_l2_cache.IN_npu_direct_wdata = tb_force_direct_wdata;
        repeat (5) @(posedge clk_core);
        #1;

        checks++;
        if (dut.npu_l2_rdata !== store_expected[beat]) begin
          errors++;
          $display("[%0t] mismatch L2 direct read beat %0d: got=%032h exp=%032h",
                   $time, beat, dut.npu_l2_rdata, store_expected[beat]);
        end

        release dut.u_l2_cache.IN_npu_direct_en;
        release dut.u_l2_cache.IN_npu_direct_we;
        release dut.u_l2_cache.IN_npu_direct_addr;
        release dut.u_l2_cache.IN_npu_direct_wdata;
        tb_force_direct_en    = 1'b0;
        tb_force_direct_we    = 1'b0;
        tb_force_direct_addr  = '0;
        tb_force_direct_wdata = '0;
      end

    end
  endtask

  task automatic issue_l2_to_host_readback(
      input src_addr_t src_addr,
      input ptr_addr_t shape_ptr
  );
    logic [16:0] exp_end;
    begin
      exp_end = src_addr + 17'(StoreBeats);
      set_memset_idle();
      set_store_idle();
      load_uop = '{
          data_dest      : from_L2_to_host,
          dest_addr      : '0,
          src_addr       : src_addr,
          shape_ptr_addr : shape_ptr,
          async          : SYNC_OP
      };
      @(posedge clk_core);
      #1;

      checks++;
      if (dut.acp_rx_start !== 1'b1 ||
          dut.acp_uop.write_en !== 1'b0 ||
          dut.acp_uop.base_addr !== src_addr ||
          dut.acp_uop.end_addr  !== exp_end) begin
        errors++;
        $display("[%0t] mismatch STORE L2->host descriptor: start=%0b we=%0b base=%0d end=%0d exp start=1 we=0 base=%0d end=%0d",
                 $time,
                 dut.acp_rx_start,
                 dut.acp_uop.write_en,
                 dut.acp_uop.base_addr,
                 dut.acp_uop.end_addr,
                 src_addr,
                 exp_end);
      end

      set_load_idle();
      @(posedge clk_core);
      #1;
    end
  endtask

  // ===| Stimulus |=============================================================
  initial begin
    rst_n_core = 1'b0;
    rst_axi_n  = 1'b0;

    set_load_idle();
    set_store_idle();
    set_memset_idle();
    cvo_uop             = '0;
    cvo_uop_valid       = 1'b0;
    cvo_data_ready      = 1'b1;
    cvo_result          = '0;
    cvo_result_valid    = 1'b0;
    gemm_result_data    = '0;
    gemm_result_valid   = 1'b0;
    tb_force_direct_en    = 1'b0;
    tb_force_direct_we    = 1'b0;
    tb_force_direct_addr  = '0;
    tb_force_direct_wdata = '0;
    s_axis_acp_fmap.tdata  = '0;
    s_axis_acp_fmap.tvalid = 1'b0;
    s_axis_acp_fmap.tlast  = 1'b0;
    s_axis_acp_fmap.tkeep  = '1;
    m_axis_acp_result.tready = 1'b1;

    repeat (4) @(posedge clk_core);
    rst_n_core = 1'b1;
    rst_axi_n  = 1'b1;
    repeat (3) @(posedge clk_core);
    #1;

    store_expected[0] = 128'h0007_0006_0005_0004_0003_0002_0001_0000;
    store_expected[1] = 128'h1007_1006_1005_1004_1003_1002_1001_1000;
    store_expected[2] = 128'h2007_2006_2005_2004_2003_2002_2001_2000;
    store_expected[3] = 128'h3007_3006_3005_3004_3003_3002_3001_3000;

    program_fmap_shape(6'd3, 16'd3, 16'd5, 16'd5);  // 75 elems -> 10 words
    program_fmap_shape(6'd9, 16'd4, 16'd4, 16'd4);  // 64 elems -> 8 words
    program_fmap_shape(6'd12, 16'd1, 16'd1, 16'd32); // 32 elems -> 4 words

    issue_gemm_store_writeback(17'd512, 6'd12);
    issue_l2_to_host_readback(17'd512, 6'd12);

    issue_acp_load("host_to_L2 uses ptr3",
                   from_host_to_L2,
                   6'd3,
                   17'd100,
                   17'd0,
                   1'b1,
                   17'd100,
                   ceil_words(3, 5, 5));

    issue_acp_load("L2_to_host uses ptr9",
                   from_L2_to_host,
                   6'd9,
                   17'd0,
                   17'd200,
                   1'b0,
                   17'd200,
                   ceil_words(4, 4, 4));

    program_fmap_shape(6'd3, 16'd1, 16'd1, 16'd9);  // overwrite: 9 elems -> 2 words

    issue_npu_load("L2_to_L1_GEMM uses overwritten ptr3",
                   from_L2_to_L1_GEMM,
                   6'd3,
                   17'd300,
                   ceil_words(1, 1, 9));

    issue_npu_load("L2_to_L1_GEMV uses ptr9",
                   from_L2_to_L1_GEMV,
                   6'd9,
                   17'd400,
                   ceil_words(4, 4, 4));

    if (errors == 0) begin
      $display("PASS: %0d cycles, mem_dispatcher shape/store writeback matches golden.", checks);
    end else begin
      $display("FAIL: %0d mismatches over %0d checks.", errors, checks);
    end
    $finish;
  end

  initial begin
    #100000 $display("FAIL: mem_dispatcher shape lookup timeout"); $finish;
  end

endmodule
