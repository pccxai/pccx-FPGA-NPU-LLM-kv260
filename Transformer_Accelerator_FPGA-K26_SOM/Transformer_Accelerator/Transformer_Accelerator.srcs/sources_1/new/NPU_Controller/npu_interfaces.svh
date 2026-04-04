// ===| AXI-Stream Interface Definition      |=====
// ===| Author: Gemini CLI (for NPU Project) |=====
// <><><><><><><> 400MHz Optimized <><><><><><><><>

`ifndef NPU_INTERFACES_SVH
`define NPU_INTERFACES_SVH

interface axis_if #(
    parameter DATA_WIDTH = 128
) ();
  logic [    DATA_WIDTH-1:0] tdata;
  logic                      tvalid;
  logic                      tready;
  logic                      tlast;
  logic [(DATA_WIDTH/8)-1:0] tkeep;

  // Slave Side (NPU Perspective: Input)
  modport slave(input tdata, tvalid, tlast, tkeep, output tready);

  // Master Side (NPU Perspective: Output)
  modport master(output tdata, tvalid, tlast, tkeep, input tready);
endinterface

// axil_if.sv
interface axil_if #(
    parameter ADDR_W = 12
) (
    input logic clk,
    rst_n
);
  logic [ADDR_W-1:0] awaddr;
  logic awvalid, awready;
  logic [31:0] wdata;
  logic [ 3:0] wstrb;
  logic wvalid, wready;
  logic [1:0] bresp;
  logic bvalid, bready;
  logic [ADDR_W-1:0] araddr;
  logic arvalid, arready;
  logic [31:0] rdata;
  logic [ 1:0] rresp;
  logic rvalid, rready;

  modport slave(
      input awaddr, awvalid, wdata, wstrb, wvalid, bready,
      input araddr, arvalid, rready,
      output awready, wready, bresp, bvalid, arready, rdata, rresp, rvalid
  );
  modport master(
      output awaddr, awvalid, wdata, wstrb, wvalid, bready,
      output araddr, arvalid, rready,
      input awready, wready, bresp, bvalid, arready, rdata, rresp, rvalid
  );
endinterface



`endif  // NPU_INTERFACES_SVH
