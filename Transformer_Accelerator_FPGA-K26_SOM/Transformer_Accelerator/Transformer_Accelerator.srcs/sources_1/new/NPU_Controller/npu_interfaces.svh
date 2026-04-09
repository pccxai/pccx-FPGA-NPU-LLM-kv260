`include "GLOBAL_CONST.svh"

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
    parameter int ADDR_W = 12,
    parameter int DATA_W = 64
) (
    input logic clk,
    input logic rst_n
);
  // AW Channel
  logic [ADDR_W-1:0] awaddr;
  logic [       2:0] awprot;
  logic awvalid, awready;

  // W Channel
  logic [    DATA_W-1:0] wdata;
  logic [(DATA_W/8)-1:0] wstrb;
  logic wvalid, wready;

  // B Channel
  logic [1:0] bresp;
  logic bvalid, bready;

  // AR Channel
  logic [ADDR_W-1:0] araddr;
  logic [       2:0] arprot;
  logic arvalid, arready;

  // R Channel
  logic [DATA_W-1:0] rdata;
  logic [       1:0] rresp;
  logic rvalid, rready;

  modport slave(
      input awaddr, awprot, awvalid, wdata, wstrb, wvalid, bready,
      input araddr, arprot, arvalid, rready,
      output awready, wready, bresp, bvalid, arready, rdata, rresp, rvalid
  );

  modport master(
      output awaddr, awprot, awvalid, wdata, wstrb, wvalid, bready,
      output araddr, arprot, arvalid, rready,
      input awready, wready, bresp, bvalid, arready, rdata, rresp, rvalid
  );
endinterface



`endif  // NPU_INTERFACES_SVH
