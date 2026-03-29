// ===| AXI-Stream Interface Definition |==========
// ===| Author: Gemini CLI (for NPU Project) |=====
// <><><><><><><> 400MHz Optimized <><><><><><><><>

`ifndef NPU_INTERFACES_SVH
`define NPU_INTERFACES_SVH

interface axis_if #(
    parameter DATA_WIDTH = 128
) ();
    logic [DATA_WIDTH-1:0] tdata;
    logic                  tvalid;
    logic                  tready;
    logic                  tlast;
    logic [(DATA_WIDTH/8)-1:0] tkeep;

    // Slave Side (NPU Perspective: Input)
    modport slave (
        input  tdata, tvalid, tlast, tkeep,
        output tready
    );

    // Master Side (NPU Perspective: Output)
    modport master (
        output tdata, tvalid, tlast, tkeep,
        input  tready
    );
endinterface

`endif // NPU_INTERFACES_SVH
