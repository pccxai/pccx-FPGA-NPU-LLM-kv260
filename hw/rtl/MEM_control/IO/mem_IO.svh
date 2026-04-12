// ===| Memory Port Mode Constants |==============================================
// Used by mem_GLOBAL_cache and mem_dispatcher to distinguish read vs. write
// accesses on the ACP and NPU ports.
// ===============================================================================

`ifndef MEM_IO_SVH
`define MEM_IO_SVH

`define PORT_MOD_E_WRITE  1'b1   // Port is in write (sink) mode
`define PORT_MOD_E_READ   1'b0   // Port is in read  (source) mode

`endif // MEM_IO_SVH
