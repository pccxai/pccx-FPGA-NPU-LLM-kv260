package isa_pkg;

  `define MOD_X64 1
  `define MOD_X32 0

  `define U_OPERATION_WIDTH 59


  `define INST_HEAD_ARCH_MOD_BIT 1

  `include "isa_x32.svh"
  `include "isa_memctrl.svh"
  `include "isa_x64.svh"

endpackage
