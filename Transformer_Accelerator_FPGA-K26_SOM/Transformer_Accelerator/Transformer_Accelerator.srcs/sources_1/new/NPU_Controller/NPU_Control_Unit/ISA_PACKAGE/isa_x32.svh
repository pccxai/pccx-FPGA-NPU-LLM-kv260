package isa_x32;
  `define X32_HEADSIZE 6
  typedef logic [16:0] dest_addr_t;
  typedef logic [7:0] loop_cnt_t;


  typedef struct packed {
    logic [31:0] data;
    logic [3:0]  byte_en;
  } x32_payload_t;

  /*─────────────────────────────────────────────
  Opcode table
  ─────────────────────────────────────────────*/
  typedef enum logic [4:0] {
    OP_VDOTM  = 4'h0,
    OP_MDOTM  = 4'h1,
    OP_MEMCPY = 4'h2
  } opcode_t;


  typedef struct packed {
    logic       to_divice;
    dest_addr_t dest_addr;
    loop_cnt_t  loop_cnt;
  } payload_memcpy_t;

  /*─────────────────────────────────────────────
  Full 32-bit instruction word
  Fixed header (6b) + union payload (26b)
  ─────────────────────────────────────────────*/
  typedef struct packed {
    logic [1:0] cmd_chaining;
    opcode_t    opcode;

    union packed {
      payload_dotm_t  dotm;   // V dot M / M dot M
      payload_memcpy_t memcpy; // memcpy
      override_memcpy_t override_memcpy;
      override_chain_memcpy_t override_chain_memcpy;
      logic [25:0]    raw;
    } payload;  // [25:0]

  } instruction_x32_t;


  //Deprecated
  typedef struct packed {
    logic       to_divice;
    dest_addr_t dest_addr;
    logic [7:0] loop_cnt;
  } memory_uop_x32_t;


endpackage
