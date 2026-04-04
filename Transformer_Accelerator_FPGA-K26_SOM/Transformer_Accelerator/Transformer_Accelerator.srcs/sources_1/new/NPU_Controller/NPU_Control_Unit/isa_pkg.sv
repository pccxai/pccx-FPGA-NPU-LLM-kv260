package isa_pkg;

  /*─────────────────────────────────────────────
  Opcode table
  ─────────────────────────────────────────────*/
  typedef enum logic [3:0] {
    OP_VDOTM  = 4'h0,
    OP_MDOTM  = 4'h1,
    OP_MEMCPY = 4'h2
    // 나머지 추가
  } opcode_t;

  /*─────────────────────────────────────────────
  Field layouts per opcode (26-bit payloads)
  ─────────────────────────────────────────────*/

  // V dot M / M dot M
  typedef struct packed {
    logic [3:0] dest;             // destination register
    logic [3:0] src1;             // source 1
    logic [3:0] src2;             // source 2
    logic [1:0] lane_idq;         // 4 lane stream
    logic       data_src;         // BF16 data source
    logic [2:0] datatype1;        // INT {4,8,16}
    logic [2:0] datatype2;        // INT {4,8,16}
    logic       find_emax_align;  // find emax align flag
    logic       accm;             // res & acc
    logic       scale;            // scale after calc
    logic [1:0] reserved;
  } payload_dotm_t;

  typedef struct packed {
    logic [1:0] dim_xyz;     // DIM XYZ select
    logic [3:0] dest_queue;  // destination queue
    logic       to_divice;
    logic [4:0] dim_x;       // Matrix Size (2^N)
    logic [4:0] dim_y;
    logic [4:0] dim_z;
    logic       data_src;
    logic [2:0] datatype;    // INT {4,8,16} / BF16
  } payload_memcpy_t;

  // ===| when matrix shape is not square of 2 |===
  // ===| when matrix Dimension is 4D |============
  // Matrix Size (x2)
  typedef struct packed {
    logic [1:0]  dim_xyzw;    // DIM XYZ select
    logic [3:0]  dest_queue;  // destination queue
    logic        to_divice;
    logic [18:0] dim_N;       // Matrix Size (x2)
    logic        data_src;
    logic [2:0]  datatype;    // INT {4,8,16} / BF16
  } override_memcpy_t;

  //chaining override,
  typedef struct packed {
    logic [1:0] dim_xyzw;   // DIM XYZ select
    logic [23:0] dim_N;     // Matrix Size (x2)
  } override_chain_memcpy_t;


  /*─────────────────────────────────────────────
  Full 32-bit instruction word
  Fixed header (6b) + union payload (26b)
  ─────────────────────────────────────────────*/
  typedef struct packed {
    opcode_t opcode;        // [31:28]
    logic    override;      // [27]
    logic    cmd_chaining;  // [26]

    union packed {
      payload_dotm_t  dotm;   // V dot M / M dot M
      payload_memcpy_t memcpy; // memcpy
      override_memcpy_t override_memcpy;
      override_chain_memcpy_t override_chain_memcpy;
      logic [25:0]    raw;    // 그냥 비트로 볼 때
    } payload;  // [25:0]

  } instruction_t;

endpackage
