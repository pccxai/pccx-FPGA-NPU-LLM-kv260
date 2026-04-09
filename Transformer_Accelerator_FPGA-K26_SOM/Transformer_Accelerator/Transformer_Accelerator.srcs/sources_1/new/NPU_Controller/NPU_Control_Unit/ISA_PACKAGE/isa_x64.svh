package isa_x64;
  //  Basic Types
  typedef logic [16:0] dest_addr_t;
  typedef logic [16:0] src_addr_t;
  typedef logic [16:0] addr_t;



  typedef logic [5:0] ptr_addr_t;  // For size and shape pointers
  typedef logic [4:0] parallel_lane_t;

  typedef logic [2:0] reserved_dot;

  typedef struct packed {
    logic [63:0] data;
    logic [7:0]  byte_en;
  } x64_payload_t;


  // npu -> host
  // host -> npu
  typedef enum logic {
    FROM_NPU  = 1'b0,
    FROM_HOST = 1'b1
  } from_device_e;

  typedef enum logic {
    TO_NPU  = 1'b0,
    TO_HOST = 1'b1
  } to_device_e;

  typedef enum logic {
    sync  = 1'b0,
    async = 1'b1
  } async_e;


  //  Flags (6-bit as per PDF spec)
  typedef struct packed {
    logic findemax;
    logic accm;     // Accumulate
    logic w_scale;
    logic [2:0] reserved;
  } flags_t;

  // Instruction format
  // instruction = x64(64bit) - head(opcode)
  typedef logic [59:0] VLIW_instruction_x64;

  //  Opcode table (4-bit)
  typedef enum logic [3:0] {
    OP_VDOTM  = 4'h0,
    OP_MDOTM  = 4'h1,
    OP_MEMCPY = 4'h2,
    OP_MEMSET = 4'h3
  } opcode_e;



  /*
//  Field layouts per opcode (60-bit payloads)
// V dot M / M dot M payload (Total 60 bits)
// reserved(3) + parallel_lane(5) + shape(6) + size(6) + flags(6) + src(17) + dest(17) = 60 bits
typedef struct packed {
  logic [2:0]     reserved;
  parallel_lane_t parallel_lane;
  ptr_addr_t      shape_ptr_addr;
  ptr_addr_t      size_ptr_addr;
  flags_t         flags;
  src_addr_t      src_addr;
  dest_addr_t     dest_addr;
} payload_dotm_t;

// Memcpy / Memset payload (Total 60 bits)
// reserved(2) + shape(6) + C(17) + B(17) + A(17) + to_device(1) = 60 bits
typedef struct packed {
  logic [1:0] reserved;
  ptr_addr_t  shape_ptr_addr;
  src_addr_t  c_addr;
  src_addr_t  b_addr;
  dest_addr_t a_addr;
  logic       to_device;       // From device To device
} payload_memcpy_t;

//  Full 64-bit instruction word
//  Payload (60b) + Fixed header (opcode 4b) = 64 bits

typedef struct packed {
  union packed {
    payload_dotm_t   dotm;
    payload_memory_t memory;
  } payload;  // [63:4]
  opcode_e opcode;  // [3:0]
} instruction_x64_t;
*/

  typedef struct packed {
    dest_addr_t     dest_reg;
    src_addr_t      src_addr;
    flags_t         flags;
    ptr_addr_t      size_ptr_addr;
    ptr_addr_t      shape_ptr_addr;
    parallel_lane_t parallel_lane;
    reserved_dot    reserved;
  } vdotm_op_x64_t;

  typedef struct packed {
    dest_addr_t     dest_reg;
    src_addr_t      src_addr;
    flags_t         flags;
    ptr_addr_t      size_ptr_addr;
    ptr_addr_t      shape_ptr_addr;
    parallel_lane_t parallel_lane;
    reserved_dot    reserved;
  } mdotm_op_x64_t;

  typedef struct packed {
    from_device_e from_device;
    to_device_e   to_device;
    dest_addr_t   dest_addr;
    src_addr_t    src_addr;
    addr_t        _addr;
    ptr_addr_t    shape_ptr_addr;
    async_e       async;
  } memcpy_op_x64_t;

  // --------------------------------------------------------
  // ===| Compute Micro-Op |=================================






  `define MEMORY_UOP_WIDTH 49

  typedef struct packed {
    flags_t    flags;
    ptr_addr_t size_ptr_addr;
    parallel_lane_t parallel_lane;
  } stlc_control_uop_t;


  typedef struct packed {
    flags_t    flags;
    ptr_addr_t size_ptr_addr;
    parallel_lane_t parallel_lane;
  } vdotm_control_uop_t;



  // ===| Compute Micro-Op |====================================
  // -----------------------------------------------------------

endpackage
