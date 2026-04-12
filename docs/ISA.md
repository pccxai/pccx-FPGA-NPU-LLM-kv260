# uCA ISA Specification

**uCA**: micro Compute Architecture — the FPGA NPU instruction set.

Target: Kria KV260 | Word width: **64-bit** | Encoding: **VLIW**

RTL source: `hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/`

---

## 1. Instruction Format

Every instruction is 64 bits wide.

```
 [63:60]   [59:0]
 OPCODE    BODY (60 bits, layout depends on opcode)
```

The top-level decoder (`ctrl_npu_decoder.sv`) strips the 4-bit opcode and routes
the remaining 60-bit body to the appropriate execution engine.

---

## 2. Opcode Table

| Opcode | Mnemonic    | Value  | Target Engine                        |
|--------|-------------|--------|--------------------------------------|
| `OP_GEMV`   | Vector–Matrix Multiply  | `4'h0` | Vector Core (μV-Cores)      |
| `OP_GEMM`   | Matrix–Matrix Multiply  | `4'h1` | Matrix Core (Systolic Array)|
| `OP_MEMCPY` | Memory Copy             | `4'h2` | MEM Dispatcher              |
| `OP_MEMSET` | Memory Set              | `4'h3` | MEM Dispatcher              |
| `OP_CVO`    | Complex Vector Op       | `4'h4` | CVO Core (μCVO-Cores)       |
| —           | Reserved                | `4'h5`–`4'hF` | —                   |

---

## 3. Instruction Encoding

### 3.1 GEMV / GEMM  (`OP_GEMV`, `OP_GEMM`)

Both share the same body layout.

```
[59:43]  dest_reg       17-bit  Destination register / address
[42:26]  src_addr       17-bit  Source address
[25:20]  flags           6-bit  Control flags (see §4)
[19:14]  size_ptr_addr   6-bit  Pointer to size descriptor
[13:8]   shape_ptr_addr  6-bit  Pointer to shape descriptor
[7:3]    parallel_lane   5-bit  Number of active parallel lanes
[2:0]    reserved        3-bit
```

### 3.2 MEMCPY  (`OP_MEMCPY`)

```
[59]     from_device     1-bit  0=FROM_NPU, 1=FROM_HOST
[58]     to_device       1-bit  0=TO_NPU,   1=TO_HOST
[57:41]  dest_addr      17-bit  Destination address
[40:24]  src_addr       17-bit  Source address
[23:7]   aux_addr       17-bit  Auxiliary address (reserved)
[6:1]    shape_ptr_addr  6-bit  Pointer to shape descriptor
[0]      async           1-bit  0=sync, 1=async transfer
```

### 3.3 MEMSET  (`OP_MEMSET`)

```
[59:58]  dest_cache      2-bit  0=fmap_shape, 1=weight_shape
[57:52]  dest_addr       6-bit  Destination pointer address (ptr_addr_t)
[51:36]  a_value        16-bit  Value A
[35:20]  b_value        16-bit  Value B
[19:4]   c_value        16-bit  Value C
[3:0]    reserved        4-bit
```

### 3.4 CVO  (`OP_CVO`)

Dispatched to the CVO Core (2× μCVO-Cores). Each μCVO-Core contains a CORDIC
unit (sin/cos) and an SFU (exp, sqrt, GELU). Required for Transformer softmax,
RMSNorm, and activation functions.

```
[59:56]  cvo_func        4-bit  Function code (see §3.4.1)
[55:39]  src_addr       17-bit  Source address in L2 cache
[38:22]  dst_addr       17-bit  Destination address in L2 cache
[21:6]   length         16-bit  Number of elements (vector length)
[5:1]    flags           5-bit  Control flags (see §3.4.2)
[0]      async           1-bit  0=sync, 1=async
```

#### 3.4.1 CVO Function Codes

| Code    | Mnemonic       | Description                                     | Hardware Unit |
|---------|----------------|-------------------------------------------------|---------------|
| `4'h0`  | `CVO_EXP`      | Element-wise exp(x)                             | SFU           |
| `4'h1`  | `CVO_SQRT`     | Element-wise sqrt(x)                            | SFU           |
| `4'h2`  | `CVO_GELU`     | Element-wise GELU(x)                            | SFU           |
| `4'h3`  | `CVO_SIN`      | Element-wise sin(x)                             | CORDIC        |
| `4'h4`  | `CVO_COS`      | Element-wise cos(x)                             | CORDIC        |
| `4'h5`  | `CVO_REDUCE_SUM` | Sum all elements → scalar at dst_addr         | SFU + Adder   |
| `4'h6`  | `CVO_SCALE`    | Element-wise multiply by scalar at src_addr+0   | SFU           |
| `4'h7`  | `CVO_RECIP`    | Element-wise 1/x                                | SFU           |
| `4'h8`–`4'hF` | —       | Reserved                                        | —             |

> **Softmax sequence** (one CVO pipeline pass):
> 1. `OP_GEMV` with `FLAG_FINDEMAX` — find e_max over attention scores
> 2. `OP_CVO CVO_EXP` with `FLAG_SUB_EMAX` — exp(x − e_max) for each score
> 3. `OP_CVO CVO_REDUCE_SUM` — sum of exps (denominator)
> 4. `OP_CVO CVO_SCALE` with `FLAG_RECIP_SCALE` — divide each exp by sum

> **RMSNorm sequence**:
> 1. `OP_GEMV` with `FLAG_FINDEMAX` during projection (emax already tracked)
> 2. `OP_CVO CVO_REDUCE_SUM` (of squares) → then
> 3. `OP_CVO CVO_SQRT` + `CVO_RECIP` → normalization factor
> 4. `OP_CVO CVO_SCALE` — apply learned weight γ

#### 3.4.2 CVO Flags (5-bit, [5:1] of body)

```
[5]  sub_emax      Subtract e_max from input before operation (requires prior FINDEMAX)
[4]  recip_scale   Use reciprocal of scalar for SCALE (divide instead of multiply)
[3]  accm          Accumulate into dst (do not overwrite)
[2:1] reserved
```

---

## 4. Flags Field for GEMV/GEMM (6-bit, [25:20])

```
[5]  findemax   Find and register the exponent maximum (e_max) for output normalization
[4]  accm       Accumulate result into destination register (do not overwrite)
[3]  w_scale    Apply weight scale factor during MAC
[2:0] reserved
```

---

## 5. Memory Routing Table (MEMCPY)

Defined in `isa_memctrl.svh` as `data_route_e`.

| Route Enum               | Encoding (`src[3:0]\|dst[3:0]`) | Description                                  |
|--------------------------|--------------------------------|----------------------------------------------|
| `from_host_to_L2`        | `8'h01`                        | Host DDR4 → L2 cache (fmap DMA in via ACP)   |
| `from_L2_to_host`        | `8'h10`                        | L2 cache → Host DDR4 (result DMA out via ACP)|
| `from_L2_to_L1_GEMM`     | `8'h12`                        | L2 → Matrix Core fmap broadcast              |
| `from_L2_to_L1_GEMV`     | `8'h13`                        | L2 → Vector Core fmap broadcast              |
| `from_L2_to_CVO`         | `8'h14`                        | L2 → CVO Core input stream                   |
| `from_GEMV_res_to_L2`    | `8'h31`                        | Vector Core result → L2 cache                |
| `from_GEMM_res_to_L2`    | `8'h21`                        | Matrix Core result → L2 cache                |
| `from_CVO_res_to_L2`     | `8'h41`                        | CVO Core result → L2 cache                   |

---

## 6. Micro-Op (uop) Structures

After decoding, the Global Scheduler splits the instruction body into
engine-specific micro-ops before dispatch.

### 6.1 GEMV / GEMM Control uop

```systemverilog
typedef struct packed {
    flags_t         flags;           // 6-bit
    ptr_addr_t      size_ptr_addr;   // 6-bit
    parallel_lane_t parallel_lane;   // 5-bit
} gemv_control_uop_t;  // = gemm_control_uop_t
```

### 6.2 Memory Control uop

```systemverilog
typedef struct packed {
    data_route_e data_dest;      // 8-bit  (source[3:0] | dest[3:0])
    dest_addr_t  dest_addr;      // 17-bit
    src_addr_t   src_addr;       // 17-bit
    ptr_addr_t   shape_ptr_addr; // 6-bit
    async_e      async;          // 1-bit
} memory_control_uop_t;
```

### 6.3 Memory Set uop

```systemverilog
typedef struct packed {
    dest_cache_e dest_cache;  // 2-bit
    ptr_addr_t   dest_addr;   // 6-bit
    a_value_t    a_value;
    b_value_t    b_value;
    c_value_t    c_value;
} memory_set_uop_t;
```

### 6.4 CVO Control uop

```systemverilog
typedef struct packed {
    cvo_func_e  cvo_func;     // 4-bit
    src_addr_t  src_addr;     // 17-bit
    dst_addr_t  dst_addr;     // 17-bit
    length_t    length;       // 16-bit
    cvo_flags_t flags;        // 5-bit
    async_e     async;        // 1-bit
} cvo_control_uop_t;
```

---

## 7. Decoupled Dataflow Pipeline

The front-end and execution engines are strictly decoupled.

```
Host (AXI-Lite) --> [AXIL_CMD_IN] --> ctrl_npu_decoder
                                            |
                   +----------+------+------+------+-----------+
                   v          v      v             v           v
              GEMV FIFO  GEMM FIFO  CVO FIFO  MEM FIFO    MEMSET FIFO
                   |          |      |             |           |
              μV-Core    Systolic  μCVO-Core  mem_dispatcher  mem_set
             (GEMV)    Array(GEMM) (CVO)
```

The front-end (`ctrl_npu_decoder`) issues instructions into per-engine FIFOs
and immediately returns — it never stalls waiting for execution to complete.
Each engine's local dispatcher independently pops from its FIFO and fires
when operands are ready.

---

## 8. AXI-Lite Register Map

Control is via `S_AXIL_CTRL` (HPM port on KV260).

| Offset | Width  | Direction | Description                                                      |
|--------|--------|-----------|------------------------------------------------------------------|
| `0x00` | 32-bit | W         | VLIW instruction [31:0] (write lower word first)                 |
| `0x04` | 32-bit | W         | VLIW instruction [63:32] (writing this word triggers NPU latch)  |
| `0x08` | 32-bit | R         | NPU status register (see §9)                                     |

---

## 9. Status Register (`0x08`)

| Bit | Name   | Description                                       |
|-----|--------|---------------------------------------------------|
| [0] | `BUSY` | NPU is executing — do not issue new instruction   |
| [1] | `DONE` | Last operation completed successfully             |
| [31:2] | —   | Reserved                                          |
