# pccx v002 ISA Quick Reference

Source of truth:
[`isa_pkg.sv`](../hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv).
All offsets below are inclusive bit ranges in the 64-bit instruction word.

## Word Format

| Bits | Field | Width | Notes |
| --- | --- | ---: | --- |
| `[63:60]` | `opcode` | 4 | Selects the instruction body decoder |
| `[59:0]` | `body` | 60 | Reinterpreted by opcode-specific packed structs |

## Opcodes

| Opcode | Value | Body layout | Decoder valid |
| --- | ---: | --- | --- |
| `OP_GEMV` | `4'h0` | `GEMV_op_x64_t` | `OUT_GEMV_op_x64_valid` |
| `OP_GEMM` | `4'h1` | `GEMM_op_x64_t` | `OUT_GEMM_op_x64_valid` |
| `OP_MEMCPY` | `4'h2` | `memcpy_op_x64_t` | `OUT_memcpy_op_x64_valid` |
| `OP_MEMSET` | `4'h3` | `memset_op_x64_t` | `OUT_memset_op_x64_valid` |
| `OP_CVO` | `4'h4` | `cvo_op_x64_t` | `OUT_cvo_op_x64_valid` |

## Field Offsets

### `OP_GEMV` / `OP_GEMM`

GEMV and GEMM share the same body layout.

| Bits | Field | Width | Type |
| --- | --- | ---: | --- |
| `[59:43]` | `dest_reg` | 17 | `dest_addr_t` |
| `[42:26]` | `src_addr` | 17 | `src_addr_t` |
| `[25:20]` | `flags` | 6 | `flags_t` |
| `[19:14]` | `size_ptr_addr` | 6 | `ptr_addr_t` |
| `[13:8]` | `shape_ptr_addr` | 6 | `ptr_addr_t` |
| `[7:3]` | `parallel_lane` | 5 | `parallel_lane_t` |
| `[2:0]` | `reserved` | 3 | Set to zero |

`flags_t`: `[5] findemax`, `[4] accm`, `[3] w_scale`, `[2:0] reserved`.

### `OP_MEMCPY`

| Bits | Field | Width | Type |
| --- | --- | ---: | --- |
| `[59]` | `from_device` | 1 | `from_device_e` |
| `[58]` | `to_device` | 1 | `to_device_e` |
| `[57:41]` | `dest_addr` | 17 | `dest_addr_t` |
| `[40:24]` | `src_addr` | 17 | `src_addr_t` |
| `[23:7]` | `aux_addr` | 17 | `addr_t` |
| `[6:1]` | `shape_ptr_addr` | 6 | `ptr_addr_t` |
| `[0]` | `async` | 1 | `async_e` |

Enums: `FROM_NPU = 1'b0`, `FROM_HOST = 1'b1`, `TO_NPU = 1'b0`,
`TO_HOST = 1'b1`, `SYNC_OP = 1'b0`, `ASYNC_OP = 1'b1`.

### `OP_MEMSET`

| Bits | Field | Width | Type |
| --- | --- | ---: | --- |
| `[59:58]` | `dest_cache` | 2 | `dest_cache_e` |
| `[57:52]` | `dest_addr` | 6 | `ptr_addr_t` |
| `[51:36]` | `a_value` | 16 | `a_value_t` |
| `[35:20]` | `b_value` | 16 | `b_value_t` |
| `[19:4]` | `c_value` | 16 | `c_value_t` |
| `[3:0]` | `reserved` | 4 | Set to zero |

`dest_cache_e`: `data_to_fmap_shape = 2'h0`,
`data_to_weight_shape = 2'h1`.

### `OP_CVO`

| Bits | Field | Width | Type |
| --- | --- | ---: | --- |
| `[59:56]` | `cvo_func` | 4 | `cvo_func_e` |
| `[55:39]` | `src_addr` | 17 | `src_addr_t` |
| `[38:22]` | `dst_addr` | 17 | `addr_t` |
| `[21:6]` | `length` | 16 | `length_t` |
| `[5:1]` | `flags` | 5 | `cvo_flags_t` |
| `[0]` | `async` | 1 | `async_e` |

`cvo_func_e`: `CVO_EXP = 4'h0`, `CVO_SQRT = 4'h1`,
`CVO_GELU = 4'h2`, `CVO_SIN = 4'h3`, `CVO_COS = 4'h4`,
`CVO_REDUCE_SUM = 4'h5`, `CVO_SCALE = 4'h6`, `CVO_RECIP = 4'h7`.

`cvo_flags_t`: `[4] sub_emax`, `[3] recip_scale`, `[2] accm`,
`[1:0] reserved`.
