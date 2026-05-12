# Packed Micro-Op Encodings

Source of truth: [`isa_pkg.sv`](../hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv).

This document records the packed micro-op field offsets currently defined by
`isa_pkg.sv`. SystemVerilog packed structs place the first declared field at
the most-significant bit, so the offsets below follow declaration order.

## 32-bit status

`isa_pkg.sv` does not define a 32-bit packed micro-op encoding. The active
micro-op structs in `isa_pkg.sv` are 17, 35, 49, 56, or 60 bits wide,
depending on the target channel. Do not infer a 32-bit firmware or FIFO
packing from these structs without adding a separate source-of-truth typedef.

The legacy `isa_x32.svh` file contains a deprecated 32-bit instruction word,
but the header comment in `isa_pkg.sv` marks the legacy `.svh` ISA files as
superseded. The field maps below therefore cover only `isa_pkg.sv`.

## Shared Flags

### `flags_t` - 6 bits

Used by GEMM and GEMV control micro-ops.

| Bits | Width | Field | Notes |
| --- | ---: | --- | --- |
| `[5]` | 1 | `findemax` | Find and register `e_max` for output normalisation. |
| `[4]` | 1 | `accm` | Accumulate into destination. |
| `[3]` | 1 | `w_scale` | Apply weight scale factor during MAC. |
| `[2:0]` | 3 | `reserved` | Reserved by `isa_pkg.sv`. |

### `cvo_flags_t` - 5 bits

Used inside `cvo_control_uop_t`.

| Bits | Width | Field | Notes |
| --- | ---: | --- | --- |
| `[4]` | 1 | `sub_emax` | Subtract `e_max` before operation. |
| `[3]` | 1 | `recip_scale` | Use reciprocal of scalar. |
| `[2]` | 1 | `accm` | Accumulate into destination. |
| `[1:0]` | 2 | `reserved` | Reserved by `isa_pkg.sv`. |

## Engine Control Micro-Ops

### `gemm_control_uop_t` - 17 bits

| Bits | Width | Field | Source type |
| --- | ---: | --- | --- |
| `[16:11]` | 6 | `flags` | `flags_t` |
| `[10:5]` | 6 | `size_ptr_addr` | `ptr_addr_t` |
| `[4:0]` | 5 | `parallel_lane` | `parallel_lane_t` |

### `GEMV_control_uop_t` - 17 bits

`GEMV_control_uop_t` uses the same packed layout as `gemm_control_uop_t`.

| Bits | Width | Field | Source type |
| --- | ---: | --- | --- |
| `[16:11]` | 6 | `flags` | `flags_t` |
| `[10:5]` | 6 | `size_ptr_addr` | `ptr_addr_t` |
| `[4:0]` | 5 | `parallel_lane` | `parallel_lane_t` |

### `memory_control_uop_t` - 49 bits

`MemoryUopWidth` is defined as `49` in `isa_pkg.sv`.

| Bits | Width | Field | Source type |
| --- | ---: | --- | --- |
| `[48:41]` | 8 | `data_dest` | `data_route_e` |
| `[40:24]` | 17 | `dest_addr` | `dest_addr_t` |
| `[23:7]` | 17 | `src_addr` | `src_addr_t` |
| `[6:1]` | 6 | `shape_ptr_addr` | `ptr_addr_t` |
| `[0]` | 1 | `async` | `async_e` |

### `memory_set_uop_t` - 56 bits

| Bits | Width | Field | Source type |
| --- | ---: | --- | --- |
| `[55:54]` | 2 | `dest_cache` | `dest_cache_e` |
| `[53:48]` | 6 | `dest_addr` | `ptr_addr_t` |
| `[47:32]` | 16 | `a_value` | `a_value_t` |
| `[31:16]` | 16 | `b_value` | `b_value_t` |
| `[15:0]` | 16 | `c_value` | `c_value_t` |

### `cvo_control_uop_t` - 60 bits

| Bits | Width | Field | Source type |
| --- | ---: | --- | --- |
| `[59:56]` | 4 | `cvo_func` | `cvo_func_e` |
| `[55:39]` | 17 | `src_addr` | `src_addr_t` |
| `[38:22]` | 17 | `dst_addr` | `addr_t` |
| `[21:6]` | 16 | `length` | `length_t` |
| `[5:1]` | 5 | `flags` | `cvo_flags_t` |
| `[0]` | 1 | `async` | `async_e` |

`cvo_flags_t` occupies bits `[5:1]` of `cvo_control_uop_t`, so its nested
fields map to uop bits `[5]` `sub_emax`, `[4]` `recip_scale`, `[3]` `accm`,
and `[2:1]` `reserved`.

## ACP / NPU Transfer Micro-Ops

### `acp_uop_t` - 35 bits

| Bits | Width | Field | Source type |
| --- | ---: | --- | --- |
| `[34]` | 1 | `write_en` | `logic` |
| `[33:17]` | 17 | `base_addr` | `logic [16:0]` |
| `[16:0]` | 17 | `end_addr` | `logic [16:0]` |

### `npu_uop_t` - 35 bits

`npu_uop_t` uses the same packed layout as `acp_uop_t`.

| Bits | Width | Field | Source type |
| --- | ---: | --- | --- |
| `[34]` | 1 | `write_en` | `logic` |
| `[33:17]` | 17 | `base_addr` | `logic [16:0]` |
| `[16:0]` | 17 | `end_addr` | `logic [16:0]` |
