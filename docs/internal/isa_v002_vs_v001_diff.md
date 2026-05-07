# ISA Package v002 vs v001 Diff

This note records how `isa_pkg.sv` changed between the early v001-style
package and the current pccx v002 package in this repository. It is based
only on `git log --follow` for `isa_pkg.sv` and the current
`hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv` contents.

## Source Trail

Relevant package-history commits:

| Commit | Date | Package change |
| --- | --- | --- |
| `48806c5` | 2026-04-05 | Initial ISA package: 32-bit instruction form with `OP_VDOTM`, `OP_MDOTM`, and `OP_MEMCPY`. |
| `6ba9bb5` | 2026-04-06 | Reworked the ISA wrapper for 64-bit mode by including `isa_x32.svh` and `isa_x64.svh`; the x64 package introduced 17-bit addresses, 60-bit payloads, and `OP_MEMSET`. |
| `e365946` | 2026-04-09 | Moved the package under `ISA_PACKAGE/` and added the separate memory-control include, `isa_memctrl.svh`. |
| `078e3d9` | 2026-04-12 | Reorganized the tree into the current `hw/rtl/.../ISA_PACKAGE/isa_pkg.sv` path. |
| `4f9e98b` | 2026-04-12 | Flattened the ISA vocabulary into `isa_pkg` and added Complex Vector Operation support. |
| `6623e3c` | 2026-04-20 | Added package `timescale 1ns / 1ps` for xsim elaboration consistency. |
| `3f33595` | 2026-05-02 | Added the v002 package contract header and shape typedefs. |
| `9b1f4a8`, `81bc3bd` | 2026-05-02 | Shape constant RAM follow-up work; package-visible shape vocabulary remains in `isa_pkg.sv`. |

## Instruction Set Changes

The v001-style package began with a 4-bit opcode enum named `opcode_t` and
three implemented opcodes:

| v001 name | v001 value | v002 name | v002 value | Notes |
| --- | ---: | --- | ---: | --- |
| `OP_VDOTM` | `4'h0` | `OP_GEMV` | `4'h0` | Vector-dot-matrix naming was replaced with GEMV terminology. |
| `OP_MDOTM` | `4'h1` | `OP_GEMM` | `4'h1` | Matrix-dot-matrix naming was replaced with GEMM terminology. |
| `OP_MEMCPY` | `4'h2` | `OP_MEMCPY` | `4'h2` | Opcode value retained, but payload fields changed. |
| none | none | `OP_MEMSET` | `4'h3` | Added in the 64-bit ISA path for constant/shape cache setup. |
| none | none | `OP_CVO` | `4'h4` | Added with Complex Vector Operation support. |

The current `opcode_e` table is a 4-bit enum with five entries:
`OP_GEMV`, `OP_GEMM`, `OP_MEMCPY`, `OP_MEMSET`, and `OP_CVO`.

## Encoding Changes

v001 encoded a 32-bit instruction as:

- `opcode_t opcode` in bits `[31:28]`
- `override` in bit `[27]`
- `cmd_chaining` in bit `[26]`
- a 26-bit packed union payload in bits `[25:0]`

v002 uses a fixed 64-bit instruction format where the opcode is stripped
before package-level payload decoding:

- `VLIW_instruction_x64` is `logic [59:0]`
- `instruction_op_x64_t` wraps the same 60-bit body
- each operation-specific struct is documented as a 60-bit body

The GEMV/GEMM body is now:

| Field | Width | Bits |
| --- | ---: | --- |
| `dest_reg` | 17 | `[59:43]` |
| `src_addr` | 17 | `[42:26]` |
| `flags` | 6 | `[25:20]` |
| `size_ptr_addr` | 6 | `[19:14]` |
| `shape_ptr_addr` | 6 | `[13:8]` |
| `parallel_lane` | 5 | `[7:3]` |
| `reserved` | 3 | `[2:0]` |

`GEMM_op_x64_t` is now a typedef alias of `GEMV_op_x64_t`, so both
compute opcodes use the same payload layout.

## Field and Type Renames

The package changed from generic or legacy engine names to v002 engine
names:

| v001 / intermediate name | Current v002 name |
| --- | --- |
| `opcode_t` | `opcode_e` |
| `payload_dotm_t`, `payload_vdotm_t`, `vdotm_op_x64_t` | `GEMV_op_x64_t` |
| `payload_dotm_t`, `payload_mdotm_t`, `mdotm_op_x64_t` | `GEMM_op_x64_t` |
| `dest` / `dest_addr` in compute payloads | `dest_reg` |
| `src1`, `src2` | `src_addr` plus shape/size pointers |
| `lane_idq` | `parallel_lane` |
| `find_emax_align` | `flags.findemax` |
| `scale` | `flags.w_scale` |
| `sync`, `async` enum values | `SYNC_OP`, `ASYNC_OP` |
| `data_to_L2_cache` | `data_to_GLOBAL_cache` |
| `data_from_L2_cache` | `data_from_GLOBAL_cache` |
| `data_to_L1_cache_stlc_in` | `data_to_L1_cache_GEMM_in` |
| `data_to_L1_cache_vdotm_in` | `data_to_L1_cache_GEMV_in` |
| `data_from_L1_cache_stlc_res` | `data_from_L1_cache_GEMM_res` |
| `data_from_L1_cache_vdotm_res` | `data_from_L1_cache_GEMV_res` |
| `from_L2_to_L1_stlc` | `from_L2_to_L1_GEMM` |
| `from_L2_to_L1_vdotm` | `from_L2_to_L1_GEMV` |
| `from_stlc_res_to_L2` | `from_GEMM_res_to_L2` |
| `from_vdotm_res_to_L2` | `from_GEMV_res_to_L2` |
| `stlc_control_uop_t` | `gemm_control_uop_t` |
| `vdotm_control_uop_t` | `GEMV_control_uop_t` |
| `acp_write_en_wire`, `acp_base_addr_wire`, `acp_end_addr` | `write_en`, `base_addr`, `end_addr` in `acp_uop_t` |
| `npu_write_en_wire`, `npu_base_addr_wire`, `npu_end_addr` | `write_en`, `base_addr`, `end_addr` in `npu_uop_t` |

## Added v002 Payloads

### MEMSET

`memset_op_x64_t` was added with:

- `dest_cache`
- `dest_addr`
- `a_value`
- `b_value`
- `c_value`
- `reserved`

The value fields use new 16-bit typedefs: `a_value_t`, `b_value_t`, and
`c_value_t`. `dest_cache_e` names the supported shape-cache destinations:
`data_to_fmap_shape` and `data_to_weight_shape`.

### CVO

`cvo_op_x64_t` was added with:

- `cvo_func`
- `src_addr`
- `dst_addr`
- `length`
- `flags`
- `async`

The CVO package vocabulary also adds:

- `length_t`
- `cvo_func_e`
- `cvo_flags_t`
- `cvo_control_uop_t`
- CVO memory routes: `from_L2_to_CVO` and `from_CVO_res_to_L2`

Current CVO function codes are `CVO_EXP`, `CVO_SQRT`, `CVO_GELU`,
`CVO_SIN`, `CVO_COS`, `CVO_REDUCE_SUM`, `CVO_SCALE`, and `CVO_RECIP`.

## Memory and Shape Vocabulary

The intermediate v001 memory-control package used `isa_memctrl.svh` as a
separate included package. The current v002 package makes memory routing
part of `isa_pkg.sv` itself:

- `data_dest_e`
- `data_source_e`
- `data_route_e`
- `memory_control_uop_t`
- `memory_set_uop_t`
- `acp_uop_t`
- `npu_uop_t`

The current memory-control uop width is named as `MemoryUopWidth = 49`,
matching route, destination address, source address, shape pointer, and
async fields.

The current package also adds explicit shape vocabulary:

- `shape_dim_t` for one 17-bit dimension
- `shape_xyz_t` for a packed `{ z, y, x }` triplet

These names document the shape constant RAM contract that earlier code
represented with raw values and dedicated fmap/weight shape routes.

## Current Package Boundary

The current `isa_pkg.sv` is the single package imported by RTL consumers.
It no longer includes `isa_x32.svh`, `isa_x64.svh`, or `isa_memctrl.svh`;
the package header marks those files as legacy vocabulary that should not
be extended. The package now owns the opcode table, instruction layouts,
routing enums, flag structs, and micro-op structs in one place.
