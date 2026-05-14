# PS-Side NPU Dispatch

This package exposes the Gemma-facing NumPy API for the pccx v002 KV260 NPU
runtime.  The hardware path is experimental and golden-vector gated; the CPU
fallback is always present so the daemon can run without a loaded bitstream.

## Interface

- `npu_gemm(W, X, layer_idx=None)` returns `X @ W` as float32 for
  `W [K_in, M_out]` and `X [B, K_in]`.
- `npu_gemv(W, x, layer_idx=None)` returns `x @ W` as float32 for
  `W [K_in, M_out]` and `x [K_in]`.
- `npu_cvo(op, x)` supports `EXP`, `SQRT`, `GELU`, `SIN`, `COS`,
  `REDUCE_SUM`, `SCALE`, and `RECIP`.
- `npu_status()` returns `mmio_hex`, `busy`, `done`, `available`, and
  `last_cycle_count`.
- `npu_available()` returns true only when `/dev/uio4` can be opened and
  mmapped and the deployed bitstream file hash matches the expected v12d list.

## NPU Path And CPU Fallback

When `/dev/uio4` is absent, mmap fails, the bitstream hash is not in the
expected list, or `PCCX_NPU_FORCE_FALLBACK=1` is set, all public calls use the
NumPy reference implementation in `cpu_fallback.py`.

If the board is present and `PCCX_NPU_EXPERIMENTAL_DISPATCH=1` is set, the
runtime can submit the matching v002 ISA word through AXIL_CMD_IN and wait for
AXIL_STAT_OUT `DONE`.  The returned tensor still uses the NumPy golden result
until DMA buffer ownership and result-region reads are verified on KV260.
This avoids claiming hardware numerical results before the golden-vector gate.

For large matrix shapes the fallback path includes M/K/N tiling with
conservative defaults.  The tile sizes are configurable through
`PCCX_NPU_TILE_M`, `PCCX_NPU_TILE_K`, and `PCCX_NPU_TILE_N`.

## Command/Status AXIL FIFO Layout

Each DataMover helper is a 4 KiB AXI-Lite page with FIFO depth 8:

| Offset | Name | Access | Description |
| --- | --- | --- | --- |
| `0x000` | `CMD_LO` | RW | Staged descriptor bits `[31:0]`. |
| `0x004` | `CMD_HI` | RW | Staged descriptor bits `[63:32]`. |
| `0x008` | `CMD_EXT` | RW | Staged descriptor bits `[71:64]` in `[7:0]`. |
| `0x00c` | `CMD_PUSH` | W | Push the staged descriptor into the command FIFO. |
| `0x010` | `STS_POP` | R | Pop one 8-bit DataMover status word. |
| `0x014` | `FLAGS` | R | command empty/full, status empty/full, sticky errors. |
| `0x018` | `CMD_LVL` | R | Command FIFO occupancy. |
| `0x01c` | `STS_LVL` | R | Status FIFO occupancy. |
| `0x020` | `ERR_W1C` | RW1C | Clear sticky error bits. |

The 72-bit DataMover descriptor packs `tag[71:68]`, reserved zero bits
`[67:64]`, `addr[63:32]`, `drr[31]=0`, `eof[30]`, `dsa[29:24]=0`,
`type[23]=1`, and `btt[22:0]`.

## BD Address Discovery

Compiled defaults match the v12d BD:

| Window | Default base |
| --- | --- |
| NPU AXI-Lite control | `0xA0000000` |
| `cmdsts_hp0` | `0xA0001000` |
| `cmdsts_hp1` | `0xA0002000` |
| `cmdsts_hp2` | `0xA0003000` |
| `cmdsts_hp3` | `0xA0004000` |
| `cmdsts_acp_fmap` | `0xA0005000` |
| `cmdsts_acp_result` | `0xA0006000` |

`discover_address_map()` first reads
`/sys/class/uio/uio4/maps/map0/addr` and `size`.  If UIO sysfs is not present,
it scans device-tree `ranges` blobs for a fabric address near the known NPU
aperture.  If discovery fails it logs a warning and returns the compiled v12d
defaults.
