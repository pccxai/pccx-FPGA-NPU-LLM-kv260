# PS-Side NPU Dispatch

This package exposes the Gemma-facing API for the pccx v002 KV260 NPU runtime.
The hardware backend is a self-contained token path: weights are loaded into
NPU L2 once, prompt tokens initialize NPU-resident activation/KV state, and each
NEXT_TOKEN command runs the full forward path inside the NPU.  The host reads
back only one 32-bit token per generated token.

## Interface

- `load_weights_to_l2(weights)` starts the one-time host-to-NPU L2 weight-load
  phase.
- `init_activation(prompt_tokens)` resets NPU-resident KV state and loads prompt
  token IDs.
- `run_one_token_step()` issues NEXT_TOKEN and returns the generated 32-bit
  token from status.
- `npu_gemm`, `npu_gemv`, and `npu_cvo` are CPU fallback compatibility helpers
  for CPU-mode Gemma modules; they do not issue hardware matrix commands.
- `npu_status()` returns `mmio_hex`, `busy`, `done`, `available`, and
  token-step status metadata.
- `npu_available()` returns true only when `/dev/uio4` can be opened and
  mmapped and the deployed bitstream file hash matches the expected v12d list.

## NPU Path And CPU Fallback

When `/dev/uio4` is absent, mmap fails, the bitstream hash is not in the
expected list, or `PCCX_NPU_FORCE_FALLBACK=1` is set, CPU-mode calls use the
NumPy reference implementation in `cpu_fallback.py`.

The NPU backend is selected only when `PCCX_NPU_TOKEN_BACKEND=1` and token-step
MMIO is available, or when `PCCX_NPU_SIM_BACKEND=1` enables the local
MMIO-compatible token simulator.  There is no host tensor readback path for
Gemma generation.

For large matrix shapes, CPU-mode fallback uses NumPy reference math.

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
