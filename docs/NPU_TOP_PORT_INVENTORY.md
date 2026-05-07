# NPU_top Port Inventory - KV260

This note documents every public port on `hw/rtl/NPU_top.sv` as of the
`docs/top-rtl-port-inventory` branch. It is an RTL boundary inventory,
not a board constraint file or timing statement.

## Evidence Source

- Top module: `hw/rtl/NPU_top.sv`
- Interface definitions: `hw/rtl/NPU_Controller/npu_interfaces.svh`
- Width constants:
  `hw/rtl/Constants/compilePriority_Order/A_const_svh/kv260_device.svh`
  and `hw/rtl/Constants/compilePriority_Order/A_const_svh/npu_arch.svh`

## Interface Conventions

`axis_if` defaults to a 128-bit AXI4-Stream payload:

| Signal | Slave direction, NPU view | Master direction, NPU view | Width | Semantics |
| --- | --- | --- | ---: | --- |
| `tdata` | input | output | `DATA_WIDTH` = 128 | Stream payload. |
| `tvalid` | input | output | 1 | Source has a valid beat. |
| `tready` | output | input | 1 | Sink can accept a beat. |
| `tlast` | input | output | 1 | Packet or burst boundary marker. |
| `tkeep` | input | output | `DATA_WIDTH/8` = 16 | Byte-lane valid mask. |

A stream beat transfers when `tvalid && tready`.

`axil_if` defaults to a 12-bit address and 64-bit data AXI4-Lite
control interface:

| Channel | NPU slave inputs | NPU slave outputs | Widths | Semantics |
| --- | --- | --- | --- | --- |
| AW | `awaddr`, `awprot`, `awvalid` | `awready` | address 12, prot 3, valid/ready 1 | Write-address handshake from PS/HPM to NPU. |
| W | `wdata`, `wstrb`, `wvalid` | `wready` | data 64, strobe 8, valid/ready 1 | Write-data handshake carrying command words. |
| B | `bready` | `bresp`, `bvalid` | response 2, valid/ready 1 | Write response. Current frontend emits OKAY-only responses. |
| AR | `araddr`, `arprot`, `arvalid` | `arready` | address 12, prot 3, valid/ready 1 | Read-address handshake for status reads. |
| R | `rready` | `rdata`, `rresp`, `rvalid` | data 64, response 2, valid/ready 1 | Read-data response. Current top-level controller wiring does not drive `mmio_npu_stat` into this path. |

## Top-Level Ports

| Port | Direction | Width / Type | Clock or reset domain | Semantics |
| --- | --- | --- | --- | --- |
| `clk_core` | input | `logic`, 1 bit | Core clock, nominal 400 MHz | Main compute/control clock. It drives `npu_controller_top`, `Global_Scheduler`, the core side of `mem_dispatcher`, the HP FIFO read side, `preprocess_fmap`, GEMM, GEMV, and CVO logic. |
| `rst_n_core` | input | `logic`, 1 bit | Active-low reset for `clk_core` | Resets core-domain state. Child modules treat it as active-low reset; release is expected synchronous to `clk_core`. |
| `clk_axi` | input | `logic`, 1 bit | AXI/HP clock, nominal 250 MHz | AXI-side clock for `mem_dispatcher` and the HP weight CDC FIFOs. In the current source, the AXI-Lite frontend is instantiated under `clk_core` through `npu_controller_top`; this port is not passed to that frontend. |
| `rst_axi_n` | input | `logic`, 1 bit | Active-low reset for `clk_axi` | Resets AXI/HP-domain state in `mem_dispatcher` and the HP weight CDC FIFOs. |
| `i_clear` | input | `logic`, 1 bit | Synchronous to `clk_core` | Active-high soft clear. Passed to the controller, feature-map preprocessing, GEMM systolic engine, and CVO engine to clear soft state without asserting the external resets. |
| `S_AXIL_CTRL` | slave interface | `axil_if.slave`, default `ADDR_W=12`, `DATA_W=64` | Consumed by current RTL under `clk_core`; interface object also carries its own `clk` and `rst_n` pins | MMIO control-plane slave. The PS/HPM side writes 64-bit VLIW command words through AXI4-Lite write channels; `ctrl_npu_frontend` queues them and the decoder emits one-hot opcode-valid pulses. The read/status channel exists in the frontend, but `NPU_top` currently leaves `mmio_npu_stat` local and the controller passes zero status inputs downstream. |
| `S_AXI_HP0_WEIGHT` | slave interface | `axis_if.slave`, default `DATA_WIDTH=128`, `tkeep=16` | Source side in `clk_axi`; converted to `clk_core` by `mem_HP_buffer` | HP0 weight-stream ingress for the matrix core. Each 128-bit beat carries 32 packed INT4 weights (`128/4`). After CDC, `NPU_top` unpacks this stream into `hp0_weight_int4[0:31]` and drives the GEMM upper INT4 weight lane. |
| `S_AXI_HP1_WEIGHT` | slave interface | `axis_if.slave`, default `DATA_WIDTH=128`, `tkeep=16` | Source side in `clk_axi`; converted to `clk_core` by `mem_HP_buffer` | HP1 weight-stream ingress for the matrix core. Each 128-bit beat carries 32 packed INT4 weights. After CDC, `NPU_top` unpacks this stream into `hp1_weight_int4[0:31]` and drives the GEMM lower INT4 weight lane. |
| `S_AXI_HP2_WEIGHT` | slave interface | `axis_if.slave`, default `DATA_WIDTH=128`, `tkeep=16` | Source side in `clk_axi`; converted to `clk_core` by `mem_HP_buffer` | HP2 weight-stream ingress for the vector core. Each 128-bit beat carries 32 packed INT4 weights. After CDC, `NPU_top` unpacks this stream into `gemv_weight_A[0:31]` and drives GEMV lane A. |
| `S_AXI_HP3_WEIGHT` | slave interface | `axis_if.slave`, default `DATA_WIDTH=128`, `tkeep=16` | Source side in `clk_axi`; converted to `clk_core` by `mem_HP_buffer` | HP3 weight-stream ingress for the vector core. Each 128-bit beat carries 32 packed INT4 weights. After CDC, `NPU_top` unpacks this stream into `gemv_weight_B[0:31]` and drives GEMV lane B; GEMV lanes C and D are tied to zero in the current two-lane configuration. |
| `S_AXIS_ACP_FMAP` | slave interface | `axis_if.slave`, default `DATA_WIDTH=128`, `tkeep=16` | ACP ingress stream; consumed in core/dispatcher paths | Coherent ACP feature-map and DMA ingress stream. The current top-level wiring passes the same interface to `mem_dispatcher` for ACP/L2 movement and to `preprocess_fmap` for GEMM/GEMV feature-map ingestion. In preprocessing, 128-bit BF16 beats are padded into the lower half of an internal 256-bit FIFO word until the future two-beat merge path is wired. |
| `M_AXIS_ACP_RESULT` | master interface | `axis_if.master`, default `DATA_WIDTH=128`, `tkeep=16` | ACP result/DMA egress stream | Coherent ACP output stream back toward the host. The current top-level source routes this interface through `mem_dispatcher` for L2-to-host/result movement. GEMM result packing exists locally as `packed_res_data`/`packed_res_valid`, but those wires are not connected to this public output in this revision. |

## Stream Port Semantics

| Port | Payload interpretation | Ready/valid owner | Current sideband use |
| --- | --- | --- | --- |
| `S_AXI_HP0_WEIGHT` | 32 INT4 GEMM upper-lane weights per 128-bit beat | External HP source drives `tvalid`; NPU HP FIFO drives `tready` | `tdata`, `tvalid`, and `tready` are connected through the FIFO path. `tlast` and `tkeep` are present by interface contract. |
| `S_AXI_HP1_WEIGHT` | 32 INT4 GEMM lower-lane weights per 128-bit beat | External HP source drives `tvalid`; NPU HP FIFO drives `tready` | `tdata`, `tvalid`, and `tready` are connected through the FIFO path. `tlast` and `tkeep` are present by interface contract. |
| `S_AXI_HP2_WEIGHT` | 32 INT4 GEMV lane-A weights per 128-bit beat | External HP source drives `tvalid`; NPU HP FIFO drives `tready` | `tdata`, `tvalid`, and `tready` are connected through the FIFO path. `tlast` and `tkeep` are present by interface contract. |
| `S_AXI_HP3_WEIGHT` | 32 INT4 GEMV lane-B weights per 128-bit beat | External HP source drives `tvalid`; NPU HP FIFO drives `tready` | `tdata`, `tvalid`, and `tready` are connected through the FIFO path. `tlast` and `tkeep` are present by interface contract. |
| `S_AXIS_ACP_FMAP` | 128-bit ACP ingress beat for feature-map load or ACP/L2 DMA | External ACP source drives `tvalid`; NPU sink path drives `tready` | `preprocess_fmap` uses `tdata`, `tvalid`, and `tready`; `mem_dispatcher` also receives the interface for ACP movement. |
| `M_AXIS_ACP_RESULT` | 128-bit ACP egress beat for result or L2-to-host DMA | NPU `mem_dispatcher` drives `tvalid`; external ACP sink drives `tready` | Sideband availability follows `axis_if.master`; concrete sideband use is owned by `mem_dispatcher`. |

## Integration Notes

- The HP weight ports are all 128-bit AXI4-Stream slave ports at the
  top boundary. Internally, `mem_HP_buffer` creates four independent
  CDC FIFOs and `NPU_top` unpacks each 128-bit word into 32 INT4
  entries before feeding GEMM or GEMV.
- `S_AXIS_ACP_FMAP` is the only top-level ACP ingress stream. It is
  shared by the memory dispatcher and preprocessing path in the current
  RTL, so command scheduling must keep the active opcode path
  unambiguous.
- `M_AXIS_ACP_RESULT` is the only public ACP egress stream. The current
  public output path is memory-dispatcher-owned; local GEMM packer
  outputs are not yet wired to this top-level port.
