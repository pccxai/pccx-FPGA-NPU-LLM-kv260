# RTL Interface & Readability Plan — pccx v002

_Internal engineering plan. Architecture-readability batch._

This document is the single place where the RTL interface and
readability landscape is recorded — what already exists, what
recurring patterns are open candidates for typed bundles or
interfaces, and the order in which the migrations should land. It is
intentionally a plan, not a code change: the only RTL the batch ships
alongside it is a single small vocabulary scaffold called out
explicitly in §6.1.

## 1. Goals (per the batch directive)

The pccx v002 RTL is already structurally sound — the protected
compute math is well-documented, the boundary modules carry contract
headers, and the ISA / data / memory packages are organised. The
direction this plan optimises for is **readability for the next
engineer**, not feature work:

- A new contributor should be able to read a module's port list and
  see the protocol, not just bits.
- Recurring bundles (handshake triples, shape triplets, status
  groups) should travel through the design as named types, so a
  rename or a width change touches one line, not forty.
- Compute math stays untouched (CLAUDE.md §6.2).
- No mass internal-signal rename (review-noise rule).

## 2. SystemVerilog interpretation rule

The batch directive includes an explicit reminder:

> Do NOT blindly use software-style classes in synthesizable RTL. For
> synthesizable RTL, translate OOP principles into package, typedef
> struct packed, enum, interface, modport, parameter/localparam
> vocabulary, deep module boundary, clear module contract comments.

Operationally this means:

- **Class OOP** (`class`, `virtual`, polymorphism) → only inside
  `tb_*` testbench files and reference models. Never in `.sv` files
  that are listed in `hw/vivado/filelist.f`.
- **Encapsulation** → `package` + `typedef struct packed` +
  `interface` + `modport`.
- **Inheritance** → `parameter` + `generate` + composition of small
  modules. There is no synthesizable inheritance.
- **Polymorphism** → `parameter`-driven specialisation, not virtual
  dispatch.

If a future verification batch wants class-based scoreboards / drivers
inside testbenches, that is a separate decision tracked in §10 below.

## 3. Existing interface infrastructure

The codebase already has a small but solid set of typed boundaries:

### 3.1 `axis_if` — AXI-Stream

Defined in `hw/rtl/NPU_Controller/npu_interfaces.svh`.

```systemverilog
interface axis_if #(parameter DATA_WIDTH = 128) ();
  logic [DATA_WIDTH-1:0]     tdata;
  logic                      tvalid;
  logic                      tready;
  logic                      tlast;
  logic [(DATA_WIDTH/8)-1:0] tkeep;
  modport slave  (input tdata, tvalid, tlast, tkeep, output tready);
  modport master (output tdata, tvalid, tlast, tkeep, input tready);
endinterface
```

Used by `mem_dispatcher` (HP / ACP), `mem_HP_buffer`, and most of the
streaming boundary modules. Style is Keller-aligned: one parameter
controls width, two modports document direction, the body holds one
real protocol.

### 3.2 `axil_if` — AXI4-Lite

Same file. Five-channel (AW / W / B / AR / R) + slave / master
modports. Used by `AXIL_CMD_IN`, `AXIL_STAT_OUT`,
`ctrl_npu_frontend`. Equally clean.

### 3.3 ISA / type vocabulary

`isa_pkg` (248 lines) already provides:

- Address types: `dest_addr_t` / `src_addr_t` / `addr_t` /
  `ptr_addr_t` / `parallel_lane_t`.
- Direction enums: `from_device_e`, `to_device_e`, `async_e`,
  `dest_cache_e`.
- Opcode and CVO function enums.
- Routing enum (`data_route_e`).
- Instruction layouts (`*_op_x64_t`).
- Micro-op structs (`*_uop_t`).
- ACP / NPU transfer uops.

`mem_pkg`, `dtype_pkg`, `vec_core_pkg`, and `device_pkg` cover the
geometry, data widths, and pipeline counts.

`perf_counter_pkg` (Stage C addition) carries `handshake_counter_t`
and the opt-in counter parameter knobs.

**Take-away:** the building blocks are present. The next
readability step is to (a) reuse them more consistently across
modules that still hand-roll equivalent bundles, and (b) add named
types for the small set of bundle patterns that recur but do not yet
have one.

## 4. Repeated signal-bundle patterns

A pass over module port lists in MAT_CORE / VEC_CORE / CVO_CORE /
PREPROCESS / MEM_control surfaced these recurring shapes. Each is a
candidate for the typed-vocabulary or typed-interface treatment.

### 4.1 Push-only stream (valid + data, no ready)

Most prevalent pattern in compute-internal pipelines. Counts come
from a `grep` over the boundary modules. Examples:

| Module                            | Port group                                             |
|---|---|
| GEMV_top                          | `IN_weight_valid`, `IN_weight[128]`                    |
| GEMV_accumulate                   | `IN_valid`, `IN_data[18]`                              |
| GEMM_systolic_array               | `i_weight_valid`, `i_weight_upper`, `i_weight_lower`   |
| GEMM_accumulator                  | `IN_valid`, `IN_partial[36]`                           |
| CVO_cordic_unit                   | `IN_valid`, `IN_angle_bf16[15:0]`                      |
| preprocess_bf16_fixed_pipeline    | `s_axis_tvalid`, `s_axis_tdata[256]` (already AXIS!)   |

Many of these can stay raw (latency-critical, push-only by design),
but a couple of `axis_if`-style consolidations would help: see §6.

### 4.2 Full handshake (valid + ready + data)

Already mostly migrated to `axis_if`. The remaining hand-rolled
instances are between `mem_dispatcher` ↔ `CVO_top`:

```systemverilog
// mem_dispatcher.sv (snippet)
output logic [15:0] OUT_cvo_data,
output logic        OUT_cvo_valid,
input  logic        IN_cvo_data_ready,

input  logic [15:0] IN_cvo_result,
input  logic        IN_cvo_result_valid,
output logic        OUT_cvo_result_ready,
```

Both pairs are full handshake on a 16-bit bus. Two `axis_if`
instances (`axis_if #(.DATA_WIDTH(16))`) replace ten lines and four
modules' worth of port discipline.

### 4.3 Shape triplet (`val0 / val1 / val2`)

Both `fmap_array_shape` and `weight_array_shape` declare:

```systemverilog
input  logic [16:0] wr_val0,  // shape: x
input  logic [16:0] wr_val1,  // shape: y
input  logic [16:0] wr_val2,  // shape: z
```

The two modules are byte-identical apart from the module name
(Keller-style shallow wrapper, see Stage C dead-module inventory
§2.4 / KELLER §6.3.1). A `shape_xyz_t` typedef in `isa_pkg`
collapses the triplet into one named struct and is the natural first
step toward the deferred shape-RAM consolidation.

### 4.4 ACP / NPU memory port bundle

`mem_dispatcher` exposes both an ACP read/write port and an NPU
read/write port to `mem_GLOBAL_cache`:

```systemverilog
output logic         OUT_acp_we,
output logic [16:0]  OUT_acp_addr,
output logic [127:0] OUT_acp_wdata,
input  logic [127:0] IN_acp_rdata,

output logic         OUT_npu_we,
output logic [16:0]  OUT_npu_addr,
output logic [127:0] OUT_npu_wdata,
input  logic [127:0] IN_npu_rdata,
```

These are textbook BRAM/URAM port groups. Either:

- A `interface l2_port_if(...)` with `master` (controller) and
  `slave` (memory) modports.
- A `typedef struct packed` for write side
  (`l2_write_t { we, addr, wdata }`) plus a flat `rdata` return.

The interface form is more Keller-consistent (one boundary, two
modports, write/read are inseparable on a single-port BRAM).

### 4.5 uop request channel

Common across `Global_Scheduler` → engine cores:

```systemverilog
input  GEMV_op_x64_t IN_gemv_op_x64,
input  logic         IN_gemv_op_x64_valid,
input  GEMM_op_x64_t IN_gemm_op_x64,
input  logic         IN_gemm_op_x64_valid,
input  cvo_op_x64_t  IN_cvo_op_x64,
input  logic         IN_cvo_op_x64_valid,
```

Each `*_op_x64_t` is already typed; only the `*_valid` bit is loose.
A small `interface uop_request_if #(type uop_t)` would name the
boundary explicitly, but the win is modest because the issue is
already a one-cycle pulse with no ready.

### 4.6 Status / observability bundle

Recurring `OUT_busy`, `OUT_done`, `OUT_*_valid`, `OUT_fifo_full`
group on most engine modules. A `module_status_t` struct
(`{busy, done, error}`) would standardise the shape across CVO_top,
GEMV_top, GEMM_systolic_top, mem_dispatcher.

Worth doing **only** alongside the Stage D counter-MVP wiring so the
struct lands together with its first consumers; otherwise the
typedef has no users and decays.

### 4.7 Clock / reset bundle

`(clk_core, rst_n_core, clk_axi, rst_axi_n)` — the cross-clock
modules in MEM_control take all four. A bundling interface would
help, but SystemVerilog interface ports cannot carry clocks
gracefully across many tools. Recommend **leave raw** with consistent
names; document the convention in `npu_interfaces.svh`.

## 5. Existing prefix / naming convention

Per the user's signal-prefix memory:

- `IN_<name>` — bare module input port.
- `OUT_<name>` — bare module output port.
- `LOC_<name>` — internal local signal.

This batch:

- Applies the prefix only to **new RTL** (perf_counter_pkg adheres,
  the Stage C scaffolds adhere).
- Does **not** mass-rename existing internal signals (review-noise
  rule). Pre-existing modules keep their `i_` / `o_` /
  `axis_if` style.

## 6. Tiered migration plan

Each tier is "land in its own commit, validate, then move on."
Higher tier numbers cost more or carry more risk.

### 6.1 Tier 1 — typed vocabulary additions (this batch)

**Lands in this batch as a single small commit:**

- Add `shape_dim_t` and `shape_xyz_t` typedefs to `isa_pkg`. The
  existing shape RAM modules can stay on raw `[16:0]` ports; future
  shape-RAM consolidation (deferred) and any new module that
  consumes shape data uses the named types from day one.

Vocabulary-only. No module imports the new types yet, so xvlog and
the existing 6 TBs stay green.

### 6.2 Tier 2 — interface migration (next batches, one boundary at a time)

Order chosen to minimise blast radius:

1. **CVO 16-bit data + result channels** → `axis_if #(.DATA_WIDTH(16))`.
   Touches `mem_dispatcher` and `CVO_top` only. No TB exists for this
   path; pair with a smoke TB in the same commit.
2. **L2 cache memory ports (ACP + NPU)** → `interface l2_port_if`.
   Touches `mem_dispatcher`, `mem_GLOBAL_cache`, and possibly
   `mem_L2_cache_fmap`. Larger blast radius — gate behind a soak
   period after #1 lands.
3. **uop request channels** → `interface uop_request_if`. Optional;
   gain is small. Defer until at least one of #1 or #2 ships.

### 6.3 Tier 3 — typed status / observability bundle (with Stage D MVP)

Add `module_status_t` to a new section of `perf_counter_pkg` (or a
sibling `npu_obs_pkg` if it grows). Wire it together with the first
counter-MVP module so the struct has at least one consumer at
introduction time. Avoid landing the typedef alone.

### 6.4 Tier 4 — module consolidation

Shape RAM consolidation (`fmap_array_shape` + `weight_array_shape`
→ `shape_const_ram`) becomes a one-commit mechanical job once
`shape_xyz_t` is in. Tracked separately as an architecture issue per
the Stage C decisions memo.

## 7. Package / parameter readability direction

The package layout (A const_svh → B device_pkg → C type_pkg → D
pipeline_pkg → E obs_pkg) is already well-organised. The directions
worth pursuing in later batches:

- **Hardware vs algorithm separation** — keep `kv260_device.svh` and
  `device_pkg.sv` strictly for KV260 board / SoC widths, and
  `dtype_pkg` / `vec_core_pkg` for algorithmic / dataflow constants.
  Today this is mostly true; the audit caught no leak.
- **Magic numbers still embedded** (per KELLER §6.4) — best wrapped
  in a topic-specific package when they reappear in the file you're
  already touching, not as a churn-only sweep:
  - `mem_CVO_stream_bridge` — FIFO depth `2048`, vector length cap
    `2048`, deser counter widths.
  - `mem_u_operation_queue` — FIFO depth `128`, prog-full `100`.
  - `mat_result_normalizer` — exponent offset `26`
    (= `dtype_pkg::FixedMantWidth - 1`).
  - `FROM_mat_result_packer` — group size `8` (= `ARRAY_SIZE / 4`).
- **Compatibility shims** — `GLOBAL_CONST.svh` is staged for removal
  (5-phase plan in `docs/internal/global_const_migration_plan.md`).
  `DEVICE_INFO.svh` was removed in Stage C. Do not delete more shims
  recklessly; honour the staged plan.

## 8. Module contract comments

Stage C standardised the contract block (Purpose / Spec ref / Clock /
Reset / Latency / Throughput / Handshake / Backpressure / Reset state /
Counters / Errors / Notes / Protected) across the boundary modules.

Convention for the next batch:

- New modules adopt the same contract block at write time.
- A module that gains a new counter knob updates its `Counters:`
  line.
- A module that swaps from raw handshake → `axis_if` updates its
  `Handshake:` line to point at the interface name.

Contract bullets must stay short. Two lines per bullet maximum;
multi-paragraph prose belongs in `docs/internal/`.

## 9. Verification side — class OOP boundary

For testbenches and reference models:

- Class-based OOP is **fine** in `hw/tb/*.sv` and any
  `hw/sim/work/<tb>/*.sv` model. Driver / monitor / scoreboard
  classes are an industry-standard idiom.
- The package compile order in `hw/vivado/filelist.f` carries
  synthesizable RTL only; testbench class libraries should stay out
  of it.
- A future verification-architecture batch may add a small
  `pccx_tb_pkg.sv` containing the common driver / monitor / golden
  model classes. This plan does **not** start that work.

## 10. Implementation order (concrete)

1. **This batch** — land this plan + `shape_dim_t` / `shape_xyz_t`
   in `isa_pkg`. Validate. Push.
2. **Next batch (counter MVP, already queued)** — wire
   `handshake_counter_t` into one engine module behind
   `EnablePerfCounters`. Optionally land `module_status_t` if the
   wiring naturally needs it.
3. **Next-after-next** — Tier-2 #1 (CVO 16-bit `axis_if`
   migration) + the smoke TB that drives it.
4. **Later** — Tier-2 #2 (L2 port `l2_port_if`), Tier-4 (shape RAM
   consolidation), GLOBAL_CONST Phase 2.
5. **Whenever a verification batch starts** — author the
   `pccx_tb_pkg` class library outside the synthesizable filelist.

## 11. Non-goals (hard limits, restated)

This plan **does not** propose any of:

- Top-level interface rewrite.
- Register map rewrite.
- Compute math edits.
- Mass internal-signal renaming.
- Shape RAM integration in this batch (only the typedef vocabulary).
- Broad GLOBAL_CONST.svh deletion.
- Public release / tag work.
- Performance / timing / KV260-readiness claims.

Anything in the audit above that does not honour these limits stays
in the deferred list — see §10.
