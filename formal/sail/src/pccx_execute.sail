/* ──────────────────────────────────────────────────────────────────────── *
 * pccx — execute.sail
 * ──────────────────────────────────────────────────────────────────────── *
 * Executable semantics — second increment.
 *
 * 1st increment shipped the cycle counter + `record_event` stub + a
 * single monolithic `step` that committed any opcode uniformly.  This
 * 2nd increment introduces **concrete MAC / DMA / SFU effects**:
 *
 *   - Per-opcode `execute_<op>` helpers, one `match` arm each.
 *   - Per-unit commit counters bumped in `pccx_regs.sail`:
 *       * MAC : GEMV, GEMM
 *       * DMA : MEMCPY, MEMSET
 *       * SFU : CVO
 *   - Async-flag observation: MEMCPY and CVO raise `async_inflight`
 *     when their ASYNC bit is set.
 *   - Program-counter advance: one architectural instruction step per
 *     `execute_*` call.
 *
 * The execute pass is still **non-interpreting w.r.t. data** — we do
 * not model the MAC array's INT4 × INT8 numeric outputs here; that
 * work belongs to the 3rd increment once the RTL MAC / CORDIC math is
 * frozen.  What this commit gives us is a refinement target: any RTL
 * dispatcher that claims to commit a GEMV must bump `mac_ops_committed`
 * exactly once.
 * ──────────────────────────────────────────────────────────────────────── */

/* ── Low-level effect hooks ──────────────────────────────────────────── */

/* `.pccx` trace hook — still a pure stub at this increment.  The
 * signature is frozen so M4.9 can slot in the real writer without
 * touching callers. */
function record_event(c : bitvector(64), op : bitvector(4), body : body_bits) -> unit = {
  let _ = c;
  let _ = op;
  let _ = body;
}

/* Advance the cycle counter by `n`. */
function advance_cycle(n : bitvector(64)) -> unit = {
  cycle = cycle + n
}

/* Advance the program counter by one instruction slot. */
function advance_pc() -> unit = {
  pc = pc + 0x0001
}

/* 32-bit "one" — used to bump the per-unit commit counters.  Sail
 * width rules prefer an explicit literal over an arithmetic coercion. */
let ops_one : bitvector(32) = 0x0000_0001

/* ── Per-opcode latency (coarse, latency-typed not latency-accurate) ──
 * Kept in lock-step with the 1st increment's values. */
function estimated_latency(i : instr) -> bitvector(64) = {
  match i {
    Gemv(_)   => 0x0000_0000_0000_0080,  /* ≈ 128 cy — GEMV tile      */
    Gemm(_)   => 0x0000_0000_0000_0100,  /*   256 cy — GEMM tile      */
    Memcpy(_) => 0x0000_0000_0000_0040,  /*    64 cy — DMA burst      */
    Memset(_) => 0x0000_0000_0000_0020,  /*    32 cy — cache preset   */
    Cvo(_)    => 0x0000_0000_0000_0060,  /*    96 cy — CVO scalar     */
  }
}

/* ── Opcode-tag / body round-trip (unchanged from 1st increment) ─────── */

function opcode_tag(i : instr) -> bitvector(4) = {
  match i {
    Gemv(_)   => encdec_opcode(OP_GEMV),
    Gemm(_)   => encdec_opcode(OP_GEMM),
    Memcpy(_) => encdec_opcode(OP_MEMCPY),
    Memset(_) => encdec_opcode(OP_MEMSET),
    Cvo(_)    => encdec_opcode(OP_CVO),
  }
}

function reencode_body(i : instr) -> body_bits = {
  match i {
    /* Widths add up explicitly; see 1st-increment commit for the full
     * rationale on the zero-tails. */
    Gemv(b)   => b.dest_reg  @ b.src_addr  @ sail_zeros(26),
    Gemm(b)   => b.dest_reg  @ b.src_addr  @ sail_zeros(26),
    Memcpy(b) => sail_zeros(1)   /* from_device (placeholder) */
               @ sail_zeros(1)   /* to_device   (placeholder) */
               @ b.dest_addr
               @ b.src_addr
               @ sail_zeros(24),
    Memset(b) => b.dest_cache @ b.dest_addr @ b.a_value @ b.b_value @ b.c_value @ sail_zeros(4),
    Cvo(b)    => encdec_cvo_func(b.cvo_func) @ b.src_addr @ b.dst_addr @ b.length @ sail_zeros(6),
  }
}

/* ── Async-flag observation helpers ─────────────────────────────────── */

/* Set `async_inflight` iff the body's async field is ASYNC_OP.  Model
 * remains single-step; the flag is an observable, not a scheduler. */
function note_async(a : async_e) -> unit = {
  match a {
    SYNC_OP  => (),
    ASYNC_OP => { async_inflight = bitone },
  }
}

/* ── Per-opcode execute bodies ──────────────────────────────────────── *
 *
 * Each helper:
 *   1. records the trace event (opcode tag + re-encoded body)
 *   2. advances the cycle counter by the estimated latency
 *   3. bumps the relevant unit's commit counter by 1
 *   4. advances the PC and marks `committed_any`
 *   5. (async ops only) raises `async_inflight`
 *
 * The execution order here is the architectural commit order — the RTL
 * refinement proof will assert: any RTL commit event on unit U bumps
 * exactly one counter, in PC order, with the same opcode tag recorded.
 * ──────────────────────────────────────────────────────────────────── */

function execute_gemv(b : gemv_body_t) -> unit = {
  record_event(cycle, encdec_opcode(OP_GEMV), reencode_body(Gemv(b)));
  advance_cycle(estimated_latency(Gemv(b)));
  mac_ops_committed = mac_ops_committed + ops_one;
  advance_pc();
  committed_any = bitone;
}

function execute_gemm(b : gemm_body_t) -> unit = {
  record_event(cycle, encdec_opcode(OP_GEMM), reencode_body(Gemm(b)));
  advance_cycle(estimated_latency(Gemm(b)));
  mac_ops_committed = mac_ops_committed + ops_one;
  advance_pc();
  committed_any = bitone;
}

function execute_memcpy(b : memcpy_body_t) -> unit = {
  record_event(cycle, encdec_opcode(OP_MEMCPY), reencode_body(Memcpy(b)));
  advance_cycle(estimated_latency(Memcpy(b)));
  dma_ops_committed = dma_ops_committed + ops_one;
  note_async(b.async);
  advance_pc();
  committed_any = bitone;
}

function execute_memset(b : memset_body_t) -> unit = {
  record_event(cycle, encdec_opcode(OP_MEMSET), reencode_body(Memset(b)));
  advance_cycle(estimated_latency(Memset(b)));
  dma_ops_committed = dma_ops_committed + ops_one;
  advance_pc();
  committed_any = bitone;
}

function execute_cvo(b : cvo_body_t) -> unit = {
  record_event(cycle, encdec_opcode(OP_CVO), reencode_body(Cvo(b)));
  advance_cycle(estimated_latency(Cvo(b)));
  sfu_ops_committed = sfu_ops_committed + ops_one;
  note_async(b.async);
  advance_pc();
  committed_any = bitone;
}

/* ── Top-level dispatcher ────────────────────────────────────────────── *
 *
 * `step` remains the outermost driver — same signature as the 1st
 * increment.  The switch from a monolithic body to per-opcode helpers
 * is invisible to callers and keeps the commit trailer stable for the
 * future trace writer. */
val step : instruction_bits -> unit
function step(word) = {
  let i = decode(word);
  match i {
    Gemv(b)   => execute_gemv(b),
    Gemm(b)   => execute_gemm(b),
    Memcpy(b) => execute_memcpy(b),
    Memset(b) => execute_memset(b),
    Cvo(b)    => execute_cvo(b),
  }
}
