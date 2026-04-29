# pccx — Sail ISA model

This directory holds the **formal specification** of the pccx v002 ISA
in the [Sail ISA semantics language](https://sail-lang.org/).  Sail is
the same DSL that underpins the official **RISC-V**, **Arm**, **CHERI**,
and **Morello** ISA specifications; using it for pccx makes the model
cross-checkable against those tools and their formal-verification back
ends (Isabelle/HOL, Coq, HOL4, Lem, C emulator, SystemVerilog).

## Ground truth

The Sail model refines the SystemVerilog package

```
../../hw/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE/isa_pkg.sv
```

Each `typedef` there has a 1:1 counterpart in `src/pccx_types.sail` so
a width mismatch fails Sail's type checker before it fails RTL.

## Layout

```
formal/sail/
├── pccx.sail_project        module manifest (prelude → types → regs → decode → execute → test)
├── Makefile                 make check / doc / clean
├── src/
│   ├── prelude.sail         minimal bitvector helpers
│   ├── pccx_types.sail      opcodes, body structs, flags, CVO funcs
│   ├── pccx_regs.sail       cycle / pc / committed_any / per-unit commit counters / last-operand observables
│   ├── pccx_decode.sail     64-bit word → typed `instr` union
│   └── pccx_execute.sail    executable semantics (3rd increment: per-opcode operand latching)
└── tests/
    └── smoke_decode.sail    typecheck-only opcode-table smoke test
```

## Running

```bash
# once, per shell:
eval $(opam env)

# type-check the whole model:
make check
# or: sail --project pccx.sail_project --all-modules --just-check

# (future) emit HTML / LaTeX model docs:
make doc
```

## Scope today

- done  Base types + register state + full 5-opcode decoder
  (`OP_GEMV`, `OP_GEMM`, `OP_MEMCPY`, `OP_MEMSET`, `OP_CVO`).
- done  Execute semantics first increment — non-interpreting cycle
  counter advance + `record_event` stub. See `src/pccx_execute.sail`.
- done  Execute semantics second increment — per-opcode `execute_*`
  dispatch, per-unit commit counters (`mac_ops_committed`,
  `dma_ops_committed`, `sfu_ops_committed`), async-flag observation,
  PC advance.
- done  Execute semantics third increment — decoded operands latched
  into per-unit "last committed" observables (`last_mac_{dst,src}`,
  `last_dma_{dst,src}`, `last_sfu_{dst,src,length}`) so refinement
  proofs have concrete decoded-field witnesses, not just commit
  counts.  Still non-interpreting w.r.t. MAC data; numeric-effect
  modelling follows the `.pccx` writer.
- WIP   Wire `record_event` to a real `.pccx` trace writer (C back end)
  so pccx-lab can diff RTL-simulated traces against the Sail reference.
- planned  Isabelle / Coq / SystemVerilog back-end exports — pccx-lab
  Phase 5E foundation (Sail → SV refinement-checked custom NPU).

## Why Sail, why now

pccx is an open NPU architecture targeting edge LLM inference — the
kind of system where silicon bugs cost tape-out cycles you cannot
afford.  A formal Sail model lets us:

- **Type-check the ISA** — opcode/field widths cannot drift between the
  RTL and the docs.
- **Generate executable reference simulators** from the same source
  the RTL refines against (Sail's C / OCaml back-ends).
- **Prove correctness properties** via Sail's Isabelle / Coq / HOL4
  back-ends once the execute pass lands.
- **Advertise rigour** — the site [hkimw.github.io/pccx][site]
  lists Sail alongside Arm / RISC-V / CHERI as one of the ISAs
  authored with real ISA-semantics tooling.

[site]: https://hkimw.github.io/pccx/en/
