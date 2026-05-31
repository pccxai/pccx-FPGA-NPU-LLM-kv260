# 2026-05-31 v22 Night Prebuild Evidence

This note records the board-free verification state after the v22 KV260
DataMover diagnosis session.

## Scope

- Board: KV260 was powered off before the night run, so this note records
  source, simulation, and generated-BD evidence only.
- Latest board boundary before poweroff: v22 AxPROT image was built, deployed,
  and loaded, but DataMover board smoke still returned HP0 `INTERR`, ACP
  `DECERR`, `OKAY 0/4`.
- AxPROT conclusion: PS-boundary `AxPROT=3'b010` is implemented and proven in
  generated/routed gates, but it is not sufficient to resolve the board
  DataMover access error.

## Verification Result

Latest board-free gate:

```text
debug/run_prebuild_gates.sh
OVERALL: PASS prebuild gates
```

Gate contents:

- Python descriptor/status contracts: PASS in the deploy workspace, 26/26.
- xsim unit suite: PASS 9 / FAIL 0.
- Generated BD topology gate: PASS.
- Generated AXI attribute gate: PASS.
- Generated AXI address gate: PASS.
- Generated AXI transaction gate: PASS.

## New Finding

The directed `tb_mem_GLOBAL_cache` testbench found a real L2 stream contract
risk before the next bitstream build.

Symptom under test:

- ACP writes three 128-bit L2 words.
- ACP result stream is held not-ready before L2-to-host readback.
- The first failing implementation could repeat stale/pre-read data when the
  read-valid pipeline opened.

Root cause:

- `mem_L2_cache_fmap` kept XPM TDP port enables permanently high.
- While the read FSM was blocked by downstream backpressure, the XPM output
  pipeline could pre-read the base address before a read beat was issued.
- Later valid beats could align with that stale/pre-read data.

Validated fix in the deploy workspace:

- Expose explicit ACP/NPU port enables in `mem_L2_cache_fmap`.
- Drive XPM enable from issued transfers.
- Keep read enable asserted while the outstanding read-valid pipeline flushes.

Directed TB coverage after the fix:

- ACP host-to-L2 write.
- ACP L2-to-host read with initial result backpressure.
- NPU L2 read path.
- Pointer advance.
- Final-word `tlast`.

Result:

```text
tb_mem_GLOBAL_cache RESULT: PASS
```

## GitHub Tracking

- KV260 memory verification issue: <https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/25>
- v002 public RTL tracking issue: <https://github.com/pccxai/pccx-v002/issues/9>
- Bring-up epic: <https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues/43>

## Next Gate

Do not start a new full build from a guess. The next build candidate should be
selected only after the remaining board-free module checks identify a concrete
RTL or BD contract change.
