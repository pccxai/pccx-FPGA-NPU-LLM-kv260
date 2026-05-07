# Vivado Synthesis Resource Policy

This note records the host-resource requirements for the KV260 Vivado
synthesis path. It is based on observed local and CI behavior, including
a PR #108 synth attempt that returned `BLOCKED_RESOURCE_TERMINATED`
before post-synthesis reports were produced.

This page is operational guidance only. It does not claim synthesis
success, implementation success, timing closure, or bitstream readiness.

## Scope

Applies to:

- `cd hw && ./vivado/build.sh synth`
- out-of-context synthesis of `NPU_top`
- Vivado runs that are expected to produce `hw/build/reports/*_post_synth.rpt`

Does not replace:

- `docs/TIMING_EVIDENCE.md` for timing-evidence wording
- full implementation / bitstream evidence
- KV260 board smoke evidence

## Observed Resource Blocker

`BLOCKED_RESOURCE_TERMINATED` means the synth attempt did not finish
because the host or runner stopped the job for resource reasons. Treat it
as missing synthesis evidence:

- no post-synthesis timing report
- no post-synthesis DRC report
- no post-synthesis utilization report
- no bitstream evidence

For review purposes, this status is different from an RTL, timing, or DRC
failure. It says the selected machine was not a suitable Vivado synth
host for this design snapshot.

## Recommended Host Target

Use a dedicated Linux host or self-hosted runner for full Vivado synth.
The practical target is:

| Resource | Recommendation |
|----------|----------------|
| RAM | 64 GiB target; 32 GiB minimum only for exploratory synth |
| Swap | 64 GiB configured swap on SSD/NVMe |
| CPU parallelism | start with `PCCX_VIVADO_JOBS=1`; raise only after one clean run |
| Disk | keep at least 20 GiB free under `hw/build/` and the Vivado temp area |

Avoid treating a standard low-memory hosted CI runner as authoritative
for full Vivado synthesis. It can still run formatting, filelist, lint,
dry-run, and script checks.

## Standard Synth Command

From repo root:

```bash
cd hw
PCCX_VIVADO_JOBS=1 ./vivado/build.sh synth
```

Record the exact command, Vivado version, host RAM, swap size, and final
status in the PR. If the job is terminated by the runner before reports
are written, record `BLOCKED_RESOURCE_TERMINATED` and do not infer any
timing or utilization result.

## Swap Setup Guidance

On a dedicated Linux synth host, provision swap before launching Vivado.
One typical setup is:

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
swapon --show
free -h
```

Use site-local policy for persistent `/etc/fstab` entries. Do not enable
swap inside shared CI jobs unless the runner owner explicitly allows it.

## Low-Memory Strategy

When a high-memory host is unavailable, use a staged evidence path:

1. Run non-Vivado checks and dry-run capture:

   ```bash
   bash scripts/v002/run-timing-evidence.sh --dry-run
   ```

2. Run RTL/filelist sanity checks that do not require full Vivado synth.
3. If Vivado is available, run with `PCCX_VIVADO_JOBS=1` and preserve the
   log tail when the runner terminates or the process is stopped.
4. Mark the result as blocked resource evidence, not as RTL, timing, or
   implementation evidence:

   ```text
   RESULT_STATUS=BLOCKED_RESOURCE_TERMINATED
   ```

5. Defer synth evidence collection to a 64 GiB target host or self-hosted
   runner.

Do not promote `impl`, bitstream packaging, release evidence, or timing
closure from a resource-terminated synth attempt.
