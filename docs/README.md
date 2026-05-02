# Documentation has moved

All uXC / pccx architecture, ISA, driver, and Gemma 3N E4B model notes have
been consolidated into the **pccx documentation site** (Sphinx, GitHub
Pages, bilingual EN / KO).

## Jump straight in

- **Latest architecture — v002** (current target of this repo):
  <https://pccxai.github.io/pccx/en/docs/v002/index.html>
- Repo-local KV260 bring-up checklist:
  [KV260_BRINGUP.md](KV260_BRINGUP.md)
- Repo-local Gemma 3N handoff boundary:
  [GEMMA3N_HANDOFF.md](GEMMA3N_HANDOFF.md)
- Repo-local xsim evidence workflow:
  [SIMULATION.md](SIMULATION.md)
- Repo-local Vivado timing evidence checklist:
  [TIMING_EVIDENCE.md](TIMING_EVIDENCE.md)
- Repo-local release evidence checklist:
  [RELEASE_EVIDENCE_CHECKLIST.md](RELEASE_EVIDENCE_CHECKLIST.md)
- Documentation root (language picker):
  <https://pccxai.github.io/pccx/en/index.html>
- pccx source repository:
  <https://github.com/pccxai/pccx>

## Where the old documents went

| Old file (deleted)               | New location                                                                                                                |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `FPGA_NPU_Architecture_v2.md`    | [v002 Architecture](https://pccxai.github.io/pccx/en/docs/v002/Architecture/index.html)                                  |
| `ISA.md`                         | [v002 ISA](https://pccxai.github.io/pccx/en/docs/v002/ISA/index.html)                                                    |
| `HW_Optimization_DSP48E2.md`     | [DSP48E2 W4A8 bit-packing](https://pccxai.github.io/pccx/en/docs/v002/Architecture/dsp48e2_w4a8.html)                    |
| `GEMMA_3N_E4B.md`                | [Gemma 3N overview](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_overview.html)                             |
| `Gemma3N_Pipeline_EN.md`         | [Gemma 3N pipeline](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_pipeline.html)                             |
| `Attention_RoPE.md`              | [Gemma 3N attention / RoPE](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_attention_rope.html)               |
| `PLE_LAuReL.md`                  | [Gemma 3N PLE & LAuReL](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_ple_laurel.html)                       |
| `FFN_Sparsity.md`                | [Gemma 3N FFN sparsity](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_ffn_sparsity.html)                     |
| *(new)* how the model runs here  | [Gemma 3N on pccx v002 execution](https://pccxai.github.io/pccx/en/docs/v002/Models/gemma3n_execution.html)              |

For the Korean mirror, swap `/en/` for `/ko/` in any of the links above.
