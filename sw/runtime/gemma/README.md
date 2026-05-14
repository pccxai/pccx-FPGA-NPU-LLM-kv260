# Gemma 3N E4B runtime package

`sw.runtime.gemma` is the v002-aware Python runtime package for the Gemma 3N
E4B text path. It replaces the older single-file scaffold with importable
modules for architecture constants, invariant checks, MMAP weights, tokenizer
wrapping, NumPy ops, attention, FFN, decoder-layer ordering, and streaming
session orchestration.

The package is intentionally conservative:

- `core_secrets.py` encodes the seven Gemma 3N invariants as executable
  constants.
- `weights.py` lazy-loads split `.npy` weights with `np.load(..., mmap_mode="r")`
  and raises the process file-descriptor soft limit for large split-weight
  directories.
- `attention.py` and `ffn.py` call `sw.runtime.npu.npu_gemm` and
  `sw.runtime.npu.npu_cvo` when that module is available, and otherwise use
  NumPy fallback math.
- `tokenizer.py` wraps `transformers.AutoTokenizer.from_pretrained(model_dir)`;
  a byte fallback lets services boot before tokenizer assets are installed.

## Public interface

```python
from sw.runtime.gemma import GemmaInferenceSession, GEMMA_3N_E4B_DEFAULTS

session = GemmaInferenceSession(
    "~/models/gemma-3n-e4b-int4/",
    use_npu=True,
    max_seq_len=1024,
    dtype="fp16",
)

for piece, event in session.generate("hello", max_new_tokens=32):
    print(piece, event["token_id"], event["npu_status"])
```

`GemmaInferenceSession.generate()` yields `(piece_text, event_dict)` per token.
Each event includes `token_id`, `tok_per_sec`, `npu_status`, and
`layer_progress`.

## NPU contract

The runtime expects the sibling module to export:

```python
def npu_gemm(W, X, *, layer_idx=None): ...
def npu_gemv(W, x, *, layer_idx=None): ...
def npu_cvo(op, x): ...
def npu_status(): ...
```

`npu_gemm` receives K-major weights shaped `[K_in, M_out]` and an activation
batch shaped `[B, K_in]`. If the sibling module is absent or disabled with
`use_npu=False`, the package uses NumPy dot products and local GELU.
