import taichi as ti
import numpy as np
import math
import gc

ti.init(arch=ti.vulkan, fast_math=True)

TILE_K = 128

# ================================================================
# Taichi GEMV kernel (version with atomic_add removed - retains previous optimizations)
# ================================================================
@ti.kernel
def _gemv_int16(
    vec:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_t:  ti.types.ndarray(dtype=ti.i16, ndim=2), # type: ignore
    scale:  ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    M: ti.i32, # type: ignore
    N: ti.i32  # type: ignore
):
    for i in range(N):
        acc = ti.f32(0.0)
        for k in range(M):
            acc += vec[k] * ti.cast(mat_t[i, k], ti.f32)
        out[i] = acc * scale[i]

@ti.kernel
def _gemv_int16_gelu(
    vec:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_t:  ti.types.ndarray(dtype=ti.i16, ndim=2), # type: ignore
    scale:  ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    M: ti.i32, # type: ignore
    N: ti.i32  # type: ignore
):
    for i in range(N):
        acc = ti.f32(0.0)
        for k in range(M):
            acc += vec[k] * ti.cast(mat_t[i, k], ti.f32)
        v = acc * scale[i]
        out[i] = 0.5 * v * (1.0 + ti.tanh(
            ti.sqrt(ti.f32(2.0 / math.pi)) * (v + ti.f32(0.044715) * v * v * v)
        ))

# ================================================================
# Memory Optimization: WeightProxy
#
# Load the float16 original matrix into VRAM (INT16) and release it from RAM.
# Replace layers["W_q"][i], etc. with this object → float16 array is GCed.
# As long as there is a .shape, igpu_matmul operates normally.
# ================================================================
class WeightProxy:
    """Lightweight proxy used instead of float16 original after VRAM upload is completed."""
    __slots__ = ("stable_id", "shape")

    def __init__(self, stable_id: int, shape: tuple):
        self.stable_id = stable_id
        self.shape     = shape   # M, N = weight.shape required in igpu_matmul

# ================================================================
# VRAM Cache: Based on reliable integer IDs
#
# Problems with the existing id(weight_mat) method:
# - id() can be reused for other objects after freeing the float16 original.
# - In that case, cache miss → re-upload
#
# New method: give sequential integer stable_id when uploading → permanently unique
#   _VRAM_CACHE[stable_id] = (ti_mat, ti_scale)
# _OBJID_TO_STABLE[id(weight_mat)] = stable_id (valid only at the time of upload)
# ================================================================
_VRAM_CACHE:       dict = {}   # stable_id -> (ti_mat, ti_scale)
_OBJID_TO_STABLE:  dict = {}   # id(array) -> stable_id
_OUTPUT_BUF_POOL:  dict = {}   # output_size -> ti.ndarray
_next_stable_id = [0]          # mutable counter (bypassing closures with lists)

def _get_or_upload_weight(weight_mat):
    """
    If WeightProxy, returns immediately from VRAM cache.
    If it is a float16/32 numpy array, quantize it to INT16 and upload it to VRAM.
    Afterwards, when the same array is recalled, it is returned from the cache.
    """
    # --- WeightProxy path: Already uploaded ---
    if isinstance(weight_mat, WeightProxy):
        return _VRAM_CACHE[weight_mat.stable_id]

    # --- numpy array path ---
    obj_id = id(weight_mat)
    if obj_id in _OBJID_TO_STABLE:
        stable_id = _OBJID_TO_STABLE[obj_id]
        return _VRAM_CACHE[stable_id]

    # Weights seen for the first time → VRAM upload after INT16 quantization
    stable_id = _next_stable_id[0]
    _next_stable_id[0] += 1

    # Can handle either float16 or float32 input
    mat_t    = weight_mat.T.astype(np.float32)
    max_vals = np.max(np.abs(mat_t), axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-8)

    scale   = (max_vals / 32767.0).flatten().astype(np.float32)
    mat_i16 = np.clip(
        np.round(mat_t / max_vals * 32767.0), -32767, 32767
    ).astype(np.int16)
    mat_i16 = np.ascontiguousarray(mat_i16)

    ti_mat   = ti.ndarray(dtype=ti.i16, shape=mat_i16.shape)
    ti_scale = ti.ndarray(dtype=ti.f32, shape=scale.shape)
    ti_mat.from_numpy(mat_i16)
    ti_scale.from_numpy(scale)

    _VRAM_CACHE[stable_id]      = (ti_mat, ti_scale)
    _OBJID_TO_STABLE[obj_id]    = stable_id
    return ti_mat, ti_scale

def _get_output_buf(size: int) -> ti.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = ti.ndarray(dtype=ti.f32, shape=(size,))
    return _OUTPUT_BUF_POOL[size]

# ================================================================
# Memory optimization core functions: preload_and_free
#
# movement:
# 1. Upload all W[key][i] (float16 array) to VRAM as INT16
# 2. Replace the corresponding item in layers dict with WeightProxy
# 3. gc.collect() → return float16 array memory
#
# Effect: 490 times per token from_numpy → 0 times
# float16 large matrix ~3.5GB → free RAM
# ================================================================
def preload_and_free(W: dict, keys: list):
    total = sum(len(W[k]) for k in keys if k in W)
    uploaded = 0

    for key in keys:
        if key not in W:
            continue
        proxies = []
        for w in W[key]:
            ti_mat, ti_scale = _get_or_upload_weight(w)   # VRAM upload
            obj_id    = id(w)
            stable_id = _OBJID_TO_STABLE[obj_id]
            proxies.append(WeightProxy(stable_id, w.shape))
            uploaded += 1
        W[key] = proxies   # float16 array dereference → GC target
        print(f" [{key}] {len(proxies)} → VRAM transfer complete")

    gc.collect()
    print(f"[Memory] Total {uploaded}/{total} weights before VRAM & free float original ✓")

# ================================================================
# Public API (maintain the same signature as before)
# ================================================================
def igpu_matmul(x_vec: np.ndarray, weight_mat) -> np.ndarray:
    x_f32   = x_vec.astype(np.float32)
    ti_w, ti_s = _get_or_upload_weight(weight_mat)
    M, N    = weight_mat.shape

    buf = _get_output_buf(N)
    _gemv_int16(x_f32, ti_w, ti_s, buf, M, N)
    return buf.to_numpy()

def igpu_matmul_gelu(x_vec: np.ndarray, weight_mat) -> np.ndarray:
    x_f32   = x_vec.astype(np.float32)
    ti_w, ti_s = _get_or_upload_weight(weight_mat)
    M, N    = weight_mat.shape

    buf = _get_output_buf(N)
    _gemv_int16_gelu(x_f32, ti_w, ti_s, buf, M, N)
    return buf.to_numpy()

def warmup():
    """
    JIT compilation + warm-up.
    To ensure that dummy matrix IDs do not conflict with real weights
    Removed from _OBJID_TO_STABLE after completing warm-up.
    """
    print("[iGPU] JIT warming up + INT16 NPU simulation mode running...")
    dummy_x     = np.zeros(2048, dtype=np.float32)
    dummy_w     = np.random.randn(2048, 2048).astype(np.float32)
    dummy_w_big = np.random.randn(2048, 8192).astype(np.float32)

    igpu_matmul(dummy_x, dummy_w)
    igpu_matmul_gelu(dummy_x, dummy_w_big)

    # Remove dummy entries (avoid polluting real weight cache by reusing ids)
    for dummy in [dummy_w, dummy_w_big]:
        d_id = id(dummy)
        if d_id in _OBJID_TO_STABLE:
            del _OBJID_TO_STABLE[d_id]

    print("[iGPU] Warming up complete! ")
