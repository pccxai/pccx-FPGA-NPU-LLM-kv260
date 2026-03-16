import taichi as ti
import numpy as np
import math
import gc

ti.init(arch=ti.vulkan, fast_math=True)

# ================================================================
# Taichi GEMV kernels
# ================================================================

@ti.kernel
def _gemv_int4(
    vec:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_p:  ti.types.ndarray(dtype=ti.u8,  ndim=2), # type: ignore
    scale:  ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    M_out:  ti.i32, # type: ignore
    K_in:   ti.i32  # type: ignore
):
    # mat_p: [M_out, K_in // 2]
    # vec: [K_in]
    # out: [M_out]
    for i in range(M_out):
        acc = ti.f32(0.0)
        for k_p in range(K_in // 2):
            packed = mat_p[i, k_p]
            
            # Unpack low 4 bits
            val_l = ti.cast(packed & ti.u8(0x0F), ti.i32)
            if val_l > 7: val_l -= 16
            
            # Unpack high 4 bits
            val_h = ti.cast((packed >> ti.u8(4)) & ti.u8(0x0F), ti.i32)
            if val_h > 7: val_h -= 16
            
            acc += vec[2 * k_p] * ti.cast(val_l, ti.f32)
            acc += vec[2 * k_p + 1] * ti.cast(val_h, ti.f32)
            
        out[i] = acc * scale[i]

@ti.kernel
def _gemv_int4_gelu(
    vec:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_p:  ti.types.ndarray(dtype=ti.u8,  ndim=2), # type: ignore
    scale:  ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    M_out:  ti.i32, # type: ignore
    K_in:   ti.i32  # type: ignore
):
    for i in range(M_out):
        acc = ti.f32(0.0)
        for k_p in range(K_in // 2):
            packed = mat_p[i, k_p]
            val_l = ti.cast(packed & ti.u8(0x0F), ti.i32)
            if val_l > 7: val_l -= 16
            val_h = ti.cast((packed >> ti.u8(4)) & ti.u8(0x0F), ti.i32)
            if val_h > 7: val_h -= 16
            acc += vec[2 * k_p] * ti.cast(val_l, ti.f32)
            acc += vec[2 * k_p + 1] * ti.cast(val_h, ti.f32)
            
        v = acc * scale[i]
        out[i] = 0.5 * v * (1.0 + ti.tanh(
            ti.sqrt(ti.f32(2.0 / math.pi)) * (v + ti.f32(0.044715) * v * v * v)
        ))

# Fallback INT16 kernels (used if input is not already quantized)
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

# ================================================================
# WeightProxy
# ================================================================
class WeightProxy:
    __slots__ = ("stable_id", "shape", "is_int4")

    def __init__(self, stable_id: int, shape: tuple, is_int4: bool = False):
        self.stable_id = stable_id
        self.shape     = shape
        self.is_int4   = is_int4

_VRAM_CACHE:       dict = {}   # stable_id -> (ti_mat, ti_scale)
_OBJID_TO_STABLE:  dict = {}   # id(obj) -> stable_id
_OUTPUT_BUF_POOL:  dict = {}   # output_size -> ti.ndarray
_next_stable_id = [0]

def _get_or_upload_weight(weight_data):
    # weight_data can be:
    # 1. WeightProxy
    # 2. tuple (packed_uint8, scale_float32) -> INT4
    # 3. numpy array -> INT16 (fallback)
    
    if isinstance(weight_data, WeightProxy):
        return _VRAM_CACHE[weight_data.stable_id]

    obj_id = id(weight_data)
    if obj_id in _OBJID_TO_STABLE:
        return _VRAM_CACHE[_OBJID_TO_STABLE[obj_id]]

    stable_id = _next_stable_id[0]
    _next_stable_id[0] += 1

    if isinstance(weight_data, tuple):
        # INT4 Path
        packed, scale = weight_data
        ti_mat = ti.ndarray(dtype=ti.u8, shape=packed.shape)
        ti_scale = ti.ndarray(dtype=ti.f32, shape=scale.shape)
        ti_mat.from_numpy(packed)
        ti_scale.from_numpy(scale.astype(np.float32))
        _VRAM_CACHE[stable_id] = (ti_mat, ti_scale)
    else:
        # INT16 Fallback Path (numpy array)
        # Assuming [out, in] or [in, out] based on shape. 
        # For simplicity, let's assume it's already in the shape we want or transpose it.
        # Original logic:
        mat_t    = weight_data.T.astype(np.float32)
        max_vals = np.max(np.abs(mat_t), axis=1, keepdims=True)
        max_vals = np.maximum(max_vals, 1e-8)
        scale   = (max_vals / 32767.0).flatten().astype(np.float32)
        mat_i16 = np.clip(np.round(mat_t / max_vals * 32767.0), -32767, 32767).astype(np.int16)
        
        ti_mat   = ti.ndarray(dtype=ti.i16, shape=mat_i16.shape)
        ti_scale = ti.ndarray(dtype=ti.f32, shape=scale.shape)
        ti_mat.from_numpy(mat_i16)
        ti_scale.from_numpy(scale)
        _VRAM_CACHE[stable_id] = (ti_mat, ti_scale)

    _OBJID_TO_STABLE[obj_id] = stable_id
    return _VRAM_CACHE[stable_id]

def _get_output_buf(size: int) -> ti.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = ti.ndarray(dtype=ti.f32, shape=(size,))
    return _OUTPUT_BUF_POOL[size]

def preload_and_free(W: dict, keys: list):
    uploaded = 0
    for key in keys:
        if key not in W: continue
        proxies = []
        for w in W[key]:
            # w might be (packed, scale) or numpy array
            ti_mat, ti_scale = _get_or_upload_weight(w)
            is_int4 = isinstance(w, tuple)
            shape = w[0].shape if is_int4 else w.shape
            # If int4, packed shape is [out, in//2], so original shape is [out, in]
            orig_shape = (shape[0], shape[1]*2) if is_int4 else shape
            
            proxies.append(WeightProxy(_OBJID_TO_STABLE[id(w)], orig_shape, is_int4))
            uploaded += 1
        W[key] = proxies
        print(f" [{key}] {len(proxies)} -> VRAM (INT4={proxies[0].is_int4})")
    gc.collect()

def igpu_matmul(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = x_vec.astype(np.float32)
    ti_w, ti_s = _get_or_upload_weight(weight_data)
    
    is_int4 = False
    if isinstance(weight_data, WeightProxy):
        is_int4 = weight_data.is_int4
        M_out, K_in = weight_data.shape
    elif isinstance(weight_data, tuple):
        is_int4 = True
        M_out = weight_data[0].shape[0]
        K_in = weight_data[0].shape[1] * 2
    else:
        M_out, K_in = weight_data.shape # [out, in] for fallback? 
        # Actually fallback uses mat_t = weight.T which is [in, out]
        # So M=in, N=out.
        pass

    if is_int4:
        buf = _get_output_buf(M_out)
        _gemv_int4(x_f32, ti_w, ti_s, buf, M_out, K_in)
        return buf.to_numpy()
    else:
        # Fallback to INT16
        # Need to match the original logic
        M, N = weight_data.shape if not isinstance(weight_data, WeightProxy) else weight_data.shape
        # Original: _gemv_int16(x_f32, ti_w, ti_s, buf, M_in, N_out)
        # So M=in, N=out.
        buf = _get_output_buf(N)
        _gemv_int16(x_f32, ti_w, ti_s, buf, M, N)
        return buf.to_numpy()

def igpu_matmul_gelu(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = x_vec.astype(np.float32)
    ti_w, ti_s = _get_or_upload_weight(weight_data)
    
    if isinstance(weight_data, WeightProxy) and weight_data.is_int4:
        M_out, K_in = weight_data.shape
        buf = _get_output_buf(M_out)
        _gemv_int4_gelu(x_f32, ti_w, ti_s, buf, M_out, K_in)
        return buf.to_numpy()
    # Simplified: only supporting INT4 for now or fallback if needed
    return igpu_matmul(x_vec, weight_data) # TODO: add int16 gelu if needed

def warmup():
    print("[iGPU] Warming up INT4 Kernels...")
    dummy_x = np.zeros(2048, dtype=np.float32)
    dummy_p = np.zeros((2048, 1024), dtype=np.uint8)
    dummy_s = np.zeros(2048, dtype=np.float32)
    
    igpu_matmul(dummy_x, (dummy_p, dummy_s))
    print("[iGPU] Warmup Complete! ")
