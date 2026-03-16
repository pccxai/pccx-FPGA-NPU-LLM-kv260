import numpy as np
import os
import ctypes

# Taichi eliminated.

base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)

# ================================================================
# C++ DLL parameter setting
# ================================================================
c_lib.run_gemv_int4.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # vec
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),   # mat_p
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # scale
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # out
    ctypes.c_int, # M_out
    ctypes.c_int  # K_in
]
c_lib.run_gemv_int4.restype = None

c_lib.run_gemv_int4_gelu.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]
c_lib.run_gemv_int4_gelu.restype = None

# Output buffer pool (avoid memory reallocation)
_OUTPUT_BUF_POOL = {}

def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]

# ================================================================
# Maintain interface (only name remains for legacy compatibility)
# ================================================================
def preload_and_free(W: dict, keys: list):
    print("[CPU_GEMV] Taichi removed. Six C++ SIMD cores directly from RAM ready for full operation mode!")
    pass

def _get_or_upload_weight(weight_data):
    pass

# ================================================================
# C++-based ultra-fast multi-core matrix multiplication operation
# ================================================================
def igpu_matmul(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    if isinstance(weight_data, tuple):
        # INT4 processing
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2
        
        out_buf = _get_output_buf(M_out)
        c_lib.run_gemv_int4(x_f32, packed, scale, out_buf, M_out, K_in)
        return out_buf.copy() # Return a copy to preserve the original
    else:
        # Fallback (general matrix multiplication)
        w_f32 = np.ascontiguousarray(weight_data.astype(np.float32))
        return np.dot(x_f32, w_f32.T)

def igpu_matmul_gelu(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    if isinstance(weight_data, tuple):
        # INT4 + GeLU fusion processing
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2
        
        out_buf = _get_output_buf(M_out)
        c_lib.run_gemv_int4_gelu(x_f32, packed, scale, out_buf, M_out, K_in)
        return out_buf.copy()
    else:
        # Fallback
        w_f32 = np.ascontiguousarray(weight_data.astype(np.float32))
        out = np.dot(x_f32, w_f32.T)
        # Python C_DLL GeLU call (already set on CPU_CORE side, but manual call is omitted for convenience)
        import CPU_CORE
        return CPU_CORE.gelu(out)

def warmup():
    print("[CPU_GEMV] Warming up multicore AVX2 SIMD engine...")
    dummy_x = np.zeros(2048, dtype=np.float32)
    dummy_p = np.zeros((2048, 1024), dtype=np.uint8)
    dummy_s = np.zeros(2048, dtype=np.float32)
    igpu_matmul(dummy_x, (dummy_p, dummy_s))
    print("[CPU_GEMV] Warm-up complete! 6 cores loaded ")
