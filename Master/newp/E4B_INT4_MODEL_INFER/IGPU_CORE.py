import numpy as np
import os
import ctypes

base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "vulkan_core.so")
vk_lib = ctypes.CDLL(dll_path)


# -----------------------------------------------------------
# init Vulkan engine

vk_lib.init_vulkan_engine.argtypes = []
vk_lib.init_vulkan_engine.restype = None

os.chdir(base_dir)

# 프로그램 켜질 때 GPU 딱 한 번 장전!
vk_lib.init_vulkan_engine()

# <><><><><><><><><><><><><><><Parameters><><><><><><><><><><><><><><

vk_lib.run_vulkan_gemv.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # x
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),   # mat_p
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # scale
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # out
    ctypes.c_int, # M_out
    ctypes.c_int  # K_in
]
vk_lib.run_vulkan_gemv.restype = None

# prefetch weight
vk_lib.prefetch_weight_async.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'), # mat_p
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int  
]
vk_lib.prefetch_weight_async.restype = None

# ping pong
vk_lib.run_vulkan_gemv_pingpong.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), 
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), 
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int  
]
vk_lib.run_vulkan_gemv_pingpong.restype = None

# <><><><><><><><><><><><><><><Parameters><><><><><><><><><><><><><><

_OUTPUT_BUF_POOL = {}
def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]

# -----------------------------------------------------------

# legacy
# # ================================================================
def preload_and_free(W: dict, keys: list): pass
def _get_or_upload_weight(weight_data): pass
def warmup(): print("[Vulkan_GEMV] shader engine load compelete ")

# ================================================================
# 4. 행렬곱 인터페이스
# ================================================================
def igpu_matmul(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    if isinstance(weight_data, tuple):
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2
        
        out_buf = _get_output_buf(M_out)
        
        vk_lib.run_vulkan_gemv(x_f32, packed, scale, out_buf, M_out, K_in)
        
        return out_buf.copy()
    else:
        w_f32 = np.ascontiguousarray(weight_data.astype(np.float32))
        return np.dot(x_f32, w_f32)

def igpu_matmul_gelu(x_vec: np.ndarray, weight_data) -> np.ndarray:
    out = igpu_matmul(x_vec, weight_data)
    import CPU_CORE
    return CPU_CORE.gelu(out)


def prefetch_weight(weight_data, buf_idx: int):
    if isinstance(weight_data, tuple):
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2
        vk_lib.prefetch_weight_async(packed, M_out, K_in, buf_idx)

def compute_pingpong(x_vec: np.ndarray, weight_data, buf_idx: int) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    if isinstance(weight_data, tuple):
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2
        
        out_buf = _get_output_buf(M_out)
        vk_lib.run_vulkan_gemv_pingpong(x_f32, scale, out_buf, M_out, K_in, buf_idx)
        return out_buf.copy()
    else:
        w_f32 = np.ascontiguousarray(weight_data.astype(np.float32))
        return np.dot(x_f32, w_f32.T)