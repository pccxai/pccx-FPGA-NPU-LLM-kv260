import numpy as np
import os
import ctypes

# Taichi 제거됨!

base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)

# ================================================================
# C++ DLL 파라미터 세팅
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

# 출력 버퍼 풀 (메모리 재할당 방지)
_OUTPUT_BUF_POOL = {}

def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]

# ================================================================
# 인터페이스 유지 (기존 호환성을 위해 이름만 남김)
# ================================================================
def preload_and_free(W: dict, keys: list):
    print("[CPU_GEMV] Taichi 제거됨. RAM에서 직접 C++ SIMD 코어 6개 풀가동 모드 준비 완료!")
    pass

def _get_or_upload_weight(weight_data):
    pass

# ================================================================
#  C++ 기반 초고속 멀티코어 행렬곱 연산
# ================================================================
def igpu_matmul(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    if isinstance(weight_data, tuple):
        # INT4 처리
        packed, scale = weight_data
        M_out = packed.shape[0]
        K_in = packed.shape[1] * 2
        
        out_buf = _get_output_buf(M_out)
        c_lib.run_gemv_int4(x_f32, packed, scale, out_buf, M_out, K_in)
        return out_buf.copy() # 원본 보존을 위해 copy 반환
    else:
        # Fallback (일반 행렬곱)
        w_f32 = np.ascontiguousarray(weight_data.astype(np.float32))
        return np.dot(x_f32, w_f32.T)

def igpu_matmul_gelu(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    if isinstance(weight_data, tuple):
        # INT4 + GeLU 퓨전 처리
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
        # 파이썬 C_DLL GeLU 호출 (이미 CPU_CORE 쪽에 세팅되어 있지만 편의상 수동 호출 생략)
        import CPU_CORE
        return CPU_CORE.gelu(out)

def warmup():
    print("[CPU_GEMV] 멀티코어 AVX2 SIMD 엔진 워밍업 중...")
    dummy_x = np.zeros(2048, dtype=np.float32)
    dummy_p = np.zeros((2048, 1024), dtype=np.uint8)
    dummy_s = np.zeros(2048, dtype=np.float32)
    igpu_matmul(dummy_x, (dummy_p, dummy_s))
    print("[CPU_GEMV] 워밍업 완료! 코어 6개 장전 완료 ")