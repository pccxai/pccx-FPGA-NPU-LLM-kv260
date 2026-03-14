import ctypes
import numpy as np
import os
import time

# 1. DLL 경로 설정 및 로드
base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)

# 2. C 함수 시그니처 정의 (매우 중요! 이거 안 하면 포인터 에러로 튕김)
# C_CONTIGUOUS: 메모리가 C언어 배열처럼 1차원으로 예쁘게 이어져 있는지 강제 확인
c_lib.run_RMSNorm_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # x
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # gamma
    ctypes.c_int # length
]
c_lib.run_RMSNorm_inplace.restype = None

# 3. 파이썬 래퍼 함수 (호출용)
def c_gelu(x_array):
    # 만약 메모리가 꼬여있다면 일렬로 정렬 (일반적으로 np.dot 결과는 이미 정렬되어 있음)
    if not x_array.flags['C_CONTIGUOUS']:
        x_array = np.ascontiguousarray(x_array)
        
    # 복사(Copy) 없이 원본 배열의 주소만 C로 넘겨서 덮어쓰기!
    c_lib.run_gelu_inplace(x_array, x_array.size)
    return x_array

# 래퍼 함수
def c_rms_norm(x_array, gamma_array):
    if not x_array.flags['C_CONTIGUOUS']:
        x_array = np.ascontiguousarray(x_array)
    if not gamma_array.flags['C_CONTIGUOUS']:
        gamma_array = np.ascontiguousarray(gamma_array)
        
    c_lib.run_RMSNorm_inplace(x_array, gamma_array, x_array.size)
    return x_array


def py_gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))

def py_rms_norm(x, gamma):
    rms = np.sqrt(np.mean(x ** 2) + 1e-6)
    return (x / rms) * gamma

# --- 테스트 및 벤치마크 ---
if __name__ == "__main__":
    # 기존 파이썬 GeLU (비교용)
    
    # Gemma 차원 크기로 테스트
    print("데이터 생성 중...")

    dim = 2048 * 2048
    test_x = np.random.randn(dim).astype(np.float32)
    test_gamma = np.random.randn(dim).astype(np.float32)

    # Gemma의 통상적인 은닉층 크기 (예: 2048 차원)
    # test_data = np.random.randn(2048 * 2048).astype(np.float32)
    
    # 공정한 대결을 위해 데이터 복사 (C함수는 원본을 덮어쓰기 때문)
    #data_for_py = test_x.copy()
    #data_for_c = test_x.copy()

    # 1. 파이썬 속도 측정
    start = time.perf_counter()
    #res_py = py_gelu(data_for_py)
    res_py = py_rms_norm(test_x.copy(), test_gamma)
    py_time = time.perf_counter() - start

    # 2. C DLL 속도 측정
    start = time.perf_counter()
    # res_c = c_gelu(data_for_c)
    res_c = c_rms_norm(test_x.copy(), test_gamma)
    c_time = time.perf_counter() - start

    print(f"\n--- 결과 ---")
    print(f"Python Time : {py_time * 1000:.5f} ms")
    print(f"C DLL Time  : {c_time * 1000:.5f} ms")
    
    if c_time > 0:
        print(f"🔥 속도 향상: 약 {py_time / c_time:.1f}배 빠름!")

    # 3. 정확도 검증 (소수점 아래 미세 오차 허용)
    diff = np.max(np.abs(res_py - res_c))
    print(f"최대 오차(Max Diff): {diff:.8f}")
    
    if diff < 1e-5:
        print("✅ 완벽하게 일치합니다! 하드웨어 포팅 대성공!")
    else:
        print("❌ 오차가 큽니다. 타입이나 컴파일러 옵션 확인 필요.")