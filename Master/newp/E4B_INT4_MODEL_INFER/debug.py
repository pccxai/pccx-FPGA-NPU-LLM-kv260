import ctypes
import numpy as np
import os
import time

# 1. Setting and loading DLL path
base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)

# 2. Define C function signature (very important! If the user don’t do this, the user will get a pointer error)
# C_CONTIGUOUS: Forcefully checks whether the memory is neatly connected in one dimension like a C language array.
c_lib.run_RMSNorm_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # x
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # gamma
    ctypes.c_int # length
]
c_lib.run_RMSNorm_inplace.restype = None

# 3. Python wrapper function (for calls)
def c_gelu(x_array):
    # Sort in line if memory is kinked (normally np.dot results are already sorted)
    if not x_array.flags['C_CONTIGUOUS']:
        x_array = np.ascontiguousarray(x_array)
        
    # Overwrite by passing only the address of the original array to C without copying.
    c_lib.run_gelu_inplace(x_array, x_array.size)
    return x_array

# wrapper function
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

# --- Tests and Benchmarks ---
if __name__ == "__main__":
    # Original Python GeLU (for comparison)
    
    # Test with Gemma dimension size
    print("Generating data...")

    dim = 2048 * 2048
    test_x = np.random.randn(dim).astype(np.float32)
    test_gamma = np.random.randn(dim).astype(np.float32)

    # Gemma's typical hidden layer size (e.g. 2048 dimensions)
    # test_data = np.random.randn(2048 * 2048).astype(np.float32)
    
    # Copy data for fair competition (since C functions overwrite the original)
    #data_for_py = test_x.copy()
    #data_for_c = test_x.copy()

    # 1. Measure Python speed
    start = time.perf_counter()
    #res_py = py_gelu(data_for_py)
    res_py = py_rms_norm(test_x.copy(), test_gamma)
    py_time = time.perf_counter() - start

    # 2. C DLL speed measurement
    start = time.perf_counter()
    # res_c = c_gelu(data_for_c)
    res_c = c_rms_norm(test_x.copy(), test_gamma)
    c_time = time.perf_counter() - start

    print(f"\n--- result ---")
    print(f"Python Time : {py_time * 1000:.5f} ms")
    print(f"C DLL Time  : {c_time * 1000:.5f} ms")
    
    if c_time > 0:
        print(f" Speedup: approximately {py_time / c_time:.1f} times faster!")

    # 3. Accuracy verification (fine errors allowed after decimal point)
    diff = np.max(np.abs(res_py - res_c))
    print(f"Max Diff: {diff:.8f}")
    
    if diff < 1e-5:
        print(" Perfect match! Hardware porting success!")
    else:
        print("The error is large. Type or compiler options need to be checked.")
