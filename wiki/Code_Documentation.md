# Code Documentation (1/8) — C++ Acceleration Layer

> **Target Files**: `vulkan_core.cpp` · `my_accelerator.cpp`
> **Role**: Low-level matrix operation kernels based on iGPU (Vulkan) and CPU (AVX2/OpenMP)

---

## 1. `vulkan_core.cpp`

### Overview

A C++ shared library that performs GEMV (General Matrix-Vector Multiply) operations on INT4 quantized weights on the iGPU using the Vulkan Compute API. It is called from Python (`IGPU_CORE.py`) via `ctypes`.

**Core Design Pattern: Ping-Pong Buffering**
While the CPU asynchronously uploads the weights for the next layer to VRAM, the GPU proceeds with operations on the previous buffer. It hides the memory transfer latency between layers by overlapping it with the computation time.

```
[Layer i]   GPU Computation (Buffer A) │ CPU Asynchronous Transfer (W[i+1] → Buffer B)
[Layer i+1] GPU Computation (Buffer B) │ CPU Asynchronous Transfer (W[i+2] → Buffer A)
```

---

### Global State

| Variable | Type | Description |
|---|---|---|
| `instance` | `VkInstance` | Vulkan instance handle |
| `physicalDevice` | `VkPhysicalDevice` | Physical GPU device handle |
| `device` | `VkDevice` | Logical device handle |
| `computeQueue` | `VkQueue` | Compute-only queue |
| `computePipeline` | `VkPipeline` | Compiled compute pipeline |
| `g_matBuf[2]` | `VkBuffer[2]` | Ping-pong weight buffers (300MB each) |
| `g_xBuf` | `VkBuffer` | Input vector buffer (MAX_K × 4 bytes) |
| `g_scaleBuf` | `VkBuffer` | INT4 dequant scale buffer |
| `g_outBuf` | `VkBuffer` | Output buffer (MAX_M × 4 bytes) |
| `g_descriptorSet[2]` | `VkDescriptorSet[2]` | Descriptor sets for each ping-pong buffer |
| `weight_loader` | `std::future<void>` | Future handle of the asynchronous weight loader |

**Constants**

```cpp
#define MAX_M 262144   // Maximum output dimension (considering LM Head vocab size)
#define MAX_K 16384    // Maximum input dimension
```

g_scaleBuf allocation: 262144 × 4 = 1,048,576 bytes
LM Head requirement: 262400 × 4 = 1,049,600 bytes
                              ─────────────────
                              Short by 1,024 bytes (256 floats)

---

### Function Reference

#### `init_vulkan_engine()`

```c
extern "C" void init_vulkan_engine()
```

**Purpose**: Called exactly once at program startup to initialize the entire Vulkan pipeline.

**Initialization Sequence**:
1. Create `VkInstance` (API version 1.2)
2. Select the first physical device (`devices[0]`)
3. Create a compute queue (queue family 0)
4. Create a descriptor set layout (binding 0~3: x, mat, scale, out)
5. Define Push Constants layout (`PushConstants` structure)
6. Load SPIR-V shader (`gemv_int4_vector4.spv`) and create a compute pipeline
7. Allocate buffers (Ping-pong weights × 2, x, scale, out)
8. Create and bind descriptor pools and descriptor sets × 2
9. Create a command pool

**Buffer Memory Layout**:
```
g_matBuf[0] (300MB) ─── Ping buffer: Weights currently being computed
g_matBuf[1] (300MB) ─── Pong buffer: Prefetching next layer weights
g_xBuf      (64KB)  ─── Input vector x (float32)
g_scaleBuf  (1MB)   ─── Dequant scale (float32)
g_outBuf    (1MB)   ─── Result output (float32)
```

All buffers are allocated in CPU-GPU zero-copy shared memory with the `HOST_VISIBLE | HOST_COHERENT` flags (Optimized for APU unified memory environment).

---

#### `prefetch_weight_async()`

```c
extern "C" void prefetch_weight_async(
    const uint8_t* mat_p,  // Source: INT4 packed weight pointer in CPU RAM
    int M_out,             // Number of output rows
    int K_in,              // Input dimension (unpacked basis)
    int buf_idx            // Target buffer index (0 or 1)
)
```

**Purpose**: Asynchronously copies weight data to the specified ping-pong buffer in a **background thread** using `std::async`.

Copy size: `M_out × (K_in / 2)` bytes (INT4 packed basis)

> `weight_loader.wait()` automatically guarantees synchronization before calling `run_vulkan_gemv_pingpong()`.

---

#### `run_vulkan_gemv_pingpong()`

```c
extern "C" void run_vulkan_gemv_pingpong(
    const float* x,        // Input vector (float32, K_in size)
    const float* scale,    // Dequant scale (float32, M_out size)
    float* out,            // Output vector (float32, M_out size)
    int M_out,
    int K_in,
    int buf_idx            // Ping-pong buffer index to use
)
```

**Purpose**: Executes GPU GEMV using the weights in the specified ping-pong buffer (`buf_idx`).

**Execution Flow**:
1. `weight_loader.wait()` — Wait for asynchronous prefetch to complete
2. `memcpy` `x`, `scale` → GPU shared buffer
3. Record command buffer: Bind pipeline → Bind descriptor → Push Constants → `Dispatch`
4. `vkQueueSubmit` + `vkQueueWaitIdle` — Synchronous execution
5. `memcpy` `g_outBuf` → `out` array

**Dispatch Size**: `ceil(M_out / 32)` workgroups (Based on shader local size of 32)

**Push Constants Structure**:
```cpp
struct PushConstants {
    uint32_t M_out;          // Number of output rows
    uint32_t K_in_vector4s;     // K_in / 32 (in uvec4 units)
};
```

---

#### `run_vulkan_gemv()` *(Legacy)*

```c
extern "C" void run_vulkan_gemv(
    const float* x,
    const uint8_t* mat_p,
    const float* scale,
    float* out,
    int M_out,
    int K_in
)
```

A legacy interface that directly copies weights to buffer[0] without ping-pong and executes synchronously. Called from `igpu_matmul()` in `IGPU_CORE.py`. It is recommended to use `run_vulkan_gemv_pingpong()` in new code.

---

#### `createBuffer()` *(Internal Utility)*

```cpp
void createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory,
    void** mappedData      // Output: Mapped pointer accessible from CPU
)
```

Handles buffer creation → Querying memory requirements → Memory allocation → Binding → `vkMapMemory` mapping in one step.

---

### Dependencies

| Item | Details |
|---|---|
| Runtime Dependencies | Vulkan SDK, SPIR-V shader (`C_DLL/gemv_int4_vector4.spv`) |
| Build Flags | `-lvulkan` |
| Python Interface | `IGPU_CORE.py` (`ctypes.CDLL`) |

---

---

## 2. `my_accelerator.cpp`

### Overview

A collection of high-performance numerical computation kernels executed on the CPU. It performs AVX2-level parallel computation utilizing OpenMP SIMD directives (`#pragma omp simd`) and the GCC auto-vectorizer. Called via `ctypes` in Python from `CPU_CORE.py` and `main.py`.

**Common Rules**:
- All functions are declared within an `extern "C"` block to allow Python `ctypes` to find the symbols.
- `float* __restrict__` keyword: Guarantees to the compiler that "this pointer memory does not overlap with any other pointer," enabling SIMD optimization.
- All operations are **In-place** (overwriting the input array with the result).

---

### Function Reference

#### `run_gelu_inplace()`

```c
void run_gelu_inplace(float* x, int length)
```

**Formula**:
$$ \text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(0.7978846 \cdot (x + 0.044715 \cdot x^3)\right)\right) $$

**Implementation Features**:
- The entire loop is SIMD parallelized using `#pragma omp simd`
- The constant `GELU_CONST = 0.7978845608028654f` is defined at compile time
- The intermediate value `cube = x³` is cached in a separate variable to prevent recalculation

**Callers**: `CPU_CORE.gelu()`, `run_gemv_int4_gelu()` within `my_accelerator.cpp`

---

#### `run_RMSNorm_inplace()`

```c
void run_RMSNorm_inplace(float* x, const float* gamma, int length)
```

**Formula**:
$$ \text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{n}\sum x_i^2 + \varepsilon}} \cdot \gamma_i \quad (\varepsilon = 10^{-6}) $$

**Implementation Features**:
- Summation loop: `#pragma omp simd reduction(+:sum)` — Final aggregation after parallel addition
- The `sum` variable is declared as `double` to prevent float32 overflow when accumulating 2048 dimensions
- Single reciprocal calculation for `inv_rms` followed by multiplication (instead of division)

**Callers**: Wrapper function `rms_norm()` in `main.py`

---

#### `run_unpack_int4_inplace()`

```c
void run_unpack_int4_inplace(
    const uint8_t* packed,   // Input: INT4 × 2 packed uint8 array
    float scale,             // Row-wise dequant scale
    float* out,              // Output: float32 array (size = packed_length × 2)
    int packed_length
)
```

**Packing Format**:
```
packed[i] = (high_nibble << 4) | low_nibble
out[2*i]   = low_nibble  (signed -8~7) × scale
out[2*i+1] = high_nibble (signed -8~7) × scale
```

Sign restoration: `if (val > 7) val -= 16` (Two's complement 4-bit → int8 conversion)

**Callers**: `CPU_CORE.embedding()` — When looking up a single embedding row as a token ID

---

#### `run_rope_inplace()`

```c
void run_rope_inplace(
    float* x,           // [num_heads × dim] continuous float32 array (in-place)
    int pos,            // Current sequence position
    float theta_base,   // RoPE base frequency (Local: 10000, Global: 1000000)
    int num_heads,      // Number of attention heads
    int dim             // Dimensions per head (fixed at 256)
)
```

**Formula**:
$$ \text{cos\_vals}[i] = \cos\!\left(\text{pos} \cdot \theta_{\text{base}}^{-2i/d}\right), \quad
x'[i] = x[i]\cos - x[i+d/2]\sin, \quad x'[i+d/2] = x[i+d/2]\cos + x[i]\sin $$

**Implementation Optimization**:
- The cos/sin values are calculated **only once**, regardless of the number of heads (`cos_vals[128]`, `sin_vals[128]` stack caching)
- Structure: Outer loop (heads) + Inner SIMD loop (dimensions)

**Callers**: `CPU_CORE.cpu_rope()`

---

#### `run_softmax_inplace()`

```c
void run_softmax_inplace(float* logits, int length, float temperature)
```

**Formula**: Temperature scaling → Subtract Max (safe exp) → Summation → Normalization

**Implementation Features**:
- Temperature division and finding the maximum are **fused into a single loop** (`reduction(max:max_val)`)
- `exp` calculation and summation are **fused into a single loop** (`reduction(+:sum_exp)`)
- `sum_exp` is accumulated as a `double` to guarantee precision for 256,000 softmax calculations
- `temperature < 1e-8` guard: Prevents division by zero

**Callers**: The `_sample()` function in `main.py`

---

#### `run_gemv_int4()`

```c
void run_gemv_int4(
    const float* vec,        // Input vector [K_in]
    const uint8_t* mat_p,    // INT4 packed weight matrix [M_out × K_in/2]
    const float* scale,      // Row-wise dequant scale [M_out]
    float* out,              // Output vector [M_out]
    int M_out,
    int K_in
)
```

**Formula**: `out[i] = scale[i] × Σ( vec[k] × dequant(mat_p[i][k]) )`

**Implementation Features**:
- `#pragma omp parallel for` — Distributes M_out rows across all CPU cores
- `#pragma omp simd reduction(+:acc)` — Processes the K loop of each row with AVX2 SIMD
- Unpacking (nibble extraction + sign extension) is handled inline in the inner loop

**Callers**: `CPU_MATRIX_CORE.igpu_matmul()` (CPU mode)

---

#### `run_gemv_int4_gelu()`

```c
void run_gemv_int4_gelu(
    const float* vec,
    const uint8_t* mat_p,
    const float* scale,
    float* out,
    int M_out,
    int K_in
)
```

Identical to `run_gemv_int4()`, but applies GELU to the output **immediately (fusion)**. Used in FFN Gate operations, it eliminates the unnecessary round trip of writing the output to memory and reading it back again.

**Callers**: `CPU_MATRIX_CORE.igpu_matmul_gelu()` (CPU mode, layer 10 and above)

---

### Build Configuration Recommendations

```bash
g++ -O3 -march=native -fopenmp -ffast-math \
    -shared -fPIC -o C_DLL/my_accelerator.so my_accelerator.cpp
```

| Flag | Reason |
|---|---|
| `-march=native` | Enables AVX2 auto-vectorization |
| `-fopenmp` | Processes `#pragma omp` directives |
| `-ffast-math` | Allows approximations to optimize `tanh`/`exp` |
| `-O3` | Highest level of optimization |

---

### Function Call Map

```
Python (main.py / CPU_CORE.py)
    │
    ├── rms_norm()         ──→  run_RMSNorm_inplace()
    ├── gelu()             ──→  run_gelu_inplace()
    ├── cpu_rope()         ──→  run_rope_inplace()
    ├── embedding()        ──→  run_unpack_int4_inplace()
    ├── _sample()          ──→  run_softmax_inplace()
    └── CPU_MATRIX_CORE
            ├── igpu_matmul()      ──→  run_gemv_int4()
            └── igpu_matmul_gelu() ──→  run_gemv_int4_gelu()
```

# Code Documentation (2/8) — Vulkan Compute Shader

> **Target Files**: `gemv_int4_vector4.comp` · `gemv_int4.comp`
> **Role**: GLSL compute shaders performing GEMV with INT4 quantized weights on the iGPU
> **Compilation**: `glslc gemv_int4_vector4.comp -o C_DLL/gemv_int4_vector4.spv`

---

## Comparison of the Two Shaders at a Glance

| Item | `gemv_int4_vector4.comp` | `gemv_int4.comp` |
| ------------------------ | --------------------------- | ------------------------ |
| **Status** | ✅ Currently in use (Production) | 🗃️ Old version (Legacy) |
| **Memory Access Unit** | `uvec4` (128-bit, 16 bytes) | `uint` (32-bit, 4 bytes) |
| **INT4 processed per loop** | 32 | 8 |
| **Push Constant Field Name** | `K_in_vector4s` | `K_in_uints` |
| **Binding 1 Type** | `uvec4[]` | `uint[]` |
| **Cache Efficiency** | High (128-bit burst) | Low (32-bit units) |

---

## 1. `gemv_int4_vector4.comp` (Currently in use)

### Overview

An optimized shader that reads a `uvec4` (128-bit vector type) at once, processing 32 INT4 values per loop. It maximizes the utilization of modern GPUs' 128-bit memory bus.

### Binding Layout

```glsl
layout(binding = 0) readonly buffer InputX  { float  x[];        };  // Input vector
layout(binding = 1) readonly buffer MatP    { uvec4  mat_vec4[]; };  // INT4 packed weights (128-bit units)
layout(binding = 2) readonly buffer Scale   { float  scale[];    };  // Row-wise dequant scale
layout(binding = 3) writeonly buffer Output { float  out_vec[];  };  // Output vector
```

### Push Constants

```glsl
layout(push_constant) uniform PushConstants {
    uint M_out;           // Number of output rows (= number of weight rows)
    uint K_in_vector4s;   // K dimension divided by uvec4 units (= K_in / 32)
} params;
```

> **Why K_in / 32?**
> 1 `uvec4` = 4 bytes × 4 = 16 bytes = 128 bits.
> Since INT4 is 4 bits, there are 2 in 1 byte, and 32 in 16 bytes → Processing 32 INT4s per 1 `uvec4`.

### Execution Structure

```
Workgroup size: local_size_x = 32
Total dispatches: ceil(M_out / 32) workgroups
→ 1 thread = Responsible for 1 output row
```

### Main Logic Detail (`main()`)

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= params.M_out) return;     // Early exit for out-of-bounds threads

    float acc = 0.0;
    uint row_offset = row * params.K_in_vector4s;  // Starting uvec4 index for the row

    for (uint k = 0; k < params.K_in_vector4s; k++) {
        uvec4 packed_128 = mat_vec4[row_offset + k];  // ← Sweep load 128 bits
        uint x_idx = k * 32;

        for (int v = 0; v < 4; v++) {          // Iterate over the 4 uint32 elements of uvec4
            uint packed_32 = packed_128[v];    // x, y, z, w components
            uint x_v_idx = x_idx + (v * 8);   // Each uint handles 8 elements of x

            for (int i = 0; i < 4; i++) {      // Decompose uint32 in 8-bit chunks 4 times
                uint byte_val = (packed_32 >> (i * 8)) & 0xFF;

                // Lower 4 bits → INT4 (signed)
                int low  = int(byte_val & 0x0F);
                if (low  > 7) low  -= 16;

                // Upper 4 bits → INT4 (signed)
                int high = int((byte_val >> 4) & 0x0F);
                if (high > 7) high -= 16;

                acc += x[x_v_idx + i*2    ] * float(low);
                acc += x[x_v_idx + i*2 + 1] * float(high);
            }
        }
    }

    out_vec[row] = acc * scale[row];  // Dequantize: Scale multiplication
}
```

### Memory Access Pattern Visualization

```
mat_vec4[row_offset + k]  →  uvec4 (128 bits)
│
├── [v=0] packed_128.x  (uint32, 4 bytes)
│   ├── [i=0] byte[0]: low=INT4, high=INT4  → x[0], x[1]
│   ├── [i=1] byte[1]: low=INT4, high=INT4  → x[2], x[3]
│   ├── [i=2] byte[2]: low=INT4, high=INT4  → x[4], x[5]
│   └── [i=3] byte[3]: low=INT4, high=INT4  → x[6], x[7]
├── [v=1] packed_128.y  → x[8]  ~ x[15]
├── [v=2] packed_128.z  → x[16] ~ x[23]
└── [v=3] packed_128.w  → x[24] ~ x[31]

1 uvec4 load = Unpacking 32 INT4s = Dot product with 32 elements of x
```

### INT4 Sign Restoration Principle

```
nibble value range: 0x0 ~ 0xF (0 ~ 15, unsigned)
signed interpretation: 0 ~ 7   → Keep positive
                       8 ~ 15  → 8 as -8, 9 as -7, ... 15 as -1 (if val > 7: val -= 16)

Example:
  nibble = 0b1010 = 10 → 10 > 7 → 10 - 16 = -6 (signed)
  nibble = 0b0011 = 3  → 3 ≤ 7  → +3 (signed)
```

### Dequantization

$$ \text{out}[row] = \text{scale}[row] \times \sum_{k} \left( x[2k] \cdot w_{low}^{(k)} + x[2k+1] \cdot w_{high}^{(k)} \right) $$

The scale is a float32 value calculated **per row** as `max(|w|) / 7.0` in `quantize.py`.

---

## 2. `gemv_int4.comp` (Legacy)

### Overview

An older version of the shader that reads memory in `uint` (32-bit) units. The predecessor to `gemv_int4_vector4.comp`, it has the same unpacking logic but has 4 times lower memory load efficiency. Since it was replaced by `gemv_int4_vector4.spv` in `vulkan_core.cpp`, it is not actually executed.

### Binding Layout

```glsl
layout(binding = 0) readonly buffer InputX  { float x[];   };
layout(binding = 1) readonly buffer MatP    { uint  mat[]; };  // ← uint[] (32-bit units, old version)
layout(binding = 2) readonly buffer Scale   { float scale[]; };
layout(binding = 3) writeonly buffer Output { float out_vec[]; };
```

### Push Constants

```glsl
layout(push_constant) uniform PushConstants {
    uint M_out;
    uint K_in_uints;   // K dimension divided by uint units (= K_in / 8)
} params;
```

> **Why K_in / 8**: 1 `uint` = 4 bytes = 32 bits. 8 INT4 values of 4 bits each = 32 bits.

### Main Logic

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= params.M_out) return;

    float acc = 0.0;
    uint row_offset = row * params.K_in_uints;

    for (uint k = 0; k < params.K_in_uints; k++) {
        uint packed_32 = mat[row_offset + k];   // ← Loads only 32 bits (limit of the old version)
        uint x_idx = k * 8;

        for (int i = 0; i < 4; i++) {           // Decompose 32 bits into 8-bit chunks 4 times
            uint byte_val = (packed_32 >> (i * 8)) & 0xFF;

            int low  = int(byte_val & 0x0F);
            if (low  > 7) low  -= 16;

            int high = int((byte_val >> 4) & 0x0F);
            if (high > 7) high -= 16;

            acc += x[x_idx + i*2    ] * float(low);
            acc += x[x_idx + i*2 + 1] * float(high);
        }
    }

    out_vec[row] = acc * scale[row];
}
```

---

## Evolution Relationship of the Two Shaders

```
gemv_int4.comp (Old version)
│
│  Problem: Reads 32 bits at a time using uint[]
│           → Processes 8 INT4s per loop
│           → 25% memory bus utilization
│
▼
gemv_int4_vector4.comp (Current)
   Improvement: Reads 128 bits at a time using uvec4[]
                → Processes 32 INT4s per loop
                → 100% memory bus utilization
                → The number of K loop iterations decreases by 4 times
```

## Build and Deploy

```bash
# Compile to SPIR-V binaries
glslc gemv_int4_vector4.comp -o C_DLL/gemv_int4_vector4.spv
glslc gemv_int4.comp         -o C_DLL/gemv_int4.spv          # Legacy, unused

# Load in vulkan_core.cpp
auto shaderCode = readFile("C_DLL/gemv_int4_vector4.spv");
```

## Complete Call Path

```
main.py
  └── hw_compute_pingpong() / hw_matmul()
        └── IGPU_CORE.py
              ├── compute_pingpong()  → vk_lib.run_vulkan_gemv_pingpong()
              └── igpu_matmul()       → vk_lib.run_vulkan_gemv()
                    └── vulkan_core.cpp
                          └── vkCmdDispatch() → gemv_int4_vector4.spv
                                                 (Executed on GPU)
```

## Performance Considerations

| Item | Value | Notes |
| ---------------- | --------- | ----------------------------------------------- |
| Workgroup Size | 32 | Recommended to match the GPU warp/wavefront size |
| Maximum M_out | 262,144 | `MAX_M` constant (LM Head vocab size) |
| Maximum K_in | 16,384 | `MAX_K` constant (FFN intermediate dimension) |
| Weight Buffer Limit | 300MB × 2 | Each ping-pong buffer. Needs splitting if FFN W_gate (~562MB) is exceeded |
| Scale Precision | float32 | Minimizes quantization error |
# Code Documentation (3/8) — CPU Calculation Layer

> **Target Files**: `CPU_CORE.py` · `CPU_MATRIX_CORE.py`
> **Role**: CPU-dedicated operations like tokenizer, attention, RoPE (`CPU_CORE`) + CPU mode INT4 GEMV interface (`CPU_MATRIX_CORE`)

---

## 1. `CPU_CORE.py`

### Overview

A module that gathers **all operations dedicated to the CPU** in the model inference pipeline.
It directly links `my_accelerator.so` (C++ DLL) using `ctypes` to delegate performance-critical kernels (GELU, RMSNorm, RoPE, INT4 unpacking, Softmax) to C++ SIMD, while the Python level handles array preparation and shape transformation.

**Module Initialization Sequence** (Automatically executed upon import):

```
1. Load AutoTokenizer  ← local_gemma_3n_int4/ directory
2. Load ctypes.CDLL("C_DLL/my_accelerator.so")
3. Register argtypes / restype for each C++ function
```

---

### C++ DLL Binding

| C++ Function | Python Wrapper | Argument Types |
| ------------------------- | ------------- | --------------------------------------------------- |
| `run_gelu_inplace` | `gelu()` | `float32[1D]`, `c_int` |
| `run_unpack_int4_inplace` | `embedding()` | `uint8[1D]`, `c_float`, `float32[1D]`, `c_int` |
| `run_rope_inplace` | `cpu_rope()` | `float32[1D]`, `c_int`, `c_float`, `c_int`, `c_int` |

> There is a bug in the code where `run_gelu_inplace.restype = None` is set **twice**
> (`run_unpack_int4_inplace.restype` is unset). It has no effect on functionality but explicit correction is recommended.

---

### Function Reference

#### `tokenize(text)`

```python
def tokenize(text: str) -> np.ndarray  # shape: [T], dtype: int64
```

Converts a string to an array of token IDs using the HuggingFace `AutoTokenizer`.
Outputs the token IDs (`print`) for debugging purposes.

```python
tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
```

---

#### `embedding(token_id, W_packed, W_scale)`

```python
def embedding(
    token_id: int,
    W_packed: np.ndarray,   # [vocab_size, hidden//2], dtype=uint8  (mmap)
    W_scale:  np.ndarray,   # [vocab_size],            dtype=float32 (mmap)
) -> np.ndarray             # [hidden], dtype=float32
```

Extracts one row of a token from the INT4 embedding table and converts it to float32.

**Execution Flow**:
```
1. W_packed[token_id]  → row_packed (uint8, 1D, contiguous guaranteed)
2. W_scale[token_id]   → row_scale  (scalar float)
3. np.empty(hidden)    → out_f32    (Pre-allocate output buffer)
4. c_lib.run_unpack_int4_inplace(row_packed, scale, out_f32, packed_len)
   └── Nibble separation + sign restoration + scale multiplication performed in C++
5. return out_f32       # [hidden] float32
```

**Memory Optimization**: Since `W_packed` and `W_scale` are arrays opened via mmap, actual disk reading occurs only for that specific row (`token_id`), which is about ~5.5 KB.

---

#### `gelu(x)`

```python
def gelu(x: np.ndarray) -> np.ndarray  # Returns the same shape as input
```

A wrapper calling C++ `run_gelu_inplace`.

```python
x_flat = np.ascontiguousarray(x.flatten().astype(np.float32))
c_lib.run_gelu_inplace(x_flat, x_flat.size)
return x_flat.reshape(x.shape)
```

Handles inputs of arbitrary shapes via `flatten()` + `reshape()`.
Works transparently for both 1D and 2D.

---

#### `cpu_qk_norm(Q, K, gamma_q, gamma_k)`

```python
def cpu_qk_norm(
    Q: np.ndarray,       # [num_heads × 256] flat
    K: np.ndarray,       # [num_heads × 256] flat
    gamma_q: np.ndarray, # [256]
    gamma_k: np.ndarray, # [256]
) -> tuple[np.ndarray, np.ndarray]  # Q_norm, K_norm (flat)
```

Applies **per-head RMSNorm** to Q and K separately.
It's a QK-Norm technique that prevents attention score explosions (a feature unique to Gemma 3N).

**Formula**:
$$ Q_{norm}[h] = \frac{Q[h]}{\sqrt{\text{mean}(Q[h]^2) + \varepsilon}} \cdot \gamma_q $$

```python
Q_reshaped = Q.reshape(-1, 256)   # [num_heads, 256]
q_rms = np.sqrt(np.mean(Q_reshaped**2, axis=1, keepdims=True) + 1e-6)
Q_norm = (Q_reshaped / q_rms) * gamma_q
return Q_norm.flatten(), K_norm.flatten()
```

> The operation is performed after forcibly casting to float32.

---

#### `cpu_rope(x, pos, theta_base)`

```python
def cpu_rope(
    x:          np.ndarray,  # [num_heads × 256] flat, float32
    pos:        int,          # Current sequence position
    theta_base: float,        # Local=10,000 / Global=1,000,000
) -> np.ndarray               # [num_heads × 256] flat, float32 (in-place result)
```

A wrapper calling C++ `run_rope_inplace`.

```python
x_flat = np.ascontiguousarray(x.astype(np.float32).flatten())
c_lib.run_rope_inplace(x_flat, int(pos), float(theta_base), num_heads, 256)
return x_flat
```

**RoPE Frequency**: `cos_vals/sin_vals` caching inside C++ (Calculated once per head).
`theta_base` is determined in `main.py` according to the layer index:

```python
theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0   # Global / Local
```

---

#### `cpu_gqa(Q_rope, K_cache_layer, V_cache_layer)`

```python
def cpu_gqa(
    Q_rope:          np.ndarray,  # [num_q_heads × 256] flat  (= 2×4×256 = 2048)
    K_cache_layer:   np.ndarray,  # [seq_len, 512] float16
    V_cache_layer:   np.ndarray,  # [seq_len, 512] float16
) -> np.ndarray                   # [2048] float32 (flat)
```

**Grouped Query Attention** (GQA) implementation.
Attention head configuration of Gemma 3N E4B:

| Item | Value |
| ----------- | ---------------------------- |
| Number of Q heads | 8 (= 2 groups × 4 heads) |
| Number of K/V heads | 2 (GQA: 4 Qs share 1 KV) |
| Head Dimension | 256 |

> **Crucial**: No scaling (`/ sqrt(256)`) — Gemma 3N's **Unscaled Attention** design.

**Execution Flow**:

```python
Q = Q_rope.reshape(2, 4, 256)          # [kv_heads, q_per_kv, head_dim]
K = K_cache.reshape(-1, 2, 256)        # [seq, kv_heads, head_dim]
V = V_cache.reshape(-1, 2, 256)

K_t = K.transpose(1, 2, 0)            # [kv_heads, head_dim, seq]
scores = Q @ K_t                       # [kv_heads, q_per_kv, seq]

# Stable softmax (subtract max)
scores -= scores.max(axis=-1, keepdims=True)
probs = exp(scores) / sum(exp(scores))

V_t = V.transpose(1, 0, 2)            # [kv_heads, seq, head_dim]
out = probs @ V_t                      # [kv_heads, q_per_kv, head_dim]
return out.flatten()                   # [2048]
```

---

#### `cpu_update_kv_cache()` *(Currently disabled)*

```python
def cpu_update_kv_cache(K_rope, V, token_cnt, layer_idx, K_cache, V_cache)
```

The function body is completely commented out. KV cache updating is handled directly inside `forward_one_token()` in `main.py`:

```python
# Inside main.py (Actual cache update location)
if i < 20:
    K_cache[i, pos, :] = K   # Write directly to pre-allocated [35, max_seq, 512] float16 array
    V_cache[i, pos, :] = V
```

---

#### `_get_rope_freqs(theta_base, dim=256)` *(Internal Utility)*

```python
_rope_freq_cache: dict = {}   # Module-level cache

def _get_rope_freqs(theta_base: float, dim: int = 256) -> np.ndarray
```

Calculates the RoPE frequency table and caches it in a module-level dictionary.
Currently, it is not directly called at the Python level because the C++ version of `run_rope_inplace` internally performs the same caching. (Auxiliary function of the legacy `cpu_rope` Python version)

---

### Dependencies and Initialization

```
When importing CPU_CORE.py:
  ├── transformers.AutoTokenizer  (HuggingFace)
  ├── ctypes.CDLL("C_DLL/my_accelerator.so")
  └── tokenizer = AutoTokenizer.from_pretrained("local_gemma_3n_int4/")
```

> The `tokenizer` is maintained as a module global variable and is also accessed directly as `CPU_CORE.tokenizer.decode()` from `main.py`.

---

---

## 2. `CPU_MATRIX_CORE.py`

### Overview

A **CPU-exclusive matrix multiplication module** that replaces `IGPU_CORE.py` when `ACCEL_MODE = "CPU"` is set.
It directly calls the `run_gemv_int4` / `run_gemv_int4_gelu` C++ kernels of `my_accelerator.so` to perform INT4 GEMV with OpenMP multicore + AVX2 SIMD.

Since it provides an **exactly identical interface** (`igpu_matmul`, `igpu_matmul_gelu`, `preload_and_free`, `warmup`) as `IGPU_CORE.py`, you can switch between CPU/GPU by merely changing the single `ACCEL_MODE` variable in `main.py`.

```python
# main.py
if ACCEL_MODE == "IGPU":
    import IGPU_CORE as FAST_MATRIX_CORE
elif ACCEL_MODE == "CPU":
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE
# Subsequent code is called identically in the form of FAST_MATRIX_CORE.igpu_matmul()
```

---

### C++ DLL Binding

```python
c_lib.run_gemv_int4.argtypes = [
    ndpointer(float32, 1D),   # vec   [K_in]
    ndpointer(uint8,   2D),   # mat_p [M_out, K_in/2]
    ndpointer(float32, 1D),   # scale [M_out]
    ndpointer(float32, 1D),   # out   [M_out]
    c_int,                    # M_out
    c_int,                    # K_in
]

c_lib.run_gemv_int4_gelu.argtypes = [...]  # Same signature
```

---

### Output Buffer Pool

```python
_OUTPUT_BUF_POOL: dict[int, np.ndarray] = {}

def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]
```

`np.empty` arrays corresponding to the output size (M_out) are **allocated only once** and reused.
This removes the memory allocation cost and GC pressure that occur with every call.

> After overwriting in the C++ kernel, you **must return it with `.copy()`**.
> Returning the pool buffer directly will contaminate the value in subsequent calls.

---

### Function Reference

#### `igpu_matmul(x_vec, weight_data)`

```python
def igpu_matmul(
    x_vec:       np.ndarray,             # [K_in] float32
    weight_data: tuple | np.ndarray,     # INT4 tuple or standard float matrix
) -> np.ndarray                          # [M_out] float32
```

**INT4 Tuple Path**:
```python
packed, scale = weight_data          # packed: uint8[M_out, K_in/2]
out_buf = _get_output_buf(M_out)
c_lib.run_gemv_int4(x_f32, packed, scale, out_buf, M_out, K_in)
return out_buf.copy()
```

**Standard Matrix Fallback Path**:
```python
return np.dot(x_f32, weight_data.astype(np.float32).T)
```

> Note the `np.dot(x, W.T)` form — assumes the weight is in the `[M_out, K_in]` layout.

---

#### `igpu_matmul_gelu(x_vec, weight_data)`

```python
def igpu_matmul_gelu(
    x_vec:       np.ndarray,
    weight_data: tuple | np.ndarray,
) -> np.ndarray
```

**INT4 Tuple Path**: Calls C++ `run_gemv_int4_gelu` — **Fuses execution** of GEMV and GELU within a single kernel.
**Standard Matrix Fallback Path**: Separate call to `CPU_CORE.gelu()` after `np.dot`.

```
Advantages of fusing GEMV + GELU:
  Separate execution: GEMV → [M_out] memory write → GELU → [M_out] memory read
  Fused execution: GEMV → acc → GELU (applied immediately at the acc stage) → [M_out] memory write once
  → Saves 1 memory round trip (Saves about 64KB when M_out=16384)
```

---

#### `preload_and_free(W, keys)` / `_get_or_upload_weight(weight_data)`

```python
def preload_and_free(W: dict, keys: list): pass
def _get_or_upload_weight(weight_data):    pass
```

These are **empty functions (no-op)** for interface compatibility with `IGPU_CORE.py`.
Since there is no concept of VRAM upload in CPU mode, they do nothing.

---

#### `warmup()`

```python
def warmup()
```

Calls the C++ kernel once with a small dummy array to warm up the OpenMP thread pool and AVX2 registers.

```python
dummy_x = np.zeros(2048, dtype=np.float32)
dummy_p = np.zeros((2048, 1024), dtype=np.uint8)
dummy_s = np.zeros(2048, dtype=np.float32)
igpu_matmul(dummy_x, (dummy_p, dummy_s))
```

Prevents latency (thread spawn, cache cold start) on the first actual inference call.

---

### Interface Correspondence Table with `IGPU_CORE.py`

| Function | CPU_MATRIX_CORE | IGPU_CORE |
| ------------------------- | ------------------------------ | ----------------------------------- |
| `igpu_matmul()` | `run_gemv_int4` (C++/CPU) | `run_vulkan_gemv` (Vulkan/iGPU) |
| `igpu_matmul_gelu()` | `run_gemv_int4_gelu` (C++/CPU) | `igpu_matmul()` + `CPU_CORE.gelu()` |
| `preload_and_free()` | no-op | no-op (Legacy VRAM optimization) |
| `_get_or_upload_weight()` | no-op | Weight VRAM upload |
| `warmup()` | OpenMP warmup | Prints shader load confirmation message |
| `prefetch_weight()` | None | `prefetch_weight_async` (Asynchronous) |
| `compute_pingpong()` | None | `run_vulkan_gemv_pingpong` (Ping-pong) |

> CPU mode does not have a ping-pong prefetch feature. Ping-pong optimization is exclusive to Vulkan iGPU mode.

---

### Module Dependencies

```
main.py
  │
  ├── ACCEL_MODE = "CPU"  →  CPU_MATRIX_CORE.py
  │                              └── C_DLL/my_accelerator.so
  │                                    ├── run_gemv_int4()
  │                                    └── run_gemv_int4_gelu()
  │
  └── (Always) CPU_CORE.py
              ├── C_DLL/my_accelerator.so
              │     ├── run_gelu_inplace()
              │     ├── run_RMSNorm_inplace()
              │     ├── run_rope_inplace()
              │     └── run_unpack_int4_inplace()
              └── transformers.AutoTokenizer
```
# Code Documentation (4/8) — iGPU Interface & Main Pipeline

> **Target Files**: `IGPU_CORE.py` · `main.py`
> **Role**: Vulkan Python binding (`IGPU_CORE`) + Orchestration of the entire inference loop (`main`)

---

## 1. `IGPU_CORE.py`

### Overview

An **iGPU acceleration interface module** that wraps `vulkan_core.so` (C++ Vulkan engine) with Python `ctypes`.
When `ACCEL_MODE = "IGPU"` is set in `main.py`, it is imported under the name `FAST_MATRIX_CORE` and provides the exact same function signatures as `CPU_MATRIX_CORE.py`.

**Module Initialization Sequence** (Automatically executed upon import):

```
1. Load ctypes.CDLL("C_DLL/vulkan_core.so")
2. os.chdir(base_dir)           ← Essential for finding .spv shader files via relative paths
3. vk_lib.init_vulkan_engine()  ← Full initialization of Vulkan instance/pipeline/buffers (Once)
4. Register argtypes for each C++ function
```

> Without `os.chdir(base_dir)`, `readFile("C_DLL/gemv_int4_vector4.spv")` in `vulkan_core.cpp`
> would search relative to the execution location and fail to find the file.

---

### C++ DLL Binding

| C++ Function | Python Wrapper | Notes |
| -------------------------- | --------------------- | ----------------------- |
| `init_vulkan_engine` | (Called directly during initialization) | Executed automatically once upon import |
| `run_vulkan_gemv` | `igpu_matmul()` | Legacy synchronous GEMV |
| `prefetch_weight_async` | `prefetch_weight()` | Asynchronous weight prefetch |
| `run_vulkan_gemv_pingpong` | `compute_pingpong()` | Ping-pong buffer specific GEMV |

**Pay attention to the argument order of `run_vulkan_gemv_pingpong`**:
```python
# C++ Signature:
#   run_vulkan_gemv_pingpong(x, scale, out, M_out, K_in, buf_idx)
# ← packed (weights) is not an argument! It is already loaded into VRAM via prefetch_weight_async()
vk_lib.run_vulkan_gemv_pingpong.argtypes = [
    float32[1D],   # x     (Input vector)
    float32[1D],   # scale (Dequant scale)
    float32[1D],   # out   (Output vector)
    c_int,         # M_out
    c_int,         # K_in
    c_int,         # buf_idx (0 or 1)
]
```

---

### Output Buffer Pool

```python
_OUTPUT_BUF_POOL: dict[int, np.ndarray] = {}

def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]
```

Allocates `np.empty` arrays by output size only once and reuses them.
When returning a result, you must call `.copy()` so the pool buffer does not become contaminated.

---

### Function Reference

#### `igpu_matmul(x_vec, weight_data)`

```python
def igpu_matmul(
    x_vec:       np.ndarray,          # [K_in] any dtype → converted to float32
    weight_data: tuple | np.ndarray,  # INT4 tuple or standard float matrix
) -> np.ndarray                       # [M_out] float32
```

**INT4 Tuple Path** (`weight_data = (packed, scale)`):
```
packed.shape[0]     → M_out
packed.shape[1] × 2 → K_in (1 uint8 = 2 INT4s)
vk_lib.run_vulkan_gemv(x, packed, scale, out_buf, M_out, K_in)
```

**Standard Matrix Path** (float matrix):
```python
return np.dot(x_f32, w_f32)
# ← Note the lack of transpose compared to np.dot(x, W.T) in CPU_MATRIX_CORE
#    weight_data must already be in [K_in, M_out] layout
```

---

#### `igpu_matmul_gelu(x_vec, weight_data)`

```python
def igpu_matmul_gelu(x_vec, weight_data) -> np.ndarray
```

Applies `CPU_CORE.gelu()` separately after calling `igpu_matmul()`.

> **Difference from CPU_MATRIX_CORE**: While CPU mode fuses GEMV+GELU in a single C++ kernel via `run_gemv_int4_gelu`, the Vulkan iGPU mode executes them separately in sequence: GEMV(GPU) → GELU(CPU). There is room for further optimization in iGPU mode by integrating GELU into the shader.

---

#### `prefetch_weight(weight_data, buf_idx)`

```python
def prefetch_weight(
    weight_data: tuple,   # (packed, scale) — no-op if not a tuple
    buf_idx:     int,     # 0 or 1 (Target ping-pong buffer)
)
```

Calls the C++ `prefetch_weight_async()` to asynchronously copy the weights to VRAM in a background thread using `std::async`. It returns immediately, and completion of the copy is guaranteed by `weight_loader.wait()` during the next `compute_pingpong()` call.

If it's not a tuple (standard float matrix), no VRAM upload is necessary, so it performs no action.

---

#### `compute_pingpong(x_vec, weight_data, buf_idx)`

```python
def compute_pingpong(
    x_vec:       np.ndarray,  # [K_in] float32
    weight_data: tuple,       # (packed, scale)
    buf_idx:     int,         # Buffer index to use
) -> np.ndarray               # [M_out] float32
```

Executes a GPU GEMV using the weights in the ping-pong buffer indicated by `buf_idx`.
The weights for this buffer must have already been loaded into VRAM by a previous `prefetch_weight(w, buf_idx)` call.

**Standard Matrix Fallback**: If not a tuple, calculates on the CPU via `np.dot(x, W.T)`.

---

#### Legacy Functions

```python
def preload_and_free(W, keys): pass   # Legacy VRAM pre-upload from previous Taichi-based version
def _get_or_upload_weight(w):  pass   # Same as above
def warmup(): print("...")            # Only prints shader load completion message
```

These are functions that existed for VRAM memory management in the older Taichi version. In the current Vulkan ping-pong architecture, weights are dynamically transferred at the time of the call, so they are unnecessary. They are kept as empty functions for interface compatibility with `main.py`.

---

---

## 2. `main.py`

### Overview

The **entry point and orchestrator** for the entire Gemma 3N E4B inference pipeline.
It controls the overall flow of Tokenization → Embedding → 35 Transformer layer iterations → Logit Decoding → Sampling.

---

### Module-Level Configuration

```python
ACCEL_MODE = "IGPU"          # Change to "CPU" to switch to CPU_MATRIX_CORE
NUM_LAYERS = 35

_IGPU_WEIGHT_KEYS = [        # List of keys targeted for ping-pong prefetch (currently passed to the no-op function)
    "W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"
]
```

Upon loading the module, it additionally registers the argtypes of two functions from `my_accelerator.so`: `run_RMSNorm_inplace` and `run_softmax_inplace`. (This structure **redundantly loads** the same DLL as `CPU_CORE.py`.)

---

### Utility Functions

#### `rms_norm(x, gamma)`

```python
def rms_norm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray
```

A wrapper exclusive to main.py that calls C++ `run_RMSNorm_inplace`.
This feature is absent in `CPU_CORE` and is only used within `main.py`.

```python
x_f32   = np.ascontiguousarray(x.astype(np.float32))     # Create an independent copy
gamma_c = np.ascontiguousarray(gamma.astype(np.float32))
c_lib.run_RMSNorm_inplace(x_f32, gamma_c, x_f32.size)    # Overwrite in-place
return x_f32
```

> Since it creates an independent copy via `np.ascontiguousarray`, the original `x` remains unmodified.

---

#### `get_router_modalities(x, w_norm, w_router)`

```python
def get_router_modalities(x, w_norm, w_router) -> np.ndarray  # [4]
```

Calculates the modality vector of the AltUp router.
Normalizes `xs[0]`, projects it using the router weights, and compresses it with Tanh.

```python
x_n = rms_norm(x, w_norm) / 2048.0      # Dimension scale correction
return np.tanh(np.dot(x_n, w_router))   # shape: [4]
```

It is called **twice**: at the beginning of the layer (AltUp Predict) and at the end of the layer (AltUp Correct).

---

#### `hw_matmul(x, w, use_gelu=False)` / `hw_prefetch(w, buf_idx)` / `hw_compute_pingpong(x, w, buf_idx, use_gelu=False)`

**Three hardware adapter functions** that abstract ping-pong optimization and mode switching.

| Function | IGPU Mode | CPU Mode |
| --------------------- | ------------------------------------- | ------------------------------ |
| `hw_matmul` | `FAST_MATRIX_CORE.igpu_matmul[_gelu]` | Inline INT4 dequant + `np.dot` |
| `hw_prefetch` | `FAST_MATRIX_CORE.prefetch_weight` | no-op |
| `hw_compute_pingpong` | `FAST_MATRIX_CORE.compute_pingpong` | `hw_matmul` fallback |

CPU mode inline dequant for `hw_matmul`:
```python
# If it's a tuple, manually convert INT4 → float32
low  = (packed & 0x0F).astype(np.int8); low[low > 7]   -= 16
high = (packed >> 4  ).astype(np.int8); high[high > 7] -= 16
w_real = interleave(low, high) * scale[:, np.newaxis]
out = np.dot(x, w_real.T)
```

---

### Core Function: `forward_one_token()`

```python
def forward_one_token(
    token_id:     int,
    pos:          int,            # Current sequence position (0-indexed)
    W:            dict,           # Dictionary of weights for the 35 layers
    W_embed:      tuple,          # (packed, scale) mmap
    W_ple_packed: np.ndarray,     # [262144, 4480] uint8 mmap
    W_ple_scale:  np.ndarray,     # [262144] float32 mmap
    norm_ple:     np.ndarray,     # [256] float32
    W_ple_proj:   tuple,          # (packed, scale) INT4
    altup_projs:  list[np.ndarray],  # [3] × [2048, 2048]
    K_cache:      np.ndarray,     # [35, max_seq, 512] float16 (pre-alloc)
    V_cache:      np.ndarray,     # [35, max_seq, 512] float16 (pre-alloc)
) -> np.ndarray                   # xs: [4, 2048] float32 (4-stream output)
```

Performs **Embedding → PLE calculation → 35 layer iterations** for a single token.

#### Phase A: Embedding and AltUp Initialization

```python
# 1. Look up INT4 embedding + Gemma 3N scaling
x0 = CPU_CORE.embedding(token_id, W_embed[0], W_embed[1])
x0 = x0 * sqrt(2048.0)               # Gemma 3N specific embedding scaling

# 2. Initialize AltUp 4-Stream
xs[0] = x0                            # Main stream (Absolutely no modifications)
xs[1..3] = dot(x0, altup_projs[0..2]) # Shadow Streams
```

#### Phase B: Calculate PLE (Per-Layer Embedding)

```python
# W_ple_proj: [2048] → [35×256] projection (IGPU)
x_proj = hw_matmul(x0, W_ple_proj) / sqrt(2048.0)
x_proj = x_proj.reshape(35, 256)
x_proj_normed = RMSNorm_perrow(x_proj) * norm_ple   # Row-wise RMSNorm

# W_ple: [vocab, 8960] → Look up corresponding token row → [35, 256]
y = embedding(token_id, W_ple_packed, W_ple_scale).reshape(35, 256) * sqrt(256.0)

# Final PLE vector (Layer-wise position embedding)
pli_all = (x_proj_normed + y) * (1/sqrt(2.0))   # shape: [35, 256]
```

#### Phase C: 35-Layer Iteration Loop

**Layer Start: AltUp Predict**
```python
modalities = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
coef_mat   = dot(W["altup_pred"][i], modalities).reshape(4, 4)  # [4, 4]
xs_pred    = xs + dot(coef_mat, xs)   # Predicted stream (temporary lens)
x          = xs_pred[0].copy()        # Attention uses pure xs_pred[0]
```

**Attention Block (Ping-Pong Order)**

```
prefetch(W_q[0], buf=0)   ← Pre-load before entering the loop

[Start of Layer i]
buf=0: Calculate W_q  │  Asynchronous: W_k → buf=1
buf=1: Calculate W_k  │  Asynchronous: W_v → buf=0
buf=0: Calculate W_v  │  Asynchronous: W_o → buf=1
       QK-Norm, RoPE, KV Cache, GQA
buf=1: Calculate W_o  │  Asynchronous: W_gate → buf=0
       Calculate LAuReL
```

**KV Cache Routing Rules**:
```python
if i < 20:
    K_cache[i, pos, :] = K      # Layers 0~19: Save to its own slot
    V_cache[i, pos, :] = V
    target_k = K_cache[i, :pos+1, :]
else:
    if i % 5 == 4:              # Global Layers (20,25,30,34): Reuse layer 19 cache
        target_k = K_cache[19, :pos+1, :]
    else:                       # Local Layers (21~24, 26~29, ...): Reuse layer 18 cache
        target_k = K_cache[18, :pos+1, :]
```

**1st Residual Connection + LAuReL**:
```python
attn_output = RMSNorm(W_o_out, post_attn_ln)
attn_output += x                             # Residual connection
# LAuReL: inputs_normalized → left → right → norm → + inputs_normalized
laurel_out_normed = inputs_normalized + RMSNorm(right(left(inputs_normalized)))
attn_output = (attn_output + laurel_out_normed) * (1/sqrt(2.0))  # Sum scaled
```

**FFN Block (Ping-Pong Order + Sparsity)**:
```
buf=0: Calculate W_gate (i≥10: Fused GELU)  │  Asynchronous: W_up → buf=1
buf=1: Calculate W_up                       │  Asynchronous: W_down → buf=0

if i < 10:   # Layers 0~9 5% sparsity surgery
    cutoff = mean(gate_out) + std(gate_out) * 1.6448536   # z=1.645 → top 5%
    sparse_gate = max(gate_out - cutoff, 0)
    hidden = gelu(sparse_gate) * up_out
else:        # Layers 10~34 dense
    hidden = gate_out * up_out

buf=0: Calculate W_down  │  Asynchronous: W_q[i+1] → buf=1 (Pre-load for next layer)
```

**2nd Residual Connection**:
```python
outputs = RMSNorm(mlp_out, post_ffn_ln) + attn_output
```

**Layer End: AltUp Correct + Inject PLE**:
```python
activated  = outputs * W["altup_scale"][i]
innovation = activated - xs_pred[0]

mod_corr   = get_router_modalities(activated, ...)
corr_coefs = dot(W["altup_corr"][i], mod_corr) + 1.0   # [4]

xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation   # Calibrate the 4 streams

# Inject PLE (Do not touch xs[0]!)
gate_ple = gelu(hw_matmul(activated, W["ple_gate"][i])) * pli_all[i]
mapped   = RMSNorm(hw_matmul(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
xs_new[1:] += mapped    # Inject into shadow streams 1, 2, and 3 only

xs = xs_new
```

---

### Core Function: `decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)`

```python
def decode_logits(...) -> np.ndarray  # [vocab_size=262400] float32
```

Transforms the 4-stream `xs`, the output of the 35 layers, into a single logit vector.

```python
# 1. Prefetch W_lm_head (Since LM Head is large, load asynchronously during CPU calculation)
hw_prefetch(W_lm_head, buf_idx=0)

# 2. Normalize magnitude of 4 streams and sum the average
target_mag = mean(xs[0]**2)**0.5
for k in 1..3:
    proj_x = dot(xs[k+1], altup_unprojs[k])
    proj_x *= target_mag / max(mean(proj_x**2)**0.5, 1e-12)  # Match magnitude
x_final = mean(stack([xs[0], proj_0, proj_1, proj_2]), axis=0)  # [2048]

# 3. Final RMSNorm + LM Head (Uses ping-pong buffer 0)
x_final = RMSNorm(x_final, W_final_norm)
logits  = hw_compute_pingpong(x_final, W_lm_head, buf_idx=0)   # [262400]
```

---

### Core Function: `_sample(logits, temperature, top_p, rep_penalty, generated)`

```python
def _sample(...) -> int  # Next token ID
```

**Execution Order**:

1. **Repetition Penalty**: Attenuate the logits of already generated tokens by `rep_penalty`(1.15)
   ```python
   logits[token] /= rep_penalty  if logits[token] > 0
   logits[token] *= rep_penalty  if logits[token] < 0
   ```

2. **Softcap**: `logits = 30.0 * tanh(logits / 30.0)` — *(Applied in `main()` before calling `_sample`)*

3. **Softmax + Temperature**: C++ `run_softmax_inplace(logits, size, temperature)`

4. **Top-p (Nucleus) Sampling**:
   ```python
   sorted_idx  = argsort(probs)[::-1]             # Sort in descending order
   cumsum       = cumsum(probs[sorted_idx])
   cutoff_mask  = cumsum - probs[sorted_idx] < top_p  # Retain only up to cumulative probability top_p
   probs_filtered[sorted_idx[cutoff_mask]] = probs[...]
   ```

5. **Token Sampling**: `np.random.choice(vocab_size, p=probs_filtered)`

> **Unimplemented Optimization**: Replacing `np.argsort` (O(n log n)) with `np.argpartition` (O(n)) in the Top-p stage could yield a significant speedup for vocab_size=262,400.

---

### `main()` — Overall Execution Flow

```
[Initialization]
  warmup()                        ← Warm up hardware
  load_local_weights()            ← Load mmap-based weights
  preload_and_free() (no-op)
  K_cache = zeros([35, 2048, 512], float16)   ← Pre-allocate KV cache
  V_cache = zeros([35, 2048, 512], float16)
  cur_pos = 0                     ← Global sequence position (Maintained across conversations)

[Conversation Loop]  while True:
  user_input = input()

  [Prefill]  for token in input_tokens:
    xs = forward_one_token(token, cur_pos, ...)
    cur_pos += 1

  [Generation]  for _ in range(MAX_NEW_TOKENS):
    logits     = decode_logits(xs, ...)
    logits     = 30 * tanh(logits / 30)        ← Softcap
    next_token = _sample(logits, ...)
    if next_token in [1, 106]: break            ← Detect EOS token

    current_text = tokenizer.decode(generated) ← Full re-decode (Prevents UTF-8 truncation in some languages)
    print(current_text[len(printed_text):])     ← Incremental output
    printed_text = current_text

    xs = forward_one_token(next_token, cur_pos, ...)
    cur_pos += 1

  gc.collect()   ← Free memory after turn ends
```

**Design Considerations**:

| Item | Value / Description |
| ---------------- | ---------------------------------------------------------------------------- |
| `cur_pos` Init | Initialized to 0 **outside** the conversation loop → KV cache continues to accumulate during multi-turn chats |
| KV Cache Limit | `MAX_NEW_TOKENS = 2048` — If exceeded, `cur_pos` goes out of bounds of the array |
| `history_tokens` | Declared but **not actually used** (Multi-turn history is unimplemented) |
| Stop Tokens | `[1, 106]` — ID 1: `<eos>`, ID 106: Gemma turn end token |
| Output Method | Full re-decode with `tokenizer.decode(generated)` then differential output to prevent UTF-8 truncation |

---

### Module Dependency Graph

```
main.py
  ├── ACCEL_MODE="IGPU" → IGPU_CORE.py
  │                           └── C_DLL/vulkan_core.so
  │                                 └── C_DLL/gemv_int4_vector4.spv (Executed on GPU)
  ├── CPU_CORE.py
  │     └── C_DLL/my_accelerator.so
  ├── safeTensor.py   (Loads weight mmap)
  └── C_DLL/my_accelerator.so  (RMSNorm, Softmax — Registered natively by main.py)
```

# Code Documentation (5/8) — Weight Loading & Conversion Pipeline

> **Target Files**: `safeTensor.py` · `Optim_tensor_load.py`
> **Role**: Loading weight mmap during inference (`safeTensor`) + Initial one-time conversion script (`Optim_tensor_load`)

---

## Overall Weight Pipeline Overview

The two files constitute a **step-by-step pipeline**.

```
[Initial One-time Execution]
Original Model (*.safetensors)
    └── quantize.py             ← INT4 quantization + generates .scale files
        └── local_gemma_3n_int4/*.safetensors

    └── Optim_tensor_load.py   ← safetensors → decomposes into individual .npy files + transpose processing
        └── mmap_weights/*.npy  (1 file per weight)

[Every Inference Execution]
    └── safeTensor.py           ← Virtually maps mmap_weights/*.npy with mmap_mode='r'
        └── Returns load_local_weights() value → main.py
```

> `safeTensor.py` will only operate normally after `Optim_tensor_load.py` has been executed.

---

## 1. `safeTensor.py`

### Overview

A loader that virtually maps the `.npy` files in the `mmap_weights/` directory with `mmap_mode='r'`, immediately returning the entire weight dictionary **without consuming RAM**.

There are two versions of the implementation within the file:

| Version | Location | Status | Source |
| ------------- | ---------------------------- | ---------- | --------------------------------------------- |
| Old Version | Top of the file (Inside the long `'''` comment) | ❌ Inactive | Parses `local_gemma_3n_int4/*.safetensors` directly |
| **Current Version**| Bottom of the file (After `'''`) | ✅ **Active** | mmap loads `mmap_weights/*.npy` |

The old version directly read files with `safetensors.torch.load_file` and converted torch tensors to numpy, but required the entire model to be loaded into RAM during loading. The current version solves this problem.

---

### Core Principle of the mmap Strategy

```python
val = np.load("mmap_weights/some_weight.npy", mmap_mode='r')
# ↑ Data read from disk at this point: 0 bytes
# ↑ OS only registers the virtual address of the file (1 page table entry)
# Actual disk read occurs only when the corresponding array element is accessed for the first time (Demand Paging)
```

As a result, the increase in RAM usage immediately after calling `load_local_weights()` is almost 0, and each weight is automatically loaded in OS page units at the time it is actually used.

**Specialized optimization for W_embed / W_ple**: Since both weights involve a pattern of querying only one row for an embedding, mmap is particularly effective. Only about 5.5 KB is actually read per token.

---

### `load_local_weights()` Function

```python
def load_local_weights(model_dir=mmap_dir) -> tuple
```

**Returned Tuple Structure**:

```python
return (
    W_embed,          # tuple (packed[262400,1024] uint8, scale[262400] f32) — mmap
    W_ple_packed,     # ndarray [262144, 4480] uint8                         — mmap
    W_ple_scale,      # ndarray [262144] float32                             — mmap
    norm_ple,         # ndarray [256] float32
    W_ple_proj,       # tuple (packed, scale) INT4
    altup_projs,      # list[3] × ndarray [2048, 2048] float32
    altup_unprojs,    # list[3] × ndarray [2048, 2048] float32
    W_final_norm,     # ndarray [2048] float32
    W_lm_head,        # tuple — Same object as W_embed (Tied Weights)
    layers,           # dict[str, list[35]] — Layer-wise weights
)
```

**`W_lm_head = W_embed` Design**:
Gemma 3N shares the same weights for the input embedding and the output LM Head (Tied Embedding). Since it references the same mmap object without separate copying, there is no memory duplication.

---

### Detailed Execution Flow

#### Step 1: Collect File List and Separate Scales

```python
all_files = glob.glob("mmap_weights/*.npy")
all_keys  = [basename(f)[:-4] for f in all_files]  # Remove .npy from filename

# Create scale file index (Fast pair search)
scales = {k[:-6]: k for k in all_keys if k.endswith(".scale")}
# Example: {"model.language_model.layers.0.self_attn.q_proj.weight":
#      "model.language_model.layers.0.self_attn.q_proj.weight.scale"}
```

#### Step 2: mmap Virtual Mapping Loop

```python
for k in all_keys:
    if k.endswith(".scale"): continue    # Scales are handled together when loading the main body

    val = np.load(f"mmap_weights/{k}.npy", mmap_mode='r')  # RAM consumption 0

    if k in scales:                      # INT4 Tensor: Bundled as a (packed, scale) tuple
        scale_val = np.load(f"mmap_weights/{scales[k]}.npy", mmap_mode='r')
        val = (val, scale_val)

    # Extract layer index and subkey using regex
    match = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", k)
    if match:
        layer_idx = int(match.group(1))  # 0 ~ 34
        sub_key   = match.group(2)       # "self_attn.q_proj.weight" etc.
        layers[KEY_MAP[sub_key]][layer_idx] = val
    else:
        globals_dict[k] = val            # Global weight not belonging to a layer
```

#### Step 3: Decompose and Return Global Weights

```python
P = "model.language_model."

W_embed      = globals_dict[P + "embed_tokens.weight"]          # tuple (mmap)
W_ple_packed,\
W_ple_scale  = globals_dict[P + "embed_tokens_per_layer.weight"] # tuple unpacking
W_ple_proj   = globals_dict[P + "per_layer_model_projection.weight"]
norm_ple     = globals_dict[P + "per_layer_projection_norm.weight"]
altup_projs  = [globals_dict[P + f"altup_projections.{i}.weight"] for i in range(3)]
altup_unprojs= [globals_dict[P + f"altup_unembed_projections.{i}.weight"] for i in range(3)]
W_final_norm = globals_dict[P + "norm.weight"]
W_lm_head    = W_embed   # References the same object (Tied Weights)
```

---

### SafeTensor Original Key → Internal Key Mapping Table

| SafeTensor Original Key (sub_key) | `layers` Dictionary Key |
| ----------------------------------- | -------------------- |
| `self_attn.q_proj.weight` | `W_q` |
| `self_attn.k_proj.weight` | `W_k` |
| `self_attn.v_proj.weight` | `W_v` |
| `self_attn.o_proj.weight` | `W_o` |
| `self_attn.q_norm.weight` | `gamma_q` |
| `self_attn.k_norm.weight` | `gamma_k` |
| `input_layernorm.weight` | `input_ln` |
| `post_attention_layernorm.weight` | `post_attn_ln` |
| `pre_feedforward_layernorm.weight` | `pre_ffn_ln` |
| `post_feedforward_layernorm.weight` | `post_ffn_ln` |
| `mlp.gate_proj.weight` | `W_gate` |
| `mlp.up_proj.weight` | `W_up` |
| `mlp.down_proj.weight` | `W_down` |
| `per_layer_input_gate.weight` | `ple_gate` |
| `per_layer_projection.weight` | `ple_proj` |
| `post_per_layer_input_norm.weight` | `ple_post_ln` |
| `laurel.linear_left.weight` | `laurel_left` |
| `laurel.linear_right.weight` | `laurel_right` |
| `laurel.post_laurel_norm.weight` | `laurel_norm` |
| `altup.router_norm.weight` | `altup_rn` |
| `altup.modality_router.weight` | `altup_router` |
| `altup.prediction_coefs.weight` | `altup_pred` |
| `altup.correction_coefs.weight` | `altup_corr` |
| `altup.correct_output_scale` | `altup_scale` |

---

---

## 2. `Optim_tensor_load.py`

### Overview

Two independent roles are **mixed in one file**.

| Section | Description | Execution Method |
| ---------- | ---------------------------------------------------- | -------------------------------------------- |
| **Top Part** | Memory usage measurement and structure inspection utility (`debug()`) | Function — Explicit call required |
| **Bottom Part** | SafeTensors → `.npy` conversion script | **Module top-level code** — Executes automatically upon `import` |

> **Warning**: The conversion code at the bottom is exposed at the module level, not wrapped in a function.
> Just running `import Optim_tensor_load` will immediately start the conversion process.
> In actual use, it should only be used to **execute directly** with `python Optim_tensor_load.py`.

---

### Top Part: Memory Inspection Utility

#### `get_real_memory_size(obj)`

```python
def get_real_memory_size(obj) -> int  # bytes
```

Recursively calculates the **actual memory footprint** of nested Python objects.

`sys.getsizeof()` only returns the shell size of a container (list, tuple) and does not include the `.nbytes` of internal numpy arrays. This function compensates for that limitation.

```python
def get_real_memory_size(obj):
    total = sys.getsizeof(obj)          # Container shell size

    if isinstance(obj, np.ndarray):
        total += obj.nbytes             # Add actual data size
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += get_real_memory_size(item)  # Recursive exploration
    return total
```

**Example — Calculating the actual size of an INT4 Tuple**:
```
W_q[0] = (packed[2048, 1024] uint8, scale[2048] float32)
  get_real_memory_size(W_q[0])
    = getsizeof(tuple)          # ~56 bytes (shell)
    + getsizeof(packed)         # ~112 bytes (ndarray object)
    + packed.nbytes             # 2048 × 1024 × 1 = 2,097,152 bytes
    + getsizeof(scale)          # ~112 bytes
    + scale.nbytes              # 2048 × 4 = 8,192 bytes
    ≈ 2,105,524 bytes (~2.0 MB)
```

---

#### `inspect_matrix_structure(name, obj)`

```python
def inspect_matrix_structure(name: str, obj) -> str
```

Describes the **nested structure and actual dimensions** of a weight object as a string.
This function is used to generate the tables in `optim_tensor_size.md`.

**Recursive Processing Rules**:

| Type | Output Format |
| ------------------------ | ------------------------------------------------------------- |
| `list` | `List[N] ──> {Structure of element 0}` (Inspects only element 0 as representative) |
| `tuple` | `Tuple( {Structure of element 1}, {Structure of element 2} )` |
| `np.ndarray` (uint8, 2D) | `[ matrix: A x B , type: uint8 , (INT4 dimension: A x B*2) ]` |
| `np.ndarray` (Others) | `[ matrix: shape , type: dtype ]` |

**INT4 Dimension Correction**: Since uint8 2D arrays pack 2 INT4s each, the actual number of columns is displayed as `shape[1] * 2`.

---

#### `format_memory_size(total_bytes)` / `calculate_memory_usage(obj)`

```python
def format_memory_size(total_bytes: int) -> str  # "GB | MB | Mb" format string
def calculate_memory_usage(obj) -> str
```

Formats the bytes obtained from `get_real_memory_size()` into three units: GB/MB/Megabits.

---

#### `debug()`

```python
def debug()
```

Calls `safeTensor.load_local_weights()` and then prints the structure and memory size of all weights in a Markdown table format. Currently, it is commented out (`#if __name__ == "__main__": debug()`) so it does not run automatically.

Output Example:
```
| name | matrix                                                                                                                         | GB       | MB     | Mb      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------ | ------- |
| W_q  | List[35] ──>  Tuple( [ matrix: 2048 x 1024 , type: uint8 , (INT4 dimension: 2048 x 2048) ], [ matrix: 2048 , type: float32 ] ) | 0.068636 | 70.284 | 562.269 |
...
```

---

### Bottom Part: SafeTensors → `.npy` Conversion Script

**Execution Conditions**: `local_gemma_3n_int4/*.safetensors` exist (After running quantize.py)
**Output Location**: `mmap_weights/*.npy` (1 file per tensor)

#### Execution Flow

```python
# 1. Create output directory
os.makedirs("mmap_weights/", exist_ok=True)

# 2. Iterate through all .safetensors files
for st_file in sorted(glob("local_gemma_3n_int4/*.safetensors")):
    tensors = load_file(st_file)      # safetensors → torch tensor dict

    # Identify INT4 tensors (those with a scale file)
    quantized_bases = [k[:-6] for k in tensors if k.endswith(".scale")]

    for k, val in tensors.items():
        # Convert bfloat16 → float32 (Handles dtype not supported by numpy)
        if val.dtype == torch.bfloat16:
            val = val.to(torch.float32)
        arr = val.numpy()

        # Determine Transpose
        is_quantized = (k in quantized_bases) or k.endswith(".scale")
        needs_transpose = False

        if not is_quantized:  # ← INT4 tensors are NEVER transposed (Core rule!)
            if any(suffix in k for suffix in TRANSPOSE_SUFFIXES):
                needs_transpose = True

        if needs_transpose:
            arr = np.ascontiguousarray(arr.T)
        else:
            arr = np.ascontiguousarray(arr)

        np.save(f"mmap_weights/{k}.npy", arr)
```

#### Transpose Target List

Among the non-quantized (float) weights, only those containing the suffixes below are transposed.

| Suffix | Corresponding Weight |
| ------------------------------------------------------------------ | -------------------- |
| `per_layer_model_projection.weight` | W_ple_proj |
| `altup_projections` | altup_projs |
| `altup_unembed_projections` | altup_unprojs |
| `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight` | Q, K, V, O |
| `gate_proj.weight`, `up_proj.weight`, `down_proj.weight` | FFN |
| `per_layer_input_gate.weight`, `per_layer_projection.weight` | ple_gate, (ple_proj) |
| `laurel.linear_left.weight`, `laurel.linear_right.weight` | LAuReL |
| `altup.modality_router.weight` | altup_router |

**Why transpose?**
Linear layer weights in SafeTensors are stored in the form `[out, in]` (Row: Output, Column: Input).
To perform an `x @ W` operation (Vector-Matrix multiplication) during inference, an `[in, out]` layout is required, so we convert it in advance with `.T`.

**Why not transpose INT4 tensors?**
During INT4 quantization, they are already stored in the `[out_dim, in_dim/2]` layout (Quantized per row, Row=Output).
Since the GEMV kernel (`run_gemv_int4` in `my_accelerator.cpp`) is designed to consume this layout directly, transposing it will rather cause malfunctions.

---

### Complete Pipeline Summary

```
[1] Execute quantize.py
    local_gemma_3n/ → local_gemma_3n_int4/
    (Original float16/32 → Generates INT4 packed + .scale files)

[2] Execute Optim_tensor_load.py  (First time only)
    local_gemma_3n_int4/*.safetensors → mmap_weights/*.npy
    - Converts bfloat16 → float32
    - Non-INT4 weights: Apply Transpose
    - INT4 weights: Store as is without transpose
    - Guarantees C-contiguous memory with np.ascontiguousarray

[3] safeTensor.py (Every execution)
    mmap_weights/*.npy → Virtually maps with mmap_mode='r'
    - Automatically pairs scale files → Constructs (packed, scale) tuple
    - Parses layer indices with Regex
    - Returns a 10-item tuple → Consumed by main.py
```
# Code Documentation (6/8) — Quantization & KV Cache Memory Management

> **Target Files**: `quantize.py` · `Memory_Manager.py`
> **Role**: Converts original model to INT4 quantization (`quantize`) + Pre-allocates KV cache utility (`Memory_Manager`)

---

## Location of the Two Files

From the perspective of the entire pipeline, both files are **one-time preparation tools**.

```
[Pre-preparation Stage]
  quantize.py        ← Original float model → INT4 safetensors (First time only)
  Memory_Manager.py  ← Tool for designing and verifying the size of the KV cache array

[Actual Inference]
  main.py            ← Consumes the outputs of the two files above (converted weights, cache design)
```

---

## 1. `quantize.py`

### Overview

A **script executed exactly once** to convert the large weights of the original Gemma 3N E4B model (float16/bfloat16) to **INT4 (4-bit) symmetric quantization** and save the conversion results in SafeTensors format.

**Execution**: `python quantize.py`

**Input/Output**:
```
Input: ORIGINAL_MODEL_DIR/local_gemma_3n/*.safetensors    (Original float model)
Output: SAVE_DIR/local_gemma_3n_int4/*.safetensors         (INT4 converted model)
```

---

### Module-Level Configuration

```python
ORIGINAL_MODEL_DIR = "/home/hwkim/.../local_gemma_3n"     # Original model path (hardcoded absolute path)
SAVE_DIR           = BASE_DIR + "/local_gemma_3n_int4"    # Output path
```

> `ORIGINAL_MODEL_DIR` is hardcoded as an absolute path.
> You must manually modify this value when executing in a different environment.

---

### List of Weights Targeted for Quantization (`_BIG_WEIGHT_SUFFIXES`)

Only 2D weights ending with the suffixes below are converted to INT4. The rest retain their original dtype.

| Suffix | Corresponding Weight | Size Before Conversion (Per Layer) | Size After Conversion |
| ----------------------------------- | ------------ | ----------------------- | ----------------------------------- |
| `q_proj.weight` | W_q | 2048×2048 f16 = 8MB | 2048×1024 u8 + 2048 f32 = ~2MB |
| `k_proj.weight` | W_k | 512×2048 f16 = 2MB | 512×1024 u8 + 512 f32 = ~0.5MB |
| `v_proj.weight` | W_v | 512×2048 f16 = 2MB | ~0.5MB |
| `o_proj.weight` | W_o | 2048×2048 f16 = 8MB | ~2MB |
| `gate_proj.weight` | W_gate | 16384×2048 f16 = 64MB | 16384×1024 u8 + 16384 f32 = ~16.3MB |
| `up_proj.weight` | W_up | 16384×2048 f16 = 64MB | ~16.3MB |
| `down_proj.weight` | W_down | 2048×16384 f16 = 64MB | ~16.3MB |
| `embed_tokens.weight` | W_embed | 262400×2048 f16 = 1.0GB | 262400×1024 u8 = ~257MB |
| `embed_tokens_per_layer.weight` | W_ple | 262144×8960 f16 = 4.4GB | ~1.1GB |
| `per_layer_input_gate.weight` | ple_gate | 256×2048 f16 = 1MB | ~0.27MB |
| `per_layer_model_projection.weight` | W_ple_proj | 8960×2048 f16 = 35MB | ~8.8MB |
| `laurel.linear_left.weight` | laurel_left | 64×2048 f16 = 0.25MB | ~0.065MB |
| `laurel.linear_right.weight` | laurel_right | 2048×64 f16 = 0.25MB | ~0.065MB |

**Weights not quantized**: All LayerNorm (gamma), altup coefficients, 1D weights, `ple_proj` (currently kept as float32 — subject to review for future INT4 conversion)

---

### Core Function: `quantize_to_int4(weight)`

```python
def quantize_to_int4(
    weight: np.ndarray    # [N, M] float16 or float32
) -> tuple[np.ndarray, np.ndarray]:
    # Returns: (packed [N, M//2] uint8, scale [N] float32)
```

Performs **Per-Row symmetric quantization**. Row = 1 output neuron.

#### Formula

$$ \text{scale}[i] = \frac{\max(|w_i|)}{7.0} $$

$$ w_q[i,j] = \text{clip}\!\left(\text{round}\!\left(\frac{w[i,j]}{\max(|w_i|)} \times 7.0\right),\ -8,\ 7\right) $$

$$ \text{packed}[i, j//2] = (w_q[i, 2j] \mathbin{\&} \texttt{0x0F})\ |\ ((w_q[i, 2j+1] \mathbin{\&} \texttt{0x0F}) \ll 4) $$

#### Step-by-Step Implementation

**Step 1: Upcasting to float32**
```python
w_f32 = weight.astype(np.float32)
# Errors can occur during max calculation with float16 precision → upcast to float32
```

**Step 2: Calculating scale per row**
```python
max_vals = np.max(np.abs(w_f32), axis=1, keepdims=True)  # [N, 1]
max_vals = np.maximum(max_vals, 1e-8)                     # Prevent division by 0
scale    = (max_vals / 7.0).flatten()                     # [N] — used during dequant
```

**Step 3: Normalization and Rounding**
```python
w_q = np.round(w_f32 / max_vals * 7.0).astype(np.int8)
w_q = np.clip(w_q, -8, 7)
# Range: [-8, 7] — Utilizes the entire range of signed 4-bit integers
# -8 is representable but introduces a slight asymmetric error during dequant
```

**Step 4: Packing 2 per uint8**
```python
w_q_low  = w_q[:, 0::2] & 0x0F   # Even columns → lower 4 bits
w_q_high = w_q[:, 1::2] & 0x0F   # Odd columns → upper 4 bits
packed   = (w_q_low | (w_q_high << 4)).astype(np.uint8)
# [N, M] int8 → [N, M//2] uint8  (Saves 50% memory)
```

**Packing Layout Visualization**:
```
Original  w_q[i]: [ a, b, c, d, e, f, ... ]  (int8, M items)
                └─┬─┘  └─┬─┘
packed[i]:    [ a|b<<4, c|d<<4, ... ]    (uint8, M/2 items)

Example: a=-3 (0b1101), b=5 (0b0101)
  a & 0x0F = 0x0D (lower 4 bits)
  b & 0x0F = 0x05 (upper 4 bits)
  packed   = 0x0D | (0x05 << 4) = 0x5D
```

---

### `main()` Execution Flow

```python
def main():
    for filename in sorted(glob("local_gemma_3n/*.safetensors")):
        tensors = load_file(filename)           # safetensors → torch tensor dict
        quantized_tensors = {}

        for name, tensor in tensors.items():
            is_big = any(name.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)

            if is_big and len(tensor.shape) == 2:   # Quantize only 2D large weights
                weight_np = tensor.to(torch.float32).numpy()
                packed, scale = quantize_to_int4(weight_np)

                quantized_tensors[name]            = torch.from_numpy(packed)  # uint8
                quantized_tensors[name + ".scale"] = torch.from_numpy(scale)   # float32

            else:                                   # 1D, small, non-targets → keep original
                quantized_tensors[name] = tensor

        save_file(quantized_tensors, SAVE_DIR + "/" + basename(filename))

        del tensors, quantized_tensors
        gc.collect()   # Release memory per file (Prevents the entire model from residing in RAM simultaneously)
```

**Reason for File-by-File Processing**: Processes SafeTensors files one by one and immediately calls `del` + `gc.collect()`, preventing the entire original model (~9GB) from being loaded into RAM at the same time.

---

### Quantization Error Characteristics

| Item | Description |
| -------------- | ------------------------------------------------------------ |
| Method | Symmetric quantization — Centered at 0, no offset |
| Expression Range | [-8, 7] — Theoretically [-7.5, 7.5], but forced to [-8, 7] via clip |
| Scale Unit | 1 per row (Output neuron) (Per-Row) |
| Theoretical Max Error | `scale × 0.5` (Rounding error) |
| Asymmetry | -8 is represented but +8 is clipped → Slight asymmetry exists in the negative direction |
| Precision Loss | float16 → INT4: About 75% bit reduction, maintains practical quality |

---

---

## 2. `Memory_Manager.py`

### Overview

A **utility module** that pre-allocates the KV cache array as a single contiguous NumPy array. Since this task is currently performed directly with `np.zeros()` in `main.py`, this module itself is not called during inference. It serves as reference code to design and verify the cache memory layout.

---

### `allocate_KVcache(layers, token, dimension)`

```python
def allocate_KVcache(
    layers:    int,   # Number of layers (35)
    token:     int,   # Maximum sequence length (2048)
    dimension: int,   # KV head dimension (512 = 2 KV heads × 256)
) -> np.ndarray       # [layers, token, dimension] float16
```

```python
A = np.zeros((layers, token, dimension), dtype=np.float16)
return A
```

---

### KV Cache Design Rationale

**Why float16?**
While restored to float32 (`K_cache.astype(np.float32)`) during attention calculation, it is stored as float16. The impact of precision loss on attention quality is negligible, but memory is halved.

**Dimension Configuration (512)**:
```
Number of KV heads: 2 (GQA structure)
Head dimension:  256
Total:       2 × 256 = 512
```

**Memory Calculation When Called With Default Values**:
```
allocate_KVcache(35, 2048, 512)
  = 35 × 2048 × 512 × 2 bytes (float16)
  = 73,400,320 bytes
  ≈ 70 MB (K + V respectively)
  K_cache + V_cache Total ≈ 140 MB
```

---

### Actual Allocation in `main.py`

Without importing `Memory_Manager.py`, it allocates directly in `main.py` with the same structure:

```python
# main.py
K_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
V_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
# = np.zeros((35, 2048, 512), dtype=np.float16) — Same design as Memory_Manager
```

**Indexing Method**:
```python
# Writing (Layers 0~19)
K_cache[layer_idx, pos, :] = K.astype(np.float16)   # In-place slice writing

# Reading
target_k = K_cache[layer_idx, :pos+1, :]             # Sequence slice accumulated so far
```

Compared to the previous dynamic growth method based on `np.concatenate`, the pre-allocation method replaces the O(N) reallocation that occurred with every token generation with an O(1) in-place write.

---

### Current Status and Future Directions

| Item | Current Status | Notes |
| -------------------------------- | ----------------------------------------- | -------------------------------- |
| `allocate_KVcache()` Usage | Unused in `main.py` | `main.py` calls `np.zeros` directly |
| `if __name__ == "__main__"` Block | For testing (`shape` print only) | Confirms `(35, 2048, 512)` |
| Layers 20~34 Slots | Allocated but not written to | Reuse 18/19 via KV routing |
| Potential OOM Risk | Index out of bounds if `cur_pos` exceeds 2048 | Needs `MAX_NEW_TOKENS` guard |

**Direction for Future Utilization**: When introducing KV cache initialization or sliding window methods in multi-turn conversations, it would be appropriate to extend this module to encapsulate the cache reset/compression logic.

---

### Position of the Two Files in the Entire Pipeline

```
[Step 1]  quantize.py
          Original float model (9GB+)
              ↓  Per-row symmetric INT4 quantization
          local_gemma_3n_int4/*.safetensors
          (Stored as packed uint8 + scale float32 pairs)

[Step 2]  Optim_tensor_load.py
          local_gemma_3n_int4/*.safetensors
              ↓  Decompose .npy by tensor + transpose processing
          mmap_weights/*.npy

[Step 3]  safeTensor.py  (Every execution)
          mmap_weights/*.npy → mmap virtual mapping
              ↓
          load_local_weights() return

[Step 4]  main.py  (Every execution)
          Based on Memory_Manager.py design
          K_cache, V_cache = np.zeros([35, 2048, 512], float16)
              ↓
          forward_one_token() → decode_logits() → _sample()
```

# Code Documentation (7/8) — FPGA NPU Engine & Architecture Design Document

> **Target Files**: `NPU_CORE.py` · `gemma3N_E4B_architecture.md`
> **Role**: Python control layer for FPGA-based NPU acceleration engine (`NPU_CORE`) + Reference document for overall model structure and hardware distribution (`gemma3N_E4B_architecture.md`)

---

## 1. `NPU_CORE.py`

### Overview

The **Python control layer for the FPGA RTL-based NPU hardware**. It is a **separate hardware target** from the current execution paths (`IGPU_CORE.py` / `CPU_MATRIX_CORE.py`) of the project, controlling a custom Systolic Array NPU implemented on a Xilinx/AMD FPGA via MMIO (Memory-Mapped I/O).

**Current Status**: Cannot be run directly without an FPGA board due to the `import MMIO` dependency. However, the `MMIO.SIMULATION_MODE` branch at the top of the function **completely mocks** the operation on a PC using NumPy.

---

### Hardware Architecture Overview

```
CPU (Python/NumPy)
    │  MMIO Register Control
    │  DMA Transfer
    ▼
Inside FPGA
  ┌─────────────────────────────────────┐
  │  AXI DMA  ──→  Ping-Pong BRAM      │
  │                    │               │
  │              Systolic Array NPU    │
  │              (32×32 PE Tile)       │
  │                    │               │
  │              ACC (Accumulator)     │
  │                    │               │
  │           RMSNorm / GeLU IP        │
  │                    │               │
  │              Result BRAM           │
  └─────────────────────────────────────┘
```

---

### MMIO Register Map

| Address | Role | Usage Example |
| ------------ | ------------------------------------------------------- | ----------------------------- |
| `0x00` | Control Register — Bit0: NPU_START (Pulse), Bit1: ACC_CLEAR | `write(0x00, 0x01)` Start calculation |
| `0x08` | RMSNorm denominator scalar (`mean_sq_val`) | `write(0x08, int(mean_sq))` |
| `0x0C` | DMA Switch — 0: Ping buffer, 1: Pong buffer selection | `write(0x0C, 0)` |
| `0x10` Bit0 | GeLU hardware IP enable | `write(0x10, 0x01)` |
| `0x10` Bit1 | Softmax IP enable | `write(0x10, 0x02)` |
| `0x10` Bit16 | NPU done flag (`w_npu_done`) — Polling target | `read(0x10) & 0x010000` |
| `0x14` | DMA stream type — 0: Token, 1: Weight | `write(0x14, 0 or 1)` |

> Register `0x04` was incorrectly used as the done flag in a previous version before being corrected to `0x10` (Code comment: `Polling bug fixed: 0x04 -> 0x10`).

---

### Core Function: `run_npu_matmul(x_vec, weight_mat, mean_sq_val, use_gelu=False)`

```python
def run_npu_matmul(
    x_vec:       np.ndarray,   # [2048] Input vector (Before RMSNorm)
    weight_mat:  np.ndarray,   # [2048, Output_Dim] Weight matrix
    mean_sq_val: float,        # RMSNorm denominator: mean(x²) value
    use_gelu:    bool = False, # FFN Gate specific GeLU hardware IP enable
) -> np.ndarray                # [Output_Dim] int16 (FPGA) or float16 (Simulation)
```

#### Simulation Path (`MMIO.SIMULATION_MODE = True`)

```python
inv_sqrt = 1.0 / sqrt(mean_sq_val + 1e-6)
x_f32    = x_vec.astype(np.float32) * inv_sqrt   # float32 upcasting essential
                                                   # (Risk of overflow in 2048D FP16 accumulation)
out = np.dot(x_f32, weight_mat.astype(np.float32))
if use_gelu:
    out = GELU(out)
return out.astype(np.float16)
```

**Reason for FP16 Upcasting**: Because accumulation across 2048 dimensions can exceed the maximum value of FP16 (65504), intermediate calculations must be done in FP32.

---

#### FPGA Execution Path — Tiling Structure

Decomposes the input (2048) and output (Output_Dim) into **32×32 tiles** for processing.

```
num_ic_tiles = 2048 // 32 = 64   (Number of input channel tiles)
num_oc_tiles = Out  // 32        (Number of output channel tiles)
total_tiles  = 64 × num_oc_tiles

Tile Sequence Example (Out=2048, total=4096):
  tile_idx 0   → oc=0, ic=0    (Output channel 0, Input channel 0)
  tile_idx 1   → oc=0, ic=1    (Output channel 0, Input channel 1)
  ...
  tile_idx 63  → oc=0, ic=63   ← Last ic: ACC read occurs
  tile_idx 64  → oc=1, ic=0    ← Start of ic: ACC_CLEAR occurs
```

---

#### FPGA Execution Path — Ping-Pong BRAM Pipeline

In each tile iteration, **calculation and DMA transfer for the next tile proceed simultaneously**.

```
[Prologue]
  Tile 0's token(32) + weight(32×32) → Ping buffer transfer (Synchronous)

[Main Loop: tile_idx = 0 → total_tiles-1]
  ┌─ 1. DMA Background Transfer (Asynchronous) ───────────────────────────────┐
  │   tile_idx Even (Ping calculating) → Next tile data → Pong Buffer   │
  │   tile_idx Odd (Pong calculating) → Next tile data → Ping Buffer   │
  │   Transfer Order: Token first (0x14=0) → Weight (0x14=1)             │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 2. Kick NPU Calculation ────────────────────────────────────────────────┐
  │   write(0x00, 0x01)  ← START pulse (Automatically returns to 0)        │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 3. Wait for Completion (Polling) ───────────────────────────────────────────┐
  │   while (read(0x10) & 0x010000) == 0: pass                      │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 4. Receive Result (Only for the last ic tile) ──────────────────────────────┐
  │   if ic == num_ic_tiles - 1:                                    │
  │       DMA recv → result_buf → final_out[oc*32:(oc+1)*32]       │
  └────────────────────────────────────────────────────────────────┘
  5. Wait for DMA transfer to complete, then proceed to the next loop

[Special Handling]
  Upon entering ic == 0: ACC_CLEAR (write(0x00, 0x02))
  → Initialize the accumulator before starting to calculate a new output channel
```

**Result Reception Timing**: The results are received via DMA only after 64 dot product accumulations calculating 1 output channel (32 output neurons) are completed (`ic == 63`). In between, the internal accumulator (ACC) in the FPGA holds the values.

---

### Wrapper Functions

```python
def npu_matmul(x, weight, mean_sq):
    """ Q, K, V, O, Down — Standard matrix multiplication """
    return run_npu_matmul(x, weight, mean_sq, use_gelu=False)

def npu_matmul_gelu(x, W_gate, mean_sq):
    """ FFN Gate exclusive — 1-Cycle GeLU hardware IP enabled immediately after matrix multiplication """
    return run_npu_matmul(x, W_gate, mean_sq, use_gelu=True)
```

---

### `npu_softmax(logits)`

```python
def npu_softmax(logits: np.ndarray) -> np.ndarray  # float16
```

**Simulation Path**: Stable Softmax (subtract max) NumPy implementation.

**FPGA Path**:
Since Softmax is not a matrix multiplication, it is sent to the dedicated **Softmax IP** instead of going through the Systolic Array.

```python
MMIO.npu_control.write(0x10, 0x02)   # Softmax_EN bit ON

for i in range(0, len(logits), 32):   # Transferred in chunks of 32
    ping_token ← logits[i:i+32]
    DMA send → NPU kick → DMA recv → probs[i:i+32]
```

> Softmax completion polling uses a **different register** `0x04 & 0x01` compared to matrix multiplication (`0x10 & 0x010000`). This indicates that the done signals of the Softmax IP and the Systolic Array are implemented separately at the hardware level.

---

### Comparison of the Three Acceleration Modes

| Item | NPU_CORE.py | IGPU_CORE.py | CPU_MATRIX_CORE.py |
| -------------- | -------------------------- | ----------------------- | ------------------------ |
| Hardware | FPGA Systolic Array | iGPU (Vulkan) | CPU AVX2/OpenMP |
| Control Method | MMIO Register + DMA | Vulkan Command Buffer | OpenMP Multicore |
| Weight Format | float16 (Built-in FPGA conversion) | INT4 uint8 (GPU unpacking) | INT4 uint8 (SIMD unpacking) |
| Tile Unit | 32×32 fixed | Workgroup 32 | AVX2 register unit |
| RMSNorm Location | Built-in NPU IP (pass mean_sq) | CPU (Separate call) | CPU (Separate call) |
| GeLU Location | Built-in NPU IP (1-Cycle) | CPU (`CPU_CORE.gelu`) | C++ SIMD inline |
| Current Usage | ❌ (FPGA target, unconnected) | ✅ Default mode | ✅ CPU mode |

---

---

## 2. `gemma3N_E4B_architecture.md`

### Overview

A **reference design document recording the entire Forward Pass** of the Gemma 3N E4B model. It includes the operating principles of each step, the rationale for CPU/IGPU distribution, and core formulas based on the example input "Hello". It serves as the **Single Source of Truth** that acts as the standard for implementation decisions in the codebase.

> **Warning**: Some code snippets in the document reflect an **older version's structure** and may differ in detail from the current implementation in `main.py`. The differences are summarized in the "Differences between the document and the current code" section below.

---

### Document Structure

| Phase | Step | Hardware | Core Content |
| ----------- | ------------------------------- | -------- | --------------------------------------- |
| **Phase 1** | 1. Tokenization + Load Weights | CPU | Text → Integer ID |
| | 2. Embedding + AltUp Init | CPU/IGPU | ID → [4, 2048] 4-stream |
| **Phase 2** | 3. AltUp Router (Predict) | IGPU | Generates `xs_pred` based on Tanh |
| | 4. Pre-Attn RMSNorm + Q,K,V | IGPU | `inputs_normalized` → Q,K,V |
| | 5. QK-Norm + RoPE | CPU | Head-wise normalization + position encoding |
| | 6. KV Cache Routing + GQA | CPU | Cache reuse for layers 20~34, unscaled |
| | 7. W_o Proj + LAuReL + 1st Residual | IGPU | `1/√2` scaled sum |
| | 8. FFN Sparsity (Layers 0~9) | IGPU/CPU | Activates only the top 5% of neurons |
| | 9. 2nd Residual Connection | IGPU | `outputs += attn_output` |
| | 10. Inject PLE (xs[1~3]) | CPU | Injects layer position info only to shadow streams |
| **Phase 3** | 11. Final Norm + LM Head | IGPU | 4-stream → vocab logit |
| | 12. Softmax + Sampling | CPU | Repetition penalty + Top-p |

---

### Core Design Principles Summary

#### 1. AltUp 4-Stream Structure

```
xs[0]  = Main Stream  ← The only input for Attention/FFN operations. Never modified directly
xs[1]  = Shadow Stream 1  ┐
xs[2]  = Shadow Stream 2  ├─ Created by altup_projs, target for PLE injection
xs[3]  = Shadow Stream 3  ┘

Layer Start: xs → xs_pred (AltUp Predict, 4×4 coefficient matrix)
Layer End:   xs_pred + innovation × corr_coefs → xs_new (AltUp Correct)
```

The core of AltUp: **Computation relies entirely on `xs[0]`**, **Information is accumulated across all 4 streams**.

---

#### 2. KV Cache Routing Rules

```
Layers 0~19:  Writes K, V to its own slot + Looks up its own cache
Layers 20~34: No cache writing
              ├── i % 5 == 4 (Global Layers: 24,29,34) → Reuses K_cache[19]
              └── The rest    (Local Layers)              → Reuses K_cache[18]
```

**Rationale**: The deeper the layer, the more the Attention pattern solidifies. Since layers 18 (Local) and 19 (Global) possess the best-learned patterns, they are reused in layers after 20. This completely eliminates the cost of storing the KV cache for 15 layers (20~34).

---

#### 3. Unscaled GQA (Unscaled Attention)

Standard Attention:
$$ \text{Attn} = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad d_k=256 $$

Gemma 3N E4B:
$$ \text{Attn} = \text{Softmax}(QK^T)V \quad \leftarrow \text{No division by } \sqrt{d_k} $$

Passes the Raw Score directly to the Softmax without scaling. There is no `/ math.sqrt(256)` operation inside `cpu_gqa()`.

---

#### 4. Extreme FFN Sparsity (Layers 0~9)

$$ \text{cutoff} = \mu(\text{gate}) + 1.6449 \cdot \sigma(\text{gate}) $$
$$ \text{sparse\_gate} = \max(\text{gate} - \text{cutoff},\ 0) $$

A normal distribution of z=1.6449 corresponds to the top 5% cutoff. Since 95% of the neurons become exactly 0, it creates an opportunity for sparse operations during the W_down matrix multiplication.

```
Layers 0~9:   sparse gate (95% zero) → W_up → W_down  (Drastically reduced computation volume)
Layers 10~34: dense gate (Fused GeLU)  → W_up → W_down  (Prioritizes speed)
```

---

#### 5. LAuReL (Layer-wise Augmented Residual Learning)

$$ \text{laurel\_out} = x_n + \text{RMSNorm}(\text{right}(\text{left}(x_n))) $$
$$ \text{attn\_final} = (\text{attn\_output} + \text{laurel\_out}) \times \frac{1}{\sqrt{2}} $$

Executed in **parallel** with the W_o projection. It bolsters expressive power by adding a short bypass path that passes through two low-rank (64-dimensional) linear layers. The `1/√2` scaling maintains variance after summing the two paths.

---

#### 6. PLE (Per-Layer Embedding) Injection

```
Calculate PLE (Before entering the layer loop, once per token):
  x0 → W_ple_proj → reshape[35, 256] → RMSNorm(row-wise) + norm_ple   = x_proj_normed
  x0 → W_ple[token_id] → reshape[35, 256] × √256                  = y
  pli_all = (x_proj_normed + y) × (1/√2)   shape: [35, 256]

Inject PLE at Layer i:
  pli = pli_all[i]                          ← Position vector for the i-th layer
  gate_ple = GELU(activated @ W_ple_gate[i]) × pli
  mapped   = RMSNorm(gate_ple @ W_ple_proj[i], ple_post_ln[i])
  xs_new[1:] += mapped   ← xs[0] is untouched
```

It accumulates layer number information exclusively in the shadow streams without polluting the main computation path.

---

### Differences between the Document and the Current Code (`main.py`)

| Item | Document (`architecture.md`) | Current Code (`main.py`) |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| KV Cache Data Structure | `K_cache = [[] for _ in range(35)]` (List) | `np.zeros([35, 2048, 512], float16)` (Pre-allocated array) |
| W_embed Format | `CPU_CORE.embedding(token_id, W_embed)` Single argument | `CPU_CORE.embedding(token_id, W_embed[0], W_embed[1])` (Tuple unpacking) |
| Residual after W_o | `attn_output += xs[0]` | `attn_output += x` (Uses `xs_pred[0].copy()`) |
| AltUp Correct | `xs_pred.copy()` then for loop | `xs_pred + corr_coefs[:, np.newaxis] * innovation` (Vectorized) |
| Ping-Pong Optimization | Not mentioned | `hw_prefetch`/`hw_compute_pingpong` in order Q→K→V→O→Gate→Up→Down |
| GeLU Location (Layers 0~9) | Applied separately after specifying `use_gelu=False` | Identical (`use_gelu=(i >= 10)`) |

---

### RoPE Layer Classification

| Layer Index | Condition | theta | Type |
| ------------------------------------------- | ------------ | --------- | ---------- |
| 4, 9, 14, 19, 24, 29, 34 | `i % 5 == 4` | 1,000,000 | **Global** |
| 0~3, 5~8, 10~13, 15~18, 20~23, 25~28, 30~33 | The rest | 10,000 | **Local** |

Global layers are responsible for capturing long-range context, while Local layers capture short-range patterns.

---

### Overall Forward Pass Data Flow Summary

```
token_id (int)
    │
    ▼ embedding() × √2048
x0 [2048]
    │
    ├──→ xs[0] = x0
    ├──→ xs[1..3] = x0 @ altup_projs[0..2]
    │
    ├──→ W_ple_proj → pli_all [35, 256]
    │
    ▼  ×35 Layers
┌──────────────────────────────────┐
│  AltUp Predict  → xs_pred        │
│  RMSNorm(xs[0]) → inputs_norm    │
│                                  │
│  Q,K,V = inputs_norm @ W_q,k,v   │
│  QK-Norm → RoPE                  │
│  KV Cache Routing                  │
│  GQA (Unscaled) → attn_raw       │
│                                  │
│  W_o + LAuReL + Residual 1           │
│                                  │
│  RMSNorm → W_gate (sparse/dense) │
│         → W_up                   │
│  hidden = gate × up              │
│  W_down + Residual 2                 │
│                                  │
│  AltUp Correct → xs_new          │
│  Inject PLE → xs_new[1:] += mapped │
└──────────────────────────────────┘
    │
    ▼
xs [4, 2048]
    │
    ▼ decode_logits()
    │  4-stream magnitude normalization + average
    │  Final RMSNorm + W_lm_head
    │
logits [262400]
    │
    ▼ _sample()
    │  Softcap(30) → Rep Penalty → Softmax → Top-p
    │
next_token (int)
```

# Code Documentation (8/8) — Chat Template & Memory Usage Reference

> **Target Files**: `chat_template.jinja` · `optim_tensor_size.md`
> **Role**: Definition of conversation prompt format (`chat_template`) + Reference table for overall weight memory usage (`optim_tensor_size`)

---

## 1. `chat_template.jinja`

### Overview

A Jinja2 template used by HuggingFace's `AutoTokenizer` when **serializing a list of multi-turn conversation messages into a single tokenizable string**. It defines the official conversation format of Gemma 3N and is referenced automatically when calling `tokenizer.apply_chat_template()`.

The `main.py` of the current project does not use this template directly and manually constructs a format string:

```python
# main.py — Simplified manual formatting
prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
```

This template serves as a standard reference for more complex scenarios, such as system prompts, multimedia content (images/audio), and strict role alternation validation.

---

### Gemma 3N Conversation Format Structure

The final string format generated by the template:

```
<bos>
<start_of_turn>user
{system_content}         ← Inserted before the first user turn if there is a system prompt

{user_message}<end_of_turn>
<start_of_turn>model
{assistant_message}<end_of_turn>
<start_of_turn>user
{user_message_2}<end_of_turn>
<start_of_turn>model
                         ← Truncated here when add_generation_prompt=True (prompts model generation)
```

**Special Tokens**:
| Token | Role |
| -------------------- | ---------------------------------------------- |
| `<bos>` | Beginning of Sequence — Inserted once at the very beginning of the sequence |
| `<start_of_turn>` | Utterance start marker |
| `<end_of_turn>` | Utterance end marker + `\n` |
| `<audio_soft_token>` | Audio input position placeholder |
| `<image_soft_token>` | Image input position placeholder |

---

### Step-by-Step Analysis of Template Logic

#### Step 1: Insert BOS Token

```jinja
{{ bos_token }}
```

Outputs the `<bos>` token once at the front of the sequence.

---

#### Step 2: System Prompt Processing

```jinja
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
```

If the first item in the message list is `role: system`:
- Save the system content to the `first_user_prefix` variable (adds `\n\n` at the end)
- Loop through the remainder of the messages, **excluding** the system message from `loop_messages`

If there is no system prompt, set `first_user_prefix = ""`.

**Content Type Branching**:
- `string` → Used directly
- `iterable` (multimodal list) → Uses the `.text` field of the first element

---

#### Step 3: Role Alternation Validation

```jinja
{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
    {{ raise_exception("Conversation roles must alternate user/assistant/...") }}
{%- endif -%}
```

The even positions of `loop.index0` (0-based index) must be `user`, and the odd positions must be `assistant`. If it doesn't match, it immediately raises an exception.

**Validation Logic Interpretation**:
```
index0=0 (even) → role must be 'user' → (True) != (True) → False → Pass
index0=1 (odd) → if role == 'user' then (True) != (False) → True → Exception!
```

---

#### Step 4: Role Name Normalization

```jinja
{%- if (message['role'] == 'assistant') -%}
    {%- set role = "model" -%}
{%- else -%}
    {%- set role = message['role'] -%}
{%- endif -%}
```

Converts the HuggingFace standard role name `"assistant"` into Gemma 3N's internal role name **`"model"`**. This is to match the `<start_of_turn>model\n` format.

---

#### Step 5: Utterance Rendering

```jinja
{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
```

- Outputs `<start_of_turn>{role}\n`
- Inserts the system prompt content (`first_user_prefix`) only into the first message (`loop.first`)

**Content Rendering Branching**:

```jinja
{%- if message['content'] is string -%}
    {{ message['content'] | trim }}          ← Standard text

{%- elif message['content'] is iterable -%}  ← Multimodal list
    {%- for item in message['content'] -%}
        {%- if item['type'] == 'audio' -%}   {{ '<audio_soft_token>' }}
        {%- elif item['type'] == 'image' -%} {{ '<image_soft_token>' }}
        {%- elif item['type'] == 'text'  -%} {{ item['text'] | trim }}
        {%- endif -%}
    {%- endfor -%}

{%- else -%}
    {{ raise_exception("Invalid content type") }}
{%- endif -%}
```

Removes leading/trailing whitespaces and line breaks using the `| trim` filter.

---

#### Step 6: Insert Generation Prompt

```jinja
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}
```

When called with `add_generation_prompt=True`, appends `<start_of_turn>model\n` at the end to prompt the model to continue generating.

---

### Usage Example

**Input Messages**:
```python
messages = [
    {"role": "system",    "content": "You are a helpful AI."},
    {"role": "user",      "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I help you?"},
    {"role": "user",      "content": "How is the weather today?"},
]
```

**Output String** (`add_generation_prompt=True`):
```
<bos><start_of_turn>user
You are a helpful AI.

Hello!<end_of_turn>
<start_of_turn>model
Hello! How can I help you?<end_of_turn>
<start_of_turn>user
How is the weather today?<end_of_turn>
<start_of_turn>model
```

---

### Comparison with `main.py` Simplified Format

| Item | `chat_template.jinja` | `main.py` Manual Format |
| --------------- | ----------------------------------------- | -------------------------- |
| BOS Token | Inserted automatically | Not inserted (Handled by tokenizer) |
| System Prompt | Supported | Unsupported |
| Multimedia | Supports `<audio/image_soft_token>` | Unsupported |
| Role Alternation Validation | Strict (Raises exception) | None |
| Multi-turn History | Supported | Unimplemented (Single turn only) |
| Usage Method | `tokenizer.apply_chat_template(messages)` | Directly constructs string with f-string |

---

---

## 2. `optim_tensor_size.md`

### Overview

A **reference measurement table for overall weight memory usage** generated by the `debug()` function in `Optim_tensor_load.py`. It records the actual RAM occupancy size per weight after INT4 quantization + mmap loading. It acts as a baseline for memory optimization efforts.

**Time of Generation**: Measured in mmap loaded state after `quantize.py` + `Optim_tensor_load.py` have finished executing.

---

### Memory Summary by Weight Category

#### Attention Weights (Per Layer × 35)

| Key | Matrix Format (packed) | INT4 Actual Dimension | 35 Layers Total |
| --------- | ------------------ | -------------- | ------------- |
| `W_q` | 2048 × 1024 uint8 | 2048 × 2048 | **70.3 MB** |
| `W_k` | 512 × 1024 uint8 | 512 × 2048 | 17.6 MB |
| `W_v` | 512 × 1024 uint8 | 512 × 2048 | 17.6 MB |
| `W_o` | 2048 × 1024 uint8 | 2048 × 2048 | **70.3 MB** |
| `gamma_q` | 256 float32 | — | 0.04 MB |
| `gamma_k` | 256 float32 | — | 0.04 MB |

> The reason W_q and W_o are 4 times larger than K and V: Number of Q heads (8) vs Number of KV heads (2), GQA structure.

---

#### FFN Weights (Per Layer × 35) — **Largest Share of Total Memory**

| Key | Matrix Format (packed) | INT4 Actual Dimension | 35 Layers Total |
| -------- | ------------------ | -------------- | ------------- |
| `W_gate` | 16384 × 1024 uint8 | 16384 × 2048 | **562.2 MB** |
| `W_up` | 16384 × 1024 uint8 | 16384 × 2048 | **562.2 MB** |
| `W_down` | 2048 × 8192 uint8 | 2048 × 16384 | **560.3 MB** |

Total of 3 FFN matrices ≈ **1.685 GB** — The largest portion in the entire model.

---

#### Normalization Weights (Per Layer × 35)

| Key | Format | 35 Layers Total |
| -------------- | -------------- | ------------- |
| `input_ln` | [2048] float32 | 0.28 MB |
| `post_attn_ln` | [2048] float32 | 0.28 MB |
| `pre_ffn_ln` | [2048] float32 | 0.28 MB |
| `post_ffn_ln` | [2048] float32 | 0.28 MB |
| `laurel_norm` | [2048] float32 | 0.28 MB |
| `ple_post_ln` | [2048] float32 | 0.28 MB |

Total of all Norm weights ≈ **1.7 MB** — Negligible level.

---

#### LAuReL Weights (Per Layer × 35)

| Key | Matrix Format (packed) | INT4 Actual Dimension | 35 Layers Total |
| -------------- | ------------------ | -------------- | ------------- |
| `laurel_left` | 64 × 1024 uint8 | 64 × 2048 | 2.2 MB |
| `laurel_right` | 2048 × 32 uint8 | 2048 × 64 | 2.5 MB |

Low-rank structure (64-dimensional bottleneck): 2048 → 64 → 2048. Extremely lightweight at a total of 4.7 MB.

---

#### AltUp Weights (Per Layer × 35)

| Key | Format | 35 Layers Total |
| -------------- | ----------------- | ------------- |
| `altup_rn` | [2048] float32 | 0.28 MB |
| `altup_router` | [2048, 4] float32 | 2.19 MB |
| `altup_pred` | [16, 4] float32 | 0.01 MB |
| `altup_corr` | [4, 4] float32 | 0.007 MB |
| `altup_scale` | [2048] float32 | 0.28 MB |

Total for all of AltUp ≈ **2.8 MB** — Very lightweight despite the 4-stream structure.

---

#### PLE Weights (Per Layer × 35)

| Key | Format | 35 Layers Total | Notes |
| ------------- | --------------------------------- | ------------- | -------------- |
| `ple_gate` | 256 × 1024 uint8 (INT4: 256×2048) | 8.8 MB | INT4 |
| `ple_proj` | **512 × 2048 float32** | **70 MB** | Kept as float32 |
| `ple_post_ln` | [2048] float32 | 0.28 MB | — |

> **`ple_proj` is currently kept as float32** — The only large weight in the entire model that did not undergo INT4 conversion. This is an unimplemented optimization point that could be reduced to ~17 MB upon INT4 quantization.

---

#### Global Weights (Independent of Layers)

| Key | Format | Size | Notes |
| --------------- | --------------------------------------- | -------------- | ------------ |
| `W_embed` | 262400 × 1024 uint8 (INT4: 262400×2048) | **257.3 MB** | mmap, Tied |
| `W_lm_head` | (Same object as W_embed) | 0 MB additional | Tied Weights |
| `W_ple` | 262144 × 4480 uint8 (INT4: 262144×8960) | **1,121.0 MB** | mmap |
| `W_ple_proj` | 8960 × 1024 uint8 (INT4: 8960×2048) | 8.8 MB | INT4 |
| `norm_ple` | [256] float32 | 0.001 MB | — |
| `altup_projs` | List[3] × [2048, 2048] float32 | 96.0 MB | Kept as float32 |
| `altup_unprojs` | List[3] × [2048, 2048] float32 | 96.0 MB | Kept as float32 |
| `W_final_norm` | [2048] float32 | 0.008 MB | — |

> **`W_ple` (1,121 MB)** is the largest among the global weights. Thanks to mmap, actual memory is consumed only up to the number of rows accessed.

---

### Overall Sum and Distribution by Category

| Category | Sum (MB) | Ratio |
| -------------------------------------- | ------------- | ------- |
| FFN (W_gate + W_up + W_down) × 35 | ~1,685 | **51%** |
| W_ple (Global, mmap) | ~1,121 | **34%** |
| W_embed / W_lm_head (Tied, mmap) | ~257 | 8% |
| altup_projs + altup_unprojs | ~192 | 6% |
| ple_proj (float32 unoptimized) | ~140 | 4% |
| Attention (W_q + W_k + W_v + W_o) × 35 | ~176 | 5% |
| The rest (Norm, AltUp, LAuReL etc.) | ~20 | 1% |
| **Total (Logical)** | **~3,591 MB** | 109% |

> `W_embed` and `W_lm_head` are the same object as Tied Weights → Occupies only 257 MB in actual RAM.
> `W_ple` and `W_embed` are mmap → Actual memory may be much less depending on access patterns.

---

### Identifying Optimization Points

Unimplemented optimization items deducible from this table:

| Item | Current | After Optimization | Savings |
| --------------------------------- | ------ | --------- | ----------- |
| `ple_proj` float32 → INT4 | 140 MB | ~17 MB | **-123 MB** |
| `altup_projs` float32 → float16 | 96 MB | 48 MB | -48 MB |
| `altup_unprojs` float32 → float16 | 96 MB | 48 MB | -48 MB |

---

### File Generation Method

```python
# Directly calls the debug() function at the top of Optim_tensor_load.py

# if __name__ == "__main__":
#     debug()   ← Execute after uncommenting

# Output: Printed to stdout in Markdown table format
# Save as optim_tensor_size.md after capturing
python Optim_tensor_load.py > optim_tensor_size.md
```

---

## Summary of Complete Documentation (1/8 ~ 8/8)

| Part | Files | Core Content |
| ---- | ---------------------------------------------- | ---------------------------------------------- |
| 1/8 | `vulkan_core.cpp` + `my_accelerator.cpp` | Vulkan ping-pong buffer structure, 6 C++ SIMD kernels |
| 2/8 | `gemv_int4_vector4.comp` + `gemv_int4.comp` | GLSL uvec4 optimization, INT4 unpacking logic |
| 3/8 | `CPU_CORE.py` + `CPU_MATRIX_CORE.py` | Attention/RoPE/Embedding CPU calculation, output buffer pool |
| 4/8 | `IGPU_CORE.py` + `main.py` | Vulkan Python binding, 35-layer full pipeline |
| 5/8 | `safeTensor.py` + `Optim_tensor_load.py` | mmap strategy, safetensors → npy conversion |
| 6/8 | `quantize.py` + `Memory_Manager.py` | INT4 symmetric quantization, KV cache pre-allocation |
| 7/8 | `NPU_CORE.py` + `gemma3N_E4B_architecture.md` | FPGA Systolic Array, overall architecture design |
| 8/8 | `chat_template.jinja` + `optim_tensor_size.md` | Conversation serialization format, baseline memory usage table |
