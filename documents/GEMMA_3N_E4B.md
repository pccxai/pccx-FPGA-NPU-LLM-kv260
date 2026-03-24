# 코드 문서화 (1/8) — C++ 가속 레이어

> **대상 파일**: `vulkan_core.cpp` · `my_accelerator.cpp`
> **역할**: iGPU(Vulkan) 및 CPU(AVX2/OpenMP) 기반 저수준 행렬 연산 커널

---

## 1. `vulkan_core.cpp`

### 개요

Vulkan Compute API를 사용하여 iGPU에서 INT4 양자화 가중치에 대한 GEMV(General Matrix-Vector Multiply) 연산을 수행하는 C++ 공유 라이브러리입니다. Python(`IGPU_CORE.py`)에서 `ctypes`를 통해 호출됩니다.

**핵심 설계 패턴: 핑퐁(Ping-Pong) 버퍼링**
CPU에서 다음 레이어 가중치를 비동기로 VRAM에 올리는 동안, GPU는 이전 버퍼로 연산을 진행합니다. 레이어 간 메모리 전송 대기 시간을 연산 시간과 겹쳐 숨깁니다.

```
[레이어 i]   GPU 연산 (버퍼 A) │ CPU 비동기 전송 (W[i+1] → 버퍼 B)
[레이어 i+1] GPU 연산 (버퍼 B) │ CPU 비동기 전송 (W[i+2] → 버퍼 A)
```

---

### 전역 상태 (Global State)

| 변수 | 타입 | 설명 |
|---|---|---|
| `instance` | `VkInstance` | Vulkan 인스턴스 핸들 |
| `physicalDevice` | `VkPhysicalDevice` | 물리 GPU 디바이스 핸들 |
| `device` | `VkDevice` | 논리 디바이스 핸들 |
| `computeQueue` | `VkQueue` | 컴퓨트 전용 큐 |
| `computePipeline` | `VkPipeline` | 컴파일된 컴퓨트 파이프라인 |
| `g_matBuf[2]` | `VkBuffer[2]` | 핑퐁 가중치 버퍼 (각 300MB) |
| `g_xBuf` | `VkBuffer` | 입력 벡터 버퍼 (MAX_K × 4 bytes) |
| `g_scaleBuf` | `VkBuffer` | INT4 dequant 스케일 버퍼 |
| `g_outBuf` | `VkBuffer` | 출력 버퍼 (MAX_M × 4 bytes) |
| `g_descriptorSet[2]` | `VkDescriptorSet[2]` | 각 핑퐁 버퍼에 대한 디스크립터 세트 |
| `weight_loader` | `std::future<void>` | 비동기 가중치 로더의 future 핸들 |

**상수**

```cpp
#define MAX_M 262144   // 최대 출력 차원 (LM Head vocab 크기 고려)
#define MAX_K 16384    // 최대 입력 차원
```

g_scaleBuf 할당: 262144 × 4 = 1,048,576 bytes
LM Head 필요량: 262400 × 4 = 1,049,600 bytes
                              ─────────────────
                              1,024 bytes (256 floats) 부족
                              
---

### 함수 레퍼런스

#### `init_vulkan_engine()`

```c
extern "C" void init_vulkan_engine()
```

**목적**: 프로그램 시작 시 단 한 번 호출되어 Vulkan 전체 파이프라인을 초기화합니다.

**초기화 순서**:
1. `VkInstance` 생성 (API 버전 1.2)
2. 첫 번째 물리 디바이스(`devices[0]`) 선택
3. 컴퓨트 큐(queue family 0) 생성
4. 디스크립터 셋 레이아웃 생성 (binding 0~3: x, mat, scale, out)
5. Push Constants 레이아웃 정의 (`PushConstants` 구조체)
6. SPIR-V 셰이더(`gemv_int4_vector4.spv`) 로드 및 컴퓨트 파이프라인 생성
7. 버퍼 할당 (핑퐁 가중치 × 2, x, scale, out)
8. 디스크립터 풀 및 디스크립터 세트 × 2 생성 및 바인딩
9. 커맨드 풀 생성

**버퍼 메모리 레이아웃**:
```
g_matBuf[0] (300MB) ─── 핑 버퍼: 현재 연산 중인 가중치
g_matBuf[1] (300MB) ─── 퐁 버퍼: 다음 레이어 가중치 프리패치
g_xBuf      (64KB)  ─── 입력 벡터 x (float32)
g_scaleBuf  (1MB)   ─── dequant 스케일 (float32)
g_outBuf    (1MB)   ─── 결과 출력 (float32)
```

모든 버퍼는 `HOST_VISIBLE | HOST_COHERENT` 플래그로 CPU-GPU 제로카피 공유 메모리에 할당됩니다 (APU unified memory 환경 최적화).

---

#### `prefetch_weight_async()`

```c
extern "C" void prefetch_weight_async(
    const uint8_t* mat_p,  // 소스: CPU RAM의 INT4 패킹 가중치 포인터
    int M_out,             // 출력 행 수
    int K_in,              // 입력 차원 (unpacked 기준)
    int buf_idx            // 대상 버퍼 인덱스 (0 또는 1)
)
```

**목적**: `std::async`를 이용해 가중치 데이터를 지정된 핑퐁 버퍼에 **백그라운드 스레드**로 비동기 복사합니다.

복사 크기: `M_out × (K_in / 2)` bytes (INT4 packed 기준)

>  `run_vulkan_gemv_pingpong()` 호출 전 `weight_loader.wait()`가 자동으로 동기화를 보장합니다.

---

#### `run_vulkan_gemv_pingpong()`

```c
extern "C" void run_vulkan_gemv_pingpong(
    const float* x,        // 입력 벡터 (float32, K_in 크기)
    const float* scale,    // dequant 스케일 (float32, M_out 크기)
    float* out,            // 출력 벡터 (float32, M_out 크기)
    int M_out,
    int K_in,
    int buf_idx            // 사용할 핑퐁 버퍼 인덱스
)
```

**목적**: 지정된 핑퐁 버퍼(`buf_idx`)의 가중치를 사용하여 GPU GEMV를 실행합니다.

**실행 흐름**:
1. `weight_loader.wait()` — 비동기 프리패치 완료 대기
2. `x`, `scale` → GPU 공유 버퍼에 `memcpy`
3. 커맨드 버퍼 기록: 파이프라인 바인딩 → 디스크립터 바인딩 → Push Constants → `Dispatch`
4. `vkQueueSubmit` + `vkQueueWaitIdle` — 동기 실행
5. `g_outBuf` → `out` 배열로 `memcpy`

**디스패치 크기**: `ceil(M_out / 32)` 워크그룹 (셰이더 로컬 크기 32 기준)

**Push Constants 구조체**:
```cpp
struct PushConstants {
    uint32_t M_out;          // 출력 행 수
    uint32_t K_in_vector4s;     // K_in / 32 (uvec4 단위)
};
```

---

#### `run_vulkan_gemv()` *(레거시)*

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

핑퐁 없이 버퍼[0]에 직접 가중치를 복사하고 동기 실행하는 레거시 인터페이스. `IGPU_CORE.py`의 `igpu_matmul()`에서 호출됩니다. 새 코드에서는 `run_vulkan_gemv_pingpong()`을 사용하는 것을 권장합니다.

---

#### `createBuffer()` *(내부 유틸리티)*

```cpp
void createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory,
    void** mappedData      // 출력: CPU에서 접근 가능한 매핑된 포인터
)
```

버퍼 생성 → 메모리 요구사항 조회 → 메모리 할당 → 바인딩 → `vkMapMemory` 매핑을 원스텝으로 처리합니다.

---

### 의존성

| 항목 | 내용 |
|---|---|
| 런타임 의존 | Vulkan SDK, SPIR-V 셰이더 (`C_DLL/gemv_int4_vector4.spv`) |
| 빌드 플래그 | `-lvulkan` |
| Python 인터페이스 | `IGPU_CORE.py` (`ctypes.CDLL`) |

---

---

## 2. `my_accelerator.cpp`

### 개요

CPU에서 실행되는 고성능 수치 연산 커널 모음입니다. OpenMP SIMD 지시어(`#pragma omp simd`)와 GCC 자동 벡터라이저를 활용하여 AVX2 수준의 병렬 연산을 수행합니다. Python에서 `ctypes`를 통해 `CPU_CORE.py` 및 `main.py`에서 호출됩니다.

**공통 규칙**:
- 모든 함수는 `extern "C"` 블록에 선언되어 Python `ctypes`에서 심볼을 찾을 수 있습니다.
- `float* __restrict__` 키워드: "이 포인터 메모리는 다른 포인터와 겹치지 않는다"고 컴파일러에게 보장, SIMD 최적화 활성화.
- 모든 연산은 **In-place** (입력 배열을 결과로 덮어씀).

---

### 함수 레퍼런스

#### `run_gelu_inplace()`

```c
void run_gelu_inplace(float* x, int length)
```

**수식**:
$$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(0.7978846 \cdot (x + 0.044715 \cdot x^3)\right)\right)$$

**구현 특징**:
- `#pragma omp simd`로 루프 전체를 SIMD 병렬화
- `GELU_CONST = 0.7978845608028654f` 상수를 컴파일 타임에 정의
- 중간 값 `cube = x³` 을 별도 변수로 캐싱하여 재계산 방지

**호출처**: `CPU_CORE.gelu()`, `my_accelerator.cpp` 내부 `run_gemv_int4_gelu()`

---

#### `run_RMSNorm_inplace()`

```c
void run_RMSNorm_inplace(float* x, const float* gamma, int length)
```

**수식**:
$$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{n}\sum x_i^2 + \varepsilon}} \cdot \gamma_i \quad (\varepsilon = 10^{-6})$$

**구현 특징**:
- 합산 루프: `#pragma omp simd reduction(+:sum)` — 병렬 덧셈 후 최종 집계
- `sum` 변수를 `double`로 선언하여 2048차원 누적 시 float32 오버플로 방지
- `inv_rms` 단일 역수 계산 후 곱셈 적용 (나눗셈 대신)

**호출처**: `main.py`의 `rms_norm()` 래퍼 함수

---

#### `run_unpack_int4_inplace()`

```c
void run_unpack_int4_inplace(
    const uint8_t* packed,   // 입력: INT4 × 2 패킹된 uint8 배열
    float scale,             // 행 단위 dequant 스케일
    float* out,              // 출력: float32 배열 (크기 = packed_length × 2)
    int packed_length
)
```

**패킹 형식**:
```
packed[i] = (high_nibble << 4) | low_nibble
out[2*i]   = low_nibble  (signed -8~7) × scale
out[2*i+1] = high_nibble (signed -8~7) × scale
```

부호 복원: `if (val > 7) val -= 16` (2의 보수 4비트 → int8 변환)

**호출처**: `CPU_CORE.embedding()` — 임베딩 행 1개를 토큰 ID로 조회할 때

---

#### `run_rope_inplace()`

```c
void run_rope_inplace(
    float* x,           // [num_heads × dim] 연속 float32 배열 (in-place)
    int pos,            // 현재 시퀀스 위치
    float theta_base,   // RoPE 기저 주파수 (Local: 10000, Global: 1000000)
    int num_heads,      // 어텐션 헤드 수
    int dim             // 헤드당 차원 (256 고정)
)
```

**수식**:
$$\text{cos\_vals}[i] = \cos\!\left(\text{pos} \cdot \theta_{\text{base}}^{-2i/d}\right), \quad
x'[i] = x[i]\cos - x[i+d/2]\sin, \quad x'[i+d/2] = x[i+d/2]\cos + x[i]\sin$$

**구현 최적화**:
- cos/sin 값은 헤드 수와 무관하게 **단 1회만** 계산 (`cos_vals[128]`, `sin_vals[128]` 스택 캐시)
- 외부 루프(헤드) + 내부 SIMD 루프(차원) 구조

**호출처**: `CPU_CORE.cpu_rope()`

---

#### `run_softmax_inplace()`

```c
void run_softmax_inplace(float* logits, int length, float temperature)
```

**수식**: Temperature scaling → Max 빼기 (안전한 exp) → 합산 → 정규화

**구현 특징**:
- 온도 나눗셈과 최댓값 탐색을 **단일 루프에 퓨전** (`reduction(max:max_val)`)
- `exp` 계산과 합산을 **단일 루프에 퓨전** (`reduction(+:sum_exp)`)
- `sum_exp`는 `double`로 누산하여 256,000개 softmax의 정밀도 보장
- `temperature < 1e-8` 가드: 0으로 나누기 방지

**호출처**: `main.py`의 `_sample()` 함수

---

#### `run_gemv_int4()`

```c
void run_gemv_int4(
    const float* vec,        // 입력 벡터 [K_in]
    const uint8_t* mat_p,    // INT4 packed 가중치 행렬 [M_out × K_in/2]
    const float* scale,      // 행 단위 dequant 스케일 [M_out]
    float* out,              // 출력 벡터 [M_out]
    int M_out,
    int K_in
)
```

**수식**: `out[i] = scale[i] × Σ( vec[k] × dequant(mat_p[i][k]) )`

**구현 특징**:
- `#pragma omp parallel for` — M_out 행을 CPU 전체 코어에 분배
- `#pragma omp simd reduction(+:acc)` — 각 행의 K 루프를 AVX2 SIMD로 처리
- 언패킹(nibble 추출 + 부호 확장)이 내부 루프에서 인라인 처리됨

**호출처**: `CPU_MATRIX_CORE.igpu_matmul()` (CPU 모드)

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

`run_gemv_int4()`와 동일하되, 출력에 GELU를 **즉시 적용(퓨전)** 합니다. FFN Gate 연산에 사용되며, 출력을 메모리에 쓰고 다시 읽는 불필요한 왕복을 제거합니다.

**호출처**: `CPU_MATRIX_CORE.igpu_matmul_gelu()` (CPU 모드, 10층 이상)

---

### 빌드 설정 권장사항

```bash
g++ -O3 -march=native -fopenmp -ffast-math \
    -shared -fPIC -o C_DLL/my_accelerator.so my_accelerator.cpp
```

| 플래그 | 이유 |
|---|---|
| `-march=native` | AVX2 자동 벡터화 활성화 |
| `-fopenmp` | `#pragma omp` 지시어 처리 |
| `-ffast-math` | `tanh`/`exp` 근사치 최적화 허용 |
| `-O3` | 최고 수준 최적화 |

---

### 함수 호출 맵

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

# 코드 문서화 (2/8) — Vulkan Compute 셰이더

> **대상 파일**: `gemv_int4_vector4.comp` · `gemv_int4.comp`
> **역할**: iGPU에서 INT4 양자화 가중치로 GEMV를 수행하는 GLSL 컴퓨트 셰이더
> **컴파일**: `glslc gemv_int4_vector4.comp -o C_DLL/gemv_int4_vector4.spv`

---

## 두 셰이더 한눈 비교

| 항목                     | `gemv_int4_vector4.comp`    | `gemv_int4.comp`         |
| ------------------------ | --------------------------- | ------------------------ |
| **상태**                 | ✅ 현재 사용 (프로덕션)      | 🗃️ 구버전 (레거시)        |
| **메모리 접근 단위**     | `uvec4` (128비트, 16바이트) | `uint` (32비트, 4바이트) |
| **1루프당 처리 INT4**    | 32개                        | 8개                      |
| **Push Constant 필드명** | `K_in_vector4s`             | `K_in_uints`             |
| **바인딩 1 타입**        | `uvec4[]`                   | `uint[]`                 |
| **캐시 효율**            | 높음 (128비트 버스트)       | 낮음 (32비트 단위)       |

---

## 1. `gemv_int4_vector4.comp` (현재 사용)

### 개요

`uvec4`(128비트 벡터 타입)를 한 번에 읽어 1루프당 INT4 32개를 처리하는 최적화된 셰이더입니다. 현대 GPU의 128비트 메모리 버스를 최대한 활용합니다.

### 바인딩 레이아웃

```glsl
layout(binding = 0) readonly buffer InputX  { float  x[];        };  // 입력 벡터
layout(binding = 1) readonly buffer MatP    { uvec4  mat_vec4[]; };  // INT4 packed 가중치 (128비트 단위)
layout(binding = 2) readonly buffer Scale   { float  scale[];    };  // 행 단위 dequant 스케일
layout(binding = 3) writeonly buffer Output { float  out_vec[];  };  // 출력 벡터
```

### Push Constants

```glsl
layout(push_constant) uniform PushConstants {
    uint M_out;           // 출력 행 수 (= 가중치 행 수)
    uint K_in_vector4s;   // K차원을 uvec4 단위로 나눈 개수 (= K_in / 32)
} params;
```

> **왜 K_in / 32인가?**  
> `uvec4` 1개 = 4바이트 × 4 = 16바이트 = 128비트.  
> INT4는 4비트이므로 1바이트에 2개, 16바이트에 32개 → 1 `uvec4`당 INT4 32개 처리.

### 실행 구조

```
워크그룹 크기: local_size_x = 32
총 디스패치: ceil(M_out / 32) 워크그룹
→ 스레드 1개 = 출력 행 1개 담당
```

### 메인 로직 상세 (`main()`)

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= params.M_out) return;     // 범위 초과 스레드 조기 종료

    float acc = 0.0;
    uint row_offset = row * params.K_in_vector4s;  // 행의 시작 uvec4 인덱스

    for (uint k = 0; k < params.K_in_vector4s; k++) {
        uvec4 packed_128 = mat_vec4[row_offset + k];  // ← 128비트 싹쓸이 로드
        uint x_idx = k * 32;

        for (int v = 0; v < 4; v++) {          // uvec4의 4개 uint32 요소 순회
            uint packed_32 = packed_128[v];    // x, y, z, w 성분
            uint x_v_idx = x_idx + (v * 8);   // uint 1개당 x 요소 8개 담당

            for (int i = 0; i < 4; i++) {      // uint32를 8비트씩 4번 분해
                uint byte_val = (packed_32 >> (i * 8)) & 0xFF;

                // 하위 4비트 → INT4 (signed)
                int low  = int(byte_val & 0x0F);
                if (low  > 7) low  -= 16;

                // 상위 4비트 → INT4 (signed)
                int high = int((byte_val >> 4) & 0x0F);
                if (high > 7) high -= 16;

                acc += x[x_v_idx + i*2    ] * float(low);
                acc += x[x_v_idx + i*2 + 1] * float(high);
            }
        }
    }

    out_vec[row] = acc * scale[row];  // dequantize: 스케일 곱셈
}
```

### 메모리 접근 패턴 시각화

```
mat_vec4[row_offset + k]  →  uvec4 (128비트)
│
├── [v=0] packed_128.x  (uint32, 4바이트)
│   ├── [i=0] byte[0]: low=INT4, high=INT4  → x[0], x[1]
│   ├── [i=1] byte[1]: low=INT4, high=INT4  → x[2], x[3]
│   ├── [i=2] byte[2]: low=INT4, high=INT4  → x[4], x[5]
│   └── [i=3] byte[3]: low=INT4, high=INT4  → x[6], x[7]
├── [v=1] packed_128.y  → x[8]  ~ x[15]
├── [v=2] packed_128.z  → x[16] ~ x[23]
└── [v=3] packed_128.w  → x[24] ~ x[31]

1 uvec4 로드 = INT4 32개 언패킹 = x 32개 원소와 내적
```

### INT4 부호 복원 원리

```
nibble 값 범위: 0x0 ~ 0xF (0 ~ 15, unsigned)
signed 해석:    0 ~ 7   → 양수 그대로
                8 ~ 15  → 8을 -8로, 9를 -7로, ... 15를 -1로 (if val > 7: val -= 16)

예시:
  nibble = 0b1010 = 10 → 10 > 7 → 10 - 16 = -6 (signed)
  nibble = 0b0011 = 3  → 3 ≤ 7  → +3 (signed)
```

### Dequantization

$$\text{out}[row] = \text{scale}[row] \times \sum_{k} \left( x[2k] \cdot w_{low}^{(k)} + x[2k+1] \cdot w_{high}^{(k)} \right)$$

스케일은 `quantize.py`에서 **행 단위**로 `max(|w|) / 7.0`으로 계산된 float32 값입니다.

---

## 2. `gemv_int4.comp` (레거시)

### 개요

`uint`(32비트) 단위로 메모리를 읽는 구버전 셰이더입니다. `gemv_int4_vector4.comp`의 전신으로, 동일한 언패킹 로직을 가지지만 메모리 로드 효율이 4배 낮습니다. 현재 `vulkan_core.cpp`에서 `gemv_int4_vector4.spv`로 교체되었으므로 실제로 실행되지 않습니다.

### 바인딩 레이아웃

```glsl
layout(binding = 0) readonly buffer InputX  { float x[];   };
layout(binding = 1) readonly buffer MatP    { uint  mat[]; };  // ← uint[] (32비트 단위, 구버전)
layout(binding = 2) readonly buffer Scale   { float scale[]; };
layout(binding = 3) writeonly buffer Output { float out_vec[]; };
```

### Push Constants

```glsl
layout(push_constant) uniform PushConstants {
    uint M_out;
    uint K_in_uints;   // K차원을 uint 단위로 나눈 개수 (= K_in / 8)
} params;
```

> **K_in / 8인 이유**: `uint` 1개 = 4바이트 = 32비트. INT4 4비트 × 8개 = 32비트.

### 메인 로직

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= params.M_out) return;

    float acc = 0.0;
    uint row_offset = row * params.K_in_uints;

    for (uint k = 0; k < params.K_in_uints; k++) {
        uint packed_32 = mat[row_offset + k];   // ← 32비트만 로드 (구버전의 한계)
        uint x_idx = k * 8;

        for (int i = 0; i < 4; i++) {           // 32비트를 8비트씩 4번 분해
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

## 두 셰이더 진화 관계

```
gemv_int4.comp (구버전)
│
│  문제: uint[] 로 32비트씩 읽음
│        → 1루프당 INT4 8개 처리
│        → 메모리 버스 활용률 25%
│
▼
gemv_int4_vector4.comp (현재)
   개선: uvec4[] 로 128비트씩 읽음
         → 1루프당 INT4 32개 처리
         → 메모리 버스 활용률 100%
         → K 루프 반복 횟수 4배 감소
```

## 빌드 및 배포

```bash
# SPIR-V 바이너리로 컴파일
glslc gemv_int4_vector4.comp -o C_DLL/gemv_int4_vector4.spv
glslc gemv_int4.comp         -o C_DLL/gemv_int4.spv          # 레거시, 미사용

# vulkan_core.cpp 에서 로드
auto shaderCode = readFile("C_DLL/gemv_int4_vector4.spv");
```

## 호출 경로 전체

```
main.py
  └── hw_compute_pingpong() / hw_matmul()
        └── IGPU_CORE.py
              ├── compute_pingpong()  → vk_lib.run_vulkan_gemv_pingpong()
              └── igpu_matmul()       → vk_lib.run_vulkan_gemv()
                    └── vulkan_core.cpp
                          └── vkCmdDispatch() → gemv_int4_vector4.spv
                                                 (GPU에서 실행)
```

## 성능 관련 주의사항

| 항목             | 값        | 비고                                            |
| ---------------- | --------- | ----------------------------------------------- |
| 워크그룹 크기    | 32        | GPU 워프/웨이브프론트 크기와 일치 권장          |
| 최대 M_out       | 262,144   | `MAX_M` 상수 (LM Head vocab 크기)               |
| 최대 K_in        | 16,384    | `MAX_K` 상수 (FFN 중간 차원)                    |
| 가중치 버퍼 한도 | 300MB × 2 | 핑퐁 각각. FFN W_gate(~562MB) 초과 시 분할 필요 |
| 스케일 정밀도    | float32   | 양자화 오차 최소화                              |

# 코드 문서화 (3/8) — CPU 연산 레이어

> **대상 파일**: `CPU_CORE.py` · `CPU_MATRIX_CORE.py`
> **역할**: 토크나이저·어텐션·RoPE 등 CPU 전담 연산(`CPU_CORE`) + CPU 모드 INT4 GEMV 인터페이스(`CPU_MATRIX_CORE`)

---

## 1. `CPU_CORE.py`

### 개요

모델 추론 파이프라인에서 **CPU가 전담하는 모든 연산**을 모아 놓은 모듈입니다.  
`my_accelerator.so` (C++ DLL)를 `ctypes`로 직접 연결해 성능이 중요한 커널(GELU, RMSNorm, RoPE, INT4 언패킹, Softmax)을 C++ SIMD로 위임하고, 파이썬 레벨에서는 배열 준비와 형상 변환만 담당합니다.

**모듈 초기화 순서** (임포트 시 자동 실행):

```
1. AutoTokenizer 로드  ← local_gemma_3n_int4/ 디렉터리
2. ctypes.CDLL("C_DLL/my_accelerator.so") 로드
3. 각 C++ 함수의 argtypes / restype 등록
```

---

### C++ DLL 바인딩

| C++ 함수                  | Python 래퍼   | 인자 타입                                           |
| ------------------------- | ------------- | --------------------------------------------------- |
| `run_gelu_inplace`        | `gelu()`      | `float32[1D]`, `c_int`                              |
| `run_unpack_int4_inplace` | `embedding()` | `uint8[1D]`, `c_float`, `float32[1D]`, `c_int`      |
| `run_rope_inplace`        | `cpu_rope()`  | `float32[1D]`, `c_int`, `c_float`, `c_int`, `c_int` |

>  `run_gelu_inplace.restype = None` 이 **두 번** 설정된 버그가 코드에 존재합니다  
> (`run_unpack_int4_inplace.restype` 미설정). 기능에는 무영향이나 명시적 수정 권장.

---

### 함수 레퍼런스

#### `tokenize(text)`

```python
def tokenize(text: str) -> np.ndarray  # shape: [T], dtype: int64
```

HuggingFace `AutoTokenizer`로 문자열을 토큰 ID 배열로 변환합니다.  
디버그용으로 토큰 ID를 출력(`print`)합니다.

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

INT4 임베딩 테이블에서 토큰 1개의 행을 꺼내 float32로 변환합니다.

**실행 흐름**:
```
1. W_packed[token_id]  → row_packed (uint8, 1D, contiguous 보장)
2. W_scale[token_id]   → row_scale  (scalar float)
3. np.empty(hidden)    → out_f32    (출력 버퍼 사전 할당)
4. c_lib.run_unpack_int4_inplace(row_packed, scale, out_f32, packed_len)
   └── C++에서 nibble 분리 + 부호 복원 + scale 곱셈 수행
5. return out_f32       # [hidden] float32
```

**메모리 최적화**: `W_packed`와 `W_scale`은 mmap으로 열린 배열이므로 실제 디스크 읽기는 해당 행(`token_id`) ~5.5 KB만 발생합니다.

---

#### `gelu(x)`

```python
def gelu(x: np.ndarray) -> np.ndarray  # 입력과 동일한 shape 반환
```

C++ `run_gelu_inplace`를 호출하는 래퍼입니다.

```python
x_flat = np.ascontiguousarray(x.flatten().astype(np.float32))
c_lib.run_gelu_inplace(x_flat, x_flat.size)
return x_flat.reshape(x.shape)
```

`flatten()` + `reshape()`로 임의 shape의 입력을 처리합니다.  
1D/2D 모두 투명하게 동작합니다.

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

Q와 K 각각에 **헤드별 RMSNorm**을 적용합니다.  
어텐션 스코어 폭발을 방지하는 QK-Norm 기법입니다 (Gemma 3N 고유 기능).

**수식**:
$$Q_{norm}[h] = \frac{Q[h]}{\sqrt{\text{mean}(Q[h]^2) + \varepsilon}} \cdot \gamma_q$$

```python
Q_reshaped = Q.reshape(-1, 256)   # [num_heads, 256]
q_rms = np.sqrt(np.mean(Q_reshaped**2, axis=1, keepdims=True) + 1e-6)
Q_norm = (Q_reshaped / q_rms) * gamma_q
return Q_norm.flatten(), K_norm.flatten()
```

> 연산은 float32로 강제 캐스팅 후 수행됩니다.

---

#### `cpu_rope(x, pos, theta_base)`

```python
def cpu_rope(
    x:          np.ndarray,  # [num_heads × 256] flat, float32
    pos:        int,          # 현재 시퀀스 위치
    theta_base: float,        # Local=10,000 / Global=1,000,000
) -> np.ndarray               # [num_heads × 256] flat, float32 (in-place 결과)
```

C++ `run_rope_inplace`를 호출하는 래퍼입니다.

```python
x_flat = np.ascontiguousarray(x.astype(np.float32).flatten())
c_lib.run_rope_inplace(x_flat, int(pos), float(theta_base), num_heads, 256)
return x_flat
```

**RoPE 주파수**: C++ 내부에서 `cos_vals/sin_vals` 캐싱 (헤드당 1회 계산).  
`theta_base`는 레이어 인덱스에 따라 `main.py`에서 결정됩니다:

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

**Grouped Query Attention** (GQA) 구현입니다.  
Gemma 3N E4B의 어텐션 헤드 구성:

| 항목        | 값                           |
| ----------- | ---------------------------- |
| Q 헤드 수   | 8 (= 2 그룹 × 4 헤드)        |
| K/V 헤드 수 | 2 (GQA: Q 4개가 KV 1개 공유) |
| 헤드 차원   | 256                          |

> **중요**: 스케일링(`/ sqrt(256)`) 없음 — Gemma 3N의 **Unscaled Attention** 설계.

**실행 흐름**:

```python
Q = Q_rope.reshape(2, 4, 256)          # [kv_heads, q_per_kv, head_dim]
K = K_cache.reshape(-1, 2, 256)        # [seq, kv_heads, head_dim]
V = V_cache.reshape(-1, 2, 256)

K_t = K.transpose(1, 2, 0)            # [kv_heads, head_dim, seq]
scores = Q @ K_t                       # [kv_heads, q_per_kv, seq]

# Stable softmax (max 빼기)
scores -= scores.max(axis=-1, keepdims=True)
probs = exp(scores) / sum(exp(scores))

V_t = V.transpose(1, 0, 2)            # [kv_heads, seq, head_dim]
out = probs @ V_t                      # [kv_heads, q_per_kv, head_dim]
return out.flatten()                   # [2048]
```

---

#### `cpu_update_kv_cache()` *(현재 비활성화)*

```python
def cpu_update_kv_cache(K_rope, V, token_cnt, layer_idx, K_cache, V_cache)
```

함수 본문이 전부 주석 처리되어 있습니다. KV 캐시 업데이트는 `main.py`의 `forward_one_token()` 안에서 직접 처리합니다:

```python
# main.py 내부 (실제 캐시 업데이트 위치)
if i < 20:
    K_cache[i, pos, :] = K   # pre-allocated [35, max_seq, 512] float16 배열에 직접 기록
    V_cache[i, pos, :] = V
```

---

#### `_get_rope_freqs(theta_base, dim=256)` *(내부 유틸)*

```python
_rope_freq_cache: dict = {}   # 모듈 레벨 캐시

def _get_rope_freqs(theta_base: float, dim: int = 256) -> np.ndarray
```

RoPE 주파수 테이블을 계산하고 모듈 레벨 딕셔너리에 캐싱합니다.  
현재는 C++ 버전의 `run_rope_inplace`가 내부적으로 같은 캐싱을 수행하므로 Python 레벨에서는 직접 호출되지 않습니다. (레거시 `cpu_rope` Python 버전의 보조 함수)

---

### 의존성 및 초기화

```
CPU_CORE.py 임포트 시:
  ├── transformers.AutoTokenizer  (HuggingFace)
  ├── ctypes.CDLL("C_DLL/my_accelerator.so")
  └── tokenizer = AutoTokenizer.from_pretrained("local_gemma_3n_int4/")
```

> `tokenizer`는 모듈 전역 변수로 유지되어 `main.py`에서 `CPU_CORE.tokenizer.decode()`로도 직접 접근합니다.

---

---

## 2. `CPU_MATRIX_CORE.py`

### 개요

`ACCEL_MODE = "CPU"` 설정 시 `IGPU_CORE.py`를 대신하는 **CPU 전용 행렬 곱셈 모듈**입니다.  
`my_accelerator.so`의 `run_gemv_int4` / `run_gemv_int4_gelu` C++ 커널을 직접 호출하여 OpenMP 멀티코어 + AVX2 SIMD로 INT4 GEMV를 수행합니다.

`IGPU_CORE.py`와 **완전히 동일한 인터페이스**(`igpu_matmul`, `igpu_matmul_gelu`, `preload_and_free`, `warmup`)를 제공하므로, `main.py`에서 `ACCEL_MODE` 변수 하나만 바꿔 CPU/GPU 전환이 가능합니다.

```python
# main.py
if ACCEL_MODE == "IGPU":
    import IGPU_CORE as FAST_MATRIX_CORE
elif ACCEL_MODE == "CPU":
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE
# 이후 코드는 FAST_MATRIX_CORE.igpu_matmul() 형태로 동일하게 호출
```

---

### C++ DLL 바인딩

```python
c_lib.run_gemv_int4.argtypes = [
    ndpointer(float32, 1D),   # vec   [K_in]
    ndpointer(uint8,   2D),   # mat_p [M_out, K_in/2]
    ndpointer(float32, 1D),   # scale [M_out]
    ndpointer(float32, 1D),   # out   [M_out]
    c_int,                    # M_out
    c_int,                    # K_in
]

c_lib.run_gemv_int4_gelu.argtypes = [...]  # 동일 시그니처
```

---

### 출력 버퍼 풀

```python
_OUTPUT_BUF_POOL: dict[int, np.ndarray] = {}

def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]
```

출력 크기(M_out)별로 `np.empty` 배열을 **한 번만 할당**하고 재사용합니다.  
매 호출마다 발생하는 메모리 할당 비용과 GC 압력을 제거합니다.

>  C++ 커널에서 덮어쓴 뒤 **반드시 `.copy()`로 반환**합니다.  
> 풀 버퍼를 직접 반환하면 다음 호출에서 값이 오염됩니다.

---

### 함수 레퍼런스

#### `igpu_matmul(x_vec, weight_data)`

```python
def igpu_matmul(
    x_vec:       np.ndarray,             # [K_in] float32
    weight_data: tuple | np.ndarray,     # INT4 튜플 또는 일반 float 행렬
) -> np.ndarray                          # [M_out] float32
```

**INT4 튜플 경로**:
```python
packed, scale = weight_data          # packed: uint8[M_out, K_in/2]
out_buf = _get_output_buf(M_out)
c_lib.run_gemv_int4(x_f32, packed, scale, out_buf, M_out, K_in)
return out_buf.copy()
```

**일반 행렬 폴백 경로**:
```python
return np.dot(x_f32, weight_data.astype(np.float32).T)
```

> `np.dot(x, W.T)` 형태임에 주의 — 가중치가 `[M_out, K_in]` 레이아웃이라고 가정합니다.

---

#### `igpu_matmul_gelu(x_vec, weight_data)`

```python
def igpu_matmul_gelu(
    x_vec:       np.ndarray,
    weight_data: tuple | np.ndarray,
) -> np.ndarray
```

**INT4 튜플 경로**: C++ `run_gemv_int4_gelu` 호출 — GEMV와 GELU를 **단일 커널에서 융합** 실행.  
**일반 행렬 폴백 경로**: `np.dot` 후 `CPU_CORE.gelu()` 별도 호출.

```
GEMV + GELU 융합의 이점:
  분리 실행: GEMV → [M_out] 메모리 write → GELU → [M_out] 메모리 read
  융합 실행: GEMV → acc → GELU (acc 단계에서 즉시 적용) → [M_out] 메모리 write 1회
  → 메모리 왕복 1회 절감 (M_out=16384일 때 약 64KB 절약)
```

---

#### `preload_and_free(W, keys)` / `_get_or_upload_weight(weight_data)`

```python
def preload_and_free(W: dict, keys: list): pass
def _get_or_upload_weight(weight_data):    pass
```

`IGPU_CORE.py`와의 인터페이스 호환성을 위한 **빈 함수(no-op)** 입니다.  
CPU 모드에서는 VRAM 업로드 개념이 없으므로 아무것도 수행하지 않습니다.

---

#### `warmup()`

```python
def warmup()
```

작은 더미 배열로 C++ 커널을 한 번 호출하여 OpenMP 스레드 풀과 AVX2 레지스터를 예열합니다.

```python
dummy_x = np.zeros(2048, dtype=np.float32)
dummy_p = np.zeros((2048, 1024), dtype=np.uint8)
dummy_s = np.zeros(2048, dtype=np.float32)
igpu_matmul(dummy_x, (dummy_p, dummy_s))
```

첫 번째 실제 추론 호출의 지연(thread spawn, cache cold start)을 방지합니다.

---

### `IGPU_CORE.py`와 인터페이스 대응표

| 함수                      | CPU_MATRIX_CORE                | IGPU_CORE                           |
| ------------------------- | ------------------------------ | ----------------------------------- |
| `igpu_matmul()`           | `run_gemv_int4` (C++/CPU)      | `run_vulkan_gemv` (Vulkan/iGPU)     |
| `igpu_matmul_gelu()`      | `run_gemv_int4_gelu` (C++/CPU) | `igpu_matmul()` + `CPU_CORE.gelu()` |
| `preload_and_free()`      | no-op                          | no-op (VRAM 최적화 레거시)          |
| `_get_or_upload_weight()` | no-op                          | 가중치 VRAM 업로드                  |
| `warmup()`                | OpenMP 예열                    | Vulkan 셰이더 로드 확인 출력        |
| `prefetch_weight()`       | 없음                           | `prefetch_weight_async` (비동기)    |
| `compute_pingpong()`      | 없음                           | `run_vulkan_gemv_pingpong` (핑퐁)   |

> CPU 모드에는 핑퐁 프리패치 기능이 없습니다. 핑퐁 최적화는 Vulkan iGPU 모드 전용입니다.

---

### 모듈 간 의존 관계

```
main.py
  │
  ├── ACCEL_MODE = "CPU"  →  CPU_MATRIX_CORE.py
  │                              └── C_DLL/my_accelerator.so
  │                                    ├── run_gemv_int4()
  │                                    └── run_gemv_int4_gelu()
  │
  └── (항상) CPU_CORE.py
              ├── C_DLL/my_accelerator.so
              │     ├── run_gelu_inplace()
              │     ├── run_RMSNorm_inplace()
              │     ├── run_rope_inplace()
              │     └── run_unpack_int4_inplace()
              └── transformers.AutoTokenizer
```

# 코드 문서화 (4/8) — iGPU 인터페이스 & 메인 파이프라인

> **대상 파일**: `IGPU_CORE.py` · `main.py`
> **역할**: Vulkan Python 바인딩(`IGPU_CORE`) + 전체 추론 루프 오케스트레이션(`main`)

---

## 1. `IGPU_CORE.py`

### 개요

`vulkan_core.so` (C++ Vulkan 엔진)를 Python `ctypes`로 감싼 **iGPU 가속 인터페이스 모듈**입니다.  
`main.py`에서 `ACCEL_MODE = "IGPU"` 설정 시 `FAST_MATRIX_CORE`라는 이름으로 임포트되며, `CPU_MATRIX_CORE.py`와 완전히 동일한 함수 시그니처를 제공합니다.

**모듈 초기화 순서** (임포트 시 자동 실행):

```
1. ctypes.CDLL("C_DLL/vulkan_core.so") 로드
2. os.chdir(base_dir)           ← .spv 셰이더 파일 경로를 상대경로로 찾기 위해 필수
3. vk_lib.init_vulkan_engine()  ← Vulkan 인스턴스/파이프라인/버퍼 전체 초기화 (1회)
4. 각 C++ 함수의 argtypes 등록
```

> `os.chdir(base_dir)` 이 없으면 `vulkan_core.cpp`의 `readFile("C_DLL/gemv_int4_vector4.spv")`가  
> 실행 위치 기준으로 경로를 찾아 파일을 못 찾습니다.

---

### C++ DLL 바인딩

| C++ 함수                   | Python 래퍼           | 비고                    |
| -------------------------- | --------------------- | ----------------------- |
| `init_vulkan_engine`       | (초기화 시 직접 호출) | 임포트 시 1회 자동 실행 |
| `run_vulkan_gemv`          | `igpu_matmul()`       | 레거시 동기 GEMV        |
| `prefetch_weight_async`    | `prefetch_weight()`   | 비동기 가중치 프리패치  |
| `run_vulkan_gemv_pingpong` | `compute_pingpong()`  | 핑퐁 버퍼 지정 GEMV     |

**`run_vulkan_gemv_pingpong` 인자 순서 주의**:
```python
# C++ 시그니처:
#   run_vulkan_gemv_pingpong(x, scale, out, M_out, K_in, buf_idx)
# ← packed(가중치)는 인자가 아님! 이미 prefetch_weight_async()로 VRAM에 올라가 있음
vk_lib.run_vulkan_gemv_pingpong.argtypes = [
    float32[1D],   # x     (입력 벡터)
    float32[1D],   # scale (dequant 스케일)
    float32[1D],   # out   (출력 벡터)
    c_int,         # M_out
    c_int,         # K_in
    c_int,         # buf_idx (0 또는 1)
]
```

---

### 출력 버퍼 풀

```python
_OUTPUT_BUF_POOL: dict[int, np.ndarray] = {}

def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]
```

출력 크기별 `np.empty` 배열을 한 번만 할당 후 재사용합니다.  
결과를 반환할 때는 반드시 `.copy()`를 호출해야 풀 버퍼가 오염되지 않습니다.

---

### 함수 레퍼런스

#### `igpu_matmul(x_vec, weight_data)`

```python
def igpu_matmul(
    x_vec:       np.ndarray,          # [K_in] any dtype → float32으로 변환
    weight_data: tuple | np.ndarray,  # INT4 튜플 또는 일반 float 행렬
) -> np.ndarray                       # [M_out] float32
```

**INT4 튜플 경로** (`weight_data = (packed, scale)`):
```
packed.shape[0]     → M_out
packed.shape[1] × 2 → K_in (uint8 1개 = INT4 2개)
vk_lib.run_vulkan_gemv(x, packed, scale, out_buf, M_out, K_in)
```

**일반 행렬 경로** (float 행렬):
```python
return np.dot(x_f32, w_f32)
# ← CPU_MATRIX_CORE의 np.dot(x, W.T)와 달리 전치 없음 주의
#    weight_data가 이미 [K_in, M_out] 레이아웃이어야 함
```

---

#### `igpu_matmul_gelu(x_vec, weight_data)`

```python
def igpu_matmul_gelu(x_vec, weight_data) -> np.ndarray
```

`igpu_matmul()` 호출 후 `CPU_CORE.gelu()`를 별도로 적용합니다.

> **CPU_MATRIX_CORE와의 차이**: CPU 모드는 `run_gemv_int4_gelu`로 GEMV+GELU를 **단일 C++ 커널에서 융합**하지만, Vulkan iGPU 모드는 GEMV(GPU) → GELU(CPU) 순서로 분리 실행합니다. iGPU 모드에서도 GELU를 셰이더에 통합하면 추가 최적화 여지가 있습니다.

---

#### `prefetch_weight(weight_data, buf_idx)`

```python
def prefetch_weight(
    weight_data: tuple,   # (packed, scale) — tuple이 아니면 no-op
    buf_idx:     int,     # 0 또는 1 (대상 핑퐁 버퍼)
)
```

C++ `prefetch_weight_async()`를 호출하여 `std::async`로 백그라운드 스레드에서 가중치를 VRAM으로 복사합니다. 호출 즉시 반환되며, 복사 완료는 다음 `compute_pingpong()` 호출 시 `weight_loader.wait()`에서 보장됩니다.

tuple이 아닌 경우(일반 float 행렬)는 VRAM 업로드 불필요이므로 아무 동작도 하지 않습니다.

---

#### `compute_pingpong(x_vec, weight_data, buf_idx)`

```python
def compute_pingpong(
    x_vec:       np.ndarray,  # [K_in] float32
    weight_data: tuple,       # (packed, scale)
    buf_idx:     int,         # 사용할 버퍼 인덱스
) -> np.ndarray               # [M_out] float32
```

`buf_idx`가 가리키는 핑퐁 버퍼의 가중치로 GPU GEMV를 실행합니다.  
이 버퍼의 가중치는 직전 `prefetch_weight(w, buf_idx)` 호출로 이미 VRAM에 올라와 있어야 합니다.

**일반 행렬 폴백**: tuple이 아닌 경우 `np.dot(x, W.T)`로 CPU 연산.

---

#### 레거시 함수들

```python
def preload_and_free(W, keys): pass   # 이전 Taichi 기반 VRAM 선업로드 레거시
def _get_or_upload_weight(w):  pass   # 동일
def warmup(): print("...")            # 셰이더 로드 완료 메시지 출력만 수행
```

이전 Taichi 버전에서 VRAM 메모리 관리를 위해 존재했던 함수들입니다. 현재 Vulkan 핑퐁 구조에서는 가중치가 호출 시점에 동적으로 전송되므로 불필요합니다. `main.py`와의 인터페이스 호환성을 위해 빈 함수로 유지됩니다.

---

---

## 2. `main.py`

### 개요

전체 Gemma 3N E4B 추론 파이프라인의 **진입점이자 오케스트레이터**입니다.  
토큰화 → 임베딩 → 35개 Transformer 레이어 반복 → 로짓 디코딩 → 샘플링의 전체 흐름을 제어합니다.

---

### 모듈 레벨 설정

```python
ACCEL_MODE = "IGPU"          # "CPU"로 변경하면 CPU_MATRIX_CORE로 전환
NUM_LAYERS = 35

_IGPU_WEIGHT_KEYS = [        # 핑퐁 프리패치 대상 키 목록 (현재 no-op 함수에 전달)
    "W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"
]
```

모듈 로드 시 `my_accelerator.so`에서 `run_RMSNorm_inplace`, `run_softmax_inplace` 두 함수의 argtypes를 추가로 등록합니다. (`CPU_CORE.py`와 동일한 DLL을 **중복 로드**하는 구조입니다.)

---

### 유틸리티 함수

#### `rms_norm(x, gamma)`

```python
def rms_norm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray
```

C++ `run_RMSNorm_inplace`를 호출하는 main.py 전용 래퍼입니다.  
`CPU_CORE`에 같은 기능이 없고, `main.py` 내부에서만 사용됩니다.

```python
x_f32   = np.ascontiguousarray(x.astype(np.float32))     # 독립 복사본 생성
gamma_c = np.ascontiguousarray(gamma.astype(np.float32))
c_lib.run_RMSNorm_inplace(x_f32, gamma_c, x_f32.size)    # in-place 덮어쓰기
return x_f32
```

> `np.ascontiguousarray`로 독립 복사본을 만들기 때문에 원본 `x`는 변경되지 않습니다.

---

#### `get_router_modalities(x, w_norm, w_router)`

```python
def get_router_modalities(x, w_norm, w_router) -> np.ndarray  # [4]
```

AltUp 라우터의 모달리티 벡터를 계산합니다.  
`xs[0]`을 정규화하고 라우터 가중치로 투영한 뒤 Tanh로 압축합니다.

```python
x_n = rms_norm(x, w_norm) / 2048.0      # 차원 스케일 보정
return np.tanh(np.dot(x_n, w_router))   # shape: [4]
```

레이어 시작(AltUp Predict)과 레이어 끝(AltUp Correct) **두 번** 호출됩니다.

---

#### `hw_matmul(x, w, use_gelu=False)` / `hw_prefetch(w, buf_idx)` / `hw_compute_pingpong(x, w, buf_idx, use_gelu=False)`

핑퐁 최적화와 모드 전환을 추상화하는 **하드웨어 어댑터 함수 3종**입니다.

| 함수                  | IGPU 모드                             | CPU 모드                       |
| --------------------- | ------------------------------------- | ------------------------------ |
| `hw_matmul`           | `FAST_MATRIX_CORE.igpu_matmul[_gelu]` | 인라인 INT4 dequant + `np.dot` |
| `hw_prefetch`         | `FAST_MATRIX_CORE.prefetch_weight`    | no-op                          |
| `hw_compute_pingpong` | `FAST_MATRIX_CORE.compute_pingpong`   | `hw_matmul` 폴백               |

`hw_matmul`의 CPU 모드 인라인 dequant:
```python
# tuple인 경우 직접 INT4 → float32 변환
low  = (packed & 0x0F).astype(np.int8); low[low > 7]   -= 16
high = (packed >> 4  ).astype(np.int8); high[high > 7] -= 16
w_real = interleave(low, high) * scale[:, np.newaxis]
out = np.dot(x, w_real.T)
```

---

### 핵심 함수: `forward_one_token()`

```python
def forward_one_token(
    token_id:     int,
    pos:          int,            # 현재 시퀀스 위치 (0-indexed)
    W:            dict,           # 35개 레이어 가중치 딕셔너리
    W_embed:      tuple,          # (packed, scale) mmap
    W_ple_packed: np.ndarray,     # [262144, 4480] uint8 mmap
    W_ple_scale:  np.ndarray,     # [262144] float32 mmap
    norm_ple:     np.ndarray,     # [256] float32
    W_ple_proj:   tuple,          # (packed, scale) INT4
    altup_projs:  list[np.ndarray],  # [3] × [2048, 2048]
    K_cache:      np.ndarray,     # [35, max_seq, 512] float16 (pre-alloc)
    V_cache:      np.ndarray,     # [35, max_seq, 512] float16 (pre-alloc)
) -> np.ndarray                   # xs: [4, 2048] float32 (4-stream 출력)
```

토큰 1개에 대해 **Embedding → PLE 계산 → 35개 레이어 반복**을 수행합니다.

#### Phase A: 임베딩 및 AltUp 초기화

```python
# 1. INT4 임베딩 조회 + Gemma 3N 스케일링
x0 = CPU_CORE.embedding(token_id, W_embed[0], W_embed[1])
x0 = x0 * sqrt(2048.0)               # Gemma 3N 고유 임베딩 스케일링

# 2. AltUp 4-Stream 초기화
xs[0] = x0                            # 메인 스트림 (절대 변형 없음)
xs[1..3] = dot(x0, altup_projs[0..2]) # 그림자 스트림 (Shadow Stream)
```

#### Phase B: PLE(Per-Layer Embedding) 계산

```python
# W_ple_proj: [2048] → [35×256] 투영 (IGPU)
x_proj = hw_matmul(x0, W_ple_proj) / sqrt(2048.0)
x_proj = x_proj.reshape(35, 256)
x_proj_normed = RMSNorm_perrow(x_proj) * norm_ple   # 행별 RMSNorm

# W_ple: [vocab, 8960] → 토큰 해당 행 조회 → [35, 256]
y = embedding(token_id, W_ple_packed, W_ple_scale).reshape(35, 256) * sqrt(256.0)

# 최종 PLE 벡터 (레이어별 위치 임베딩)
pli_all = (x_proj_normed + y) * (1/sqrt(2.0))   # shape: [35, 256]
```

#### Phase C: 35개 레이어 반복 루프

**레이어 시작: AltUp Predict**
```python
modalities = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
coef_mat   = dot(W["altup_pred"][i], modalities).reshape(4, 4)  # [4, 4]
xs_pred    = xs + dot(coef_mat, xs)   # 예측 스트림 (임시 렌즈)
x          = xs_pred[0].copy()        # Attention은 순수 xs_pred[0] 사용
```

**Attention 블록 (핑퐁 순서)**

```
prefetch(W_q[0], buf=0)   ← 루프 진입 전 선행 로드

[레이어 i 시작]
buf=0: 계산 W_q  │  비동기: W_k → buf=1
buf=1: 계산 W_k  │  비동기: W_v → buf=0
buf=0: 계산 W_v  │  비동기: W_o → buf=1
       QK-Norm, RoPE, KV Cache, GQA
buf=1: 계산 W_o  │  비동기: W_gate → buf=0
       LAuReL 계산
```

**KV 캐시 라우팅 규칙**:
```python
if i < 20:
    K_cache[i, pos, :] = K      # 레이어 0~19: 자기 슬롯에 저장
    V_cache[i, pos, :] = V
    target_k = K_cache[i, :pos+1, :]
else:
    if i % 5 == 4:              # Global 레이어 (20,25,30,34): 레이어 19 캐시 재사용
        target_k = K_cache[19, :pos+1, :]
    else:                       # Local  레이어 (21~24, 26~29, ...): 레이어 18 캐시 재사용
        target_k = K_cache[18, :pos+1, :]
```

**1차 잔차 연결 + LAuReL**:
```python
attn_output = RMSNorm(W_o_out, post_attn_ln)
attn_output += x                             # 잔차 연결
# LAuReL: inputs_normalized → left → right → norm → + inputs_normalized
laurel_out_normed = inputs_normalized + RMSNorm(right(left(inputs_normalized)))
attn_output = (attn_output + laurel_out_normed) * (1/sqrt(2.0))  # 스케일 합산
```

**FFN 블록 (핑퐁 순서 + 희소성)**:
```
buf=0: 계산 W_gate (i≥10: GELU 융합)  │  비동기: W_up → buf=1
buf=1: 계산 W_up                       │  비동기: W_down → buf=0

if i < 10:   # 0~9층 5% 희소성 수술
    cutoff = mean(gate_out) + std(gate_out) * 1.6448536   # z=1.645 → 상위 5%
    sparse_gate = max(gate_out - cutoff, 0)
    hidden = gelu(sparse_gate) * up_out
else:        # 10~34층 dense
    hidden = gate_out * up_out

buf=0: 계산 W_down  │  비동기: W_q[i+1] → buf=1 (다음 레이어 선행 로드)
```

**2차 잔차 연결**:
```python
outputs = RMSNorm(mlp_out, post_ffn_ln) + attn_output
```

**레이어 끝: AltUp Correct + PLE 주입**:
```python
activated  = outputs * W["altup_scale"][i]
innovation = activated - xs_pred[0]

mod_corr   = get_router_modalities(activated, ...)
corr_coefs = dot(W["altup_corr"][i], mod_corr) + 1.0   # [4]

xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation   # 4개 스트림 보정

# PLE 주입 (xs[0]은 건드리지 않음!)
gate_ple = gelu(hw_matmul(activated, W["ple_gate"][i])) * pli_all[i]
mapped   = RMSNorm(hw_matmul(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
xs_new[1:] += mapped    # 그림자 스트림 1, 2, 3에만 주입

xs = xs_new
```

---

### 핵심 함수: `decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)`

```python
def decode_logits(...) -> np.ndarray  # [vocab_size=262400] float32
```

35개 레이어 출력인 4-stream `xs`를 하나의 로짓 벡터로 변환합니다.

```python
# 1. W_lm_head 프리패치 (LM Head는 크기가 크므로 CPU 계산 중 비동기 로드)
hw_prefetch(W_lm_head, buf_idx=0)

# 2. 4개 스트림 크기 정규화 후 평균 합산
target_mag = mean(xs[0]**2)**0.5
for k in 1..3:
    proj_x = dot(xs[k+1], altup_unprojs[k])
    proj_x *= target_mag / max(mean(proj_x**2)**0.5, 1e-12)  # 크기 맞춤
x_final = mean(stack([xs[0], proj_0, proj_1, proj_2]), axis=0)  # [2048]

# 3. Final RMSNorm + LM Head (핑퐁 버퍼 0 사용)
x_final = RMSNorm(x_final, W_final_norm)
logits  = hw_compute_pingpong(x_final, W_lm_head, buf_idx=0)   # [262400]
```

---

### 핵심 함수: `_sample(logits, temperature, top_p, rep_penalty, generated)`

```python
def _sample(...) -> int  # 다음 토큰 ID
```

**실행 순서**:

1. **Repetition Penalty**: 이미 생성된 토큰의 로짓을 `rep_penalty`(1.15)로 감쇠
   ```python
   logits[token] /= rep_penalty  if logits[token] > 0
   logits[token] *= rep_penalty  if logits[token] < 0
   ```

2. **Softcap**: `logits = 30.0 * tanh(logits / 30.0)` — *(`_sample` 호출 전 `main()`에서 적용)*

3. **Softmax + Temperature**: C++ `run_softmax_inplace(logits, size, temperature)`

4. **Top-p (Nucleus) Sampling**:
   ```python
   sorted_idx  = argsort(probs)[::-1]             # 내림차순 정렬
   cumsum       = cumsum(probs[sorted_idx])
   cutoff_mask  = cumsum - probs[sorted_idx] < top_p  # 누적확률 top_p 미만만 유지
   probs_filtered[sorted_idx[cutoff_mask]] = probs[...]
   ```

5. **토큰 샘플링**: `np.random.choice(vocab_size, p=probs_filtered)`

> **미구현 최적화**: Top-p 단계에서 `np.argsort` (O(n log n)) 대신 `np.argpartition` (O(n))으로 교체하면 vocab_size=262,400 기준 유의미한 속도 향상 가능.

---

### `main()` — 전체 실행 흐름

```
[초기화]
  warmup()                        ← 하드웨어 예열
  load_local_weights()            ← mmap 기반 가중치 로드
  preload_and_free() (no-op)
  K_cache = zeros([35, 2048, 512], float16)   ← 사전 할당 KV 캐시
  V_cache = zeros([35, 2048, 512], float16)
  cur_pos = 0                     ← 전역 시퀀스 포지션 (대화 간 유지)

[대화 루프]  while True:
  user_input = input()

  [Prefill]  for token in input_tokens:
    xs = forward_one_token(token, cur_pos, ...)
    cur_pos += 1

  [Generation]  for _ in range(MAX_NEW_TOKENS):
    logits     = decode_logits(xs, ...)
    logits     = 30 * tanh(logits / 30)        ← Softcap
    next_token = _sample(logits, ...)
    if next_token in [1, 106]: break            ← EOS 토큰 감지

    current_text = tokenizer.decode(generated) ← 전체 재디코딩 (UTF-8 한글 깨짐 방지)
    print(current_text[len(printed_text):])     ← 증분 출력
    printed_text = current_text

    xs = forward_one_token(next_token, cur_pos, ...)
    cur_pos += 1

  gc.collect()   ← 턴 종료 후 메모리 정리
```

**설계 주의사항**:

| 항목             | 값 / 설명                                                                    |
| ---------------- | ---------------------------------------------------------------------------- |
| `cur_pos` 초기화 | 대화 루프 **밖**에서 0으로 초기화 → 멀티턴 대화 시 KV 캐시가 계속 누적됨     |
| KV 캐시 한도     | `MAX_NEW_TOKENS = 2048` — 초과 시 `cur_pos`가 배열 범위를 벗어남             |
| `history_tokens` | 선언만 되고 **실제로 사용되지 않음** (멀티턴 히스토리 미구현)                |
| Stop 토큰        | `[1, 106]` — ID 1: `<eos>`, ID 106: Gemma turn end 토큰                      |
| 한글 출력 방식   | `tokenizer.decode(generated)` 전체 재디코딩 후 차분 출력으로 UTF-8 절단 방지 |

---

### 모듈 의존 그래프

```
main.py
  ├── ACCEL_MODE="IGPU" → IGPU_CORE.py
  │                           └── C_DLL/vulkan_core.so
  │                                 └── C_DLL/gemv_int4_vector4.spv (GPU 실행)
  ├── CPU_CORE.py
  │     └── C_DLL/my_accelerator.so
  ├── safeTensor.py   (가중치 mmap 로드)
  └── C_DLL/my_accelerator.so  (RMSNorm, Softmax — main.py 자체 등록)
```

# 코드 문서화 (5/8) — 가중치 로딩 & 변환 파이프라인

> **대상 파일**: `safeTensor.py` · `Optim_tensor_load.py`
> **역할**: 추론 시 가중치 mmap 로드(`safeTensor`) + 최초 1회 변환 스크립트(`Optim_tensor_load`)

---

## 전체 가중치 파이프라인 개요

두 파일은 **단계적 파이프라인**을 구성합니다.

```
[최초 1회 실행]
원본 모델 (*.safetensors)
    └── quantize.py             ← INT4 양자화 + .scale 파일 생성
        └── local_gemma_3n_int4/*.safetensors

    └── Optim_tensor_load.py   ← safetensors → 개별 .npy 파일 분해 + 전치 처리
        └── mmap_weights/*.npy  (가중치당 파일 1개)

[매 추론 실행]
    └── safeTensor.py           ← mmap_weights/*.npy를 mmap_mode='r'로 가상 매핑
        └── load_local_weights() 반환값 → main.py
```

> `Optim_tensor_load.py` 실행 이후에야 `safeTensor.py`가 정상 동작합니다.

---

## 1. `safeTensor.py`

### 개요

`mmap_weights/` 디렉터리의 `.npy` 파일들을 `mmap_mode='r'`로 가상 매핑하여, **RAM을 소비하지 않고** 전체 가중치 딕셔너리를 즉시 반환하는 로더입니다.

파일 안에는 두 버전의 구현이 존재합니다:

| 버전          | 위치                         | 상태       | 소스                                          |
| ------------- | ---------------------------- | ---------- | --------------------------------------------- |
| 구버전        | 파일 상단 (긴 `'''` 주석 안) | ❌ 비활성   | `local_gemma_3n_int4/*.safetensors` 직접 파싱 |
| **현재 버전** | 파일 하단 (`'''` 이후)       | ✅ **활성** | `mmap_weights/*.npy` mmap 로드                |

구버전은 `safetensors.torch.load_file`로 직접 파일을 읽고 torch 텐서를 numpy로 변환했으나, 로딩 시 전체 모델을 RAM에 올려야 했습니다. 현재 버전은 이 문제를 해결합니다.

---

### mmap 전략의 핵심 원리

```python
val = np.load("mmap_weights/some_weight.npy", mmap_mode='r')
# ↑ 이 시점에서 디스크에서 읽히는 데이터: 0 bytes
# ↑ OS가 파일의 가상 주소만 등록 (페이지 테이블 항목 1개)
# 실제 디스크 읽기는 해당 배열 원소에 처음 접근할 때만 발생 (Demand Paging)
```

결과적으로 `load_local_weights()` 호출 직후 RAM 사용량 증가는 거의 0이며, 각 가중치는 실제로 사용되는 시점에 OS 페이지 단위로 자동 로드됩니다.

**W_embed / W_ple 특화 최적화**: 두 가중치는 행 1개만 접근하는 임베딩 조회 패턴이므로, mmap이 특히 효과적입니다. 토큰 1개당 약 5.5 KB만 실제로 읽힙니다.

---

### `load_local_weights()` 함수

```python
def load_local_weights(model_dir=mmap_dir) -> tuple
```

**반환 튜플 구조**:

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
    W_lm_head,        # tuple — W_embed와 동일 객체 (Tied Weights)
    layers,           # dict[str, list[35]] — 레이어별 가중치
)
```

**`W_lm_head = W_embed` 설계**:  
Gemma 3N은 입력 임베딩과 출력 LM Head가 같은 가중치를 공유합니다 (Tied Embedding). 별도 복사 없이 동일 mmap 객체를 참조하므로 메모리 중복이 없습니다.

---

### 실행 흐름 상세

#### 1단계: 파일 목록 수집 및 scale 분리

```python
all_files = glob.glob("mmap_weights/*.npy")
all_keys  = [basename(f)[:-4] for f in all_files]  # 파일명에서 .npy 제거

# scale 파일 인덱스 생성 (빠른 pair 탐색)
scales = {k[:-6]: k for k in all_keys if k.endswith(".scale")}
# 예: {"model.language_model.layers.0.self_attn.q_proj.weight":
#      "model.language_model.layers.0.self_attn.q_proj.weight.scale"}
```

#### 2단계: mmap 가상 매핑 루프

```python
for k in all_keys:
    if k.endswith(".scale"): continue    # scale은 본체 로드 시 함께 처리

    val = np.load(f"mmap_weights/{k}.npy", mmap_mode='r')  # RAM 소비 0

    if k in scales:                      # INT4 텐서: (packed, scale) 튜플로 묶음
        scale_val = np.load(f"mmap_weights/{scales[k]}.npy", mmap_mode='r')
        val = (val, scale_val)

    # 정규식으로 레이어 인덱스와 서브키 추출
    match = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", k)
    if match:
        layer_idx = int(match.group(1))  # 0 ~ 34
        sub_key   = match.group(2)       # "self_attn.q_proj.weight" 등
        layers[KEY_MAP[sub_key]][layer_idx] = val
    else:
        globals_dict[k] = val            # 레이어 소속 아닌 전역 가중치
```

#### 3단계: 전역 가중치 분해 및 반환

```python
P = "model.language_model."

W_embed      = globals_dict[P + "embed_tokens.weight"]          # tuple (mmap)
W_ple_packed,\
W_ple_scale  = globals_dict[P + "embed_tokens_per_layer.weight"] # tuple 언패킹
W_ple_proj   = globals_dict[P + "per_layer_model_projection.weight"]
norm_ple     = globals_dict[P + "per_layer_projection_norm.weight"]
altup_projs  = [globals_dict[P + f"altup_projections.{i}.weight"] for i in range(3)]
altup_unprojs= [globals_dict[P + f"altup_unembed_projections.{i}.weight"] for i in range(3)]
W_final_norm = globals_dict[P + "norm.weight"]
W_lm_head    = W_embed   # 동일 객체 참조 (Tied Weights)
```

---

### SafeTensor 원본 키 → 내부 키 매핑표

| SafeTensor 원본 키 (sub_key)        | `layers` 딕셔너리 키 |
| ----------------------------------- | -------------------- |
| `self_attn.q_proj.weight`           | `W_q`                |
| `self_attn.k_proj.weight`           | `W_k`                |
| `self_attn.v_proj.weight`           | `W_v`                |
| `self_attn.o_proj.weight`           | `W_o`                |
| `self_attn.q_norm.weight`           | `gamma_q`            |
| `self_attn.k_norm.weight`           | `gamma_k`            |
| `input_layernorm.weight`            | `input_ln`           |
| `post_attention_layernorm.weight`   | `post_attn_ln`       |
| `pre_feedforward_layernorm.weight`  | `pre_ffn_ln`         |
| `post_feedforward_layernorm.weight` | `post_ffn_ln`        |
| `mlp.gate_proj.weight`              | `W_gate`             |
| `mlp.up_proj.weight`                | `W_up`               |
| `mlp.down_proj.weight`              | `W_down`             |
| `per_layer_input_gate.weight`       | `ple_gate`           |
| `per_layer_projection.weight`       | `ple_proj`           |
| `post_per_layer_input_norm.weight`  | `ple_post_ln`        |
| `laurel.linear_left.weight`         | `laurel_left`        |
| `laurel.linear_right.weight`        | `laurel_right`       |
| `laurel.post_laurel_norm.weight`    | `laurel_norm`        |
| `altup.router_norm.weight`          | `altup_rn`           |
| `altup.modality_router.weight`      | `altup_router`       |
| `altup.prediction_coefs.weight`     | `altup_pred`         |
| `altup.correction_coefs.weight`     | `altup_corr`         |
| `altup.correct_output_scale`        | `altup_scale`        |

---

---

## 2. `Optim_tensor_load.py`

### 개요

두 가지 독립적인 역할이 **한 파일 안에 혼재**합니다.

| 섹션       | 내용                                                 | 실행 방식                                    |
| ---------- | ---------------------------------------------------- | -------------------------------------------- |
| **상단부** | 메모리 사용량 측정 및 구조 검사 유틸리티 (`debug()`) | 함수 — 명시적 호출 필요                      |
| **하단부** | SafeTensors → `.npy` 변환 스크립트                   | **모듈 최상위 코드** — `import` 시 자동 실행 |

>  **주의**: 하단부 변환 코드가 함수 안에 감싸져 있지 않고 모듈 레벨에 노출되어 있습니다.  
> `import Optim_tensor_load`만 해도 변환 작업이 즉시 시작됩니다.  
> 실제 사용 시에는 `python Optim_tensor_load.py`로 **직접 실행**하는 용도로만 사용해야 합니다.

---

### 상단부: 메모리 검사 유틸리티

#### `get_real_memory_size(obj)`

```python
def get_real_memory_size(obj) -> int  # bytes
```

중첩된 Python 객체의 **실제 메모리 점유량**을 재귀적으로 계산합니다.

`sys.getsizeof()`는 컨테이너(list, tuple)의 껍데기 크기만 반환하고 내부 numpy 배열의 `.nbytes`는 포함하지 않습니다. 이 함수는 해당 한계를 보완합니다.

```python
def get_real_memory_size(obj):
    total = sys.getsizeof(obj)          # 컨테이너 껍데기 크기

    if isinstance(obj, np.ndarray):
        total += obj.nbytes             # 실제 데이터 크기 추가
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += get_real_memory_size(item)  # 재귀 탐색
    return total
```

**예시 — INT4 튜플의 실제 크기 계산**:
```
W_q[0] = (packed[2048, 1024] uint8, scale[2048] float32)
  get_real_memory_size(W_q[0])
    = getsizeof(tuple)          # ~56 bytes (껍데기)
    + getsizeof(packed)         # ~112 bytes (ndarray 객체)
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

가중치 객체의 **중첩 구조와 실제 차원을 문자열로 서술**합니다.  
`optim_tensor_size.md`의 표를 생성하는 데 사용된 함수입니다.

**재귀 처리 규칙**:

| 타입                     | 출력 형식                                                     |
| ------------------------ | ------------------------------------------------------------- |
| `list`                   | `List[N] ──> {0번 원소 구조}` (0번만 대표로 검사)             |
| `tuple`                  | `Tuple( {원소1 구조}, {원소2 구조} )`                         |
| `np.ndarray` (uint8, 2D) | `[ matrix: A x B , type: uint8 , (INT4 dimension: A x B*2) ]` |
| `np.ndarray` (기타)      | `[ matrix: shape , type: dtype ]`                             |

**INT4 차원 보정**: uint8 2D 배열은 INT4가 2개씩 팩킹되어 있으므로 실제 열 수를 `shape[1] * 2`로 표시합니다.

---

#### `format_memory_size(total_bytes)` / `calculate_memory_usage(obj)`

```python
def format_memory_size(total_bytes: int) -> str  # "GB | MB | Mb" 형식 문자열
def calculate_memory_usage(obj) -> str
```

`get_real_memory_size()`로 얻은 바이트를 GB/MB/Megabit 세 단위로 포맷합니다.

---

#### `debug()`

```python
def debug()
```

`safeTensor.load_local_weights()`를 호출한 뒤 모든 가중치의 구조와 메모리 크기를 마크다운 표 형식으로 출력합니다. 현재는 주석 처리(`#if __name__ == "__main__": debug()`)되어 자동 실행되지 않습니다.

출력 예시:
```
| name | matrix                                                                                                                         | GB       | MB     | Mb      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------ | ------- |
| W_q  | List[35] ──>  Tuple( [ matrix: 2048 x 1024 , type: uint8 , (INT4 dimension: 2048 x 2048) ], [ matrix: 2048 , type: float32 ] ) | 0.068636 | 70.284 | 562.269 |
...
```

---

### 하단부: SafeTensors → `.npy` 변환 스크립트

**실행 조건**: `local_gemma_3n_int4/*.safetensors` 존재 (quantize.py 실행 후)  
**출력 위치**: `mmap_weights/*.npy` (텐서당 파일 1개)

#### 실행 흐름

```python
# 1. 출력 디렉터리 생성
os.makedirs("mmap_weights/", exist_ok=True)

# 2. 모든 .safetensors 파일 순회
for st_file in sorted(glob("local_gemma_3n_int4/*.safetensors")):
    tensors = load_file(st_file)      # safetensors → torch tensor dict

    # INT4 텐서 식별 (scale 파일이 있는 것)
    quantized_bases = [k[:-6] for k in tensors if k.endswith(".scale")]

    for k, val in tensors.items():
        # bfloat16 → float32 변환 (numpy 미지원 dtype 처리)
        if val.dtype == torch.bfloat16:
            val = val.to(torch.float32)
        arr = val.numpy()

        # 전치(Transpose) 결정
        is_quantized = (k in quantized_bases) or k.endswith(".scale")
        needs_transpose = False

        if not is_quantized:  # ← INT4 텐서는 절대 전치하지 않음 (핵심 규칙!)
            if any(suffix in k for suffix in TRANSPOSE_SUFFIXES):
                needs_transpose = True

        if needs_transpose:
            arr = np.ascontiguousarray(arr.T)
        else:
            arr = np.ascontiguousarray(arr)

        np.save(f"mmap_weights/{k}.npy", arr)
```

#### 전치(Transpose) 적용 대상 목록

비양자화(float) 가중치 중 아래 suffix를 포함하는 것들만 전치합니다.

| suffix                                                             | 해당 가중치          |
| ------------------------------------------------------------------ | -------------------- |
| `per_layer_model_projection.weight`                                | W_ple_proj           |
| `altup_projections`                                                | altup_projs          |
| `altup_unembed_projections`                                        | altup_unprojs        |
| `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight` | Q, K, V, O           |
| `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`           | FFN                  |
| `per_layer_input_gate.weight`, `per_layer_projection.weight`       | ple_gate, (ple_proj) |
| `laurel.linear_left.weight`, `laurel.linear_right.weight`          | LAuReL               |
| `altup.modality_router.weight`                                     | altup_router         |

**왜 전치하는가?**  
SafeTensors에서 Linear 레이어 가중치는 `[out, in]` (행: 출력, 열: 입력) 형태로 저장됩니다.  
추론 시 `x @ W` 연산(벡터-행렬 곱)을 수행하려면 `[in, out]` 레이아웃이 필요하므로 `.T`로 미리 변환합니다.

**왜 INT4 텐서는 전치하지 않는가?**  
INT4 양자화 시 이미 `[out_dim, in_dim/2]` (행 단위 양자화, 행=출력) 레이아웃으로 저장됩니다.  
GEMV 커널(`my_accelerator.cpp`의 `run_gemv_int4`)이 이 레이아웃을 직접 소비하도록 설계되어 있으므로 전치하면 오히려 오작동합니다.

---

### 전체 파이프라인 요약

```
[1] quantize.py 실행
    local_gemma_3n/ → local_gemma_3n_int4/
    (원본 float16/32 → INT4 packed + .scale 파일 생성)

[2] Optim_tensor_load.py 실행  (최초 1회)
    local_gemma_3n_int4/*.safetensors → mmap_weights/*.npy
    - bfloat16 → float32 변환
    - 비INT4 가중치: 전치(Transpose) 적용
    - INT4 가중치: 전치 없이 그대로 저장
    - np.ascontiguousarray로 C-연속 메모리 보장

[3] safeTensor.py (매 실행)
    mmap_weights/*.npy → mmap_mode='r' 가상 매핑
    - scale 파일 자동 페어링 → (packed, scale) 튜플 구성
    - 정규식으로 레이어 인덱스 파싱
    - 10개 항목 튜플 반환 → main.py 소비
```


# 코드 문서화 (6/8) — 양자화 & KV 캐시 메모리 관리

> **대상 파일**: `quantize.py` · `Memory_Manager.py`
> **역할**: 원본 모델 INT4 양자화 변환(`quantize`) + KV 캐시 사전 할당 유틸리티(`Memory_Manager`)

---

## 두 파일의 위치

전체 파이프라인 관점에서 두 파일 모두 **1회성 준비 도구**입니다.

```
[사전 준비 단계]
  quantize.py        ← 원본 float 모델 → INT4 safetensors (최초 1회)
  Memory_Manager.py  ← KV 캐시 배열 크기 설계 및 검증 도구

[실제 추론]
  main.py            ← 위 두 파일의 결과물(변환된 가중치, 캐시 설계)을 소비
```

---

## 1. `quantize.py`

### 개요

원본 Gemma 3N E4B 모델(float16/bfloat16)의 대형 가중치를 **INT4(4비트) 대칭 양자화**로 변환하고, 변환 결과를 SafeTensors 포맷으로 저장하는 **최초 1회 실행 스크립트**입니다.

**실행**: `python quantize.py`

**입출력**:
```
입력: ORIGINAL_MODEL_DIR/local_gemma_3n/*.safetensors    (원본 float 모델)
출력: SAVE_DIR/local_gemma_3n_int4/*.safetensors         (INT4 변환 모델)
```

---

### 모듈 레벨 설정

```python
ORIGINAL_MODEL_DIR = "/home/hwkim/.../local_gemma_3n"     # 원본 모델 경로 (절대 경로 하드코딩)
SAVE_DIR           = BASE_DIR + "/local_gemma_3n_int4"    # 출력 경로
```

>  `ORIGINAL_MODEL_DIR`이 절대 경로로 하드코딩되어 있습니다.  
> 다른 환경에서 실행 시 이 값을 수동으로 수정해야 합니다.

---

### 양자화 대상 가중치 목록 (`_BIG_WEIGHT_SUFFIXES`)

아래 suffix로 끝나는 2D 가중치만 INT4로 변환합니다. 나머지는 원본 dtype 그대로 유지됩니다.

| suffix                              | 해당 가중치  | 변환 전 크기 (레이어당) | 변환 후 크기                        |
| ----------------------------------- | ------------ | ----------------------- | ----------------------------------- |
| `q_proj.weight`                     | W_q          | 2048×2048 f16 = 8MB     | 2048×1024 u8 + 2048 f32 = ~2MB      |
| `k_proj.weight`                     | W_k          | 512×2048 f16 = 2MB      | 512×1024 u8 + 512 f32 = ~0.5MB      |
| `v_proj.weight`                     | W_v          | 512×2048 f16 = 2MB      | ~0.5MB                              |
| `o_proj.weight`                     | W_o          | 2048×2048 f16 = 8MB     | ~2MB                                |
| `gate_proj.weight`                  | W_gate       | 16384×2048 f16 = 64MB   | 16384×1024 u8 + 16384 f32 = ~16.3MB |
| `up_proj.weight`                    | W_up         | 16384×2048 f16 = 64MB   | ~16.3MB                             |
| `down_proj.weight`                  | W_down       | 2048×16384 f16 = 64MB   | ~16.3MB                             |
| `embed_tokens.weight`               | W_embed      | 262400×2048 f16 = 1.0GB | 262400×1024 u8 = ~257MB             |
| `embed_tokens_per_layer.weight`     | W_ple        | 262144×8960 f16 = 4.4GB | ~1.1GB                              |
| `per_layer_input_gate.weight`       | ple_gate     | 256×2048 f16 = 1MB      | ~0.27MB                             |
| `per_layer_model_projection.weight` | W_ple_proj   | 8960×2048 f16 = 35MB    | ~8.8MB                              |
| `laurel.linear_left.weight`         | laurel_left  | 64×2048 f16 = 0.25MB    | ~0.065MB                            |
| `laurel.linear_right.weight`        | laurel_right | 2048×64 f16 = 0.25MB    | ~0.065MB                            |

**양자화되지 않는 가중치**: 모든 LayerNorm(gamma), altup 계수, 1D 가중치, `ple_proj` (현재 float32 유지 — 향후 INT4 변환 검토 대상)

---

### 핵심 함수: `quantize_to_int4(weight)`

```python
def quantize_to_int4(
    weight: np.ndarray    # [N, M] float16 또는 float32
) -> tuple[np.ndarray, np.ndarray]:
    # 반환: (packed [N, M//2] uint8, scale [N] float32)
```

**행 단위(Per-Row) 대칭 양자화**를 수행합니다. 행 = 출력 뉴런 1개.

#### 수식

$$\text{scale}[i] = \frac{\max(|w_i|)}{7.0}$$

$$w_q[i,j] = \text{clip}\!\left(\text{round}\!\left(\frac{w[i,j]}{\max(|w_i|)} \times 7.0\right),\ -8,\ 7\right)$$

$$\text{packed}[i, j//2] = (w_q[i, 2j] \mathbin{\&} \texttt{0x0F})\ |\ ((w_q[i, 2j+1] \mathbin{\&} \texttt{0x0F}) \ll 4)$$

#### 단계별 구현

**1단계: float32 업캐스팅**
```python
w_f32 = weight.astype(np.float32)
# float16 정밀도로는 max 계산 시 오차 발생 가능 → float32로 업캐스팅
```

**2단계: 행별 스케일 계산**
```python
max_vals = np.max(np.abs(w_f32), axis=1, keepdims=True)  # [N, 1]
max_vals = np.maximum(max_vals, 1e-8)                     # 0으로 나누기 방지
scale    = (max_vals / 7.0).flatten()                     # [N] — dequant 시 사용
```

**3단계: 정규화 및 반올림**
```python
w_q = np.round(w_f32 / max_vals * 7.0).astype(np.int8)
w_q = np.clip(w_q, -8, 7)
# 범위: [-8, 7] — 4비트 부호 있는 정수 전체 범위 활용
# -8은 표현 가능하지만 dequant 시 약간의 비대칭 오차 발생
```

**4단계: 2개씩 uint8 패킹**
```python
w_q_low  = w_q[:, 0::2] & 0x0F   # 짝수 열 → 하위 4비트
w_q_high = w_q[:, 1::2] & 0x0F   # 홀수 열 → 상위 4비트
packed   = (w_q_low | (w_q_high << 4)).astype(np.uint8)
# [N, M] int8 → [N, M//2] uint8  (50% 메모리 절약)
```

**패킹 레이아웃 시각화**:
```
원본  w_q[i]: [ a, b, c, d, e, f, ... ]  (int8, M개)
                └─┬─┘  └─┬─┘
packed[i]:    [ a|b<<4, c|d<<4, ... ]    (uint8, M/2개)

예시: a=-3 (0b1101), b=5 (0b0101)
  a & 0x0F = 0x0D (하위 4비트)
  b & 0x0F = 0x05 (상위 4비트)
  packed   = 0x0D | (0x05 << 4) = 0x5D
```

---

### `main()` 실행 흐름

```python
def main():
    for filename in sorted(glob("local_gemma_3n/*.safetensors")):
        tensors = load_file(filename)           # safetensors → torch tensor dict
        quantized_tensors = {}

        for name, tensor in tensors.items():
            is_big = any(name.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)

            if is_big and len(tensor.shape) == 2:   # 2D 대형 가중치만 양자화
                weight_np = tensor.to(torch.float32).numpy()
                packed, scale = quantize_to_int4(weight_np)

                quantized_tensors[name]            = torch.from_numpy(packed)  # uint8
                quantized_tensors[name + ".scale"] = torch.from_numpy(scale)   # float32

            else:                                   # 1D, 소형, 비대상 → 원본 유지
                quantized_tensors[name] = tensor

        save_file(quantized_tensors, SAVE_DIR + "/" + basename(filename))

        del tensors, quantized_tensors
        gc.collect()   # 파일 단위 메모리 해제 (전체 모델이 RAM에 동시 상주하지 않도록)
```

**파일 단위 처리의 이유**: SafeTensors 파일 1개씩 처리하고 즉시 `del` + `gc.collect()`를 호출하여, 원본 모델 전체(~9GB)가 RAM에 동시에 올라오지 않도록 합니다.

---

### 양자화 오차 특성

| 항목           | 내용                                                         |
| -------------- | ------------------------------------------------------------ |
| 방식           | 대칭 양자화 (Symmetric) — 0 중심, 오프셋 없음                |
| 표현 범위      | [-8, 7] — 이론상 [-7.5, 7.5]이나 clip으로 [-8, 7] 강제       |
| 스케일 단위    | 행(출력 뉴런) 1개당 1개 (Per-Row)                            |
| 이론 최대 오차 | `scale × 0.5` (반올림 오차)                                  |
| 비대칭성       | -8은 표현되지만 +8은 clip → 음의 방향으로 미세한 비대칭 존재 |
| 정밀도 손실    | float16 → INT4: 약 75% 비트 감소, 실용적 품질 유지           |

---

---

## 2. `Memory_Manager.py`

### 개요

KV 캐시 배열을 단일 연속 NumPy 배열로 사전 할당하는 **유틸리티 모듈**입니다. 현재는 `main.py`에서 직접 `np.zeros()`로 동일 작업을 수행하므로, 이 모듈 자체가 추론 중 호출되지는 않습니다. 캐시 메모리 레이아웃을 설계·검증하는 참조용 코드로 역할합니다.

---

### `allocate_KVcache(layers, token, dimension)`

```python
def allocate_KVcache(
    layers:    int,   # 레이어 수 (35)
    token:     int,   # 최대 시퀀스 길이 (2048)
    dimension: int,   # KV 헤드 차원 (512 = 2 KV헤드 × 256)
) -> np.ndarray       # [layers, token, dimension] float16
```

```python
A = np.zeros((layers, token, dimension), dtype=np.float16)
return A
```

---

### KV 캐시 설계 근거

**왜 float16인가?**  
어텐션 계산 시에는 float32로 복원(`K_cache.astype(np.float32)`)하지만, 저장은 float16으로 합니다. 정밀도 손실이 어텐션 품질에 미치는 영향이 미미한 반면 메모리는 절반으로 줄어듭니다.

**차원 구성 (512)**:
```
KV 헤드 수: 2개 (GQA 구조)
헤드 차원:  256
합계:       2 × 256 = 512
```

**기본값으로 호출 시 메모리 계산**:
```
allocate_KVcache(35, 2048, 512)
  = 35 × 2048 × 512 × 2 bytes (float16)
  = 73,400,320 bytes
  ≈ 70 MB (K + V 각각)
  K_cache + V_cache 합계 ≈ 140 MB
```

---

### `main.py`에서의 실제 할당

`Memory_Manager.py`를 import하지 않고 `main.py`에서 직접 동일 구조로 할당합니다:

```python
# main.py
K_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
V_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
# = np.zeros((35, 2048, 512), dtype=np.float16) — Memory_Manager와 동일 설계
```

**인덱싱 방식**:
```python
# 쓰기 (레이어 0~19)
K_cache[layer_idx, pos, :] = K.astype(np.float16)   # in-place 슬라이스 기록

# 읽기
target_k = K_cache[layer_idx, :pos+1, :]             # 현재까지 누적된 시퀀스 슬라이스
```

이전 `np.concatenate` 기반 동적 성장 방식 대비, 사전 할당 방식은 토큰 생성마다 발생하던 O(N) 재할당을 O(1) in-place 쓰기로 대체합니다.

---

### 현재 상태 및 개선 방향

| 항목                             | 현재 상태                                 | 비고                             |
| -------------------------------- | ----------------------------------------- | -------------------------------- |
| `allocate_KVcache()` 사용 여부   | `main.py`에서 미사용                      | `main.py`가 직접 `np.zeros` 호출 |
| `if __name__ == "__main__"` 블록 | 테스트 용도 (`shape` 출력만)              | `(35, 2048, 512)` 확인           |
| 레이어 20~34 슬롯                | 할당은 되어 있으나 기록 안 됨             | KV 라우팅으로 18/19번 재사용     |
| 잠재 OOM 위험                    | `cur_pos`가 2048 초과 시 인덱스 범위 초과 | `MAX_NEW_TOKENS` 가드 필요       |

**향후 활용 방향**: 멀티턴 대화에서 KV 캐시 초기화 또는 슬라이딩 윈도우 방식을 도입할 경우, 이 모듈을 확장하여 캐시 리셋·압축 로직을 캡슐화하는 것이 적합합니다.

---

### 두 파일의 전체 파이프라인 위치

```
[Step 1]  quantize.py
          원본 float 모델 (9GB+)
              ↓  행별 대칭 INT4 양자화
          local_gemma_3n_int4/*.safetensors
          (packed uint8 + scale float32 쌍으로 저장)

[Step 2]  Optim_tensor_load.py
          local_gemma_3n_int4/*.safetensors
              ↓  텐서별 .npy 분해 + 전치 처리
          mmap_weights/*.npy

[Step 3]  safeTensor.py  (매 실행 시)
          mmap_weights/*.npy → mmap 가상 매핑
              ↓
          load_local_weights() 반환

[Step 4]  main.py  (매 실행 시)
          Memory_Manager.py 설계 기반
          K_cache, V_cache = np.zeros([35, 2048, 512], float16)
              ↓
          forward_one_token() → decode_logits() → _sample()
```

# 코드 문서화 (7/8) — FPGA NPU 엔진 & 아키텍처 설계 문서

> **대상 파일**: `NPU_CORE.py` · `gemma3N_E4B_architecture.md`
> **역할**: FPGA 기반 NPU 가속 엔진 (`NPU_CORE`) + 전체 모델 구조 및 하드웨어 분배 레퍼런스 문서 (`gemma3N_E4B_architecture.md`)

---

## 1. `NPU_CORE.py`

### 개요

**FPGA RTL 기반 NPU 하드웨어의 Python 제어 레이어**입니다. 현재 프로젝트의 실행 경로(`IGPU_CORE.py` / `CPU_MATRIX_CORE.py`)와는 **별개의 하드웨어 타겟**으로, Xilinx/AMD FPGA에 구현된 커스텀 Systolic Array NPU를 MMIO(Memory-Mapped I/O)로 제어합니다.

**현재 상태**: `import MMIO` 의존성으로 인해 FPGA 보드 없이는 직접 실행 불가. 단, 함수 최상단의 `MMIO.SIMULATION_MODE` 분기가 PC에서의 동작을 **NumPy로 완전히 모킹(Mocking)** 합니다.

---

### 하드웨어 아키텍처 개요

```
CPU (Python/NumPy)
    │  MMIO 레지스터 제어
    │  DMA 전송
    ▼
FPGA 내부
  ┌─────────────────────────────────────┐
  │  AXI DMA  ──→  Ping-Pong BRAM      │
  │                    │               │
  │              Systolic Array NPU    │
  │              (32×32 PE 타일)       │
  │                    │               │
  │              ACC (누산기)          │
  │                    │               │
  │           RMSNorm / GeLU IP        │
  │                    │               │
  │              Result BRAM           │
  └─────────────────────────────────────┘
```

---

### MMIO 레지스터 맵

| 주소         | 역할                                                    | 사용 예                       |
| ------------ | ------------------------------------------------------- | ----------------------------- |
| `0x00`       | 제어 레지스터 — Bit0: NPU_START (펄스), Bit1: ACC_CLEAR | `write(0x00, 0x01)` 계산 시작 |
| `0x08`       | RMSNorm 분모 스칼라 (`mean_sq_val`)                     | `write(0x08, int(mean_sq))`   |
| `0x0C`       | DMA 스위치 — 0: Ping 버퍼, 1: Pong 버퍼 선택            | `write(0x0C, 0)`              |
| `0x10` Bit0  | GeLU 하드웨어 IP 활성화                                 | `write(0x10, 0x01)`           |
| `0x10` Bit1  | Softmax IP 활성화                                       | `write(0x10, 0x02)`           |
| `0x10` Bit16 | NPU 완료 플래그 (`w_npu_done`) — 폴링 대상              | `read(0x10) & 0x010000`       |
| `0x14`       | DMA 스트림 타입 — 0: Token, 1: Weight                   | `write(0x14, 0 또는 1)`       |

>  `0x04` 레지스터는 이전 버전에서 완료 플래그로 잘못 사용되었다가 `0x10`으로 수정된 이력이 있습니다 (코드 주석 `Polling bug fixed: 0x04 -> 0x10`).

---

### 핵심 함수: `run_npu_matmul(x_vec, weight_mat, mean_sq_val, use_gelu=False)`

```python
def run_npu_matmul(
    x_vec:       np.ndarray,   # [2048] 입력 벡터 (RMSNorm 적용 전)
    weight_mat:  np.ndarray,   # [2048, Output_Dim] 가중치 행렬
    mean_sq_val: float,        # RMSNorm 분모: mean(x²) 값
    use_gelu:    bool = False, # FFN Gate 전용 GeLU 하드웨어 활성화
) -> np.ndarray                # [Output_Dim] int16 (FPGA) 또는 float16 (시뮬레이션)
```

#### 시뮬레이션 경로 (`MMIO.SIMULATION_MODE = True`)

```python
inv_sqrt = 1.0 / sqrt(mean_sq_val + 1e-6)
x_f32    = x_vec.astype(np.float32) * inv_sqrt   # float32 업캐스팅 필수
                                                   # (2048차원 FP16 누산 → 오버플로 위험)
out = np.dot(x_f32, weight_mat.astype(np.float32))
if use_gelu:
    out = GELU(out)
return out.astype(np.float16)
```

**FP16 업캐스팅 이유**: 2048차원 누산 시 FP16의 최대값(65504)을 초과할 수 있어 반드시 FP32로 중간 계산합니다.

---

#### FPGA 실행 경로 — 타일링 구조

입력(2048)과 출력(Output_Dim)을 **32×32 타일**로 분해하여 처리합니다.

```
num_ic_tiles = 2048 // 32 = 64   (입력 채널 타일 수)
num_oc_tiles = Out  // 32        (출력 채널 타일 수)
total_tiles  = 64 × num_oc_tiles

타일 순서 예시 (Out=2048, total=4096):
  tile_idx 0   → oc=0, ic=0    (출력채널 0번, 입력채널 0번)
  tile_idx 1   → oc=0, ic=1    (출력채널 0번, 입력채널 1번)
  ...
  tile_idx 63  → oc=0, ic=63   ← ic 마지막: ACC 읽기 발생
  tile_idx 64  → oc=1, ic=0    ← ic 시작: ACC_CLEAR 발생
```

---

#### FPGA 실행 경로 — Ping-Pong BRAM 파이프라인

각 타일 반복에서 **계산과 다음 타일 DMA 전송이 동시에** 진행됩니다.

```
[Prologue]
  타일 0의 token(32) + weight(32×32) → Ping 버퍼 전송 (동기)

[메인 루프: tile_idx = 0 → total_tiles-1]
  ┌─ 1. DMA 백그라운드 전송 (비동기) ───────────────────────────────┐
  │   tile_idx 짝수(Ping 계산 중) → 다음 tile 데이터 → Pong 버퍼   │
  │   tile_idx 홀수(Pong 계산 중) → 다음 tile 데이터 → Ping 버퍼   │
  │   전송 순서: Token 먼저 (0x14=0) → Weight (0x14=1)             │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 2. NPU 계산 킥 ────────────────────────────────────────────────┐
  │   write(0x00, 0x01)  ← START 펄스 (자동으로 0으로 복귀)        │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 3. 완료 대기 (폴링) ───────────────────────────────────────────┐
  │   while (read(0x10) & 0x010000) == 0: pass                      │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 4. 결과 수신 (ic 마지막 타일만) ──────────────────────────────┐
  │   if ic == num_ic_tiles - 1:                                    │
  │       DMA recv → result_buf → final_out[oc*32:(oc+1)*32]       │
  └────────────────────────────────────────────────────────────────┘
  5. DMA 전송 완료 대기 후 다음 루프

[특수 처리]
  ic == 0 진입 시: ACC_CLEAR (write(0x00, 0x02))
  → 새 출력 채널 계산 시작 전 누산기 초기화
```

**결과 수신 타이밍**: 출력 채널 1개(32개 출력 뉴런)를 계산하는 64번의 내적 누산이 완료된 뒤(`ic == 63`)에만 DMA로 결과를 받습니다. 중간에는 FPGA 내부 누산기(ACC)가 값을 유지합니다.

---

### 래퍼 함수

```python
def npu_matmul(x, weight, mean_sq):
    """ Q, K, V, O, Down — 일반 행렬곱 """
    return run_npu_matmul(x, weight, mean_sq, use_gelu=False)

def npu_matmul_gelu(x, W_gate, mean_sq):
    """ FFN Gate 전용 — 행렬곱 직후 1-Cycle GeLU 하드웨어 IP 활성화 """
    return run_npu_matmul(x, W_gate, mean_sq, use_gelu=True)
```

---

### `npu_softmax(logits)`

```python
def npu_softmax(logits: np.ndarray) -> np.ndarray  # float16
```

**시뮬레이션 경로**: Stable Softmax (max 빼기) NumPy 구현.

**FPGA 경로**:  
Softmax는 행렬곱이 아니므로 Systolic Array를 거치지 않고 전용 **Softmax IP**로 전달합니다.

```python
MMIO.npu_control.write(0x10, 0x02)   # Softmax_EN 비트 ON

for i in range(0, len(logits), 32):   # 32개씩 분할 전송
    ping_token ← logits[i:i+32]
    DMA send → NPU kick → DMA recv → probs[i:i+32]
```

> Softmax의 완료 폴링은 `0x04 & 0x01` 방식으로 행렬곱(`0x10 & 0x010000`)과 **다른 레지스터를 사용**합니다. Softmax IP와 Systolic Array의 done 신호가 하드웨어 레벨에서 별도로 구현되어 있음을 나타냅니다.

---

### 세 가속 모드 비교

| 항목           | NPU_CORE.py                | IGPU_CORE.py            | CPU_MATRIX_CORE.py       |
| -------------- | -------------------------- | ----------------------- | ------------------------ |
| 하드웨어       | FPGA Systolic Array        | iGPU (Vulkan)           | CPU AVX2/OpenMP          |
| 제어 방식      | MMIO 레지스터 + DMA        | Vulkan Command Buffer   | OpenMP 멀티코어          |
| 가중치 포맷    | float16 (FPGA 내장 변환)   | INT4 uint8 (GPU 언패킹) | INT4 uint8 (SIMD 언패킹) |
| 타일 단위      | 32×32 고정                 | 워크그룹 32             | AVX2 레지스터 단위       |
| RMSNorm 위치   | NPU 내장 IP (mean_sq 전달) | CPU (별도 호출)         | CPU (별도 호출)          |
| GeLU 위치      | NPU 내장 IP (1-Cycle)      | CPU (`CPU_CORE.gelu`)   | C++ SIMD 인라인          |
| 현재 사용 여부 | ❌ (FPGA 타겟, 미연결)      | ✅ 기본 모드             | ✅ CPU 모드               |

---

---

## 2. `gemma3N_E4B_architecture.md`

### 개요

Gemma 3N E4B 모델의 **Forward Pass 전체를 레퍼런스로 기록한 설계 문서**입니다. "안녕하세요" 라는 예시 입력을 기준으로 각 단계의 동작 원리, CPU/IGPU 분배 근거, 핵심 수식을 포함합니다. 코드베이스에서 구현 판단의 기준이 되는 **단일 진실 소스(Single Source of Truth)** 역할을 합니다.

> **주의**: 문서 내 일부 코드 스니펫은 **이전 버전 구조**를 반영하며, 현재 `main.py`의 구현과 세부적으로 다를 수 있습니다. 차이점은 아래 "문서와 현재 코드의 차이" 항목에서 정리합니다.

---

### 문서 구조

| Phase       | 단계                            | 하드웨어 | 핵심 내용                               |
| ----------- | ------------------------------- | -------- | --------------------------------------- |
| **Phase 1** | 1. 토큰화 + 가중치 로드         | CPU      | 텍스트 → 정수 ID                        |
|             | 2. 임베딩 + AltUp 초기화        | CPU/IGPU | ID → [4, 2048] 4-스트림                 |
| **Phase 2** | 3. AltUp Router (Predict)       | IGPU     | Tanh 기반 `xs_pred` 생성                |
|             | 4. Pre-Attn RMSNorm + Q,K,V     | IGPU     | `inputs_normalized` → Q,K,V             |
|             | 5. QK-Norm + RoPE               | CPU      | 헤드별 정규화 + 위치 인코딩             |
|             | 6. KV Cache 라우팅 + GQA        | CPU      | 20~34층 캐시 재사용, 무스케일           |
|             | 7. W_o Proj + LAuReL + 1차 잔차 | IGPU     | `1/√2` 스케일 합산                      |
|             | 8. FFN Sparsity (0~9층)         | IGPU/CPU | 상위 5% 뉴런만 활성화                   |
|             | 9. 2차 잔차 연결                | IGPU     | `outputs += attn_output`                |
|             | 10. PLE 주입 (xs[1~3])          | CPU      | 그림자 스트림에만 레이어 위치 정보 주입 |
| **Phase 3** | 11. Final Norm + LM Head        | IGPU     | 4-스트림 → vocab 로짓                   |
|             | 12. Softmax + 샘플링            | CPU      | 반복 패널티 + Top-p                     |

---

### 핵심 설계 원칙 요약

#### 1. AltUp 4-Stream 구조

```
xs[0]  = 메인 스트림  ← Attention/FFN 연산의 유일한 입력. 절대 직접 수정 안 됨
xs[1]  = 그림자 스트림 1  ┐
xs[2]  = 그림자 스트림 2  ├─ altup_projs로 생성, PLE 주입 대상
xs[3]  = 그림자 스트림 3  ┘

레이어 시작: xs → xs_pred (AltUp Predict, 4×4 계수 행렬)
레이어 끝:   xs_pred + innovation × corr_coefs → xs_new (AltUp Correct)
```

AltUp의 핵심: **연산은 `xs[0]` 하나만**, **정보 축적은 4개 스트림 전부**.

---

#### 2. KV 캐시 라우팅 규칙

```
레이어 0~19:  자신의 슬롯에 K, V 기록 + 자신의 캐시 조회
레이어 20~34: 캐시 기록 없음
              ├── i % 5 == 4 (Global 레이어: 24,29,34) → K_cache[19] 재사용
              └── 나머지    (Local  레이어)              → K_cache[18] 재사용
```

**근거**: 심층 레이어일수록 Attention 패턴이 고착화됩니다. 18번(Local)과 19번(Global) 레이어가 가장 잘 학습된 패턴을 보유하므로, 20번 이후 레이어에서 이를 재사용합니다. 레이어 15개(20~34)의 KV 캐시 저장 비용을 완전히 제거합니다.

---

#### 3. Unscaled GQA (무스케일 어텐션)

표준 어텐션:
$$\text{Attn} = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad d_k=256$$

Gemma 3N E4B:
$$\text{Attn} = \text{Softmax}(QK^T)V \quad \leftarrow \sqrt{d_k} \text{ 나눗셈 없음}$$

스케일 없이 Raw Score를 그대로 Softmax에 전달합니다. `cpu_gqa()` 내부에서 `/ math.sqrt(256)` 연산이 존재하지 않습니다.

---

#### 4. FFN 극단적 희소성 (0~9층)

$$\text{cutoff} = \mu(\text{gate}) + 1.6449 \cdot \sigma(\text{gate})$$
$$\text{sparse\_gate} = \max(\text{gate} - \text{cutoff},\ 0)$$

정규분포 z=1.6449는 상위 5% 커트라인에 해당합니다. 95%의 뉴런이 정확히 0이 되어 W_down 행렬곱에서 sparse 연산 기회가 생깁니다.

```
0~9층:   sparse gate (95% 제로) → W_up → W_down  (연산량 대폭 감소)
10~34층: dense gate (GeLU 융합)  → W_up → W_down  (속도 우선)
```

---

#### 5. LAuReL (Layer-wise Augmented Residual Learning)

$$\text{laurel\_out} = x_n + \text{RMSNorm}(\text{right}(\text{left}(x_n)))$$
$$\text{attn\_final} = (\text{attn\_output} + \text{laurel\_out}) \times \frac{1}{\sqrt{2}}$$

W_o 프로젝션과 **병렬**로 실행됩니다. 저랭크(64차원) 두 선형 레이어를 통과하는 짧은 우회 경로를 추가해 표현력을 보강합니다. `1/√2` 스케일링은 두 경로 합산 후 분산을 유지합니다.

---

#### 6. PLE (Per-Layer Embedding) 주입

```
PLE 계산 (레이어 루프 진입 전, 토큰 1개당 1회):
  x0 → W_ple_proj → reshape[35, 256] → RMSNorm(행별) + norm_ple   = x_proj_normed
  x0 → W_ple[token_id] → reshape[35, 256] × √256                  = y
  pli_all = (x_proj_normed + y) × (1/√2)   shape: [35, 256]

레이어 i에서 PLE 주입:
  pli = pli_all[i]                          ← i번 레이어의 위치 벡터
  gate_ple = GELU(activated @ W_ple_gate[i]) × pli
  mapped   = RMSNorm(gate_ple @ W_ple_proj[i], ple_post_ln[i])
  xs_new[1:] += mapped   ← xs[0]는 건드리지 않음
```

레이어 번호 정보를 메인 연산 경로에 오염시키지 않고 그림자 스트림에만 누적합니다.

---

### 문서와 현재 코드(`main.py`)의 차이점

| 항목             | 문서 (`architecture.md`)                          | 현재 코드 (`main.py`)                                              |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| KV 캐시 자료구조 | `K_cache = [[] for _ in range(35)]` (리스트)      | `np.zeros([35, 2048, 512], float16)` (사전 할당 배열)              |
| W_embed 형태     | `CPU_CORE.embedding(token_id, W_embed)` 단일 인자 | `CPU_CORE.embedding(token_id, W_embed[0], W_embed[1])` (튜플 분해) |
| W_o 이후 잔차    | `attn_output += xs[0]`                            | `attn_output += x` (`xs_pred[0].copy()` 사용)                      |
| AltUp Correct    | `xs_pred.copy()` 후 for 루프                      | `xs_pred + corr_coefs[:, np.newaxis] * innovation` (벡터화)        |
| 핑퐁 최적화      | 미기재                                            | Q→K→V→O→Gate→Up→Down 순서로 `hw_prefetch`/`hw_compute_pingpong`    |
| GeLU 위치(0~9층) | `use_gelu=False` 명시 후 별도 적용                | 동일 (`use_gelu=(i >= 10)`)                                        |

---

### RoPE 레이어 분류

| 레이어 인덱스                               | 조건         | theta     | 타입       |
| ------------------------------------------- | ------------ | --------- | ---------- |
| 4, 9, 14, 19, 24, 29, 34                    | `i % 5 == 4` | 1,000,000 | **Global** |
| 0~3, 5~8, 10~13, 15~18, 20~23, 25~28, 30~33 | 나머지       | 10,000    | **Local**  |

Global 레이어는 긴 범위 문맥 포착, Local 레이어는 근거리 패턴 포착을 담당합니다.

---

### 전체 Forward Pass 데이터 흐름 요약

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
    ▼  ×35 레이어
┌──────────────────────────────────┐
│  AltUp Predict  → xs_pred        │
│  RMSNorm(xs[0]) → inputs_norm    │
│                                  │
│  Q,K,V = inputs_norm @ W_q,k,v   │
│  QK-Norm → RoPE                  │
│  KV Cache 라우팅                  │
│  GQA (Unscaled) → attn_raw       │
│                                  │
│  W_o + LAuReL + 잔차 1           │
│                                  │
│  RMSNorm → W_gate (sparse/dense) │
│         → W_up                   │
│  hidden = gate × up              │
│  W_down + 잔차 2                 │
│                                  │
│  AltUp Correct → xs_new          │
│  PLE 주입 → xs_new[1:] += mapped │
└──────────────────────────────────┘
    │
    ▼
xs [4, 2048]
    │
    ▼ decode_logits()
    │  4-스트림 크기 정규화 + 평균
    │  Final RMSNorm + W_lm_head
    │
logits [262400]
    │
    ▼ _sample()
    │  Softcap(30) → Rep Penalty → Softmax → Top-p
    │
next_token (int)
```

# 코드 문서화 (8/8) — 챗 템플릿 & 메모리 사용량 레퍼런스

> **대상 파일**: `chat_template.jinja` · `optim_tensor_size.md`
> **역할**: 대화 프롬프트 포맷 정의(`chat_template`) + 전체 가중치 메모리 사용량 기준표(`optim_tensor_size`)

---

## 1. `chat_template.jinja`

### 개요

HuggingFace `AutoTokenizer`가 멀티턴 대화 메시지 리스트를 **단일 토큰화 가능한 문자열로 직렬화**할 때 사용하는 Jinja2 템플릿입니다. Gemma 3N의 공식 대화 포맷을 정의하며, `tokenizer.apply_chat_template()` 호출 시 자동으로 참조됩니다.

현재 프로젝트의 `main.py`는 이 템플릿을 직접 사용하지 않고 수동으로 포맷 문자열을 구성합니다:

```python
# main.py — 단순화된 수동 포맷
prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
```

이 템플릿은 시스템 프롬프트, 멀티미디어 콘텐츠(이미지/오디오), 엄격한 역할 교대 검증 등 더 복잡한 시나리오에서 표준 참조용으로 활용됩니다.

---

### Gemma 3N 대화 포맷 구조

템플릿이 생성하는 최종 문자열 형식:

```
<bos>
<start_of_turn>user
{system_content}         ← 시스템 프롬프트가 있을 경우 첫 번째 user 턴 앞에 삽입

{user_message}<end_of_turn>
<start_of_turn>model
{assistant_message}<end_of_turn>
<start_of_turn>user
{user_message_2}<end_of_turn>
<start_of_turn>model
                         ← add_generation_prompt=True 시 여기서 절단 (모델 생성 유도)
```

**특수 토큰**:
| 토큰                 | 역할                                           |
| -------------------- | ---------------------------------------------- |
| `<bos>`              | Beginning of Sequence — 시퀀스 최앞에 1회 삽입 |
| `<start_of_turn>`    | 발화 시작 마커                                 |
| `<end_of_turn>`      | 발화 종료 마커 + `\n`                          |
| `<audio_soft_token>` | 오디오 입력 위치 플레이스홀더                  |
| `<image_soft_token>` | 이미지 입력 위치 플레이스홀더                  |

---

### 템플릿 로직 단계별 분석

#### 1단계: BOS 토큰 삽입

```jinja
{{ bos_token }}
```

시퀀스 맨 앞에 `<bos>` 토큰을 1회 출력합니다.

---

#### 2단계: 시스템 프롬프트 처리

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

메시지 리스트의 첫 항목이 `role: system`이면:
- 시스템 내용을 `first_user_prefix` 변수에 저장 (뒤에 `\n\n` 추가)
- `loop_messages`에서 시스템 메시지를 **제외**한 나머지로 루프 진행

시스템 프롬프트가 없으면 `first_user_prefix = ""`로 설정합니다.

**콘텐츠 타입 분기**:
- `string` → 직접 사용
- `iterable` (멀티모달 리스트) → 첫 번째 원소의 `.text` 필드 사용

---

#### 3단계: 역할 교대 검증

```jinja
{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
    {{ raise_exception("Conversation roles must alternate user/assistant/...") }}
{%- endif -%}
```

`loop.index0`(0-based 인덱스)의 짝수 위치에는 반드시 `user`, 홀수 위치에는 반드시 `assistant`여야 합니다. 어긋나면 즉시 예외를 발생시킵니다.

**검증 로직 해석**:
```
index0=0 (짝수) → role == 'user' 이어야 함 → (True) != (True) → False → 통과
index0=1 (홀수) → role == 'user' 이면 (True) != (False) → True → 예외!
```

---

#### 4단계: 역할명 정규화

```jinja
{%- if (message['role'] == 'assistant') -%}
    {%- set role = "model" -%}
{%- else -%}
    {%- set role = message['role'] -%}
{%- endif -%}
```

HuggingFace 표준 역할명 `"assistant"`를 Gemma 3N의 내부 역할명 **`"model"`** 로 변환합니다. `<start_of_turn>model\n` 포맷에 맞추기 위함입니다.

---

#### 5단계: 발화 렌더링

```jinja
{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
```

- `<start_of_turn>{role}\n` 출력
- 첫 번째 메시지(`loop.first`)에만 시스템 프롬프트 내용(`first_user_prefix`) 삽입

**콘텐츠 렌더링 분기**:

```jinja
{%- if message['content'] is string -%}
    {{ message['content'] | trim }}          ← 일반 텍스트

{%- elif message['content'] is iterable -%}  ← 멀티모달 리스트
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

`| trim` 필터로 앞뒤 공백 및 줄바꿈을 제거합니다.

---

#### 6단계: 생성 프롬프트 삽입

```jinja
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}
```

`add_generation_prompt=True`로 호출 시 마지막에 `<start_of_turn>model\n`을 추가하여 모델이 이어서 생성하도록 유도합니다.

---

### 사용 예시

**입력 메시지**:
```python
messages = [
    {"role": "system",    "content": "당신은 친절한 AI입니다."},
    {"role": "user",      "content": "안녕하세요!"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
    {"role": "user",      "content": "오늘 날씨가 어때요?"},
]
```

**출력 문자열** (`add_generation_prompt=True`):
```
<bos><start_of_turn>user
당신은 친절한 AI입니다.

안녕하세요!<end_of_turn>
<start_of_turn>model
안녕하세요! 무엇을 도와드릴까요?<end_of_turn>
<start_of_turn>user
오늘 날씨가 어때요?<end_of_turn>
<start_of_turn>model
```

---

### `main.py`의 단순화 포맷과 비교

| 항목            | `chat_template.jinja`                     | `main.py` 수동 포맷        |
| --------------- | ----------------------------------------- | -------------------------- |
| BOS 토큰        | 자동 삽입                                 | 미삽입 (토크나이저가 처리) |
| 시스템 프롬프트 | 지원                                      | 미지원                     |
| 멀티미디어      | `<audio/image_soft_token>` 지원           | 미지원                     |
| 역할 교대 검증  | 엄격 (예외 발생)                          | 없음                       |
| 멀티턴 히스토리 | 지원                                      | 미구현 (단일 턴만)         |
| 사용 방법       | `tokenizer.apply_chat_template(messages)` | 문자열 f-string 직접 구성  |

---

---

## 2. `optim_tensor_size.md`

### 개요

`Optim_tensor_load.py`의 `debug()` 함수로 생성된 **전체 가중치 메모리 사용량 기준 측정표**입니다. INT4 양자화 + mmap 로딩 이후의 실제 RAM 점유 크기를 가중치별로 기록합니다. 메모리 최적화 작업의 기준선(Baseline) 역할을 합니다.

**생성 시점**: `quantize.py` + `Optim_tensor_load.py` 실행 완료 후 mmap 로드 상태에서 측정.

---

### 가중치 분류별 메모리 요약

#### Attention 가중치 (레이어당 × 35)

| 키        | 행렬 형태 (packed) | INT4 실제 차원 | 35레이어 합계 |
| --------- | ------------------ | -------------- | ------------- |
| `W_q`     | 2048 × 1024 uint8  | 2048 × 2048    | **70.3 MB**   |
| `W_k`     | 512 × 1024 uint8   | 512 × 2048     | 17.6 MB       |
| `W_v`     | 512 × 1024 uint8   | 512 × 2048     | 17.6 MB       |
| `W_o`     | 2048 × 1024 uint8  | 2048 × 2048    | **70.3 MB**   |
| `gamma_q` | 256 float32        | —              | 0.04 MB       |
| `gamma_k` | 256 float32        | —              | 0.04 MB       |

> W_q, W_o는 K, V 대비 4배 큰 이유: Q 헤드 수(8) vs KV 헤드 수(2), GQA 구조.

---

#### FFN 가중치 (레이어당 × 35) — **전체 메모리의 최대 비중**

| 키       | 행렬 형태 (packed) | INT4 실제 차원 | 35레이어 합계 |
| -------- | ------------------ | -------------- | ------------- |
| `W_gate` | 16384 × 1024 uint8 | 16384 × 2048   | **562.2 MB**  |
| `W_up`   | 16384 × 1024 uint8 | 16384 × 2048   | **562.2 MB**  |
| `W_down` | 2048 × 8192 uint8  | 2048 × 16384   | **560.3 MB**  |

FFN 3개 행렬 합계 ≈ **1.685 GB** — 전체 모델에서 가장 큰 비중.

---

#### Normalization 가중치 (레이어당 × 35)

| 키             | 형태           | 35레이어 합계 |
| -------------- | -------------- | ------------- |
| `input_ln`     | [2048] float32 | 0.28 MB       |
| `post_attn_ln` | [2048] float32 | 0.28 MB       |
| `pre_ffn_ln`   | [2048] float32 | 0.28 MB       |
| `post_ffn_ln`  | [2048] float32 | 0.28 MB       |
| `laurel_norm`  | [2048] float32 | 0.28 MB       |
| `ple_post_ln`  | [2048] float32 | 0.28 MB       |

모든 Norm 가중치 합계 ≈ **1.7 MB** — 무시 가능한 수준.

---

#### LAuReL 가중치 (레이어당 × 35)

| 키             | 행렬 형태 (packed) | INT4 실제 차원 | 35레이어 합계 |
| -------------- | ------------------ | -------------- | ------------- |
| `laurel_left`  | 64 × 1024 uint8    | 64 × 2048      | 2.2 MB        |
| `laurel_right` | 2048 × 32 uint8    | 2048 × 64      | 2.5 MB        |

저랭크 구조(64차원 병목): 2048 → 64 → 2048. 총 4.7 MB로 매우 경량.

---

#### AltUp 가중치 (레이어당 × 35)

| 키             | 형태              | 35레이어 합계 |
| -------------- | ----------------- | ------------- |
| `altup_rn`     | [2048] float32    | 0.28 MB       |
| `altup_router` | [2048, 4] float32 | 2.19 MB       |
| `altup_pred`   | [16, 4] float32   | 0.01 MB       |
| `altup_corr`   | [4, 4] float32    | 0.007 MB      |
| `altup_scale`  | [2048] float32    | 0.28 MB       |

AltUp 전체 합계 ≈ **2.8 MB** — 4-스트림 구조임에도 매우 경량.

---

#### PLE 가중치 (레이어당 × 35)

| 키            | 형태                              | 35레이어 합계 | 비고           |
| ------------- | --------------------------------- | ------------- | -------------- |
| `ple_gate`    | 256 × 1024 uint8 (INT4: 256×2048) | 8.8 MB        | INT4           |
| `ple_proj`    | **512 × 2048 float32**            | **70 MB**     | float32 유지 |
| `ple_post_ln` | [2048] float32                    | 0.28 MB       | —              |

> **`ple_proj`는 현재 float32로 유지** — 전체 모델에서 유일하게 INT4 변환이 적용되지 않은 대형 가중치입니다. INT4로 양자화 시 ~17 MB로 감소 가능한 미구현 최적화 포인트입니다.

---

#### 전역 가중치 (레이어 독립)

| 키              | 형태                                    | 크기           | 비고         |
| --------------- | --------------------------------------- | -------------- | ------------ |
| `W_embed`       | 262400 × 1024 uint8 (INT4: 262400×2048) | **257.3 MB**   | mmap, Tied   |
| `W_lm_head`     | (W_embed와 동일 객체)                   | 0 MB 추가      | Tied Weights |
| `W_ple`         | 262144 × 4480 uint8 (INT4: 262144×8960) | **1,121.0 MB** | mmap         |
| `W_ple_proj`    | 8960 × 1024 uint8 (INT4: 8960×2048)     | 8.8 MB         | INT4         |
| `norm_ple`      | [256] float32                           | 0.001 MB       | —            |
| `altup_projs`   | List[3] × [2048, 2048] float32          | 96.0 MB        | float32 유지 |
| `altup_unprojs` | List[3] × [2048, 2048] float32          | 96.0 MB        | float32 유지 |
| `W_final_norm`  | [2048] float32                          | 0.008 MB       | —            |

> **`W_ple` (1,121 MB)**가 전역 가중치 중 가장 큼. mmap 덕분에 실제 메모리는 접근한 행만큼만 소비.

---

### 전체 합산 및 카테고리별 분포

| 카테고리                               | 합계 (MB)     | 비율    |
| -------------------------------------- | ------------- | ------- |
| FFN (W_gate + W_up + W_down) × 35      | ~1,685        | **51%** |
| W_ple (전역, mmap)                     | ~1,121        | **34%** |
| W_embed / W_lm_head (Tied, mmap)       | ~257          | 8%      |
| altup_projs + altup_unprojs            | ~192          | 6%      |
| ple_proj (float32 미최적화)            | ~140          | 4%      |
| Attention (W_q + W_k + W_v + W_o) × 35 | ~176          | 5%      |
| 나머지 (Norm, AltUp, LAuReL 등)        | ~20           | 1%      |
| **총합 (논리적)**                      | **~3,591 MB** | 109%      |

> `W_embed`와 `W_lm_head`는 Tied Weights로 동일 객체 → 실제 RAM에는 257 MB만 점유.  
> `W_ple`, `W_embed`는 mmap → 접근 패턴에 따라 실제 메모리는 훨씬 적을 수 있음.

---

### 최적화 포인트 식별

이 표에서 도출할 수 있는 미구현 최적화 항목:

| 항목                              | 현재   | 최적화 후 | 절감        |
| --------------------------------- | ------ | --------- | ----------- |
| `ple_proj` float32 → INT4         | 140 MB | ~17 MB    | **-123 MB** |
| `altup_projs` float32 → float16   | 96 MB  | 48 MB     | -48 MB      |
| `altup_unprojs` float32 → float16 | 96 MB  | 48 MB     | -48 MB      |

---

### 파일 생성 방법

```python
# Optim_tensor_load.py 상단부의 debug() 함수를 직접 호출

# if __name__ == "__main__":
#     debug()   ← 주석 해제 후 실행

# 출력: 마크다운 표 형식으로 stdout에 출력
# 캡처 후 optim_tensor_size.md 로 저장
python Optim_tensor_load.py > optim_tensor_size.md
```

---

## 전체 문서화 완료 요약 (1/8 ~ 8/8)

| 회차 | 파일                                           | 핵심 내용                                      |
| ---- | ---------------------------------------------- | ---------------------------------------------- |
| 1/8  | `vulkan_core.cpp` + `my_accelerator.cpp`       | Vulkan 핑퐁 버퍼 구조, C++ SIMD 커널 6종       |
| 2/8  | `gemv_int4_vector4.comp` + `gemv_int4.comp`    | GLSL uvec4 최적화, INT4 언패킹 로직            |
| 3/8  | `CPU_CORE.py` + `CPU_MATRIX_CORE.py`           | 어텐션/RoPE/임베딩 CPU 연산, 출력 버퍼 풀      |
| 4/8  | `IGPU_CORE.py` + `main.py`                     | Vulkan Python 바인딩, 35레이어 전체 파이프라인 |
| 5/8  | `safeTensor.py` + `Optim_tensor_load.py`       | mmap 전략, safetensors → npy 변환              |
| 6/8  | `quantize.py` + `Memory_Manager.py`            | INT4 대칭 양자화, KV 캐시 사전 할당            |
| 7/8  | `NPU_CORE.py` + `gemma3N_E4B_architecture.md`  | FPGA Systolic Array, 전체 아키텍처 설계        |
| 8/8  | `chat_template.jinja` + `optim_tensor_size.md` | 대화 직렬화 포맷, 메모리 사용량 기준표         |
