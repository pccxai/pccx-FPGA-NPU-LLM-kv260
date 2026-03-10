import taichi as ti
import numpy as np
import math

ti.init(arch=ti.vulkan, fast_math=True)

TILE_K = 128

# 🚀 NPU(FPGA) 완벽 모사: INT4 (W4A16) 초고속 GEMV 커널
@ti.kernel
def _gemv_int4_packed(
    vec:          ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_packed:   ti.types.ndarray(dtype=ti.u8,  ndim=2), # type: ignore
    scales:       ti.types.ndarray(dtype=ti.f32, ndim=2), # type: ignore
    out:          ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out_features: ti.i32, # type: ignore
    in_features:  ti.i32 # type: ignore
):
    ti.loop_config(block_dim=128)
    for i in range(out_features):
        sum_val = ti.f32(0.0)
        # 1바이트에 2개의 가중치가 들어있으므로 절반만 돎
        for j_half in range(in_features // 2):
            j_base = j_half * 2
            # 안전하게 i32로 캐스팅 후 비트 연산 수행
            packed_val = ti.cast(mat_packed[i, j_half], ti.i32)

            # [하드웨어 모사] 하위 4비트 추출 및 부호 확장
            low_4 = packed_val & 15
            low_val = ti.f32(low_4)
            if low_4 > 7:
                low_val -= 16.0
            
            # [하드웨어 모사] 상위 4비트 추출 및 부호 확장
            high_4 = (packed_val >> 4) & 15
            high_val = ti.f32(high_4)
            if high_4 > 7:
                high_val -= 16.0
            
            # 32개 단위 그룹 스케일 적용
            scale = scales[i, j_base // 32]
            
            sum_val += vec[j_base] * low_val * scale
            sum_val += vec[j_base + 1] * high_val * scale
            
        out[i] = sum_val

@ti.kernel
def _gemv_int4_packed_gelu(
    vec:          ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_packed:   ti.types.ndarray(dtype=ti.u8,  ndim=2), # type: ignore
    scales:       ti.types.ndarray(dtype=ti.f32, ndim=2), # type: ignore
    out:          ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out_features: ti.i32,# type: ignore
    in_features:  ti.i32 # type: ignore
):
    ti.loop_config(block_dim=128)
    for i in range(out_features):
        sum_val = ti.f32(0.0)
        for j_half in range(in_features // 2):
            j_base = j_half * 2
            packed_val = ti.cast(mat_packed[i, j_half], ti.i32)

            # if문 다 날리고 비트 시프트로 부호 확장!
            # 하위 4비트: 왼쪽 28칸 밀고 오른쪽 28칸 밀면 자동으로 -8~+7 부호가 생김
            low_val = ti.cast((packed_val << 28) >> 28, ti.f32)
            
            # 상위 4비트: 왼쪽 24칸 밀고 오른쪽 28칸 밀기
            high_val = ti.cast((packed_val << 24) >> 28, ti.f32)
            
            scale = scales[i, j_base // 32]
            sum_val += vec[j_base] * low_val * scale
            sum_val += vec[j_base + 1] * high_val * scale
            
        # 커널 융합: GeLU 인라인 처리
        v = sum_val
        out[i] = 0.5 * v * (1.0 + ti.tanh(
            ti.sqrt(ti.f32(2.0 / math.pi)) * (v + ti.f32(0.044715) * v * v * v)
        ))

# 기존 INT16 커널 (양자화 안 한 원본 가중치용)
@ti.kernel
def _gemv_int16(
    vec:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_t:  ti.types.ndarray(dtype=ti.i16, ndim=2), # type: ignore
    scale:  ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    M: ti.i32, # type: ignore
    N: ti.i32  # type: ignore
):
    ti.loop_config(block_dim=TILE_K)
    for i, k_base in ti.ndrange(N, (M + TILE_K - 1) // TILE_K):
        partial = ti.f32(0.0)
        for dk in range(TILE_K):
            k = k_base * TILE_K + dk
            if k < M:
                partial += vec[k] * ti.cast(mat_t[i, k], ti.f32)
        ti.atomic_add(out[i], partial * scale[i])

@ti.kernel
def _gemv_int16_gelu(
    vec:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    mat_t:  ti.types.ndarray(dtype=ti.i16, ndim=2), # type: ignore
    scale:  ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    out:    ti.types.ndarray(dtype=ti.f32, ndim=1), # type: ignore
    M: ti.i32, # type: ignore
    N: ti.i32  # type: ignore
):
    ti.loop_config(block_dim=TILE_K)
    for i, k_base in ti.ndrange(N, (M + TILE_K - 1) // TILE_K):
        partial = ti.f32(0.0)
        for dk in range(TILE_K):
            k = k_base * TILE_K + dk
            if k < M:
                partial += vec[k] * ti.cast(mat_t[i, k], ti.f32)
        ti.atomic_add(out[i], partial * scale[i])

    for i in range(N):
        v = out[i]
        out[i] = 0.5 * v * (1.0 + ti.tanh(
            ti.sqrt(ti.f32(2.0 / math.pi)) * (v + ti.f32(0.044715) * v * v * v)
        ))

@ti.kernel
def _zero_buffer(buf: ti.types.ndarray(dtype=ti.f32, ndim=1), N: ti.i32): # type: ignore
    for i in range(N):
        buf[i] = 0.0

# VRAM 폭발 방지 & 캐싱 로직
_CPU_WEIGHT_CACHE = {}  
_WEIGHT_BUF_POOL = {}   
_SCALE_BUF_POOL = {}    
_OUTPUT_BUF_POOL = {}

_INT4_WEIGHT_POOL = {}
_INT4_SCALE_POOL = {}

def _stream_int4_gpu_weight(mat_packed: np.ndarray, scales: np.ndarray):
    w_id = id(mat_packed)
    if w_id not in _INT4_WEIGHT_POOL:
        # safeTensor에서 float32로 캐스팅된 경우를 대비해 uint8로 강제 복원
        mat_u8 = mat_packed.astype(np.uint8)
        scales_f32 = scales.astype(np.float32)
        
        ti_mat = ti.ndarray(dtype=ti.u8, shape=mat_u8.shape)
        ti_scale = ti.ndarray(dtype=ti.f32, shape=scales_f32.shape)
        
        ti_mat.from_numpy(mat_u8)
        ti_scale.from_numpy(scales_f32)
        
        _INT4_WEIGHT_POOL[w_id] = ti_mat
        _INT4_SCALE_POOL[w_id] = ti_scale
        
    return _INT4_WEIGHT_POOL[w_id], _INT4_SCALE_POOL[w_id]

def _stream_int16_gpu_weight(weight_mat: np.ndarray):
    w_id = id(weight_mat)
    if w_id not in _CPU_WEIGHT_CACHE:
        mat_t = weight_mat.T.astype(np.float32)
        max_vals = np.max(np.abs(mat_t), axis=1, keepdims=True)
        max_vals = np.maximum(max_vals, 1e-8)
        
        scale = (max_vals / 32767.0).flatten().astype(np.float32)
        mat_i16 = np.clip(np.round(mat_t / max_vals * 32767.0), -32767, 32767).astype(np.int16)
        
        _CPU_WEIGHT_CACHE[w_id] = (np.ascontiguousarray(mat_i16), scale)
        
    mat_i16, scale = _CPU_WEIGHT_CACHE[w_id]
    shape = mat_i16.shape
    N = scale.shape[0]
    
    if shape not in _WEIGHT_BUF_POOL:
        _WEIGHT_BUF_POOL[shape] = ti.ndarray(dtype=ti.i16, shape=shape)
        _SCALE_BUF_POOL[shape] = ti.ndarray(dtype=ti.f32, shape=(N,))
        
    ti_mat = _WEIGHT_BUF_POOL[shape]
    ti_scale = _SCALE_BUF_POOL[shape]
    
    ti_mat.from_numpy(mat_i16) 
    ti_scale.from_numpy(scale)
    
    return ti_mat, ti_scale

def _get_output_buf(size: int) -> ti.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = ti.ndarray(dtype=ti.f32, shape=(size,))
    return _OUTPUT_BUF_POOL[size]

# Public API
def igpu_matmul(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    # 딕셔너리로 양자화 데이터가 들어오면 INT4 (W4A16) 파이프라인 탑승!
    if isinstance(weight_data, dict) and "packed" in weight_data:
        ti_mat, ti_scales = _stream_int4_gpu_weight(weight_data["packed"], weight_data["scales"])
        out_features = weight_data["packed"].shape[0]
        in_features = weight_data["packed"].shape[1] * 2
        
        buf = _get_output_buf(out_features)
        _zero_buffer(buf, out_features)
        _gemv_int4_packed(x_f32, ti_mat, ti_scales, buf, out_features, in_features)
        return buf.to_numpy()
        
    # 원본 Bfloat16 데이터면 기존 INT16 파이프라인 탑승!
    else:
        ti_w, ti_s = _stream_int16_gpu_weight(weight_data)
        M, N = weight_data.shape
        buf = _get_output_buf(N)
        _zero_buffer(buf, N)
        _gemv_int16(x_f32, ti_w, ti_s, buf, M, N)
        return buf.to_numpy()

def igpu_matmul_gelu(x_vec: np.ndarray, weight_data) -> np.ndarray:
    x_f32 = np.ascontiguousarray(x_vec.astype(np.float32))
    
    if isinstance(weight_data, dict) and "packed" in weight_data:
        ti_mat, ti_scales = _stream_int4_gpu_weight(weight_data["packed"], weight_data["scales"])
        out_features = weight_data["packed"].shape[0]
        in_features = weight_data["packed"].shape[1] * 2
        
        buf = _get_output_buf(out_features)
        _zero_buffer(buf, out_features)
        _gemv_int4_packed_gelu(x_f32, ti_mat, ti_scales, buf, out_features, in_features)
        return buf.to_numpy()
        
    else:
        ti_w, ti_s = _stream_int16_gpu_weight(weight_data)
        M, N = weight_data.shape
        buf = _get_output_buf(N)
        _zero_buffer(buf, N)
        _gemv_int16_gelu(x_f32, ti_w, ti_s, buf, M, N)
        return buf.to_numpy()

def warmup():
    print("[iGPU] JIT 워밍업 + INT4/INT16 NPU 시뮬레이션 모드 가동 중...")
    dummy_x = np.zeros(2048, dtype=np.float32)
    dummy_w = np.random.randn(2048, 2048).astype(np.float32)
    igpu_matmul(dummy_x, dummy_w)
    
    # INT4 워밍업 더미
    dummy_packed = {"packed": np.zeros((2048, 1024), dtype=np.uint8), "scales": np.ones((2048, 64), dtype=np.float32)}
    igpu_matmul_gelu(dummy_x, dummy_packed)
    print("[iGPU] warmup() complete")