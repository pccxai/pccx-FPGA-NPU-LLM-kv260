import taichi as ti
import numpy as np
import math
import gc

ti.init(arch=ti.vulkan, fast_math=True)

TILE_K = 128

# ================================================================
# Taichi GEMV 커널 (atomic_add 제거 버전 - 이전 최적화 유지)
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
#  메모리 최적화: WeightProxy
#
# float16 원본 행렬을 VRAM(INT16)에 올린 뒤 RAM에서 해제.
# layers["W_q"][i] 등을 이 객체로 교체 → float16 배열이 GC됨.
# .shape만 있으면 igpu_matmul이 정상 동작.
# ================================================================
class WeightProxy:
    """VRAM 업로드 완료 후 float16 원본 대신 쓰는 경량 프록시."""
    __slots__ = ("stable_id", "shape")

    def __init__(self, stable_id: int, shape: tuple):
        self.stable_id = stable_id
        self.shape     = shape   # igpu_matmul에서 M, N = weight.shape 가 필요

# ================================================================
#  VRAM 캐시: 안정적인 정수 ID 기반
#
# 기존 id(weight_mat) 방식의 문제:
#   - float16 원본 해제 후 id()가 다른 객체에 재사용될 수 있음
#   - 그 경우 캐시 미스 → 재업로드
#
# 새 방식: 업로드 시 순차 정수 stable_id 부여 → 영구 고유
#   _VRAM_CACHE[stable_id] = (ti_mat, ti_scale)
#   _OBJID_TO_STABLE[id(weight_mat)] = stable_id  (업로드 당시만 유효)
# ================================================================
_VRAM_CACHE:       dict = {}   # stable_id -> (ti_mat, ti_scale)
_OBJID_TO_STABLE:  dict = {}   # id(array) -> stable_id
_OUTPUT_BUF_POOL:  dict = {}   # output_size -> ti.ndarray
_next_stable_id = [0]          # mutable counter (리스트로 클로저 우회)

def _get_or_upload_weight(weight_mat):
    """
    WeightProxy이면 VRAM 캐시에서 즉시 반환.
    float16/32 numpy 배열이면 INT16으로 양자화 후 VRAM 업로드.
    이후 동일 배열 재호출 시 캐시에서 반환.
    """
    # --- WeightProxy 경로: 이미 업로드됨 ---
    if isinstance(weight_mat, WeightProxy):
        return _VRAM_CACHE[weight_mat.stable_id]

    # --- numpy 배열 경로 ---
    obj_id = id(weight_mat)
    if obj_id in _OBJID_TO_STABLE:
        stable_id = _OBJID_TO_STABLE[obj_id]
        return _VRAM_CACHE[stable_id]

    # 처음 보는 가중치 → INT16 양자화 후 VRAM 업로드
    stable_id = _next_stable_id[0]
    _next_stable_id[0] += 1

    # float16 또는 float32 입력 모두 처리 가능
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
#  메모리 최적화 핵심 함수: preload_and_free
#
# 동작:
#   1. W[key][i] (float16 배열)를 모두 VRAM에 INT16으로 업로드
#   2. layers dict의 해당 항목을 WeightProxy로 교체
#   3. gc.collect() → float16 배열 메모리 반환
#
# 효과: 토큰당 490회 from_numpy → 0회
#        float16 큰 행렬 ~3.5GB → RAM 해제
# ================================================================
def preload_and_free(W: dict, keys: list):
    total = sum(len(W[k]) for k in keys if k in W)
    uploaded = 0

    for key in keys:
        if key not in W:
            continue
        proxies = []
        for w in W[key]:
            ti_mat, ti_scale = _get_or_upload_weight(w)   # VRAM 업로드
            obj_id    = id(w)
            stable_id = _OBJID_TO_STABLE[obj_id]
            proxies.append(WeightProxy(stable_id, w.shape))
            uploaded += 1
        W[key] = proxies   # float16 배열 참조 해제 → GC 대상
        print(f"  [{key}] {len(proxies)}개 → VRAM 이전 완료")

    gc.collect()
    print(f"[메모리] 총 {uploaded}/{total}개 가중치 VRAM 이전 & float 원본 해제 ✓")

# ================================================================
# Public API (기존과 동일한 시그니처 유지)
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
    JIT 컴파일 + 워밍업.
    더미 행렬 ID가 실제 가중치와 충돌하지 않도록
    워밍업 완료 후 _OBJID_TO_STABLE 에서 제거.
    """
    print("[iGPU] JIT 워밍업 + INT16 NPU 시뮬레이션 모드 가동 중...")
    dummy_x     = np.zeros(2048, dtype=np.float32)
    dummy_w     = np.random.randn(2048, 2048).astype(np.float32)
    dummy_w_big = np.random.randn(2048, 8192).astype(np.float32)

    igpu_matmul(dummy_x, dummy_w)
    igpu_matmul_gelu(dummy_x, dummy_w_big)

    # 더미 항목 제거 (id 재사용으로 실제 가중치 캐시 오염 방지)
    for dummy in [dummy_w, dummy_w_big]:
        d_id = id(dummy)
        if d_id in _OBJID_TO_STABLE:
            del _OBJID_TO_STABLE[d_id]

    print("[iGPU] 워밍업 완료! ")
