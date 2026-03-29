Gemma 3N (INT4 + AltUp) Pipeline 상세 연산 흐름도 (HW 최적화 반영본)

이 문서는 파이프라인에서 사용되는 기본 연산들의 실제 수학적 동작을 정의하고, **KV260 NPU 가속기 및 메모리 병목 현상을 해결하기 위한 하드웨어-소프트웨어 공동 설계(Co-design) 최적화 기법**이 적용된 파이프라인 단계를 순서대로 설명한다.
기본 차원은 $D = 2048$, 라우터 차원은 $D_{mod}$, 패치 임베딩 차원은 $256$ 으로 가정하며, 어텐션 헤드는 **Q헤드 8개, KV헤드 2개(Head Dim=256)** 로 구성된다.

### 0. 사전 정의: 핵심 함수들의 실제 수학 연산

파이프라인 전체에서 반복적으로 사용되는 함수들이 내부적으로 어떤 연산을 수행하는지 정의한다.

**Embedding (임베딩)**: 단어의 ID(정수)를 받아 미리 학습된 거대한 가중치 행렬에서 해당 ID번째 줄(행)을 통째로 뽑아오는 연산이다.

$$
Output = W_{embed}[token\_id, :]
$$

**RMSNorm (Root Mean Square Normalization)**: 입력 벡터의 값들이 너무 커지거나 작아지지 않게 평균적인 크기로 나누어주고, 학습 가능한 가중치( $\gamma$ )를 곱해주는 연산이다.

$$
RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_{i}^{2} + 10^{-6}}
$$

$$
Output = \left(\frac{x}{RMS}\right) \times \gamma
$$

**GELU (Gaussian Error Linear Unit)**: 비선형 활성화 함수.

$$
Output = 0.5 \times x \times \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^{3})\right)\right)
$$

**ROPE (Rotary Position Embedding)**: 위치 정보를 위한 회전 변환.

$$
Output_{2i} = x_{2i} \times \cos(\theta) - x_{2i+1} \times \sin(\theta)
$$

$$
Output_{2i+1} = x_{2i} \times \sin(\theta) + x_{2i+1} \times \cos(\theta)
$$

### 1. 토큰 임베딩 (Token Embedding) & 메모리 티어링

가장 먼저 들어온 정수 형태의 토큰 ID를 벡터로 변환한다.

**[ HW 최적화 포인트 1: 상수 폴딩 (Constant Folding) ]**
기존 알고리즘의 $\times \sqrt{2048.0}$ 연산은 런타임에 수행하지 않는다. 오프라인(Python Host)에서 모델 가중치를 추출할 때 $W_{embed}$ 자체에 미리 곱해둔 $W_{embed\_precal}$ 을 사용한다. (DSP 소모 0)

**[ HW 최적화 포인트 2: Vocab Pruning & 메모리 티어링 (RAM vs USB) ]**
전체 25.6만 개의 토큰을 모두 램에 올리면 약 1GB의 용량이 필요하여 메모리 병목이 발생한다.

* **자주 쓰는 토큰 (한/영/특수기호 약 5만 개):** KV260 보드의 **DDR4 RAM**에 상주시켜 NPU가 대기 시간 없이 즉각 접근.

* **안 쓰는 토큰 (기타 언어):** 보드에 연결된 **USB/SSD에 메모리 맵(mmap)** 형태로 배치하여, 희귀 토큰 입력 시에만 Zero-copy로 읽어오도록 분기 처리.

**연산 과정:**

$$
x_{0} = Embedding(token\_id, W_{embed\_precal})
$$

*입력: int (스칼라 값 1개)*
*출력 크기: $ 1 \times 2048 $*

### 2. AltUp 초기 투영 (AltUp Initial Projections)

원본 벡터 $x_{0}$ 를 3개의 서로 다른 가중치 행렬과 곱해서 총 4개의 모달리티 벡터 모음( $ xs $ )을 만든다.

**연산 과정:**

$$
xs_{1} = x_{0} \cdot altup\_projs[0]
$$

$$
xs_{2} = x_{0} \cdot altup\_projs[1]
$$

$$
xs_{3} = x_{0} \cdot altup\_projs[2]
$$

*출력 크기: 완성된 $ xs \to 4 \times 2048 $*

### 3. 위치 및 패치 임베딩 세팅 (PLE Setup)

35개의 트랜스포머 레이어 전체에서 사용할 보조 벡터( $pli\_all$ )를 미리 한 번에 계산해 둔다.

**[ HW 최적화 포인트 3: 수학적 상쇄 (Mathematical Cancellation) ]**
원래 $x_{proj}$ 를 구할 때 $\div \sqrt{2048.0}$ 을 수행해야 하지만, 바로 다음 연산이 RMSNorm이므로 스케일 값이 분모/분자에서 완벽히 상쇄된다. **따라서 무거운 하드웨어 나눗셈기를 완전히 생략한다.**

**[ HW 최적화 포인트 4: 사전 연산 병합 ]**
$pli\_all$ 을 구할 때 마지막에 곱해지는 $\times \frac{1}{\sqrt{2.0}}$ 와 $y$ 를 구할 때 곱해지는 $\times \sqrt{256.0}$ (또는 4-bit left shift) 연산 역시 런타임에 하지 않는다. 오프라인에서 $norm_{ple}$ 와 $W_{ple\_packed}$ 에 미리 상수를 모두 병합해 둔 가중치( `_precal` )를 사용한다.

**연산 과정 (나눗셈 및 스칼라 곱셈 제거 완료):**

$$
x_{proj} = x_{0} \cdot W_{ple\_proj}
$$

*(결과를* $35 \times 256$ *크기로 reshape)*

$$
x_{proj\_normed} = RMSNorm(x_{proj}) \times norm_{ple\_precal}
$$

$$
y = Embedding(token\_id, W_{ple\_packed\_precal})
$$

**최종 하드웨어 연산 (단순 덧셈 1회로 단축):**

$$
pli\_all = x_{proj\_normed} + y
$$

*출력 크기: $ pli_all \to 35 \times 256 $*

### 4. 트랜스포머 레이어 (35번 반복)

A. AltUp 라우터 및 혼합 (Router & Pred)

$$
x_{n} = RMSNorm(xs_{0}, W_{altup\_rn\_precal})
$$

*(기존 나눗셈* $\div 2048.0$ *은 가중치에 미리 반영됨)*

$$
modalities = \tanh(x_{n} \cdot W_{altup\_router})
$$

$$
coef\_mat = (W_{altup\_pred} \cdot modalities).reshape(4, 4)
$$

$$
xs_{pred} = xs + (coef\_mat \cdot xs)
$$

B. 어텐션 (Attention Q, K, V & GQA)

$$
x_{input} = xs_{pred}[0]
$$

$$
x_{norm} = RMSNorm(x_{input}, W_{input\_ln})
$$

$$
Q = x_{norm} \cdot W_{q}, \quad K = x_{norm} \cdot W_{k}, \quad V = x_{norm} \cdot W_{v}
$$

$$
Q^{head}_{i} = \frac{Q^{head}_{i}}{RMS(Q^{head}_{i})} \times \gamma_{q}, \quad K^{head}_{j} = \frac{K^{head}_{j}}{RMS(K^{head}_{j})} \times \gamma_{k}
$$

$$
Q_{rope} = ROPE(Q_{norm}, \theta), \quad K_{rope} = ROPE(K_{norm}, \theta)
$$

$$
attn\_raw = GQA(Q_{rope}, target\_K\_cache, target\_V\_cache)
$$

$$
attn\_output = attn\_raw \cdot W_{o}
$$

C. Laurel 보조 신경망 및 어텐션 출력 결합

$$
laurel\_x = (x_{norm} \cdot W_{laurel\_left}) \cdot W_{laurel\_right}
$$

$$
laurel\_out\_normed = \mathbf{x_{norm}} + RMSNorm(laurel\_x, W_{laurel\_norm\_precal})
$$

$$
attn\_output\_normed = RMSNorm(attn\_output, W_{post\_attn\_ln\_precal}) + x_{input\_precal}
$$

**최종 결합 (상수 곱 연산 제거):**

$$
x_{attn} = attn\_output\_normed + laurel\_out\_normed
$$

D. 피드포워드 네트워크 (FFN - Gate, Up, Down)

$$
x_{n2} = RMSNorm(x_{attn}, W_{pre\_ffn\_ln})
$$

$$
gate\_raw = x_{n2} \cdot W_{gate}
$$

$$
up\_out = x_{n2} \cdot W_{up}
$$

*(레이어 10 기준 분기 처리 생략 - 내부 구현 동일)*

$$
hidden = GELU(gate\_raw) \times up\_out
$$

$$
mlp\_out = hidden \cdot W_{down}
$$

$$
outputs = RMSNorm(mlp\_out, W_{post\_ffn\_ln}) + x_{attn}
$$

E. 모달리티 업데이트 (AltUp Correction)

$$
activated = outputs \times W_{altup\_scale}
$$

$$
innovation = activated - xs_{pred}[0]
$$

$$
x_{n3} = RMSNorm(activated, W_{altup\_rn\_precal})
$$

$$
mod\_corr = \tanh(x_{n3} \cdot W_{altup\_router})
$$

$$
corr\_coefs = (W_{altup\_corr} \cdot mod\_corr) + 1.0
$$

$$
xs_{new} = xs_{pred} + (corr\_coefs_{[:,1]} \times innovation_{[1,:]})
$$

$$
gate\_ple = GELU(activated \cdot W_{ple\_gate}) \times pli
$$

$$
mapped = RMSNorm(gate\_ple \cdot W_{ple\_proj}, W_{ple\_post\_ln})
$$

$$
xs_{new}[1:] = xs_{new}[1:] + mapped
$$

### 5. 로짓 디코딩 (Decode Logits)

$$
target\_mag = \sqrt{Mean(xs[0]^{2})}
$$

$$
proj\_x_{k} = xs[k+1] \cdot altup\_unprojs[k] \quad (k=0,1,2)
$$

$$
new\_mag_{k} = \sqrt{Mean(proj\_x_{k}^{2})}
$$

$$
proj\_x_{k} = proj\_x_{k} \times \frac{target\_mag}{\max(new\_mag_{k}, 10^{-12})}
$$

$$
x_{final} = Mean([xs[0], proj\_x_{0}, proj\_x_{1}, proj\_x_{2}])
$$

$$
x_{final\_norm} = RMSNorm(x_{final}, W_{final\_norm})
$$

$$
Logits\_Raw = x_{final\_norm} \cdot W_{lm\_head}
$$

$$
Logits = 30.0 \times \tanh\left(\frac{Logits\_Raw}{30.0}\right)
$$

### 6. 샘플링 로직 (Generation & Sampling)

**Repetition Penalty (반복 패널티)**:

$$
Logits_{t} = Logits_{t} \times \rho \quad (\text{if } Logits_{t} < 0)
$$

$$
Logits_{t} = \frac{Logits_{t}}{\rho} \quad (\text{if } Logits_{t} \ge 0)
$$

**Temperature Softmax**:

$$
probs_i = \frac{\exp(Logits_i / T)}{\sum \exp(Logits_j / T)}
$$

*(이후 Top-P Cut-off 및 랜덤 샘플링 진행)*

### 7. 시스템 및 메모리 최적화 아키텍처 (Hardware Integration)

* **오프라인 가중치 전처리 (Pre-computation):** 런타임 스칼라 곱셈( $ \sqrt{2048} $, $1/\sqrt{2}$ 등 )을 모두 가중치 텐서에 사전 병합하여 연산 사이클과 DSP 자원 소모를 0으로 만듦.

* **어휘 사전 분할 (Vocab Tiering):** 자주 쓰이는 한국어/영어 토큰은 고속 DDR4 램에, 미사용 희귀 토큰은 SSD/USB에 MMAP으로 구성하여 대역폭 한계 극복.

* **나눗셈 연산 스킵 (Math Cancellation):** RMSNorm 직전의 스케일 나눗셈을 수학적 원리에 기반하여 하드웨어에서 전면 생략.

* **Ping-Pong 더블 버퍼링:** 가속기가 행렬곱을 계산하는 동안 백그라운드 스레드에서 다음 계산용 가중치를 프리패치.

* **In-place 메모리 덮어쓰기:** RMSNorm, GELU 커널 최적화로 추가 메모리 할당 방지.

### 8. 메모리 최적화 및 텐서 RAM 상주 사이즈 변화

HW 설계 및 배포(Serving) 환경에서 메모리 병목 현상을 방지하기 위해 각 텐서들이 차지하는 RAM 상주 용량을 분석한 표이다.
가장 큰 용량을 차지하는 거대 어휘 텐서(`W_embed`, `W_ple`, `W_lm_head`)의 경우, 자주 쓰는 어휘 **약 5만 개만 RAM에 상주(Caching)** 시키고 나머지는 USB로 분리(MMAP Zero-copy)하는 **메모리 티어링**을 적용하여 용량을 극적으로 압축했다.

타입 변환 시 주의점: **BF16 모델**은 기본 Float32 텐서들의 용량을 절반으로 줄여주지만, INT4 양자화가 들어간 Tuple 텐서의 경우 압축된 INT4 파츠는 그대로고 뒤에 붙는 스케일(Scale) 벡터에만 BF16이 적용되므로 용량 감소가 미미하다.

| 텐서명          | 파라미터 구조 및 기본 타입                | 최적화 내역                      | 기존 RAM (FP32) | 최적화 상주 RAM (FP32) | 최적화 상주 RAM (BF16) |
| --------------- | ----------------------------------------- | -------------------------------- | --------------- | ---------------------- | ---------------------- |
| `altup_rn`      | List\[35\] ──> \[ 2048 , float32 \]       | `_precal` 연산 병합              | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `altup_router`  | List\[35\] ──> \[ 2048 x 4 , float32 \]   | 유지                             | 2.192 MB        | 2.192 MB               | **1.096 MB**           |
| `altup_pred`    | List\[35\] ──> \[ 16 x 4 , float32 \]     | 유지                             | 0.013 MB        | 0.013 MB               | **0.006 MB**           |
| `input_ln`      | List\[35\] ──> \[ 2048 , float32 \]       | 유지                             | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `W_q`           | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 70.284 MB       | 70.284 MB              | **70.144 MB**          |
| `W_k`           | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 17.579 MB       | 17.579 MB              | **17.540 MB**          |
| `W_v`           | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 17.579 MB       | 17.579 MB              | **17.540 MB**          |
| `gamma_q`       | List\[35\] ──> \[ 256 , float32 \]        | 유지                             | 0.038 MB        | 0.038 MB               | **0.019 MB**           |
| `gamma_k`       | List\[35\] ──> \[ 256 , float32 \]        | 유지                             | 0.038 MB        | 0.038 MB               | **0.019 MB**           |
| `W_o`           | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 70.284 MB       | 70.284 MB              | **70.144 MB**          |
| `laurel_left`   | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 2.206 MB        | 2.206 MB               | **2.190 MB**           |
| `laurel_right`  | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 2.471 MB        | 2.471 MB               | **2.200 MB**           |
| `laurel_norm`   | List\[35\] ──> \[ 2048 , float32 \]       | `_precal` 연산 병합              | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `post_attn_ln`  | List\[35\] ──> \[ 2048 , float32 \]       | `_precal` 연산 병합              | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `pre_ffn_ln`    | List\[35\] ──> \[ 2048 , float32 \]       | 유지                             | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `W_gate`        | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 562.198 MB      | 562.198 MB             | **561.108 MB**         |
| `W_up`          | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 562.198 MB      | 562.198 MB             | **561.108 MB**         |
| `W_down`        | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 560.284 MB      | 560.284 MB             | **559.194 MB**         |
| `post_ffn_ln`   | List\[35\] ──> \[ 2048 , float32 \]       | 유지                             | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `altup_scale`   | List\[35\] ──> \[ 2048 , float32 \]       | 유지                             | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `altup_corr`    | List\[35\] ──> \[ 4 x 4 , float32 \]      | 유지                             | 0.007 MB        | 0.007 MB               | **0.003 MB**           |
| `ple_gate`      | List\[35\] ──> Tuple(uint8, float32)      | 유지 (Scale만 반갈)              | 8.794 MB        | 8.794 MB               | **8.760 MB**           |
| `ple_proj`      | List\[35\] ──> \[ 256 x 2048 , float32 \] | 유지                             | 140.005 MB      | 140.005 MB             | **70.002 MB**          |
| `ple_post_ln`   | List\[35\] ──> \[ 2048 , float32 \]       | 유지                             | 0.277 MB        | 0.277 MB               | **0.138 MB**           |
| `W_embed`       | Tuple( \[262400 x 1024, uint8\], ... )    | `_precal` & **5만 토큰 Pruning** | 257.251 MB      | **49.020 MB**          | **48.920 MB**          |
| `W_ple`         | Tuple( \[262144 x 4480, uint8\], ... )    | `_precal` & **5만 토큰 Pruning** | 1121.000 MB     | **213.810 MB**         | **213.710 MB**         |
| `norm_ple`      | \[ 256 , float32 \]                       | `_precal` 연산 병합              | 0.001 MB        | 0.001 MB               | **0.000 MB**           |
| `W_ple_proj`    | Tuple( \[8960 x 1024, uint8\], ... )      | `_precal` 연산 병합              | 8.784 MB        | 8.784 MB               | **8.750 MB**           |
| `altup_projs`   | List\[3\] ──> \[ 2048 x 2048 , float32 \] | 유지                             | 96.000 MB       | 96.000 MB              | **48.000 MB**          |
| `altup_unprojs` | List\[3\] ──> \[ 2048 x 2048 , float32 \] | 유지                             | 96.000 MB       | 96.000 MB              | **48.000 MB**          |
| `W_final_norm`  | \[ 2048 , float32 \]                      | 유지                             | 0.008 MB        | 0.008 MB               | **0.004 MB**           |
| `W_lm_head`     | Tuple( \[262400 x 1024, uint8\], ... )    | **5만 토큰 Pruning**             | 257.251 MB      | **49.020 MB**          | **48.920 MB**          |

> **최종 요약**
>
> * 기존 램 상주 용량 합계: 약 **3,500 MB** (3.5 GB)
>
> * 최적화 후 (FP32 기준): 약 **2,150 MB**
>
> * 최적화 후 (BF16 변환 기준): 약 **2,050 MB**