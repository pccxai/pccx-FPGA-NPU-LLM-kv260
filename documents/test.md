Gemma 3N (INT4 + AltUp) Pipeline 상세 연산 흐름도 (코드 100% 일치 완벽본)

이 문서는 파이프라인에서 사용되는 기본 연산들의 실제 수학적 동작을 먼저 정의하고, 이후 각 파이프라인 단계별로 크기 변환과 연산 과정을 순서대로 설명한다.
기본 차원은 $D = 2048$, 라우터 차원은 $D_{mod}$, 패치 임베딩 차원은 $256$으로 가정하며, 어텐션 헤드는 **Q헤드 8개, KV헤드 2개(Head Dim=256)**로 구성된다.

0. 사전 정의: 핵심 함수들의 실제 수학 연산

파이프라인 전체에서 반복적으로 사용되는 함수들이 내부적으로 어떤 연산을 수행하는지 정의한다.

Embedding (임베딩): 단어의 ID(정수)를 받아 미리 학습된 거대한 가중치 행렬에서 해당 ID번째 줄(행)을 통째로 뽑아오는 연산이다.

$$Output = W_{embed}[token\_id, :]$$

RMSNorm (Root Mean Square Normalization): 입력 벡터의 값들이 너무 커지거나 작아지지 않게 평균적인 크기로 나누어주고, 학습 가능한 가중치($\gamma$)를 곱해주는 연산이다.

$$RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_{i}^{2} + 10^{-6}}$$

$$Output = \left(\frac{x}{RMS}\right) \times \gamma$$

GELU (Gaussian Error Linear Unit): 단순히 0 이하를 버리는 ReLU와 달리, 정규분포를 이용해 부드럽게 꺾이는 비선형 활성화 함수다. 근사 수식은 다음과 같다.

$$Output = 0.5 \times x \times \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^{3})\right)\right)$$

ROPE (Rotary Position Embedding): 단어의 위치 정보를 주기 위해 짝수/홀수 인덱스별로 회전 변환($\sin, \cos$)을 곱해주는 연산이다.

$$Output_{2i} = x_{2i} \times \cos(\theta) - x_{2i+1} \times \sin(\theta)$$

$$Output_{2i+1} = x_{2i} \times \sin(\theta) + x_{2i+1} \times \cos(\theta)$$

1. 토큰 임베딩 (Token Embedding)

가장 먼저 들어온 정수 형태의 토큰 ID를 벡터로 변환하고 스케일을 키워준다.

연산 과정:

$$x_{0} = Embedding(token\_id, W_{embed}) \times \sqrt{2048.0}$$

입력: int (스칼라 값 1개)

가중치 크기: $Vocab\_Size \times 2048$ (타입: INT4 튜플)

출력 크기: $1 \times 2048$

2. AltUp 초기 투영 (AltUp Initial Projections)

원본 벡터 $x_{0}$를 3개의 서로 다른 가중치 행렬과 곱해서 총 4개의 모달리티 벡터 모음($xs$)을 만든다.

연산 과정: $xs$라는 빈 행렬의 0번째 행에 $x_{0}$를 넣고, 1~3번째 행은 내적(dot product)으로 채운다.

$$xs_{1} = x_{0} \cdot altup\_projs[0]$$

$$xs_{2} = x_{0} \cdot altup\_projs[1]$$

$$xs_{3} = x_{0} \cdot altup\_projs[2]$$

출력 크기: 완성된 $xs \to 4 \times 2048$

3. 위치 및 패치 임베딩 세팅 (PLE Setup)

35개의 트랜스포머 레이어 전체에서 사용할 보조 벡터($pli\_all$)를 미리 한 번에 계산해 둔다.

연산 과정:

$$x_{proj} = \frac{x_{0} \cdot W_{ple\_proj}}{\sqrt{2048.0}}$$

위 수식의 결과를 $35 \times 256$ 크기로 리쉐이프(reshape) 한다. 그 다음 이 값을 정규화하고 패치 임베딩 값을 더해준다.

$$x_{proj\_normed} = RMSNorm(x_{proj}) \times norm_{ple}$$

$$y = Embedding(token\_id, W_{ple\_packed}) \times \sqrt{256.0}$$

$$pli\_all = (x_{proj\_normed} + y) \times \frac{1}{\sqrt{2.0}}$$

출력 크기: $pli\_all \to 35 \times 256$

4. 트랜스포머 레이어 (35번 반복)

루프를 돌며 35번 반복되는 핵심 구간이다.

A. AltUp 라우터 및 혼합 (Router & Pred)

4개의 벡터가 담긴 $xs$를 서로 섞어주는 과정이다.

$$x_{n} = \frac{RMSNorm(xs_{0}, W_{altup\_rn})}{2048.0}$$

$$modalities = \tanh(x_{n} \cdot W_{altup\_router})$$

$$coef\_mat = (W_{altup\_pred} \cdot modalities).reshape(4, 4)$$

$$xs_{pred} = xs + (coef\_mat \cdot xs)$$

B. 어텐션 (Attention Q, K, V & GQA)

섞인 벡터 중 첫 번째 것을 꺼내어 입력으로 쓴다.

$$x_{input} = xs_{pred}[0]$$

$$x_{norm} = RMSNorm(x_{input}, W_{input\_ln})$$

Q, K, V 투영 및 Head-wise QK-Norm:
Q와 K 행렬 전체를 한 번에 정규화하지 않고, 256차원(Head 크기) 단위로 그룹을 나누어 각각 RMSNorm을 적용한다.

$$Q = x_{norm} \cdot W_{q}, \quad K = x_{norm} \cdot W_{k}, \quad V = x_{norm} \cdot W_{v}$$

$$Q^{head}_{i} = \frac{Q^{head}_{i}}{RMS(Q^{head}_{i})} \times \gamma_{q}, \quad K^{head}_{j} = \frac{K^{head}_{j}}{RMS(K^{head}_{j})} \times \gamma_{k}$$

동적 주파수 ROPE 및 비대칭 KV Cache Sharing:
레이어 인덱스($i$)에 따라 회전 주파수($\theta$)를 변경하며, 20층 이상부터는 VRAM 절약을 위해 18, 19층의 캐시를 불균형하게 공유한다.

$$\theta = 1,000,000 \quad (\text{if } i \% 5 == 4) \quad \text{else} \quad 10,000$$

$$Q_{rope} = ROPE(Q_{norm}, \theta), \quad K_{rope} = ROPE(K_{norm}, \theta)$$

Cache 전략 (공유 규칙): * $i < 20$: 현재 $K_{rope}, V$를 자체 캐시에 저장 및 사용.

$i \ge 20$: 캐시를 저장하지 않고 재사용하되, $i \% 5 == 4$ 인 레이어만 Layer 19의 캐시를 사용하고, 나머지 모든 레이어는 Layer 18의 캐시를 사용한다.

$$attn\_raw = GQA(Q_{rope}, target\_K\_cache, target\_V\_cache)$$

$$attn\_output = attn\_raw \cdot W_{o}$$

C. Laurel 보조 신경망 및 어텐션 출력 결합

Laurel의 잔차 연결(Residual Connection) 시 원본 입력이 아닌 **정규화된 입력($x_{norm}$)**을 더해준다.

$$laurel\_x = (x_{norm} \cdot W_{laurel\_left}) \cdot W_{laurel\_right}$$

$$laurel\_out\_normed = \mathbf{x_{norm}} + RMSNorm(laurel\_x, W_{laurel\_norm})$$

$$attn\_output = RMSNorm(attn\_output, W_{post\_attn\_ln}) + x_{input}$$

$$x_{attn} = (attn\_output + laurel\_out\_normed) \times \frac{1}{\sqrt{2.0}}$$

D. 피드포워드 네트워크 (FFN - Gate, Up, Down)

$$x_{n2} = RMSNorm(x_{attn}, W_{pre\_ffn\_ln})$$

$$gate\_raw = x_{n2} \cdot W_{gate}$$

$$up\_out = x_{n2} \cdot W_{up}$$

레이어 10 이상 (표준 GELU Gate 적용):
HW 가속기 내부에서 곧바로 GELU가 씌워진 채 반환된다.

$$gate\_out = GELU(gate\_raw)$$

$$hidden = gate\_out \times up\_out$$

레이어 10 미만 (Sparse Gate 적용):

$$cutoff = Mean(gate\_raw) + Std(gate\_raw) \times 1.6448536$$

$$sparse\_gate = \max(gate\_raw - cutoff, 0.0)$$

$$hidden = GELU(sparse\_gate) \times up\_out$$

최종 FFN 출력 결합:

$$mlp\_out = hidden \cdot W_{down}$$

$$outputs = RMSNorm(mlp\_out, W_{post\_ffn\_ln}) + x_{attn}$$

E. 모달리티 업데이트 (AltUp Correction)

FFN을 통과한 값을 기준으로 나머지 3개의 모달리티 벡터들을 다음 레이어를 위해 업데이트한다.

$$activated = outputs \times W_{altup\_scale}$$

$$innovation = activated - xs_{pred}[0]$$

$$x_{n3} = \frac{RMSNorm(activated, W_{altup\_rn})}{2048.0}$$

$$mod\_corr = \tanh(x_{n3} \cdot W_{altup\_router})$$

$$corr\_coefs = (W_{altup\_corr} \cdot mod\_corr) + 1.0$$

텐서 차원 보정 (Broadcasting): $corr\_coefs$는 $(4, 1)$ 크기로 재구성되어 $(1, 2048)$ 크기의 $innovation$ 벡터에 각 행별로 곱해진다.

$$xs_{new} = xs_{pred} + (corr\_coefs_{[:,1]} \times innovation_{[1,:]})$$

마지막으로 $pli$ 벡터를 섞어준다.

$$gate\_ple = GELU(activated \cdot W_{ple\_gate}) \times pli$$

$$mapped = RMSNorm(gate\_ple \cdot W_{ple\_proj}, W_{ple\_post\_ln})$$

$$xs_{new}[1:] = xs_{new}[1:] + mapped$$

5. 로짓 디코딩 (Decode Logits)

역투영 시 벡터 크기(Magnitude)를 강제로 보정하여 합치는 것이 핵심이다.

$$target\_mag = \sqrt{Mean(xs[0]^{2})}$$

$$proj\_x_{k} = xs[k+1] \cdot altup\_unprojs[k] \quad (k=0,1,2)$$

크기 매칭 (Magnitude Matching):

$$new\_mag_{k} = \sqrt{Mean(proj\_x_{k}^{2})}$$

$$proj\_x_{k} = proj\_x_{k} \times \frac{target\_mag}{\max(new\_mag_{k}, 10^{-12})}$$

보정된 4개의 벡터를 평균내고 최종 행렬을 곱한다.

$$x_{final} = Mean([xs[0], proj\_x_{0}, proj\_x_{1}, proj\_x_{2}])$$

$$x_{final\_norm} = RMSNorm(x_{final}, W_{final\_norm})$$

$$Logits\_Raw = x_{final\_norm} \cdot W_{lm\_head}$$

Logit Soft-Capping:

$$Logits = 30.0 \times \tanh\left(\frac{Logits\_Raw}{30.0}\right)$$

6. 샘플링 로직 (Generation & Sampling)

생성된 Logit을 기반으로 다음 토큰을 선택한다.

Repetition Penalty (반복 패널티): 이전에 생성된 토큰 $t$ 에 대하여, Logit의 부호에 따라 패널티($\rho = 1.15$) 연산을 분기 처리한다.


$$Logits_{t} = Logits_{t} \times \rho \quad (\text{if } Logits_{t} < 0)$$

$$Logits_{t} = \frac{Logits_{t}}{\rho} \quad (\text{if } Logits_{t} \ge 0)$$

Temperature Softmax: Temperature($T=0.65$)를 나누어 분포를 조정한 뒤, C++ 최적화 SIMD 커널을 통해 고속 Softmax를 적용하여 확률($probs$)을 얻는다.


$$probs_i = \frac{\exp(Logits_i / T)}{\sum \exp(Logits_j / T)}$$

Top-P Sampling: 확률이 높은 순으로 정렬한 뒤, 누적 확률이 Top-P (0.9) 미만인 토큰들만 남기고 나머지는 잘라낸(Cut-off) 후 랜덤 샘플링을 진행한다.

7. 시스템 및 메모리 최적화 아키텍처 (Hardware Integration)

향후 FPGA(KV260) 설계 시 버스 및 제어기 구현을 위한 핵심 스펙

Ping-Pong 더블 버퍼링 (hw_compute_pingpong):
GPU/가속기가 현재 레이어의 행렬곱(예: $K$)을 계산하는 동안, 백그라운드 스레드에서 다음 계산에 필요한 가중치(예: $V$)를 반대쪽 버퍼에 미리 프리패치(Prefetch)하여 I/O 대기시간(Latency)을 0으로 은닉한다.

In-place 메모리 덮어쓰기 (__restrict__):
메모리 대역폭이 극단적으로 부족한 환경을 극복하기 위해, RMSNorm, GELU, Softmax 연산 시 별도의 출력 텐서를 생성하지 않고 원본 메모리 공간의 값을 직접 덮어쓴다.

MMAP Zero-Copy 스트리밍:
RAM에 모델 전체를 올리지 않고 OS의 페이징(Page Fault)을 이용해 SSD에서 필요한 행렬 1줄씩만 C-Contiguous 포인터 형태로 직접 스트리밍한다.