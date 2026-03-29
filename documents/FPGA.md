Gemma 3N NPU 하드웨어 (SystemVerilog) 파이프라인 상세 흐름도

이 문서는 NPU_top을 중심으로, 메모리(AXI)에서 들어온 데이터가 시스톨릭 어레이를 거쳐 다시 출력될 때까지의 하드웨어 파이프라인 연산을 순서대로 설명한다.
기본적으로 가로(Weight) 32채널, 세로(FMap) 32채널 파이프라인으로 구성된 400MHz 타겟 Zero-Bubble 구조를 따른다.

0. 사전 정의: 핵심 하드웨어 연산의 실제 동작

하드웨어 내부에서 반복적으로 사용되는 비트 연산과 DSP 블록의 동작을 정의한다.

1. LOD (Leading One Detection) 가장 높은 자릿수(MSB)부터 탐색하여 처음으로 '1'이 등장하는 위치를 찾는 하드웨어 로직이다. 정규화(Normalization) 단계에서 부동소수점의 지수(Exponent)를 결정할 때 쓴다. 실제 stlc_result_normalizer.sv 코드에서는 48비트 데이터 중 47번 부호 비트를 제외하고 46번부터 0번 비트까지만 탐색한다.

$$	ext{pos} = \max \{ i \mid x[i] == 1, \ 0 \le i \le 46 \}$$

2. Barrel Shift (배럴 시프트) 클럭 지연 없이 한 번에 원하는 비트 수만큼 데이터를 왼쪽이나 오른쪽으로 밀어버리는 조합 회로 연산이다. 멀티플렉서(MUX)를 이용해 구현되며, BF16을 고정소수점으로 맞추거나 되돌릴 때 쓴다.

$$	ext{Output} = x \gg 	ext{shift\_amount} \quad 	ext{또는} \quad x \ll 	ext{shift\_amount}$$

3. DSP48 MAC (Multiply-Accumulate) Xilinx DSP48E2 프리미티브 내부에서 일어나는 핵심 곱셈-누산 연산이다. 가중치(B)와 특징맵(A)을 곱하고, 이전 행에서 내려온 부분합(PCIN)과 더한다.

$$P = P_{prev} + (A 	imes B)$$

1. Feature Map 전처리 (BF16 to Fixed-Point)

모듈: stlc_bf16_fixed_pipeline.sv

AXI-Stream(HPC0, HPC1 병합)으로 들어온 256-bit(16 x BF16) 특징맵 데이터를 시스톨릭 어레이가 계산하기 편하도록 27-bit 고정소수점으로 변환하는 과정이다. 
[업데이트 내역]: 기존 1-Lane 구조에서 **16-Lane 병렬 파이프라인**으로 확장되어, 2클럭 만에 32개(1블록)의 Global emax를 도출하고 일괄 시프트(Shift)한다.

연산 과정: 먼저 256-bit 데이터에서 16개씩 2사이클에 걸쳐(Ping-Pong) 32개 타일 내의 가장 큰 지수 값(Global emax)을 파싱한다.

$$	ext{emax} = \max(x_0.exp, x_1.exp, \dots, x_{31}.exp)$$

Stage 1: 각 값의 지수와 emax의 차이(delta_e)를 구하고, 지수가 0인지 여부에 따라 숨겨진 비트(Hidden Bit)를 포함하여 27비트 베이스 벡터를 생성한다.

$$	ext{delta\_e} = 	ext{emax} - x_i.exp$$

$$	ext{base\_vec} = (x_i.exp == 0) \ ? \ \{8'h00, x_i.mantissa, 12'b0\} : \{8'h01, x_i.mantissa, 12'b0\}$$

Stage 2 & 3: 16-Lane 병렬 배럴 시프터를 통해 16개의 베이스 벡터를 동시에 시프트하여 고정소수점 값으로 출력한다.

$$x_{fixed} = (	ext{delta\_e} \ge 27) \ ? \ 0 : (	ext{base\_vec} \gg 	ext{delta\_e})$$

입력 크기: $1 	imes 256$   bits (16 x BF16)

내부 emax 크기: $1 	imes 8$   bits

출력 크기: $1 	imes 432$   bits (16 x 27-bit)

데이터 타입: BF16   $ightarrow$   Fixed-point

2. FMap 캐싱 및 대각선 딜레이 (Cache & Staggered Delay)

모듈: stlc_fmap_cache.sv & TO_stlc_fmap_staggered_delay.sv

변환된 27-bit FMap 데이터를 비대칭 XPM BRAM(SRAM)에 저장해두고, 시스톨릭 어레이의 32개 세로 열(Column)에 동시에 뿌려주되, 2D 어레이의 데이터 도착 타이밍(Wavefront)에 맞게 대각선으로 지연시킨다.
[업데이트 내역]: Asymmetric(비대칭) 포트 규격을 적용하여 **Write는 432-bit(16개) 한 방에, Read는 27-bit 1개를 32개 레인으로 브로드캐스트**하도록 수정. 또한 **e_max 전용 캐시**를 추가하여 데이터와 e_max의 파이프라인 타이밍을 완벽히 동기화.

연산 과정: SRAM에서 읽어온 1개의 데이터를 32개 열로 복사(Broadcast)한다.

$$	ext{broadcast\_data}[c] = 	ext{SRAM\_Read}(	ext{addr}) \quad (c = 0 \dots 31)$$

열 번호   $c$  에 비례하여 시프트 레지스터 체인을 통해 클럭 지연(Delay)을 먹인다. (0번 열은 지연 없음, 31번 열은 31클럭 지연)

$$	ext{row\_data}[c] = 	ext{broadcast\_data}[c] 	imes z^{-c}$$

입력 크기: $1 	imes 432$   bits (Write)

출력 크기: $32$  개의   $1 	imes 27$   bits 배열 (Read Broadcast)

데이터 타입: Fixed-point

3. 가중치 언패킹 및 분배 (Weight Dispatch)

모듈: TO_stlc_weight_dispatcher.sv

메모리 대역폭을 꽉 채워서 들어온 128-bit 뭉탱이 가중치를 시스톨릭 어레이가 개별적으로 받을 수 있도록 4-bit짜리 32개로 쪼개어 가로(Row) 방향으로 밀어 넣는다.

연산 과정: 128-bit 데이터를 4-bit 단위로 슬라이싱하여 레지스터에 저장한다.

$$W_{out}[i] = 	ext{fifo\_data}[ (i 	imes 4 + 3) : (i 	imes 4) ] \quad (i = 0 \dots 31)$$

입력 크기: $1 	imes 128$   bits

출력 크기: $32$  개의   $1 	imes 4$   bits 배열

데이터 타입: Packed Bits   $ightarrow$   INT4

4. 시스톨릭 어레이 연산 (DSP MAC)

모듈: stlc_NxN_array.sv & stlc_dsp_unit.sv

실질적인 행렬곱이 일어나는 32x32 심장부다. VLIW 명령어와 타이밍 제어에 따라 데이터를 곱하고 누산하며, FPGA 물리적 배선 한계를 피하기 위한 브레이크(Break) 구조가 적용되어 있다.

연산 과정: A. 데이터 패딩 및 부호 확장: 들어온 FMap(27-bit)을 DSP A포트 규격에 맞춰 상단에 '0' 3개를 채워 30-bit로 패딩한다. INT4 가중치는 B포트 규격에 맞게 최상위 부호 비트를 14번 반복하여 18-bit로 부호 확장(Sign-extension)한다.

$$A = \{ 3'b0, 	ext{FMap}_{27} \}$$

$$B = \{ 14 	imes W_{int4}[3], W_{int4} \}$$

B. MAC 연산 및 캐스케이드 (Cascade): DSP 내부에서 곱셈과 누산을 수행한다. 일반적인 행(Row 0~15, 17~31)은 전용 고속 라우팅인 PCIN 체인을 통해 이전 부분합을 받아온다. (Z_MUX = 001)

$$P = 	ext{PCIN} + (A 	imes B)$$

C. 캐스케이드 브레이크 (Row 16): 칩 내부의 SLR이나 클럭 영역 경계로 인한 타이밍 위반을 막기 위해, 16번째 행에서는 전용 체인을 끊고 일반 패브릭(Fabric) 배선망을 이용해 이전 부분합을 C 포트로 우회시켜 받는다. (Z_MUX = 011)

$$P = C + (A 	imes B)$$

D. 플러시 (Flush) 제어: 명령어 파이프라인에서 Flush 비트가 감지되면, DSP 내부 OPMODE를 0으로 덮어씌워 누산기 버퍼를 초기화한다.

$$P = 0 \quad (	ext{if is\_flushing} == 1)$$

입력 A (FMap): $1 	imes 30$   bits

입력 B (Weight): $1 	imes 18$   bits

입력 PCIN / C (이전 부분합): $1 	imes 48$   bits

출력 PCOUT / P: $1 	imes 48$   bits

데이터 타입: Fixed-point  &  INT4   $ightarrow$   48-bit Accumulation

5. 최종 누산 (Accumulator)

모듈: stlc_accumulator.sv

시스톨릭 어레이의 가장 마지막 행(Bottom Row)에서 흘러나온 48-bit 부분합들을 최종적으로 누적해서 더한다.

연산 과정: DSP의 곱셈기(Multiplier)를 끄고(USE_MULT="NONE"), OPMODE를 00_001_00_10로 설정하여 순수하게 내부 레지스터(P)에 새로 들어온 값(PCIN)을 덧셈만 한다.

$$P_{final} = P_{final} + 	ext{PCIN}$$

입력 크기: $1 	imes 48$   bits

출력 크기: $1 	imes 48$   bits

데이터 타입: 48-bit 2's Complement

6. 결과 정규화 (Result Normalization)

모듈: stlc_result_normalizer.sv

커질 대로 커진 48-bit 고정소수점 누산 결과를 다시 16-bit 부동소수점(BF16 유사 포맷)으로 압축하여 메모리로 돌려보낼 준비를 한다. 4클럭 단계(4-Stage Pipeline)로 구성된다.

연산 과정: Stage 1: 2의 보수 체계를 부호(Sign)와 절댓값(Magnitude)으로 분리한다.

$$	ext{Sign} = x[47]$$

$$	ext{Abs\_Data} = 	ext{Sign} \ ? \ (\sim x + 1) : x$$

Stage 2: LOD 회로를 돌려 절댓값 중 가장 높은 '1'의 위치(first_one_pos)를 찾는다.

$$	ext{pos} = 	ext{LOD}(	ext{Abs\_Data})$$

Stage 3: 배럴 시프터로 가수를 정렬하고, 1단계에서 딜레이 라인을 타고 넘어온 원본 emax 값을 이용해 새로운 지수(New Exp)를 계산한다. (포맷 편향값 26 기준 적용)

$$	ext{New\_Exp} = emax + 	ext{pos} - 26$$

$$	ext{Mantissa} = 	ext{Abs\_Data}[ 	ext{pos}-1 : 	ext{pos}-7 ] \quad (	ext{pos} \ge 7 	ext{일 경우})$$

Stage 4: 부호, 지수, 가수를 하나로 합쳐서 최종 패킹한다.

$$	ext{Data\_Out} = \{ 	ext{Sign}, 	ext{New\_Exp}[7:0], 	ext{Mantissa}[6:0] \}$$

입력 크기: $1 	imes 48$   bits (데이터),   $1 	imes 8$   bits (emax)

출력 크기: $1 	imes 16$   bits

데이터 타입: 48-bit 2's Complement   $ightarrow$   BF16 Format

7. 차세대 아키텍처 예고: LUT 기반 INT4 곱셈 및 DSP Repurposing

기존의 DSP에 의존하던 MAC 연산 구조를 탈피하여, INT4 양자화의 특성을 100% 활용하는 **LUT 기반 Shift-and-Add 구조**로 패러다임을 전환할 예정이다. 

이 방식을 통해 묶여있던 수많은 DSP를 범용 행렬 곱셈기나 고정밀도 덧셈기로 재배치(Repurposing)할 수 있으며, 궁극적으로 **LUT + DSP 병렬 실행(Parallel Execution)** 을 통해 처리량(Throughput)을 극대화할 수 있다.

연산 원리 (BF16(A) * INT4(B) with Shift-and-Add):
INT4 가중치의 절댓값 특성을 이용해 무거운 곱셈기를 배럴 시프터(Shift)와 덧셈(Add)만으로 쪼개어 일반 LUT에서 처리한다. 부호(Sign)는 마지막에 별도로 결정하므로 음수 곱셈을 고려할 필요가 없다.

- B = 0  $ightarrow$  0
- B = 1  $ightarrow$  $A$
- B = -1 $ightarrow$  $-A$
- B = 2  $ightarrow$  $A \ll 1$
- B = 3  $ightarrow$  $(A \ll 1) + A$
- B = 4  $ightarrow$  $A \ll 2$
- B = 5  $ightarrow$  $(A \ll 2) + A$
- B = 6  $ightarrow$  $(A \ll 2) + (A \ll 1)$
- B = 7  $ightarrow$  $(A \ll 3) - A$
- B = 8  $ightarrow$  $A \ll 3$

기대 효과:
1. 배선 딜레이 및 Fan-out 병목 완화 (DSP 포트 의존도 감소)
2. DSP 블록 해방으로 인한 복잡한 추가 연산(Attention Score 계산, 벡터 덧셈 등) 오프로딩
3. 아키텍처의 완전한 스케일 아웃(Scale-out) 가능성 확보
