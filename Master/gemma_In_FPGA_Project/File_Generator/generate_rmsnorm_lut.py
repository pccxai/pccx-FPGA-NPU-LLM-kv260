import numpy as np

def generate_inv_sqrt_pwl_1024(slope_file="rmsnorm_slope.mem", inter_file="rmsnorm_inter.mem"):
    print(" [하이엔드 버전] RMSNorm 1/sqrt(x) 1024분할 BRAM 컨닝페이퍼 생성 시작...")

    # 입력값은 32비트 (0 ~ 42억)
    # 상위 10비트 (1024개 구간), 하위 22비트는 구간 내의 위치(0 ~ 4,194,303)
    NUM_SEGMENTS = 1024
    SEGMENT_SIZE = 2**22  # 4,194,304

    # 하드웨어 스케일링 팩터
    # 절편(b)은 32비트 통을 다 쓰기 위해 2^30으로 스케일링 (정밀도 극대화!)
    SCALE_B = 2**30

    slope_hex_list = []
    inter_hex_list = []

    for i in range(NUM_SEGMENTS):
        # 1. 구간의 시작점(x1)과 끝점(x2)
        # 0번 구간의 시작점은 1/0 에러 방지를 위해 1로 보정
        x1 = max(1, i * SEGMENT_SIZE)
        x2 = (i + 1) * SEGMENT_SIZE - 1

        # 2. 실제 정답 (1/sqrt(x)) 계산
        y1 = 1.0 / np.sqrt(x1)
        y2 = 1.0 / np.sqrt(x2)

        # 3. 기울기(a)와 y절편(b) 계산
        # 구간 내 하위 22비트(x_frac)가 0 ~ (2^22-1) 범위를 가지므로
        # 하드웨어 연산: y_hw = (a_hw * x_frac) + b_hw

        # 절편 b_hw는 그 구간의 시작점 y1을 스케일링한 값!
        b_hw = y1 * SCALE_B

        # 기울기 a_hw는 22비트(x_frac)를 곱했을 때 (y2 - y1) * SCALE_B 가 되도록 스케일링!
        a_hw = ((y2 - y1) * SCALE_B) / SEGMENT_SIZE

        # 4. 하드웨어 비트 수에 맞게 정수로 변환 및 클리핑
        int_inter = int(np.round(b_hw))
        int_slope = int(np.round(a_hw))

        # 절편은 32비트 양수 (0 ~ 2^32-1)
        int_inter = int(np.clip(int_inter, 0, 4294967295))

        # 기울기는 음수이므로 16비트 Signed (-32768 ~ 32767) 범위에 클리핑
        int_slope = int(np.clip(int_slope, -32768, 32767))

        # 5. Hex 문자열 변환 (기울기는 4자리, 절편은 8자리)
        slope_hex_list.append(f"{int_slope & 0xFFFF:04X}")
        inter_hex_list.append(f"{int_inter & 0xFFFFFFFF:08X}")

    # 파일 굽기
    with open(slope_file, 'w') as fs, open(inter_file, 'w') as fi:
        for s, i in zip(slope_hex_list, inter_hex_list):
            fs.write(f"{s}\n")
            fi.write(f"{i}\n")

    print(f" 기울기(Slope, 16-bit) 1024개 저장 완료! -> {slope_file}")
    print(f" Y절편(Intercept, 32-bit) 1024개 저장 완료! -> {inter_file}")
    print(" DSP48E2 (27x18) 곱셈기 1개로 처리할 완벽한 BRAM 데이터 준비 완료!")

if __name__ == "__main__":
    generate_inv_sqrt_pwl_1024()