import numpy as np

def generate_softmax_frac_lut(filename="softmax_frac.mem"):
    print(" Softmax 2^x (소수부) 1024분할 BRAM 컨닝페이퍼 생성 시작...")

    NUM_ENTRIES = 1024

    #  하드웨어 스케일링 팩터 (Q1.15 포맷)
    # 1.0이라는 숫자를 하드웨어의 32768 (2^15)로 뻥튀기!
    # 왜냐? 2^0.999... 는 거의 2.0에 가까우므로,
    # 2.0 * 32768 = 65536 이 되어 16비트(0~65535) 통에 소름 돋게 딱 꽉 차게 들어감!
    SCALE_FACTOR = 32768.0

    hex_list = []

    for i in range(NUM_ENTRIES):
        # 1. 구간 인덱스(0~1023)를 0.0 ~ 0.999... 의 소수점 값으로 변환
        frac_val = i / float(NUM_ENTRIES)

        # 2. 2^(소수부) 정답 계산 (결과는 1.0 ~ 1.999... 사이로 나옴)
        y = np.power(2.0, frac_val)

        # 3. 하드웨어 정수(16비트)로 스케일링 및 반올림
        int_y = int(np.round(y * SCALE_FACTOR))

        # 4. 16비트 Unsigned 범위(0 ~ 65535)로 클리핑 (오버플로우 방지)
        int_y = int(np.clip(int_y, 0, 65535))

        # 5. 4자리 Hex 문자열로 이쁘게 포장
        hex_list.append(f"{int_y & 0xFFFF:04X}")

    # 파일로 굽기
    with open(filename, 'w') as f:
        for h in hex_list:
            f.write(f"{h}\n")

    print(f" 2^x 소수부 컨닝페이퍼(16-bit) {NUM_ENTRIES}개 저장 완료! -> {filename}")
    print(" 이제 FPGA는 e^x 지수함수를 덧셈과 시프트만으로 씹어먹습니다!")

if __name__ == "__main__":
    generate_softmax_frac_lut()