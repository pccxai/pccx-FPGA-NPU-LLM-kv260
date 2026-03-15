import numpy as np

def generate_gemma_tile(filename=r"C:\Users\breadk\Desktop\FPGA_Project\TinyNPU-RTL\gemma_tile.mem",
                        lut_filename=r"C:\Users\breadk\Desktop\FPGA_Project\TinyNPU-RTL\gelu_table.mem",
                        size=32):
    print(f" Gemma 3N 32x32 Tile 생성 시작...")

    np.random.seed(42)
    matrix_A = np.clip(np.random.normal(0, 30, (size, size)), -128, 127).astype(np.int8)
    matrix_B = np.clip(np.random.normal(0, 30, (size, size)), -128, 127).astype(np.int8)

    # 1. 타일 데이터 생성 및 저장
    with open(filename, 'w') as f:
        for k in range(size):
            hex_A = ""
            hex_B = ""
            for i in reversed(range(size)):
                val_a = int(matrix_A[i][k])
                val_b = int(matrix_B[k][i])
                hex_A += f"{val_a & 0xFF:02X}"
                hex_B += f"{val_b & 0xFF:02X}"
            f.write(f"{hex_B}{hex_A}\n")

    # 2. 파이썬 원본 MAC 정답 계산
    golden_mac = np.dot(matrix_A.astype(np.int32), matrix_B.astype(np.int32))

    #  3. HW처럼 gelu_table.mem을 직접 읽어서 GeLU 정답 도출!
    gelu_lut = []
    with open(lut_filename, 'r') as f:
        for line in f:
            val = int(line.strip(), 16)
            # 8비트 2의 보수 헥사값을 파이썬의 Signed 정수로 변환!
            if val >= 128:
                val -= 256
            gelu_lut.append(val)

    # 4. MAC 결과값을 인덱스로 써서 GeLU 배열 완성
    golden_gelu = np.zeros_like(golden_mac, dtype=np.int8)
    for i in range(size):
        for j in range(size):
            # MAC 결과(16비트 Signed)를 16비트 비트열(0~65535)로 변환해서 LUT 참조!
            mac_val = int(golden_mac[i][j])
            lut_index = mac_val & 0xFFFF
            golden_gelu[i][j] = gelu_lut[lut_index]

    print(" 타일 메모리 & LUT 매핑 완료!")
    print(f" 파이썬 PE(0,0)   | MAC: {golden_mac[0][0]:>5} -> GeLU: {golden_gelu[0][0]}")
    print(f" 파이썬 PE(31,31) | MAC: {golden_mac[31][31]:>5} -> GeLU: {golden_gelu[31][31]}")

if __name__ == "__main__":
    generate_gemma_tile()