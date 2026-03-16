import numpy as np

def generate_gemma_tile(filename=r"C:\Users\breadk\Desktop\FPGA_Project\TinyNPU-RTL\gemma_tile.mem",
                        lut_filename=r"C:\Users\breadk\Desktop\FPGA_Project\TinyNPU-RTL\gelu_table.mem",
                        size=32):
    print(f" Gemma 3N 32x32 Tile creation begins...")

    np.random.seed(42)
    matrix_A = np.clip(np.random.normal(0, 30, (size, size)), -128, 127).astype(np.int8)
    matrix_B = np.clip(np.random.normal(0, 30, (size, size)), -128, 127).astype(np.int8)

    # 1. Create and save tile data
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

    # 2. Calculate the original MAC answer in Python
    golden_mac = np.dot(matrix_A.astype(np.int32), matrix_B.astype(np.int32))

    # 3. Derive the GeLU answer by reading gelu_table.mem directly like HW.
    gelu_lut = []
    with open(lut_filename, 'r') as f:
        for line in f:
            val = int(line.strip(), 16)
            # Convert 8-bit 2's complement hexadecimal value to signed integer in Python.
            if val >= 128:
                val -= 256
            gelu_lut.append(val)

    # 4. Complete the GeLU array using the MAC result as an index.
    golden_gelu = np.zeros_like(golden_mac, dtype=np.int8)
    for i in range(size):
        for j in range(size):
            # Convert the MAC result (16-bit Signed) to a 16-bit bit string (0~65535) and refer to the LUT.
            mac_val = int(golden_mac[i][j])
            lut_index = mac_val & 0xFFFF
            golden_gelu[i][j] = gelu_lut[lut_index]

    print(" Tile memory & LUT mapping complete!")
    print(f" Python PE(0,0) | MAC: {golden_mac[0][0]:>5} -> GeLU: {golden_gelu[0][0]}")
    print(f" Python PE(31,31) | MAC: {golden_mac[31][31]:>5} -> GeLU: {golden_gelu[31][31]}")

if __name__ == "__main__":
    generate_gemma_tile()