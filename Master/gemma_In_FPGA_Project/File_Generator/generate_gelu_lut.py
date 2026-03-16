import numpy as np

SCALE_IN = 0.001
SCALE_OUT = 0.001

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def generate_lut_file(filename="C:/Users/breadk/Desktop/FPGA_Project/TinyNPU-RTL/gelu_table.mem"):
    with open(filename, 'w') as f:
        for i in range(65536):
            # Hardware 16-bit address scheme mapping
            signed_i = i if i < 32768 else i - 65536

            float_val = signed_i * SCALE_IN
            gelu_val = gelu(float_val)

            int8_val = np.round(gelu_val / SCALE_OUT)
            int8_clipped = int(np.clip(int8_val, -128, 127))

            # Save exactly 2 characters (8 bits) in Hex only.
            hex_str = f"{int8_clipped & 0xFF:02X}"
            f.write(f"{hex_str}\n")

    print(" Decontamination complete! Pure 64KB GeLU LUT creation complete!")

if __name__ == "__main__":
    generate_lut_file()