import numpy as np

def generate_softmax_frac_lut(filename="softmax_frac.mem"):
    print(" Start creating Softmax 2^x (decimal part) 1024 division BRAM cheat sheet...")

    NUM_ENTRIES = 1024

    # Hardware scaling factor (Q1.15 format)
    # The number 1.0 is changed to 32768 (2^15) in hardware.
    # Why? Since 2^0.999... is close to 2.0,
    # 2.0 * 32768 = 65536, which fills the 16-bit (0~65535) box to a scary degree.
    SCALE_FACTOR = 32768.0

    hex_list = []

    for i in range(NUM_ENTRIES):
        # 1. Convert the interval index (0 to 1023) to a decimal value of 0.0 to 0.999...
        frac_val = i / float(NUM_ENTRIES)

        # 2. Calculate the correct answer with 2^(decimal part) (results are between 1.0 and 1.999...)
        y = np.power(2.0, frac_val)

        # 3. Scaling and rounding to hardware integer (16 bits)
        int_y = int(np.round(y * SCALE_FACTOR))

        # 4. Clipping to 16-bit Unsigned range (0 to 65535) (overflow prevention)
        int_y = int(np.clip(int_y, 0, 65535))

        # 5. Beautifully packaged with 4-digit hex string
        hex_list.append(f"{int_y & 0xFFFF:04X}")

    # Burn to file
    with open(filename, 'w') as f:
        for h in hex_list:
            f.write(f"{h}\n")

    print(f" 2^x decimal cheat sheet (16-bit) {NUM_ENTRIES} saved! -> {filename}")
    print("Now the FPGA eats the e^x exponential function with just addition and shift!")

if __name__ == "__main__":
    generate_softmax_frac_lut()