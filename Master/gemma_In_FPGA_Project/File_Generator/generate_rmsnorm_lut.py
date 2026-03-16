import numpy as np

def generate_inv_sqrt_pwl_1024(slope_file="rmsnorm_slope.mem", inter_file="rmsnorm_inter.mem"):
    print(" [High-end version] RMSNorm 1/sqrt(x) 1024 split BRAM cheat sheet creation start...")

    # Input value is 32 bits (0 to 4.2 billion)
    # The upper 10 bits (1024 sections), the lower 22 bits are the positions within the section (0 ~ 4,194,303)
    NUM_SEGMENTS = 1024
    SEGMENT_SIZE = 2**22  # 4,194,304

    # Hardware scaling factor
    # The intercept (b) is scaled to 2^30 to use up all 32 bits (maximizing precision!)
    SCALE_B = 2**30

    slope_hex_list = []
    inter_hex_list = []

    for i in range(NUM_SEGMENTS):
        # 1. Starting point (x1) and ending point (x2) of the section
        # The starting point of section 0 is corrected to 1 to prevent 1/0 error.
        x1 = max(1, i * SEGMENT_SIZE)
        x2 = (i + 1) * SEGMENT_SIZE - 1

        # 2. Calculate the actual correct answer (1/sqrt(x))
        y1 = 1.0 / np.sqrt(x1)
        y2 = 1.0 / np.sqrt(x2)

        # 3. Calculate slope (a) and y-intercept (b)
        # Since the lower 22 bits (x_frac) in the section range from 0 to (2^22-1),
        # Hardware calculation: y_hw = (a_hw * x_frac) + b_hw

        # The intercept b_hw is a scaled value of the starting point y1 of the interval.
        b_hw = y1 * SCALE_B

        # The slope a_hw is scaled so that when multiplied by 22 bits (x_frac), it becomes (y2 - y1) * SCALE_B.
        a_hw = ((y2 - y1) * SCALE_B) / SEGMENT_SIZE

        # 4. Convert to integer and clip to fit hardware bit count
        int_inter = int(np.round(b_hw))
        int_slope = int(np.round(a_hw))

        # Intercept is a 32-bit positive number (0 to 2^32-1)
        int_inter = int(np.clip(int_inter, 0, 4294967295))

        # The slope is negative, so it clips to the 16-bit Signed (-32768 to 32767) range.
        int_slope = int(np.clip(int_slope, -32768, 32767))

        # 5. Hex string conversion (4 digits for slope, 8 digits for intercept)
        slope_hex_list.append(f"{int_slope & 0xFFFF:04X}")
        inter_hex_list.append(f"{int_inter & 0xFFFFFFFF:08X}")

    # burn files
    with open(slope_file, 'w') as fs, open(inter_file, 'w') as fi:
        for s, i in zip(slope_hex_list, inter_hex_list):
            fs.write(f"{s}\n")
            fi.write(f"{i}\n")

    print(f" 1024 slopes (16-bit) saved! -> {slope_file}")
    print(f" 1024 Y intercepts (Intercept, 32-bit) saved! -> {inter_file}")
    print(" Complete BRAM data ready to be processed by one DSP48E2 (27x18) multiplier!")

if __name__ == "__main__":
    generate_inv_sqrt_pwl_1024()