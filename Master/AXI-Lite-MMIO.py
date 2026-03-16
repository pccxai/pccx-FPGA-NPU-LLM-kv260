from pynq import Overlay, allocate
import numpy as np
import time

# 1. Load bitstream and block design (Overlay) hardware
# This is the step of programming the FPGA area by pushing the bit file and hwh file the user synthesized onto the board.
print("Loading FPGA overlay...")
ol = Overlay("gemma3n_npu_design.bit")

# 2. IP module binding
# The name set in the Vivado block design is imported as a Python object.
# For example, assume that the DMA IP name is 'axi_dma_0' and the AXI Lite wrapper for NPU control is 'npu_axi_wrapper_0'.
dma = ol.axi_dma_0
npu_axi = ol.npu_axi_wrapper_0

# 3. Memory buffer allocation for physically contiguous DMA (CMA area)
# The data type must be set according to the hardware bus width (e.g. 32 bit).
# If the user are going to pass on Gemma's bfloat16 data, it is standard to pack two 16 bits into 32 bits and send them.
# Here, assuming it matches the 32x32 Systolic Array input size, will set the size to (1024,).
print("Allocating physical memory (CMA) buffer...")
input_buffer = allocate(shape=(1024,), dtype=np.uint32)
output_buffer = allocate(shape=(1024,), dtype=np.uint32)

# 1. Python single data conversion function (Float -> INT16 Fixed-Point Q4.12)
# Q4.12 format means using 12 bits after the decimal point.
# The scale factor becomes 2^12 = 4096.0.
SCALE_FACTOR = 4096.0

def float_to_q4_12(float_array):
    # Multiply a floating point array by a scale factor
    scaled = float_array * SCALE_FACTOR
    # Cut it without exceeding the int16 range with np.clip, then cast it to int16.
    # The range for int16 is -32768 to 32767
    q_array = np.clip(scaled, -32768, 32767).astype(np.int16)
    return q_array

# float32 data for testing (assuming Gemma weights or input values)
gemma_data_float = np.array([1.5, -0.75, 3.14, -2.5], dtype=np.float32)

# Convert to INT16
gemma_data_int16 = float_to_q4_12(gemma_data_float)

# In order to shoot with DMA, two pieces of 16-bit data must be combined into one according to the 32-bit bus.
# Even indexes are packed into the lower 16 bits, and odd indices are packed into the upper 16 bits.
packed_data = np.zeros(len(gemma_data_int16) // 2, dtype=np.uint32)
for i in range(len(packed_data)):
    low_16 = gemma_data_int16[2*i] & 0xFFFF
    high_16 = (gemma_data_int16[2*i + 1] & 0xFFFF) << 16
    packed_data[i] = high_16 | low_16

print("Original Float:", gemma_data_float)
print("Q4.12 INT16:", gemma_data_int16)
print("32-bit packed data for DMA transmission (Hex):", [hex(x) for x in packed_data])

# 4. Fill the input buffer with test data
# In the actual project, the Gemma weights/input values ​​read from safetensors are packed and inserted here.
for i in range(1024):
    input_buffer[i] = i

# --- NPU pipeline execution routine ---

# 5. Hardware initialization (Clear)
# Assume that MMIO address 0x04 is the accumulator and state machine initialization register.
# Apply rule number 5: If the user write 1 in Python, the hardware (Verilog) must automatically lower it to 0 at the next clock.
npu_axi.write(0x04, 1)

# 6. Data transmission via DMA (CPU -> BRAM)
print("Start DMA data transfer (TX)...")
# sendchannel is a channel that sends data from the CPU to the hardware.
dma.sendchannel.transfer(input_buffer)

# Wait until all data has passed. A cache flush is also performed internally.
dma.sendchannel.wait()
print("DMA TX completed. Data loaded into BRAM.")

# 7. Trigger to start NPU operation
# Assuming MMIO address 0x00 is the NPU Start register.
# Since the BRAM is full of data, a command is given to the FSM inside the NPU to start computation.
# Likewise, this must operate with an auto clear pulse to be safe.
npu_axi.write(0x00, 1)

# 8. Waiting for computation completion (Polling)
# Apply rule number 6: RTL memory map and Python address must match perfectly.
# Assume MMIO address 0x10 is the NPU Done status register.
# Checks the hardware status by running an infinite loop until the value becomes 1.
print("NPU operation in progress... waiting for polling")
while True:
    npu_status = npu_axi.read(0x10)
    if npu_status == 1:
        break
    time.sleep(0.001) # Very short wait to prevent CPU utilization from skyrocketing to 100%

print("NPU calculation complete!")

# 9. Receiving results via DMA (BRAM -> CPU)
# Now that the NPU has completed the calculation and written the result to BRAM, it is now brought to the receiving channel.
dma.recvchannel.transfer(output_buffer)
dma.recvchannel.wait()
print("DMA RX completed. Result data copied to memory.")

# 10. Verification of results (sample)
print("Check the front of the output buffer:")
print(output_buffer[:10])

# Physical memory that has been used must be released. Otherwise, memory leak will occur.
input_buffer.close()
output_buffer.close()