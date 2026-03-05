# Full-Stack Integration: AXI4-Lite & PYNQ (Software Control)

This covers the process of wrapping our pure logic circuit (Verilog) with a standard interface, allowing the ARM CPU (Zynq PS) to communicate with it in the real world.

## 1. The Magic of Memory Mapped I/O (MMIO)
The CPU cannot physically press hardware switches. Instead, it utilizes the AXI4-Lite interconnect: **"Writing a value to a specific memory address translates into electrical signals that toggle the hardware switches."**

```mermaid
sequenceDiagram
    participant PY as Python (Jupyter)
    participant AXI as AXI Bus (0x4000_0000)
    participant HW as NPU Hardware

    Note over PY, HW: DMA Data Write
    PY->>AXI: npu.write(0x04, 1) (dma_we = 1)
    PY->>AXI: npu.write(0x0C, 15) (dma_wdata = 15)
    AXI->>HW: Saves 15 into BRAM

    Note over PY, HW: Trigger Computation
    PY->>AXI: npu.write(0x00, 1) (start_mac = 1)
    AXI->>HW: FSM State: IDLE -> RUN
```