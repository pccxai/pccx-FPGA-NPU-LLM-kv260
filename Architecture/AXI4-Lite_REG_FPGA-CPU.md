# AXI LITE Register FPGA <-> CPU

`0x00` (Control Register - Write Only)

* `Bit[0]` : START (Setting 1 kicks off NPU computation! CUDA Kernel Launch)

* `Bit[1]` : ACC_CLEAR (Setting 1 resets internal accumulators in the Systolic Array to 0)
    
`0x04` (Status Register - Read Only)

* `Bit[0]` : DONE (1 means computation complete. CPU polls this while waiting)

`0x08` (RMSNorm Param - Write Only)

* `Bit[31:0]` : `mean_sq` value (Plugs in a 32-bit scalar for denominator calculation)

`0x0C` (Ping-Pong MUX - Write Only)

* `Bit[0]` : 0 means DMA->Ping / 1 means DMA->Pong

`0x10` (Mode / Command - Write Only)

* `Bit[0]` : `GeLU_EN` (1 applies 1-Cycle GeLU before output)

* `Bit[1]` : `Softmax_EN` (1 applies Softmax before output)

`0x14` (Reserved / Debug) : Reserved (currently empty)
