Gemma 3N (INT4 + AltUp) Pipeline Detailed Operation Flowchart (100% Code Matched Perfect Version)

This document first defines the actual mathematical behavior of the basic operations used in the pipeline, and then sequentially explains the size transformation and operation process at each pipeline stage.
It is assumed that the base dimension is $D = 2048$, the router dimension is $D_{mod}$, the patch embedding dimension is $256$, and the attention heads consist of **8 Q heads and 2 KV heads (Head Dim=256)**.

0. Pre-definition: Actual Mathematical Operations of Core Functions

Defines what mathematical operations the core functions used repeatedly throughout the pipeline perform internally.

Embedding: Takes the ID (integer) of a word and extracts the entire row corresponding to that ID from a pre-trained massive weight matrix.

$$ Output = W_{embed}[token\_id, :] $$

RMSNorm (Root Mean Square Normalization): An operation that divides the input vector values by their average size so they don't get too large or too small, and multiplies them by a learnable weight ($\gamma$).

$$ RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_{i}^{2} + 10^{-6}} $$

$$ Output = \left(\frac{x}{RMS}\right) \times \gamma $$

GELU (Gaussian Error Linear Unit): Unlike ReLU which simply discards values below 0, it is a non-linear activation function that bends smoothly using a normal distribution. The approximate formula is as follows.

$$ Output = 0.5 \times x \times \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^{3})\right)\right) $$

ROPE (Rotary Position Embedding): An operation that multiplies a rotation transformation ($\sin, \cos$) for each even/odd index to give positional information to the word.

$$ Output_{2i} = x_{2i} \times \cos(\theta) - x_{2i+1} \times \sin(\theta) $$

$$ Output_{2i+1} = x_{2i} \times \sin(\theta) + x_{2i+1} \times \cos(\theta) $$

1. Token Embedding

First, the incoming integer-type token ID is converted into a vector and its scale is increased.

Operation Process:

$$ x_{0} = Embedding(token\_id, W_{embed}) \times \sqrt{2048.0} $$

Input: int (1 scalar value)

Weight Size: $Vocab\_Size \times 2048$ (Type: INT4 tuple)

Output Size: $1 \times 2048$

2. AltUp Initial Projections

The original vector $x_{0}$ is multiplied with 3 different weight matrices to create a collection of 4 modality vectors ($xs$).

Operation Process: Put $x_{0}$ into the 0th row of an empty matrix called $xs$, and fill the 1st~3rd rows with the dot product.

$$ xs_{1} = x_{0} \cdot altup\_projs[0] $$

$$ xs_{2} = x_{0} \cdot altup\_projs[1] $$

$$ xs_{3} = x_{0} \cdot altup\_projs[2] $$

Output Size: Completed $xs \to 4 \times 2048$

3. Position and Patch Embedding Setup (PLE Setup)

The auxiliary vectors ($pli\_all$) to be used across all 35 transformer layers are pre-calculated at once.

Operation Process:

$$ x_{proj} = \frac{x_{0} \cdot W_{ple\_proj}}{\sqrt{2048.0}} $$

The result of the above formula is reshaped to a size of $35 \times 256$. Then, this value is normalized and the patch embedding value is added.

$$ x_{proj\_normed} = RMSNorm(x_{proj}) \times norm_{ple} $$

$$ y = Embedding(token\_id, W_{ple\_packed}) \times \sqrt{256.0} $$

$$ pli\_all = (x_{proj\_normed} + y) \times \frac{1}{\sqrt{2.0}} $$

Output Size: $pli\_all \to 35 \times 256$

4. Transformer Layer (Repeated 35 Times)

This is the core section repeated 35 times in a loop.

A. AltUp Router & Pred

This is the process of mixing the $xs$ containing 4 vectors.

$$ x_{n} = \frac{RMSNorm(xs_{0}, W_{altup\_rn})}{2048.0} $$

$$ modalities = \tanh(x_{n} \cdot W_{altup\_router}) $$

$$ coef\_mat = (W_{altup\_pred} \cdot modalities).reshape(4, 4) $$

$$ xs_{pred} = xs + (coef\_mat \cdot xs) $$

B. Attention Q, K, V & GQA

Take out the first of the mixed vectors and use it as input.

$$ x_{input} = xs_{pred}[0] $$

$$ x_{norm} = RMSNorm(x_{input}, W_{input\_ln}) $$

Q, K, V Projection and Head-wise QK-Norm:
Instead of normalizing the entire Q and K matrices at once, they are divided into groups of 256 dimensions (Head size) and RMSNorm is applied to each.

$$ Q = x_{norm} \cdot W_{q}, \quad K = x_{norm} \cdot W_{k}, \quad V = x_{norm} \cdot W_{v} $$

$$ Q^{head}_{i} = \frac{Q^{head}_{i}}{RMS(Q^{head}_{i})} \times \gamma_{q}, \quad K^{head}_{j} = \frac{K^{head}_{j}}{RMS(K^{head}_{j})} \times \gamma_{k} $$

Dynamic Frequency ROPE and Asymmetric KV Cache Sharing:
The rotation frequency ($\theta$) is changed according to the layer index ($i$), and from layer 20 onwards, the caches of layers 18 and 19 are shared asymmetrically to save VRAM.

$$ \theta = 1,000,000 \quad (\text{if } i \% 5 == 4) \quad \text{else} \quad 10,000 $$

$$ Q_{rope} = ROPE(Q_{norm}, \theta), \quad K_{rope} = ROPE(K_{norm}, \theta) $$

Cache Strategy (Sharing Rules):
* $i < 20$: Save and use the current $K_{rope}, V$ in its own cache.

$i \ge 20$: Reuse the cache without saving it, but only layers where $i \% 5 == 4$ use the cache of Layer 19, and all other layers use the cache of Layer 18.

$$ attn\_raw = GQA(Q_{rope}, target\_K\_cache, target\_V\_cache) $$

$$ attn\_output = attn\_raw \cdot W_{o} $$

C. Laurel Auxiliary Neural Network and Attention Output Combination

During Laurel's Residual Connection, the **normalized input ($x_{norm}$)** is added instead of the original input.

$$ laurel\_x = (x_{norm} \cdot W_{laurel\_left}) \cdot W_{laurel\_right} $$

$$ laurel\_out\_normed = \mathbf{x_{norm}} + RMSNorm(laurel\_x, W_{laurel\_norm}) $$

$$ attn\_output = RMSNorm(attn\_output, W_{post\_attn\_ln}) + x_{input} $$

$$ x_{attn} = (attn\_output + laurel\_out\_normed) \times \frac{1}{\sqrt{2.0}} $$

D. Feed-Forward Network (FFN - Gate, Up, Down)

$$ x_{n2} = RMSNorm(x_{attn}, W_{pre\_ffn\_ln}) $$

$$ gate\_raw = x_{n2} \cdot W_{gate} $$

$$ up\_out = x_{n2} \cdot W_{up} $$

Layer 10 and above (Standard GELU Gate applied):
Returned with GELU applied directly inside the HW accelerator.

$$ gate\_out = GELU(gate\_raw) $$

$$ hidden = gate\_out \times up\_out $$

Below Layer 10 (Sparse Gate applied):

$$ cutoff = Mean(gate\_raw) + Std(gate\_raw) \times 1.6448536 $$

$$ sparse\_gate = \max(gate\_raw - cutoff, 0.0) $$

$$ hidden = GELU(sparse\_gate) \times up\_out $$

Final FFN Output Combination:

$$ mlp\_out = hidden \cdot W_{down} $$

$$ outputs = RMSNorm(mlp\_out, W_{post\_ffn\_ln}) + x_{attn} $$

E. Modality Update (AltUp Correction)

Update the remaining 3 modality vectors for the next layer based on the value that passed through FFN.

$$ activated = outputs \times W_{altup\_scale} $$

$$ innovation = activated - xs_{pred}[0] $$

$$ x_{n3} = \frac{RMSNorm(activated, W_{altup\_rn})}{2048.0} $$

$$ mod\_corr = \tanh(x_{n3} \cdot W_{altup\_router}) $$

$$ corr\_coefs = (W_{altup\_corr} \cdot mod\_corr) + 1.0 $$

Tensor Dimension Correction (Broadcasting): $corr\_coefs$ is reconstructed into a $(4, 1)$ size and multiplied row by row to the $(1, 2048)$ size $innovation$ vector.

$$ xs_{new} = xs_{pred} + (corr\_coefs_{[:,1]} \times innovation_{[1,:]}) $$

Finally, mix the $pli$ vector.

$$ gate\_ple = GELU(activated \cdot W_{ple\_gate}) \times pli $$

$$ mapped = RMSNorm(gate\_ple \cdot W_{ple\_proj}, W_{ple\_post\_ln}) $$

$$ xs_{new}[1:] = xs_{new}[1:] + mapped $$

5. Decode Logits

The key here is to forcibly correct the vector size (Magnitude) when back-projecting and combine them.

$$ target\_mag = \sqrt{Mean(xs[0]^{2})} $$

$$ proj\_x_{k} = xs[k+1] \cdot altup\_unprojs[k] \quad (k=0,1,2) $$

Magnitude Matching:

$$ new\_mag_{k} = \sqrt{Mean(proj\_x_{k}^{2})} $$

$$ proj\_x_{k} = proj\_x_{k} \times \frac{target\_mag}{\max(new\_mag_{k}, 10^{-12})} $$

Average the 4 corrected vectors and multiply by the final matrix.

$$ x_{final} = Mean([xs[0], proj\_x_{0}, proj\_x_{1}, proj\_x_{2}]) $$

$$ x_{final\_norm} = RMSNorm(x_{final}, W_{final\_norm}) $$

$$ Logits\_Raw = x_{final\_norm} \cdot W_{lm\_head} $$

Logit Soft-Capping:

$$ Logits = 30.0 \times \tanh\left(\frac{Logits\_Raw}{30.0}\right) $$

6. Generation & Sampling

Select the next token based on the generated Logit.

Repetition Penalty: For a previously generated token $t$, the penalty ($\rho = 1.15$) operation branches depending on the sign of the Logit.

$$ Logits_{t} = Logits_{t} \times \rho \quad (\text{if } Logits_{t} < 0) $$

$$ Logits_{t} = \frac{Logits_{t}}{\rho} \quad (\text{if } Logits_{t} \ge 0) $$

Temperature Softmax: Divide by Temperature ($T=0.65$) to adjust the distribution, and apply a high-speed Softmax through the C++ optimized SIMD kernel to get probabilities ($probs$).

$$ probs_i = \frac{\exp(Logits_i / T)}{\sum \exp(Logits_j / T)} $$

Top-P Sampling: Sort in descending order of probability, leave only tokens whose cumulative probability is less than Top-P (0.9), cut off the rest, and then proceed with random sampling.

7. System and Memory Optimization Architecture (Hardware Integration)

Core specifications for implementing bus and controller during future FPGA (KV260) design

Ping-Pong Double Buffering (hw_compute_pingpong):
While the GPU/accelerator calculates the matrix multiplication (e.g., $K$) of the current layer, it prefetches the weights required for the next calculation (e.g., $V$) to the opposite buffer in a background thread to hide the I/O waiting time (Latency) to 0.

In-place Memory Overwriting (__restrict__):
To overcome environments with extremely limited memory bandwidth, operations like RMSNorm, GELU, and Softmax overwrite the values in the original memory space directly without creating separate output tensors.

MMAP Zero-Copy Streaming:
Instead of loading the entire model into RAM, it streams directly in the form of C-Contiguous pointers one line of the necessary matrix at a time from the SSD using the OS's paging (Page Fault).


scp /home/hwkim/Desktop/github/TinyNPU-RTL/Master/newp/E4B_INT4_MODEL_INFER/NPU_wrapper.hwh ubuntu@222.100.3.239:/home/ubuntu/NPU-FPGA-Transformer-Accelerator/Master/newp/E4B_INT4_MODEL_INFER/NPU.hwh

scp/home/hwkim/Desktop/github/TinyNPU-RTL/Master/newp/E4B_INT4_MODEL_INFER/NPU.bit ubuntu@222.100.3.239:/home/ubuntu/NPU-FPGA-Transformer-Accelerator/Master/newp/E4B_INT4_MODEL_INFER/NPU.bit