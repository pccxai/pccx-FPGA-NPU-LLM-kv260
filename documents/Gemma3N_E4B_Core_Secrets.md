# Gemma 3N E4B Architecture Core Secrets

The Gemma 3N model completely subverts the standard transformer conventions established by models like LLaMA and Gemma 1/2.

**CRITICAL WARNING:** To prevent catastrophic failures (e.g., NaN explosions, representation collapse) during future code development or debugging, you must strictly adhere to the following core architectural rules.

---

## 1. Absolute Prohibition of RMSNorm `+ 1.0`

- **Rule:** Gemma 3N utilizes `scale_plus_one=False`.
- **Action:** You must use the raw weights exactly as they are.
- **Consequence of Violation:** Blindly adding `+ 1.0` to the scale weights, as was common in past architectures, will immediately cause exponential numerical explosion and result in `NaN`s.

## 2. AltUp Router Uses Tanh, NOT Softmax

- **Rule:** The 4-Stream mixing ratio calculation relies on a scaled `Tanh` function, not Softmax.
- **Action:** You must scale the input $ x $ by the dimension size (2048.0) before applying the routing weights and the $ \tanh $ activation.
- **Formula:** `np.tanh(np.dot(x_n / 2048.0, w_router))`

## 3. The True Nature of AltUp Residual Connections

- **Rule:** The mixed data (`xs_pred[0]`) is **never** the primary input to the Attention or FFN computation blocks.
- **Action:**
  - The unmodified, pure original stream (`xs[0]`) must be passed into the Attention and FFN blocks.
  - The mixed stream (`xs_pred[0]`) acts only as a 'temporary lens'. It is bypassed until the very end of the layer, where it serves solely as the base for calculating the residual delta and updating the shadow streams.

## 4. Abolition of Attention Scaling and Softcap

- **Rule:** The Attention mechanism is completely unscaled and unconstrained.
- **Action:**
  - Do not divide the $ \mathbf{Q} \cdot \mathbf{K}^T $ dot product by $ \sqrt{256} $.
  - Do not apply any Softcap function to the Attention Logits or the Final Logits. Set them to `NONE`.

## 5. Dynamic Alternating RoPE Angles

- **Rule:** The RoPE `theta_base` is not constant across the 35 layers.
- **Action:** Implement a 5-layer repeating cycle pattern: `[Local, Local, Local, Local, Global]`.
  - Local layers use a `theta_base` of 10,000.
  - Global layers use a `theta_base` of 1,000,000.

## 6. Extreme FFN Gaussian Top-K Sparsity

- **Rule:** Early FFN layers enforce aggressive neuron pruning.
- **Action:** For the first 10 layers (Layers 0 through 9), calculate the mean and standard deviation of the Gate output. Keep only the top 5% of activations (using the statistical threshold $ \mu + 1.645 \cdot \sigma $) and forcefully zero out the remaining 95% (like a ReLU).

## 7. Precise Injection Points for LAuReL and PLE

- **Rule:** Calibration modules must be injected at specific locations with specific scalings.
- **Action for LAuReL:** Execute in parallel with the Attention computation, sum the results, and scale the combined output by $ 1 / \sqrt{2.0} $.
- **Action for PLE:** Never inject PLE into the main stream (`xs[0]`) at the start of a layer. It must be selectively injected **only** into the 'shadow streams' (`xs[1~3]`) at the very end of the layer.
