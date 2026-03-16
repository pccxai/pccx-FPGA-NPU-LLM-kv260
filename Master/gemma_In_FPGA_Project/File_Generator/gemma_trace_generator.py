import numpy as np

def generate_hi_token_trace():
    print(" 'hi' token NPU pipeline golden model (answer sheet) generation is in operation!\n")

    # =====================================================================
    # 1. Virtual “hi” token data setting (input value)
    # =====================================================================
    # The text "hi" goes through the Tokenizer and becomes an embedding vector,
    # Assuming that it has been converted to mean square and attention score through MAC Array.

    mean_sq_val = 16777216  # Mean square to enter RMSNorm (e.g. 2^24)
    attn_score_val = -3     # Attention Score to be entered into Softmax (Q*K result)

    # =====================================================================
    # 2. 100% accurate mathematical operations in Python (FP64 floating point)
    # =====================================================================
    rmsnorm_true = 1.0 / np.sqrt(mean_sq_val)
    softmax_true = np.exp(attn_score_val)

    # =====================================================================
    # 3. Hardware scaling (conversion to Q1.15 format)
    # =====================================================================
    SCALE_Q15 = 32768.0

    rmsnorm_hw_expected = int(np.round(rmsnorm_true * SCALE_Q15))
    softmax_hw_expected = int(np.round(softmax_true * SCALE_Q15))

    # =====================================================================
    # 4. Output results (for copy-and-paste in Testbench)
    # =====================================================================
    print("==================================================")
    print(" [Input data (value to be injected into Testbench with syringe)]")
    print(f" - i_token_mean_sq  : {mean_sq_val} (32'd{mean_sq_val})")
    print(f" - i_attn_score_raw : {attn_score_val} (-16'd{-attn_score_val})")
    print("==================================================")
    print(" [Hardware answer sheet (golden numbers that RTL must spit out)]")
    print(f" - o_rmsnorm_val : {rmsnorm_hw_expected} (actual math value: {rmsnorm_true:.6f})")
    print(f" - o_softmax_prob : {softmax_hw_expected} (actual math value: {softmax_true:.6f})")
    print("==================================================")
    print("\n [NEXT STEP] Proceed to take this answer sheet and go verify gemma_layer_top!")

if __name__ == "__main__":
    generate_hi_token_trace()