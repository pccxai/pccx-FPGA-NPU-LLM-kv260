import numpy as np

def generate_hi_token_trace():
    print(" 'hi' 토큰 NPU 파이프라인 골든 모델(정답지) 생성 가동!\n")

    # =====================================================================
    # 1. 가상의 "hi" 토큰 데이터 세팅 (입력값)
    # =====================================================================
    # 텍스트 "hi"가 Tokenizer를 거쳐 임베딩 벡터가 되고,
    # MAC Array를 거쳐 제곱평균과 어텐션 스코어로 변했다고 가정!

    mean_sq_val = 16777216  # RMSNorm에 들어갈 제곱평균 (예: 2^24)
    attn_score_val = -3     # Softmax에 들어갈 Attention Score (Q*K 결과값)

    # =====================================================================
    # 2. 파이썬으로 100% 정확한 수학 연산 (FP64 부동소수점)
    # =====================================================================
    rmsnorm_true = 1.0 / np.sqrt(mean_sq_val)
    softmax_true = np.exp(attn_score_val)

    # =====================================================================
    # 3. 하드웨어 스케일링 (Q1.15 포맷으로 변환)
    # =====================================================================
    SCALE_Q15 = 32768.0

    rmsnorm_hw_expected = int(np.round(rmsnorm_true * SCALE_Q15))
    softmax_hw_expected = int(np.round(softmax_true * SCALE_Q15))

    # =====================================================================
    # 4. 결과 출력 (Testbench 복붙용)
    # =====================================================================
    print("==================================================")
    print(" [입력 데이터 (Testbench에 주사기로 꽂아넣을 값)]")
    print(f" - i_token_mean_sq  : {mean_sq_val} (32'd{mean_sq_val})")
    print(f" - i_attn_score_raw : {attn_score_val} (-16'd{-attn_score_val})")
    print("==================================================")
    print(" [하드웨어 정답지 (RTL이 무조건 뱉어내야 하는 황금 숫자)]")
    print(f" - o_rmsnorm_val  : {rmsnorm_hw_expected} (실제 수학값: {rmsnorm_true:.6f})")
    print(f" - o_softmax_prob : {softmax_hw_expected} (실제 수학값: {softmax_true:.6f})")
    print("==================================================")
    print("\n [NEXT STEP] 이 정답지를 들고 gemma_layer_top을 검증하러 갑시다!")

if __name__ == "__main__":
    generate_hi_token_trace()