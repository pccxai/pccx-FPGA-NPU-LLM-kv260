import numpy as np
import time

# =====================================================================
# 1. 하드웨어 메모리 & NPU 모사 세팅
# =====================================================================
class HardwareKVCacheManager:
    """ FPGA DDR4에 할당된 연속된 정적 메모리 공간 모사 """
    def __init__(self, max_seq_len, embed_dim):
        self.max_seq_len = max_seq_len
        self.current_pos = 0 
        
        # [seq_len, embed_dim] 크기의 물리적 BRAM/DDR4 영역 할당 (동적 할당 X)
        self.k_cache = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
        self.v_cache = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
        
    def write_cache(self, new_k, new_v):
        """ NPU가 계산한 새 토큰의 K, V를 캐시에 기록 (Pointer 증가) """
        self.k_cache[self.current_pos, :] = new_k
        self.v_cache[self.current_pos, :] = new_v
        self.current_pos += 1
        
    def read_active_cache(self):
        """ 0번부터 현재 포인터까지의 모든 K, V 데이터를 NPU로 DMA 전송 모사 """
        return self.k_cache[:self.current_pos, :], self.v_cache[:self.current_pos, :]

def mock_npu_gemv(vector, weight_matrix, name=""):
    """ NPU의 64x64 Systolic Array 타일링 연산 모사 """
    # 실제론 여기서 타일 단위로 쪼개서 BRAM에서 읽어와 곱하지만, 파이프라인 가독성을 위해 np.dot으로 압축
    print(f"   ⚡ [NPU 가동] {name} 행렬 곱셈 (GEMV) 처리 중...")
    time.sleep(0.1) # 하드웨어 연산 딜레이 모사
    return np.dot(vector, weight_matrix)

# =====================================================================
# 2. Gemma 3N 미니 디코드 파이프라인
# =====================================================================
def run_gemma_mini_pipeline():
    # 간단한 한글 토큰 사전 (테스트 벡터용)
    vocab = {0: "안", 1: "녕", 2: "하", 3: "세", 4: "요", 5: "<EOS>"}
    
    # 하드웨어 스펙 세팅 (미니 버전)
    embed_dim = 64
    max_seq_len = 10
    
    # 가짜 뇌 (Gemma 3N의 가중치라고 가정)
    W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
    W_k = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
    W_v = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
    
    # 메모리 풀 초기화
    kv_manager = HardwareKVCacheManager(max_seq_len, embed_dim)
    
    print("====================================================")
    print("NPU Auto-regressive Decode Loop 시작!")
    print("====================================================\n")
    
    # 1. Prefill 단계 모사 (사용자가 "안녕" 이라고 입력함)
    print("[Prefill 단계] 프롬프트 '안녕' 처리 중...")
    input_tokens = [0, 1] # "안", "녕"
    
    # (원래는 병렬로 처리하지만, 구조 이해를 위해 토큰 단위로 밀어넣음)
    for token in input_tokens:
        dummy_embed = np.random.randn(1, embed_dim) # 단어의 임베딩 벡터
        
        # NPU 연산: Q, K, V 뽑아내기
        q = mock_npu_gemv(dummy_embed, W_q, name="W_q")
        k = mock_npu_gemv(dummy_embed, W_k, name="W_k")
        v = mock_npu_gemv(dummy_embed, W_v, name="W_v")
        
        # KV 캐시에 저장
        kv_manager.write_cache(k, v)
    
    print(f"Prefill 완료! 현재 KV Cache 포인터: {kv_manager.current_pos}\n")
    
    # 2. Decode 단계 (한 글자씩 "하", "세", "요" 생성)
    # 방금 처리한 "녕"을 시작점으로 잡음
    current_token_idx = 1 # "녕"
    generated_text = "안녕"
    
    # 타겟 정답(Test Vector) 강제 매핑 (하->세->요->EOS)
    target_sequence = [2, 3, 4, 5] 
    
    for step in range(4):
        print(f"[Decode Step {step+1}] 이전 단어 처리 중...")
        
        # [Step 1] 현재 단어를 NPU에 넣고 Q, K, V 연산
        dummy_embed = np.random.randn(1, embed_dim)
        q = mock_npu_gemv(dummy_embed, W_q, name="W_q")
        k = mock_npu_gemv(dummy_embed, W_k, name="W_k")
        v = mock_npu_gemv(dummy_embed, W_v, name="W_v")
        
        # [Step 2] 방금 만든 K, V를 캐시에 추가 기록
        kv_manager.write_cache(k, v)
        
        # [Step 3] Attention 연산 (KV Cache 통째로 읽어오기!) 🔥
        # 이게 바로 LLM의 메모리 대역폭을 다 잡아먹는 주범이야!
        past_k, past_v = kv_manager.read_active_cache()
        print(f"  [Memory] KV Cache에서 {past_k.shape[0]}개의 과거 토큰(K,V) DMA 로드 완료!")
        
        # NPU 가동: Q x K^T (현재 단어가 과거 단어들과 얼마나 연관있는지 점수 매기기)
        attn_scores = np.dot(q, past_k.T) 
        attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores)) # Softmax
        
        # NPU 가동: Score x V (최종 문맥 벡터 생성)
        context_vector = np.dot(attn_probs, past_v)
        
        # [Step 4] 원래라면 여기서 복잡한 MLP와 Output Projection을 거쳐 
        # 확률이 가장 높은 단어(argmax)를 뽑지만, 우리는 구조 검증용이므로 정해진 타겟을 뱉음!
        next_token_idx = target_sequence[step]
        next_word = vocab[next_token_idx]
        
        generated_text += next_word
        print(f"[출력] NPU가 다음 단어를 예측했습니다 -> **{next_word}**\n")
        
        if next_token_idx == 5: # <EOS> 토큰
            print("<EOS> 감지! 생성을 종료합니다.")
            break
            
        # 뱉어낸 단어를 다음 스텝의 입력으로 피드백! (Auto-regressive)
        current_token_idx = next_token_idx

    print("====================================================")
    print(f"최종 생성된 문장: {generated_text}")
    print("====================================================")

if __name__ == "__main__":
    run_gemma_mini_pipeline()