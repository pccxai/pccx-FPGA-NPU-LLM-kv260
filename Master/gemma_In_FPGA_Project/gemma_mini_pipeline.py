import numpy as np
import time

# =====================================================================
# 1. Hardware memory & NPU simulation settings
# =====================================================================
class HardwareKVCacheManager:
    """ Simulates contiguous static memory space allocated to FPGA DDR4 """
    def __init__(self, max_seq_len, embed_dim):
        self.max_seq_len = max_seq_len
        self.current_pos = 0

        # [seq_len, embed_dim] physical BRAM/DDR4 area allocation of size (dynamic allocation
        self.k_cache = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
        self.v_cache = np.zeros((max_seq_len, embed_dim), dtype=np.float32)

    def write_cache(self, new_k, new_v):
        """ Write the K and V of the new token calculated by the NPU to the cache (Pointer increases) """
        self.k_cache[self.current_pos, :] = new_k
        self.v_cache[self.current_pos, :] = new_v
        self.current_pos += 1

    def read_active_cache(self):
        """ Simulate DMA transfer of all K, V data from 0 to the current pointer to NPU """
        return self.k_cache[:self.current_pos, :], self.v_cache[:self.current_pos, :]

def mock_npu_gemv(vector, weight_matrix, name=""):
    """ NPU's 64x64 Systolic Array tiling operation simulation """
    # In reality, it is divided into tile units, read from BRAM, and multiplied, but for pipeline readability, it is compressed into np.dot.
    print(f" [NPU up] Processing {name} matrix multiplication (GEMV)...")
    time.sleep(0.1) # Hardware operation delay simulation
    return np.dot(vector, weight_matrix)

# =====================================================================
# 2. Gemma 3N mini decode pipeline
# =====================================================================
def run_gemma_mini_pipeline():
    # Simple Korean token dictionary (for test vectors)
    vocab = {0: "An", 1: "Nyeong", 2: "Ha", 3: "Se", 4: "Yo", 5: "<EOS>"}

    # Setting hardware specifications (mini version)
    embed_dim = 64
    max_seq_len = 10

    # Fake brain (assuming weights of Gemma 3N)
    W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
    W_k = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01
    W_v = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.01

    # Initialize memory pool
    kv_manager = HardwareKVCacheManager(max_seq_len, embed_dim)

    print("====================================================")
    print("NPU Auto-regressive Decode Loop begins!")
    print("====================================================\n")

    # 1. Simulate the Prefill step (user enters “Hi”)
    print("[Prefill stage] Processing prompt 'Hello'...")
    input_tokens = [0, 1] # "hi"

    # (Originally processed in parallel, but pushed into token units to understand the structure)
    for token in input_tokens:
        dummy_embed = np.random.randn(1, embed_dim) # Embedding vector of word

        # NPU operations: extracting Q, K, V
        q = mock_npu_gemv(dummy_embed, W_q, name="W_q")
        k = mock_npu_gemv(dummy_embed, W_k, name="W_k")
        v = mock_npu_gemv(dummy_embed, W_v, name="W_v")

        # Save to KV Cache
        kv_manager.write_cache(k, v)

    print(f"Prefill completed! Current KV Cache pointer: {kv_manager.current_pos}\n")

    # 2. Decode step (generating “Ha”, “Se”, and “Yo” one by one)
    # Take the "Nyeong" just processed as a starting point
    current_token_idx = 1 # “Nyeong”
    generated_text = "Hello"

    # Forced mapping of target answer (Test Vector) (Ha->Se->Yo->EOS)
    target_sequence = [2, 3, 4, 5]

    for step in range(4):
        print(f"[Decode Step {step+1}] Processing previous word...")

        # [Step 1] Insert the current word into the NPU and calculate Q, K, V
        dummy_embed = np.random.randn(1, embed_dim)
        q = mock_npu_gemv(dummy_embed, W_q, name="W_q")
        k = mock_npu_gemv(dummy_embed, W_k, name="W_k")
        v = mock_npu_gemv(dummy_embed, W_v, name="W_v")

        # [Step 2] Add the K and V just created to the cache.
        kv_manager.write_cache(k, v)

        # [Step 3] Attention operation (Read the entire KV Cache!)
        # This is the main culprit that eats up all of LLM’s memory bandwidth.
        past_k, past_v = kv_manager.read_active_cache()
        print(f" [Memory] DMA load of {past_k.shape[0]} past tokens (K,V) from KV Cache completed!")

        # NPU operation: Q x K^T (score how related the current word is to past words)
        attn_scores = np.dot(q, past_k.T)
        attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores)) # Softmax

        # NPU operation: Score x V (generates final context vector)
        context_vector = np.dot(attn_probs, past_v)

        # [Step 4] Originally, this would be done through complex MLP and Output Projection.
        # We pick the word with the highest probability (argmax), but since we are doing it for structural verification, we spit out a set target.
        next_token_idx = target_sequence[step]
        next_word = vocab[next_token_idx]

        generated_text += next_word
        print(f"[output] NPU predicted next word -> **{next_word}**\n")

        if next_token_idx == 5: # <EOS> Token
            print("<EOS> detected! End creation.")
            break

        # Feedback the words the user spit out as input for the next step! (Auto-regressive)
        current_token_idx = next_token_idx

    print("====================================================")
    print(f"Final generated sentence: {generated_text}")
    print("====================================================")

if __name__ == "__main__":
    run_gemma_mini_pipeline()