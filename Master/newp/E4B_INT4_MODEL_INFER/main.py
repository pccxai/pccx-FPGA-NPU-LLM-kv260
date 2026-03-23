import numpy as np
import CPU_CORE
import safeTensor
import math

import gc
import sys
import ctypes

# see ram usage
import os
import psutil

ACCEL_MODE = "IGPU"
#ACCEL_MODE = "CPU"

_IGPU_WEIGHT_KEYS = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"]
NUM_LAYERS = 35

if ACCEL_MODE == "IGPU":
    import IGPU_CORE as FAST_MATRIX_CORE
elif ACCEL_MODE == "CPU":
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE
# -----------------------------------------------------------
# C - DLL porting (load and init set)

base_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(base_dir, "C_DLL", "my_accelerator.so")
c_lib = ctypes.CDLL(dll_path)

# ><><><><><><><><Parameters><><><><><><><><

# RMS Norm
c_lib.run_RMSNorm_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]
c_lib.run_RMSNorm_inplace.restype = None

# Softmax
c_lib.run_softmax_inplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_float
]
c_lib.run_softmax_inplace.restype = None

# ><><><><><><><><><><><><><><><><><><><><><

# -----------------------------------------------------------





def hw_matmul(x, w, use_gelu=False):
    if ACCEL_MODE == "IGPU":
        return FAST_MATRIX_CORE.igpu_matmul_gelu(x, w) if use_gelu else FAST_MATRIX_CORE.igpu_matmul(x, w)
    else:
        if isinstance(w, tuple):
            packed, scale = w
            low = (packed & 0x0F).astype(np.int8)
            low[low > 7] -= 16
            high = (packed >> 4).astype(np.int8)
            high[high > 7] -= 16
            res = np.empty((packed.shape[0], packed.shape[1]*2), dtype=np.float32)
            res[:, 0::2] = low
            res[:, 1::2] = high
            w_real = res * scale[:, np.newaxis]
            out = np.dot(x, w_real.T)
        else:
            out = np.dot(x, w)
        return CPU_CORE.gelu(out) if use_gelu else out
'''
def rms_norm(x, gamma):
    x_f32 = x.astype(np.float32)
    rms   = np.sqrt(np.mean(x_f32 ** 2) + 1e-6)
    return (x_f32 / rms) * gamma
'''
def hw_prefetch(w, buf_idx):
    """If the next weight is an INT4 tuple, fetch it secretly in a background thread"""
    if ACCEL_MODE == "IGPU" and isinstance(w, tuple):
        FAST_MATRIX_CORE.prefetch_weight(w, buf_idx)

def hw_compute_pingpong(x, w, buf_idx, use_gelu=False):
    """Fire calculation with current buffer (buf_idx)! If it is not a tuple, bypass to general matrix multiplication"""
    if ACCEL_MODE == "IGPU" and isinstance(w, tuple):
        out = FAST_MATRIX_CORE.compute_pingpong(x, w, buf_idx)
        return CPU_CORE.gelu(out) if use_gelu else out
    else:
        return hw_matmul(x, w, use_gelu)
    
def rms_norm(x, gamma):
    # Create an independent contiguous array while converting input x to float32 (maintaining the original Python logic)
    x_f32 = np.ascontiguousarray(x.astype(np.float32))
    
    # The gamma weights are also safely guaranteed to be float32 contiguous arrays.
    gamma_c = np.ascontiguousarray(gamma.astype(np.float32))
    
    # C++ In-place overwrite operation! (x_f32 internal value is replaced by result)
    c_lib.run_RMSNorm_inplace(x_f32, gamma_c, x_f32.size)
    
    return x_f32

def get_router_modalities(x, w_norm, w_router):
    x_n = rms_norm(x, w_norm) / 2048.0
    return np.tanh(np.dot(x_n, w_router))

def forward_one_token(token_id, pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                      W_ple_proj, altup_projs, K_cache, V_cache):

    safe_token_id = int(min(token_id, W_ple_packed.shape[0] - 1))
    x0 = CPU_CORE.embedding(safe_token_id, W_embed[0], W_embed[1])
    x0 = x0 * math.sqrt(2048.0)

    xs = np.zeros((4, 2048), dtype=np.float32)
    xs[0] = x0
    for k in range(3):
        xs[k + 1] = np.dot(x0, altup_projs[k])
        
    x_proj = hw_matmul(x0, W_ple_proj) / math.sqrt(2048.0)
    x_proj = x_proj.reshape(35, 256)
    x_proj_f32 = x_proj.astype(np.float32)
    rms_vals   = np.sqrt(np.mean(x_proj_f32 ** 2, axis=1, keepdims=True) + 1e-6)
    x_proj_normed = (x_proj_f32 / rms_vals) * norm_ple

    unpacked_w_ple = CPU_CORE.embedding(safe_token_id, W_ple_packed, W_ple_scale)
    y = unpacked_w_ple.reshape(35, 256) * math.sqrt(256.0)
    pli_all = (x_proj_normed + y) * (1.0 / math.sqrt(2.0))


    # Just before entering the dataflow pipeline, load the first weight (Q).
    ping_pong = 0
    hw_prefetch(W["W_q"][0], ping_pong)

    for i in range(NUM_LAYERS):
        modalities  = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
        coef_mat    = np.dot(W["altup_pred"][i], modalities).reshape(4, 4)
        xs_pred     = xs + np.dot(coef_mat, xs)

        x                 = xs_pred[0].copy()
        inputs_normalized = rms_norm(x, W["input_ln"][i])

        # ----------------------------------------------------
        # 1. Q
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_k"][i], next_buf)
        Q = hw_compute_pingpong(inputs_normalized, W["W_q"][i], curr_buf)
        ping_pong = next_buf
        
        # 2. K
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_v"][i], next_buf)
        K = hw_compute_pingpong(inputs_normalized, W["W_k"][i], curr_buf)
        ping_pong = next_buf
        
        # 3. V
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_o"][i], next_buf)
        V = hw_compute_pingpong(inputs_normalized, W["W_v"][i], curr_buf)
        ping_pong = next_buf
        # ----------------------------------------------------

        Q, K = CPU_CORE.cpu_qk_norm(Q,   K, W["gamma_q"][i], W["gamma_k"][i])
        theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0
        Q     = CPU_CORE.cpu_rope(Q, pos=pos, theta_base=theta)
        K     = CPU_CORE.cpu_rope(K, pos=pos, theta_base=theta)

        if i < 20:
            K_cache[i, pos, :] = K
            V_cache[i, pos, :] = V
            target_k_cache = K_cache[i, :pos + 1, :]
            target_v_cache = V_cache[i, :pos + 1, :]
        else:
            if i % 5 == 4:
                target_k_cache = K_cache[19, :pos + 1, :]
                target_v_cache = V_cache[19, :pos + 1, :]
            else:
                target_k_cache = K_cache[18, :pos + 1, :]
                target_v_cache = V_cache[18, :pos + 1, :]

        attn_raw = CPU_CORE.cpu_gqa(Q, target_k_cache, target_v_cache)

        # ----------------------------------------------------
        # 4. O
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_gate"][i], next_buf)
        attn_output = hw_compute_pingpong(attn_raw, W["W_o"][i], curr_buf)
        ping_pong = next_buf
        # ----------------------------------------------------

        laurel_x          = hw_matmul(inputs_normalized, W["laurel_left"][i])
        laurel_x          = hw_matmul(laurel_x, W["laurel_right"][i])
        laurel_out_normed = inputs_normalized + rms_norm(laurel_x, W["laurel_norm"][i])

        attn_output  = rms_norm(attn_output, W["post_attn_ln"][i])
        attn_output += x
        attn_output  = (attn_output + laurel_out_normed) * (1.0 / math.sqrt(2.0))

        x_n2 = rms_norm(attn_output, W["pre_ffn_ln"][i])

        # ----------------------------------------------------
        # 5. Gate
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_up"][i], next_buf)
        gate_out = hw_compute_pingpong(x_n2, W["W_gate"][i], curr_buf, use_gelu=(i >= 10))
        ping_pong = next_buf

        # 6. Up
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        hw_prefetch(W["W_down"][i], next_buf)
        up_out = hw_compute_pingpong(x_n2, W["W_up"][i], curr_buf)
        ping_pong = next_buf
        # ----------------------------------------------------

        if i < 10:
            cutoff      = np.mean(gate_out) + np.std(gate_out) * 1.6448536
            sparse_gate = np.maximum(gate_out - cutoff, 0.0)
            hidden = CPU_CORE.gelu(sparse_gate) * up_out
        else:
            hidden = gate_out * up_out

        # ----------------------------------------------------
        # 7. Down
        curr_buf = ping_pong; next_buf = 1 - ping_pong
        # If the next layer exists, load the first batter (Q) of the next layer.
        if i < NUM_LAYERS - 1:
            hw_prefetch(W["W_q"][i+1], next_buf)
            
        mlp_out = hw_compute_pingpong(hidden, W["W_down"][i], curr_buf)
        ping_pong = next_buf
        # ----------------------------------------------------

        outputs  = rms_norm(mlp_out, W["post_ffn_ln"][i])
        outputs += attn_output

        activated  = outputs * W["altup_scale"][i]
        innovation = activated - xs_pred[0]
        mod_corr   = get_router_modalities(activated, W["altup_rn"][i], W["altup_router"][i])
        corr_coefs = np.dot(W["altup_corr"][i], mod_corr) + 1.0

        xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation
        pli      = pli_all[i]
        gate_ple = CPU_CORE.gelu(hw_matmul(activated, W["ple_gate"][i])) * pli
        mapped   = rms_norm(hw_matmul(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
        xs_new[1:] += mapped
        xs = xs_new

    return xs

def decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head):
    # While the CPU is calculating, start loading the ultra-large W_lm_head into buffer 0! (asynchronous)
    hw_prefetch(W_lm_head, 0)
    
    target_mag = np.mean(xs[0] ** 2) ** 0.5
    unembedded = [xs[0]]
    for k in range(3):
        proj_x  = np.dot(xs[k + 1], altup_unprojs[k])
        new_mag = np.mean(proj_x ** 2) ** 0.5
        proj_x *= target_mag / max(new_mag, 1e-12)
        unembedded.append(proj_x)
    x_final = np.mean(np.stack(unembedded, axis=0), axis=0)
    x_final = rms_norm(x_final, W_final_norm)
    
    # Immediately fire the x_final that has completed the calculation into the previously loaded buffer number 0.
    logits = hw_compute_pingpong(x_final, W_lm_head, buf_idx=0)
    return logits

def _sample(logits: np.ndarray, temperature: float, top_p: float,
            rep_penalty: float, generated: list) -> int:
    
    # 1. Repetition Penalty (maintain Python logic - overhead is minimal due to small number of tokens)
    if rep_penalty != 1.0 and len(generated) > 0:
        for token in set(generated):
            if logits[token] < 0:
                logits[token] *= rep_penalty
            else:
                logits[token] /= rep_penalty
                
    if temperature == 0.0: 
        return int(np.argmax(logits))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # softmax in dll(C) - SIMD

    # Confirmed as float32 contiguous array to prevent memory kinks
    logits_f32 = np.ascontiguousarray(logits.astype(np.float32))
    
    # When thrown into the C++ kernel, the values ​​inside the logits_f32 array are completely changed to 'probs' values.
    c_lib.run_softmax_inplace(logits_f32, logits_f32.size, float(temperature))
    
    probs = logits_f32 # Map only names to probs
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Top-p sampling (this part can also be transferred to C++ later, but maintained for now)
    if top_p < 1.0:
        sorted_idx  = np.argsort(probs)[::-1]
        cumsum      = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumsum - probs[sorted_idx] < top_p
        probs_filtered = np.zeros_like(probs)
        probs_filtered[sorted_idx[cutoff_mask]] = probs[sorted_idx[cutoff_mask]]
        if probs_filtered.sum() == 0: probs_filtered[sorted_idx[0]] = 1.0
        probs = probs_filtered / probs_filtered.sum()
        
    return int(np.random.choice(len(probs), p=probs))

'''python ver
def _sample(logits: np.ndarray, temperature: float, top_p: float,
            rep_penalty: float, generated: list) -> int:
    if rep_penalty != 1.0 and len(generated) > 0:
        for token in set(generated):
            if logits[token] < 0:
                logits[token] *= rep_penalty
            else:
                logits[token] /= rep_penalty
    if temperature == 0.0: return int(np.argmax(logits))
    logits = logits / max(temperature, 1e-8)
    logits_safe = logits - np.max(logits)
    probs  = np.exp(logits_safe) / np.sum(np.exp(logits_safe))
    if top_p < 1.0:
        sorted_idx  = np.argsort(probs)[::-1]
        cumsum      = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumsum - probs[sorted_idx] < top_p
        probs_filtered = np.zeros_like(probs)
        probs_filtered[sorted_idx[cutoff_mask]] = probs[sorted_idx[cutoff_mask]]
        if probs_filtered.sum() == 0: probs_filtered[sorted_idx[0]] = 1.0
        probs = probs_filtered / probs_filtered.sum()
    return int(np.random.choice(len(probs), p=probs))
'''

# Function to print current RAM usage (RSS) in MB
def print_ram_usage(step_name):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    print(f"[{step_name}] RAM Usage: {rss_mb:.2f} MB")

def main():
    print_ram_usage("1. Before Model Load")
    
    TEMPERATURE    = 0.65    # 0.3 is too rigid and easy to fall into repetition. Upgraded to 0.6.
    TOP_P          = 0.9    
    REP_PENALTY    = 1.15   # Significant increase from 1.02 to 1.15! (repeat loop absolute defense)
    MAX_NEW_TOKENS = 2048   # Max sequence len
    KV_CACHE_DIM   = 512

    FAST_MATRIX_CORE.warmup()
    print("\nGemma 3N [INT4 Optimized] - Chat Mode")
    W_embed, W_ple_packed, W_ple_scale, norm_ple, W_ple_proj, altup_projs, altup_unprojs, \
        W_final_norm, W_lm_head, W = safeTensor.load_local_weights()
    


    print("[Memory] Optimizing weighted VRAM...")
    FAST_MATRIX_CORE.preload_and_free(W, _IGPU_WEIGHT_KEYS)
    FAST_MATRIX_CORE._get_or_upload_weight(W_lm_head)

    print_ram_usage("2. After Model Load")

    # Full conversation history (simple form)
    history_tokens = []
    #K_cache = [None for _ in range(35)]
    #V_cache = [None for _ in range(35)]
    K_cache = np.zeros((NUM_LAYERS,MAX_NEW_TOKENS,KV_CACHE_DIM), dtype=np.float16)
    V_cache = np.zeros((NUM_LAYERS,MAX_NEW_TOKENS,KV_CACHE_DIM), dtype=np.float16)

    cur_pos = 0
    print_ram_usage("3. After KV Cache Allocation")

    print("\n--- Start conversation (end: 'exit' or 'quit') ---")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]: break
            if not user_input.strip(): continue

            # Apply Chat Template (simplification)
            prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
            input_tokens = CPU_CORE.tokenize(prompt)
            
            print("Model: ", end="", flush=True)
            
            xs = None
            # Prefill (process only new input)
            for token_id in input_tokens:
                xs = forward_one_token(token_id, cur_pos, W, W_embed, W_ple_packed, W_ple_scale, norm_ple,
                                       W_ple_proj, altup_projs, K_cache, V_cache)
                cur_pos += 1
            
            print_ram_usage("4. After Prefill")
            
            # Generation
            generated = []
            STOP_TOKENS = [1, 106]
            printed_text = ""  


            for _ in range(MAX_NEW_TOKENS):
                logits = decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)

                logits = 30.0 * np.tanh(logits / 30.0)

                next_token = _sample(logits, TEMPERATURE, TOP_P, REP_PENALTY, generated)
                
                if next_token in STOP_TOKENS: break
                
                generated.append(next_token)

                # [Edit] Prevent truncation of UTF-8 Korean characters and hide special tokens (<unused>)
                # Decode all tokens collected so far, skipping any special symbols.
                current_text = CPU_CORE.tokenizer.decode(generated, skip_special_tokens=True)
                
                # Only the newly added parts (new letters) are cut and displayed on the screen.
                new_text = current_text[len(printed_text):]
                print(new_text, end="", flush=True)
                
                # Update the status of printed text
                printed_text = current_text

                xs = forward_one_token(next_token, cur_pos, W, W_embed, W_ple_packed, W_ple_scale,
                                       norm_ple, W_ple_proj, altup_projs, K_cache, V_cache)
                cur_pos += 1
                
            print() # New line after model response
            gc.collect() # Turn-based Memory Cleanup
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    print_ram_usage("5. After Generation")

    print("\n[Complete] The conversation has ended.")

#import size_check

if __name__ == "__main__":
    main()
    #size_check.debug()
