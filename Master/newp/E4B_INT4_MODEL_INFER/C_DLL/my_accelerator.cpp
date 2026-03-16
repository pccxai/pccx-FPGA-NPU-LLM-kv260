#include <cmath>
#include <cstdint> // uint8_t
#include <omp.h>   

#define GELU_CONST 0.7978845608028654f

// Force export to C specification so Python ctypes can find function names.
extern "C" {
    // __restrict__: A keyword that swears to the compiler that “this pointer memory is for me only!”
    // Without this, the compiler is afraid and cannot perform SIMD parallelization properly.

    // Overwrite (In-place) operation by receiving only the pointer (*x) and length of the array
    void run_gelu_inplace(float* __restrict__ x, int length) {

        #pragma omp simd
        for (int i = 0; i < length; i++) {
            float val = x[i];
            float cube = val * val * val; 
            float inner = GELU_CONST * (val + 0.044715f * cube);
            
            x[i] = 0.5f * val * (1.0f + std::tanh(inner)); 
        }
    }

    // RMSNorm
    void run_RMSNorm_inplace(float* __restrict__ x, const float* __restrict__ gamma, int length) {
        double sum = 0.0f;

        // separately add and add up after
        #pragma omp simd reduction(+ : sum)        
        for(int i = 0; i < length; i++){ 
            float val = x[i];
            sum += val * val;
        }

        float inv_rms = 1.0f / std::sqrt((sum / (float)length) + 1e-6f);
        
        #pragma omp simd
        for(int i = 0; i < length; i++){
            x[i] = x[i] * inv_rms * gamma[i];
        }    
    }

    void run_unpack_int4_inplace(const uint8_t *__restrict__ packed, float scale, float *__restrict__ out, int packed_length)
    {
        #pragma omp simd
        for (int i = 0; i < packed_length; i++)
        {
            uint8_t p = packed[i];

            // 1. extract low 4bits
            int8_t low = p & 0x0F;
            if (low > 7)
                low -= 16;

            // extract high 4bits
            int8_t high = (p >> 4) & 0x0F;
            if (high > 7)
                high -= 16;

            out[2 * i] = (float)low * scale;
            out[2 * i + 1] = (float)high * scale;
        }
    }

    void run_rope_inplace(float *__restrict__ x, int pos, float theta_base, int num_heads, int dim)
    {
        int half = dim / 2;

        // No matter how many heads there are, the angle is the same, so it is calculated only once (128 times) and placed in the cache.
        float cos_vals[128];
        float sin_vals[128];

        #pragma omp simd
        for (int i = 0; i < half; i++)
        {
            // Frequency calculation: 1.0 / (theta_base ^ (2 * i / dim))
            float exp_val = (2.0f * (float)i) / (float)dim;
            float freq = 1.0f / std::pow(theta_base, exp_val);
            float angle = (float)pos * freq;

            cos_vals[i] = std::cos(angle);
            sin_vals[i] = std::sin(angle);
        }

        // Rotation is applied to each head using the calculated cos and sin values ​​(In-place overwrite)
        for (int h = 0; h < num_heads; h++)
        {
            int head_offset = h * dim;
            float *x_head = x + head_offset;

            #pragma omp simd
            for (int i = 0; i < half; i++)
            {
                float x0 = x_head[i];
                float x1 = x_head[i + half];

                float cos_a = cos_vals[i];
                float sin_a = sin_vals[i];

                x_head[i] = x0 * cos_a - x1 * sin_a;
                x_head[i + half] = x1 * cos_a + x0 * sin_a;
            }
        }
    }

    // Softmax Acceleration (Temperature scaling , In-place overwrite)
    void run_softmax_inplace(float *__restrict__ logits, int length, float temperature)
    {
        // prevent divide by zero
        float temp = (temperature > 1e-8f) ? temperature : 1e-8f;
        float inv_temp = 1.0f / temp;

        float max_val = -INFINITY;

        // 1. Temperature divide(mult inver) & find max val in one loop(fusion)
        #pragma omp simd reduction(max : max_val)
        for (int i = 0; i < length; i++)
        {
            logits[i] *= inv_temp;
            if (logits[i] > max_val)
            {
                max_val = logits[i];
            }
        }

        double sum_exp = 0.0;

        // 2. safe Exp expression
        // fusion Sum Up in single loop (using double for Enhance Precision)
        #pragma omp simd reduction(+ : sum_exp)
        for (int i = 0; i < length; i++)
        {
            logits[i] = std::exp(logits[i] - max_val);
            sum_exp += (double)logits[i];
        }

        // normalize
        float inv_sum = (float)(1.0 / sum_exp);

        #pragma omp simd
        for (int i = 0; i < length; i++)
        {
            logits[i] *= inv_sum;
        }
    }


    void run_gemv_int4(const float *__restrict__ vec, const uint8_t *__restrict__ mat_p, const float *__restrict__ scale, float *__restrict__ out, int M_out, int K_in)
    {
        int K_packed = K_in / 2;

        // use all core
        #pragma omp parallel for
        for (int i = 0; i < M_out; i++)
        {
            float acc = 0.0f;
            const uint8_t *row_p = mat_p + i * K_packed;

            // Using AVX2 SIMD (union of 8 data calc) in single core
            #pragma omp simd reduction(+ : acc)
            for (int k = 0; k < K_packed; k++)
            {
                uint8_t p = row_p[k];

                int8_t low = p & 0x0F;
                if (low > 7)
                    low -= 16;

                int8_t high = (p >> 4) & 0x0F;
                if (high > 7)
                    high -= 16;

                acc += vec[2 * k] * (float)low + vec[2 * k + 1] * (float)high;
            }
            out[i] = acc * scale[i];
        }
    }

    // INT4 GEMV + GeLU fusion (FFN gate, reduce memory Access)
    void run_gemv_int4_gelu(
        const float *__restrict__ vec, 
        const uint8_t *__restrict__ mat_p, 
        const float *__restrict__ scale, 
        float *__restrict__ out, 
        int M_out, 
        int K_in)
    {
        int K_packed = K_in / 2;

        #pragma omp parallel for
        for (int i = 0; i < M_out; i++)
        {
            float acc = 0.0f;
            const uint8_t *row_p = mat_p + i * K_packed;

            #pragma omp simd reduction(+ : acc)
            for (int k = 0; k < K_packed; k++)
            {
                uint8_t p = row_p[k];

                int8_t low = p & 0x0F;
                if (low > 7)
                    low -= 16;

                int8_t high = (p >> 4) & 0x0F;
                if (high > 7)
                    high -= 16;

                acc += vec[2 * k] * (float)low + vec[2 * k + 1] * (float)high;
            }
            float v = acc * scale[i];

            // GeLU
            float cube = v * v * v;
            float inner = GELU_CONST * (v + 0.044715f * cube);
            out[i] = 0.5f * v * (1.0f + std::tanh(inner));
        }
    }
}