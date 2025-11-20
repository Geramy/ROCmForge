#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

extern "C" {

__device__ __forceinline__ void rope_single(
    float& x_real,
    float& x_imag,
    float freq_real,
    float freq_imag
) {
    float new_real = x_real * freq_real - x_imag * freq_imag;
    float new_imag = x_real * freq_imag + x_imag * freq_real;
    x_real = new_real;
    x_imag = new_imag;
}

__global__ void rope_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int head_dim,
    int num_heads,
    int seq_len,
    int rope_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (dim_idx >= rope_dim / 2) return;
    
    int input_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    int cache_offset = seq_idx * rope_dim / 2 + dim_idx;
    
    // Get cos and sin values
    float cos_val = cos_cache[cache_offset];
    float sin_val = sin_cache[cache_offset];
    
    // Get input values (complex numbers)
    float x_real = __half2float(input[input_offset + dim_idx]);
    float x_imag = __half2float(input[input_offset + dim_idx + rope_dim / 2]);
    
    // Apply RoPE rotation
    rope_single(x_real, x_imag, cos_val, sin_val);
    
    // Write output
    output[input_offset + dim_idx] = __float2half(x_real);
    output[input_offset + dim_idx + rope_dim / 2] = __float2half(x_imag);
    
    // Copy remaining dimensions unchanged
    if (dim_idx < (head_dim - rope_dim) / 2) {
        int copy_idx = rope_dim + dim_idx * 2;
        if (copy_idx < head_dim) {
            output[input_offset + copy_idx] = input[input_offset + copy_idx];
            if (copy_idx + 1 < head_dim) {
                output[input_offset + copy_idx + 1] = input[input_offset + copy_idx + 1];
            }
        }
    }
}

} // extern "C"