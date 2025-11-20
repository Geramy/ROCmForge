#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

extern "C" {

__global__ void softmax_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int vocab_size,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int vocab_idx = threadIdx.x;
    
    if (vocab_idx >= vocab_size) return;
    
    // Find maximum value for numerical stability
    __shared__ float max_val;
    
    if (vocab_idx == 0) {
        max_val = -INFINITY;
    }
    __syncthreads();
    
    float val = __half2float(input[batch_idx * vocab_size + vocab_idx]);
    atomicMax(&max_val, val);
    __syncthreads();
    
    // Compute exp and sum
    __shared__ float sum_exp;
    
    if (vocab_idx == 0) {
        sum_exp = 0.0f;
    }
    __syncthreads();
    
    float exp_val = expf(val - max_val);
    atomicAdd(&sum_exp, exp_val);
    __syncthreads();
    
    // Compute softmax probability
    float prob = exp_val / sum_exp;
    output[batch_idx * vocab_size + vocab_idx] = __float2half(prob);
}

__global__ void scaled_softmax_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    float scale,
    int vocab_size,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int vocab_idx = threadIdx.x;
    
    if (vocab_idx >= vocab_size) return;
    
    // Apply scale and find maximum value
    __shared__ float max_val;
    
    if (vocab_idx == 0) {
        max_val = -INFINITY;
    }
    __syncthreads();
    
    float val = __half2float(input[batch_idx * vocab_size + vocab_idx]) * scale;
    atomicMax(&max_val, val);
    __syncthreads();
    
    // Compute exp and sum
    __shared__ float sum_exp;
    
    if (vocab_idx == 0) {
        sum_exp = 0.0f;
    }
    __syncthreads();
    
    float exp_val = expf(val - max_val);
    atomicAdd(&sum_exp, exp_val);
    __syncthreads();
    
    // Compute softmax probability
    float prob = exp_val / sum_exp;
    output[batch_idx * vocab_size + vocab_idx] = __float2half(prob);
}

} // extern "C"