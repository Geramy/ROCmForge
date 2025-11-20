#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

extern "C" {

__global__ void layer_norm_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    float epsilon,
    int hidden_size,
    int seq_len
) {
    int seq_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (hidden_idx >= hidden_size) return;
    
    // Compute mean and variance for this sequence
    __shared__ float sum;
    __shared__ float sum_sq;
    
    if (hidden_idx == 0) {
        sum = 0.0f;
        sum_sq = 0.0f;
    }
    __syncthreads();
    
    float val = __half2float(input[seq_idx * hidden_size + hidden_idx]);
    atomicAdd(&sum, val);
    atomicAdd(&sum_sq, val * val);
    __syncthreads();
    
    float mean = sum / hidden_size;
    float var = (sum_sq / hidden_size) - (mean * mean);
    float std_dev = sqrtf(var + epsilon);
    
    // Apply layer normalization
    float normalized = (val - mean) / std_dev;
    float w = __half2float(weight[hidden_idx]);
    float b = __half2float(bias[hidden_idx]);
    
    output[seq_idx * hidden_size + hidden_idx] = __float2half(normalized * w + b);
}

} // extern "C"