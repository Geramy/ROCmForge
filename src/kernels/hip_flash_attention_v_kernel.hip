//! HIP Kernel for Attention @ V Computation - Phase E
//!
//! Computes attention-weighted values: attention @ V
//! Input shapes:
//!   attention: [seq_len, cache_len] - attention weights after softmax
//!   v: [cache_len, num_heads, head_dim] - value vectors
//! Output shape:
//!   output: [seq_len, num_heads, head_dim] - attention-weighted values
//!
//! Optimized for ROCm with shared memory and coalesced access patterns.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

extern "C" {

/// Attention @ V kernel
///
/// Computes final attention output by weighting V with attention scores.
/// Uses shared memory for V tiles and optimized memory access patterns.
///
/// @param attention Attention weights [seq_len, cache_len]
/// @param v Value vectors [cache_len, num_heads, head_dim]
/// @param output Output tensor [seq_len, num_heads, head_dim]
/// @param seq_len Current sequence length
/// @param cache_len Total cache length
/// @param num_heads Number of attention heads
/// @param head_dim Dimension per attention head
__global__ void flash_attention_v_kernel(
    const float* __restrict__ attention,
    const float* __restrict__ v,
    float* __restrict__ output,
    const int seq_len,
    const int cache_len,
    const int num_heads,
    const int head_dim
) {
    // Thread and block indexing
    const int seq_idx = blockIdx.x;  // Each block handles one sequence position
    const int head_idx = blockIdx.y;  // Each block handles one attention head
    const int dim_idx = threadIdx.x; // Each thread handles part of head dimension
    
    // Bounds checking
    if (seq_idx >= seq_len || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    
    // Shared memory for V tile
    extern __shared__ float v_tile[];
    
    // Load attention weights for this sequence position
    const float* attention_row = attention + seq_idx * cache_len;
    
    // Compute attention @ V for this head and dimension
    float sum = 0.0f;
    
    // Process cache in tiles for better memory access
    const int tile_size = blockDim.x;
    const int num_tiles = (cache_len + tile_size - 1) / tile_size;
    
    for (int tile = 0; tile < num_tiles; ++tile) {
        int cache_start = tile * tile_size;
        int cache_end = min(cache_start + tile_size, cache_len);
        
        // Load V tile into shared memory
        for (int i = threadIdx.x; i < (cache_end - cache_start) * head_dim; i += blockDim.x) {
            int cache_pos = cache_start + i / head_dim;
            int dim_pos = i % head_dim;
            
            if (cache_pos < cache_end && dim_pos < head_dim) {
                int v_idx = cache_pos * num_heads * head_dim + head_idx * head_dim + dim_pos;
                v_tile[i] = v[v_idx];
            }
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = cache_start; k < cache_end; ++k) {
            int tile_k = k - cache_start;
            float attention_weight = attention_row[k];
            
            if (dim_idx < head_dim) {
                float v_value = v_tile[tile_k * head_dim + dim_idx];
                sum += attention_weight * v_value;
            }
        }
        
        __syncthreads();
    }
    
    // Write result to output
    int output_idx = seq_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
    output[output_idx] = sum;
}

/// Optimized attention @ V kernel with vectorized loads
///
/// Uses float4 loads for better memory bandwidth utilization.
/// Suitable for larger head dimensions that are multiples of 4.
__global__ void flash_attention_v_vectorized_kernel(
    const float* __restrict__ attention,
    const float* __restrict__ v,
    float* __restrict__ output,
    const int seq_len,
    const int cache_len,
    const int num_heads,
    const int head_dim
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int vec_idx = threadIdx.x; // Process 4 elements at once
    
    const int vec_size = 4;
    const int num_vecs = head_dim / vec_size;
    
    if (seq_idx >= seq_len || head_idx >= num_heads || vec_idx >= num_vecs) {
        return;
    }
    
    // Shared memory for attention weights
    __shared__ float attention_shared[1024]; // Adjust size based on expected cache_len
    
    // Load attention weights for this sequence
    for (int i = threadIdx.x; i < cache_len; i += blockDim.x) {
        attention_shared[i] = attention[seq_idx * cache_len + i];
    }
    __syncthreads();
    
    // Process 4 elements at a time using float4
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int k = 0; k < cache_len; ++k) {
        float attention_weight = attention_shared[k];
        
        // Load 4 V values at once
        int v_base = k * num_heads * head_dim + head_idx * head_dim + vec_idx * vec_size;
        const float4* v_vec = reinterpret_cast<const float4*>(&v[v_base]);
        float4 v_values = *v_vec;
        
        // Multiply and accumulate
        result.x += attention_weight * v_values.x;
        result.y += attention_weight * v_values.y;
        result.z += attention_weight * v_values.z;
        result.w += attention_weight * v_values.w;
    }
    
    // Store result
    int output_base = seq_idx * num_heads * head_dim + head_idx * head_dim + vec_idx * vec_size;
    float4* output_vec = reinterpret_cast<float4*>(&output[output_base]);
    *output_vec = result;
}

/// Warp-level optimized attention @ V kernel
///
/// Uses warp-level primitives for reduction and optimized memory access.
/// Best for performance on modern AMD GPUs.
__global__ void flash_attention_v_warp_kernel(
    const float* __restrict__ attention,
    const float* __restrict__ v,
    float* __restrict__ output,
    const int seq_len,
    const int cache_len,
    const int num_heads,
    const int head_dim
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    if (seq_idx >= seq_len || head_idx >= num_heads) {
        return;
    }
    
    // Each warp processes a portion of the head dimension
    const int dim_per_warp = (head_dim + 31) / 32;
    const int dim_start = warp_id * dim_per_warp;
    const int dim_end = min(dim_start + dim_per_warp, head_dim);
    
    // Shared memory for attention weights
    __shared__ float attention_shared[2048]; // Support larger cache lengths
    
    // Load attention weights cooperatively
    for (int i = threadIdx.x; i < cache_len; i += blockDim.x) {
        attention_shared[i] = attention[seq_idx * cache_len + i];
    }
    __syncthreads();
    
    // Process assigned dimensions
    for (int d = dim_start + lane_id; d < dim_end; d += 32) {
        float sum = 0.0f;
        
        // Compute attention @ V for this dimension
        for (int k = 0; k < cache_len; ++k) {
            float attention_weight = attention_shared[k];
            int v_idx = k * num_heads * head_dim + head_idx * head_dim + d;
            sum += attention_weight * v[v_idx];
        }
        
        // Store result
        int output_idx = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        output[output_idx] = sum;
    }
}

} // extern "C"