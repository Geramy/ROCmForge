/**
 * Debug kernel: Compute QK^T only to verify correctness
 */

#include <hip/hip_runtime.h>

constexpr int BLOCK_SIZE = 256;

extern "C" __global__ void debug_qk_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int head_dim
) {
    const int batch_idx = blockIdx.z;
    const int query_pos = blockIdx.y;
    const int key_pos = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || query_pos >= seq_len || key_pos >= seq_len) {
        return;
    }

    // Compute dot product Q[query_pos, :] @ K[key_pos, :]
    const int batch_offset = batch_idx * seq_len * head_dim;
    const float* Q_batch = Q + batch_offset;
    const float* K_batch = K + batch_offset;

    // Regular (non-transposed) matmul for testing
    float partial = 0.0f;
    for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
        partial += Q_batch[query_pos * head_dim + i] * K_batch[key_pos * head_dim + i];
    }

    __shared__ float s_partial[BLOCK_SIZE];
    s_partial[tid] = partial;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_idx = batch_idx * seq_len * seq_len + query_pos * seq_len + key_pos;
        output[out_idx] = s_partial[0];
    }
}
