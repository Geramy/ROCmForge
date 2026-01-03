/**
 * weighted_matmul.hip - softmax × V matrix multiply kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 32 threads (1 wave of 32) - exact wavefront size
 * Shared memory: 32 floats for wave32 reduction
 * Accumulator precision: FP32
 *
 * Computes softmax × V per attention head:
 *   weights [batch, heads, seq_q, seq_k]
 *   V       [batch, heads, seq_k, dim]
 *   output  [batch, heads, seq_q, dim]
 *
 * For each (batch, head, query_pos) triple:
 *   output[query_pos, dim] = sum over seq_k of (weights[query_pos, key_pos] * V[key_pos, dim])
 *
 * Thread safety: Block size == WARP_SIZE (32), so all threads participate
 * in reduction. No "stray" threads that could cause hazards.
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size
constexpr int MAX_HEAD_DIM = 128; // Max head dimension for register tiling

/**
 * Weighted matmul kernel (softmax weights × V)
 *
 * Each block computes one output row:
 * - For a given (batch, head, query_pos), compute output[dim]
 *
 * Grid: (seq_q, num_heads, batch_size) blocks
 * Block: WARP_SIZE threads (exactly one wavefront, no partial waves)
 *
 * @param weights   Attention weights [batch, heads, seq_q, seq_k]
 * @param V         Value tensor [batch, heads, seq_k, dim]
 * @param output    Output tensor [batch, heads, seq_q, dim]
 * @param batch_size Number of batches
 * @param seq_q     Query sequence length
 * @param seq_k     Key sequence length (weights dim, V rows)
 * @param num_heads Number of attention heads
 * @param head_dim  Dimension per head
 */
extern "C" __global__ void weighted_matmul_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ V,
    float* __restrict__ output,
    const int batch_size,
    const int seq_q,
    const int seq_k,
    const int num_heads,
    const int head_dim
) {
    // Block indexing: each block handles one (query_pos, head, batch) triple
    const int query_pos = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;

    // Bounds check
    if (batch_idx >= batch_size || head_idx >= num_heads || query_pos >= seq_q) {
        return;
    }

    // Shared memory for wave32 reduction
    __shared__ float s_partial[WARP_SIZE];

    // === Layout: [batch, heads, seq, dim] ===
    // Index = batch * heads * seq * dim + head * seq * dim + seq * dim + d
    const int weights_head_offset = batch_idx * num_heads * seq_q * seq_k
                                  + head_idx * seq_q * seq_k;
    const int v_head_offset = batch_idx * num_heads * seq_k * head_dim
                            + head_idx * seq_k * head_dim;
    const int out_head_offset = batch_idx * num_heads * seq_q * head_dim
                             + head_idx * seq_q * head_dim;

    // For each output dimension, compute weighted sum over seq_k
    for (int d = 0; d < head_dim; d++) {
        float partial_sum = 0.0f;

        // Each thread computes partial sum for a subset of key positions
        for (int key_pos = tid; key_pos < seq_k; key_pos += WARP_SIZE) {
            // weights[query_pos, key_pos]
            const int w_idx = weights_head_offset + query_pos * seq_k + key_pos;
            // V[key_pos, d]
            const int v_idx = v_head_offset + key_pos * head_dim + d;

            partial_sum += weights[w_idx] * V[v_idx];
        }

        // Store partial in shared memory
        s_partial[tid] = partial_sum;
        __syncthreads();

        // Wave32 reduction: sum all partials
        // Start at stride=16 (half of wave32)
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial[tid] += s_partial[tid + stride];
            }
            __syncthreads();
        }

        // Write output: [batch, heads, seq_q, dim]
        if (tid == 0) {
            const int out_idx = out_head_offset + query_pos * head_dim + d;
            output[out_idx] = s_partial[0];
        }
        __syncthreads();
    }
}
