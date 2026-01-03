/**
 * qkt_matmul.hip - QK^T matrix multiply kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 32 threads (1 wave of 32) - exact wavefront size
 * Shared memory: 32 floats for wave32 reduction
 * Accumulator precision: FP32
 *
 * Computes QK^T per attention head:
 *   Q [batch, heads, seq_q, dim]
 *   K [batch, heads, seq_k, dim]
 *   Output [batch, heads, seq_q, seq_k]
 *
 * For each (batch, head) pair, for each query position:
 *   output[query_pos, key_pos] = sum(Q[query_pos, i] * K[key_pos, i])
 *
 * Thread safety: Block size == WARP_SIZE (32), so all threads participate
 * in reduction. No "stray" threads that could cause hazards.
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size
constexpr int MAX_HEAD_DIM = 128; // Max head dimension for register tiling

/**
 * QK^T matrix multiplication kernel with optional scale fusion
 *
 * Each block computes one row of the output:
 * - For a given (batch, head, query_pos), compute all key_pos scores
 *
 * Grid: (seq_q, num_heads, batch_size) blocks
 * Block: WARP_SIZE threads (exactly one wavefront, no partial waves)
 *
 * @param Q          Input tensor [batch, heads, seq_q, dim]
 * @param K          Input tensor [batch, heads, seq_k, dim]
 * @param output     Output tensor [batch, heads, seq_q, seq_k]
 * @param scale      Optional scale factor (e.g., 1.0 / sqrt(dim)), use 1.0 for no scaling
 * @param batch_size Number of batches
 * @param seq_q      Query sequence length
 * @param seq_k      Key sequence length
 * @param num_heads  Number of attention heads
 * @param head_dim   Dimension per head
 */
extern "C" __global__ void qkt_matmul_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ output,
    const float scale,
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
    const int q_head_offset = batch_idx * num_heads * seq_q * head_dim
                           + head_idx * seq_q * head_dim;
    const int k_head_offset = batch_idx * num_heads * seq_k * head_dim
                           + head_idx * seq_k * head_dim;

    // Q row for this query position
    const int q_row_offset = q_head_offset + query_pos * head_dim;

    // Load Q row into registers (cooperative load across wave32)
    float q_row[MAX_HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < MAX_HEAD_DIM; i += WARP_SIZE) {
        int idx = i + tid;
        if (idx < head_dim) {
            q_row[idx] = Q[q_row_offset + idx];
        } else {
            q_row[idx] = 0.0f;
        }
    }
    __syncthreads();

    // Compute QK^T: for each key position, compute dot product
    for (int key_pos = 0; key_pos < seq_k; key_pos++) {
        // K row for this key position
        const int k_row_offset = k_head_offset + key_pos * head_dim;

        // Each thread computes partial dot product
        float partial_score = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_HEAD_DIM; i += WARP_SIZE) {
            int idx = i + tid;
            if (idx < head_dim) {
                partial_score += q_row[idx] * K[k_row_offset + idx];
            }
        }

        // Store partial in shared memory
        s_partial[tid] = partial_score;
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

        // Write output: [batch, heads, seq_q, seq_k]
        // Apply scale during write (fused scale operation)
        if (tid == 0) {
            const int out_offset = batch_idx * num_heads * seq_q * seq_k
                                 + head_idx * seq_q * seq_k
                                 + query_pos * seq_k
                                 + key_pos;
            output[out_offset] = s_partial[0] * scale;
        }
        __syncthreads();
    }
}
