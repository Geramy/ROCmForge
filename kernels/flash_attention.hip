/**
 * flash_attention.hip - Fused Attention kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Fuses: QK^T matmul, scale, mask, softmax, softmax*V into single kernel
 * Eliminates CPU-GPU round-trips for better performance
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * FlashAttention kernel - computes attention in a single fused pass
 *
 * Algorithm per block (one query position):
 *   1. Compute QK^T for one query position vs all key positions
 *   2. Apply scaling factor (1.0 / sqrt(head_dim))
 *   3. Apply causal mask if enabled
 *   4. Compute softmax (row-wise)
 *   5. Compute output: softmax × V
 *
 * Tensor layouts (row-major):
 *   Q: [batch_size, seq_len, num_heads, head_dim]
 *   K: [batch_size, seq_len, num_heads, head_dim]
 *   V: [batch_size, seq_len, num_heads, head_dim]
 *   output: [batch_size, seq_len, num_heads, head_dim]
 *   mask: [batch_size, seq_len, seq_len] (optional, row-major)
 *
 * @param Q         Query tensor
 * @param K         Key tensor
 * @param V         Value tensor
 * @param output    Output tensor (pre-allocated)
 * @param mask      Optional causal mask (use nullptr for no mask)
 * @param scale     Scaling factor (typically 1.0 / sqrt(head_dim))
 * @param batch_size Number of batches
 * @param seq_len   Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim  Dimension per head
 */
extern "C" __global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const float* __restrict__ mask,
    const float scale,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    // Each thread block processes one (batch, head, query_pos) triple
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_pos = blockIdx.x;

    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || query_pos >= seq_len) {
        return;
    }

    // Shared memory for intermediate results
    // s_partial: for thread-local partial results during reduction
    // s_scores: for final attention scores (seq_len entries)
    __shared__ float s_partial[BLOCK_SIZE];  // Partial reduction results
    __shared__ float s_scores[256];          // Final attention scores
    __shared__ float s_max;                  // Max for numerical stability
    __shared__ float s_sum;                  // Sum for normalization

    // Initialize shared memory to avoid garbage values
    s_partial[tid] = 0.0f;
    if (tid < seq_len) {
        s_scores[tid] = 0.0f;
    }
    if (tid == 0) {
        s_max = 0.0f;
        s_sum = 0.0f;
    }
    __syncthreads();

    // Base pointers for this batch
    // Tensor layout: [batch_size, seq_len, head_dim] (3D, not 4D)
    const int batch_offset = batch_idx * seq_len * head_dim;
    const float* Q_batch = Q + batch_offset;
    const float* K_batch = K + batch_offset;
    const float* V_batch = V + batch_offset;
    float* output_batch = output + batch_offset;

    // Load Q row for this query position
    // Q_row[head_dim] - all head dimensions for this query position
    float q_row[128];  // Max head_dim we support (register limit)
    const int q_row_offset = query_pos * head_dim;

    // Load Q row (each thread loads multiple elements if head_dim > BLOCK_SIZE)
    for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
        if (i < 128) {
            q_row[i] = Q_batch[q_row_offset + i];
        } else {
            // Fallback for larger head_dim - this would need more complex handling
            // For now, we support head_dim <= 128
        }
    }

    // Compute QK^T scores for this query position vs all key positions
    // CPU matmul formula: C[i,j] = sum_l A[i,l] * B[l,j] where B is accessed transposed
    // For attention: C[query_pos, key_pos] = sum_l Q[query_pos, l] * K[l, key_pos]
    for (int key_pos = 0; key_pos < seq_len; key_pos++) {
        // Clear partial reduction memory
        s_partial[tid] = 0.0f;
        __syncthreads();

        // Each thread computes a partial dot product
        float partial_score = 0.0f;

        for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
            if (i < 128) {
                // CPU matmul accesses K as transposed: K[l * seq_len + j]
                // For QK^T[i,j], we sum over l: Q[i,l] * K[l,j]
                // Here: i=query_pos, j=key_pos, l=i (loop variable)
                partial_score += q_row[i] * K_batch[i * seq_len + key_pos];
            }
        }

        // Store partial result in s_partial (separate from final scores)
        s_partial[tid] = partial_score;
        __syncthreads();

        // Reduce partial scores (wave32 optimized)
        for (int stride = 16; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial[tid] += s_partial[tid + stride];
            }
            __syncthreads();
        }

        // First thread handles the score
        if (tid == 0) {
            float score = s_partial[0] * scale;

            // Apply mask if provided
            if (mask != nullptr) {
                const int mask_idx = batch_idx * seq_len * seq_len + query_pos * seq_len + key_pos;
                float mask_val = mask[mask_idx];
                if (mask_val < -1e30f) {
                    score = mask_val;
                }
            }
            // Note: No automatic causal mask - use mask tensor if causal masking is needed

            s_scores[key_pos] = score;  // Store at key_pos index for softmax pass
        }
        __syncthreads();

        // After computing all scores, we need to compute softmax
        // But we need to store all scores first - using s_scores as a ring buffer
        // For seq_len <= BLOCK_SIZE, we can store all scores
        // For larger seq_len, we'd need multiple passes (future optimization)
    }

    // Compute softmax (reduce to find max, then sum of exp)
    // First pass: find max (use s_partial for reduction, preserve s_scores)
    float max_val = -1e30f;
    for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
        float val = s_scores[j];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Reduce max across threads using s_partial (don't corrupt s_scores!)
    s_partial[tid] = max_val;
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = s_partial[tid];
            float b = s_partial[tid + stride];
            s_partial[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    max_val = s_partial[0];
    s_max = max_val;
    __syncthreads();

    // Second pass: compute exp and sum (use s_partial for sum reduction)
    float sum = 0.0f;
    for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
        float val = expf(s_scores[j] - max_val);
        sum += val;
    }

    // Reduce sum across threads using s_partial
    s_partial[tid] = sum;
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    sum = s_partial[0];
    s_sum = sum;
    __syncthreads();

    // Normalize scores: compute softmax and store back in s_scores
    float inv_sum = 1.0f / (sum + 1e-6f);  // Avoid div by zero
    for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
        // Re-compute exp from original scores
        float val = expf(s_scores[j] - max_val) * inv_sum;
        s_scores[j] = val;  // Now s_scores contains softmax weights
    }
    __syncthreads();

    // Compute output: softmax × V
    // output[query_pos, dim] = sum over key_pos of (softmax[key_pos] * V[key_pos, dim])
    const int out_row_offset = query_pos * head_dim;

    for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {
        s_partial[tid] = 0.0f;
        __syncthreads();

        float partial_sum = 0.0f;
        for (int j = tid; j < seq_len; j += BLOCK_SIZE) {
            // V[j, dim_idx]
            const int v_offset = j * head_dim + dim_idx;
            partial_sum += s_scores[j] * V_batch[v_offset];
        }

        s_partial[tid] = partial_sum;
        __syncthreads();

        // Reduce partial sums using s_partial
        for (int stride = 16; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_partial[tid] += s_partial[tid + stride];
            }
            __syncthreads();
        }

        // Write output
        if (tid == 0 && dim_idx < 128) {
            output_batch[out_row_offset + dim_idx] = s_partial[0];
        }
    }
}
