/**
 * softmax.hip - Row-wise softmax kernel with numerical stability
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 * Shared memory: 512 floats (2×256 for max + sum reduction)
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * Row-wise softmax with numerical stability
 *
 * Applies softmax independently to each row of the scores matrix.
 * Each row represents attention weights for one query position.
 *
 * Algorithm:
 * 1. Find max value in row (for numerical stability)
 * 2. Compute exp(x - max) for each element
 * 3. Sum the exponentiated values
 * 4. Normalize by dividing by sum
 *
 * @param scores     Flattened [batch_size * seq_len * seq_len] attention scores (row-major)
 *                     Modified in-place with softmax output
 * @param batch_size Number of batches
 * @param seq_len    Sequence length (also row length in scores)
 */
extern "C" __global__ void softmax_kernel(
    float* __restrict__ scores,
    const int batch_size,
    const int seq_len
) {
    // Shared memory for parallel reduction
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    // One block per row
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_rows = batch_size * seq_len;

    if (row_idx >= total_rows) return;

    // Pointer to this row
    float* row = scores + row_idx * seq_len;

    // === Step 1: Find max value in row (numerical stability) ===
    float max_val = -1e20f;  // Effectively -inf
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, row[i]);
    }

    s_max[tid] = max_val;
    __syncthreads();

    // Wave32 reduction (start at 16, half of wavefront size)
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    max_val = s_max[0];

    // === Step 2: Compute exp(x - max) and sum ===
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        float val = expf(row[i] - max_val);
        sum += val;
        row[i] = val;  // Store unnormalized exp temporarily
    }

    s_sum[tid] = sum;
    __syncthreads();

    // Wave32 reduction for sum
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    // === Step 3: Normalize by dividing by sum ===
    const float inv_sum = 1.0f / s_sum[0];
    for (int i = tid; i < seq_len; i += BLOCK_SIZE) {
        row[i] *= inv_sum;
    }
}
