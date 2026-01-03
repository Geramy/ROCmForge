/**
 * rms_norm.hip - RMSNorm normalization kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * RMSNorm formula: output = input / sqrt(mean(input^2) + eps) * weight
 * where mean is computed over the last dimension (hidden_size)
 *
 * This is a row-wise operation: each row [seq_len, hidden_size] is normalized independently.
 *
 * Layout: row-major [seq_len, hidden_size]
 * For each row: compute RMS of all elements, then normalize each element by RMS and multiply by weight
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * RMSNorm kernel
 *
 * For each row: output[row, j] = input[row, j] / sqrt(mean(input[row, :]^2) + eps) * weight[j]
 *
 * Grid: (seq_len, 1, 1) - one block per row
 * Block: BLOCK_SIZE threads
 * Shared memory: BLOCK_SIZE floats for reduction
 *
 * @param input      Input tensor [seq_len, hidden_size]
 * @param weight     Weight tensor [hidden_size]
 * @param output     Output tensor [seq_len, hidden_size]
 * @param seq_len    Sequence length (number of rows)
 * @param hidden_size Hidden size (row length)
 * @param eps        Epsilon for numerical stability
 */
extern "C" __global__ void rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int seq_len,
    const int hidden_size,
    const float eps
) {
    // Each block processes one row
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_idx >= seq_len) {
        return;
    }

    // Shared memory for reduction
    __shared__ float s_sum_sq[BLOCK_SIZE];

    // Pointer to this row
    const float* row = input + row_idx * hidden_size;

    // Step 1: Compute sum of squares for this row
    float sum_sq = 0.0f;
    for (int j = tid; j < hidden_size; j += BLOCK_SIZE) {
        float val = row[j];
        sum_sq += val * val;
    }
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    // Step 2: Reduce to get total sum (wave reduction)
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Compute RMS = sqrt(mean(sum_sq) + eps)
    float mean_sq = s_sum_sq[0] / (float)hidden_size;
    float rms = rsqrtf(mean_sq + eps);

    // Step 4: Normalize and apply weight
    for (int j = tid; j < hidden_size; j += BLOCK_SIZE) {
        output[row_idx * hidden_size + j] = row[j] * rms * weight[j];
    }
}
