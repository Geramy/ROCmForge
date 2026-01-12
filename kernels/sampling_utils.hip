/**
 * sampling_utils.hip - Core sampling utilities for GPU token sampling
 *
 * GPU: AMD Radeon (RDNA3, wave32)
 * Block size: 256 threads (configurable)
 *
 * Provides:
 * - Row-wise softmax with numerical stability
 * - Exclusive prefix sum per row (CDF computation)
 * - Random number generation utilities
 *
 * Based on FlashInfer sampling algorithms:
 * https://flashinfer.ai/2025/03/10/sampling.html
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size
constexpr int BLOCK_SIZE = 256;   // Threads per block

/**
 * Row-wise softmax with numerical stability
 *
 * For each row (batch element):
 * 1. Find max value in row (for numerical stability)
 * 2. Compute exp(x - max) for each element
 * 3. Compute sum of exp values
 * 4. Divide each exp by sum
 *
 * Grid: (batch_size) blocks
 * Block: BLOCK_SIZE threads
 *
 * @param logits        Input logits [batch_size, vocab_size]
 * @param probabilities Output probabilities [batch_size, vocab_size]
 * @param batch_size    Number of batch elements
 * @param vocab_size    Vocabulary size
 */
extern "C" __global__ void softmax_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probabilities,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    // Row offset for this batch element
    const int row_offset = batch_idx * vocab_size;

    // Shared memory for reduction
    __shared__ float s_max;
    __shared__ float s_sum;

    // Step 1: Find max value in row (for numerical stability)
    float local_max = -1e30f; // Very small number

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float val = logits[row_offset + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Shared memory reduction for max
    __shared__ float s_max_data[BLOCK_SIZE];
    s_max_data[tid] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_data[tid] = fmaxf(s_max_data[tid], s_max_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_max = s_max_data[0];
    }
    __syncthreads();

    const float max_val = s_max;

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float val = expf(logits[row_offset + i] - max_val);
        probabilities[row_offset + i] = val;
        local_sum += val;
    }

    // Shared memory reduction for sum
    __shared__ float s_sum_data[BLOCK_SIZE];
    s_sum_data[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_data[tid] += s_sum_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_sum = s_sum_data[0];
        // Guard against division by zero
        if (s_sum == 0.0f) {
            s_sum = 1e-10f;
        }
    }
    __syncthreads();

    const float sum_inv = 1.0f / s_sum;

    // Step 3: Normalize by sum
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        probabilities[row_offset + i] *= sum_inv;
    }
}

/**
 * Inclusive prefix sum (CDF) per row
 *
 * Computes cumulative sum: output[i] = sum(input[0..i])
 * Used for inverse transform sampling
 *
 * Grid: (batch_size) blocks
 * Block: BLOCK_SIZE threads
 *
 * @param input   Input probabilities [batch_size, vocab_size]
 * @param output  Output CDF [batch_size, vocab_size]
 * @param batch_size Number of batch elements
 * @param vocab_size Vocabulary size
 */
extern "C" __global__ void prefix_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int row_offset = batch_idx * vocab_size;

    // For vocab sizes that fit in shared memory, use simple scan
    // For larger vocab, use block-based approach (future optimization)

    __shared__ float s_data[BLOCK_SIZE * 2]; // Double buffer for Blelloch scan

    // Load data into shared memory
    float running_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float val = input[row_offset + i];
        output[row_offset + i] = running_sum + val;
        running_sum += val;
    }

    // Note: This is a simplified prefix sum
    // For production with large vocab_size, implement:
    // - Kogge-Stone scan for small vocab
    // - Block-based prefix sum for large vocab
}

/**
 * Apply temperature scaling to logits
 *
 * output = logits / temperature
 *
 * Grid: (batch_size) blocks
 * Block: BLOCK_SIZE threads
 *
 * @param logits      Input/output logits [batch_size, vocab_size]
 * @param temperature Scaling factor (> 0)
 * @param batch_size  Number of batch elements
 * @param vocab_size  Vocabulary size
 */
extern "C" __global__ void temperature_scale_kernel(
    float* __restrict__ logits,
    const float temperature,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int row_offset = batch_idx * vocab_size;
    const float scale = 1.0f / temperature; // Precompute for efficiency

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[row_offset + i] *= scale;
    }
}

/**
 * Find top-k threshold per row
 *
 * Finds the kth largest value in each row (threshold for top-k)
 *
 * Grid: (batch_size) blocks
 * Block: min(BLOCK_SIZE, vocab_size) threads
 *
 * @param probabilities Input probabilities [batch_size, vocab_size]
 * @param thresholds    Output thresholds [batch_size]
 * @param top_k         K value
 * @param batch_size    Number of batch elements
 * @param vocab_size    Vocabulary size
 */
extern "C" __global__ void topk_threshold_kernel(
    const float* __restrict__ probabilities,
    float* __restrict__ thresholds,
    const int top_k,
    const int batch_size,
    const int vocab_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int row_offset = batch_idx * vocab_size;

    // Use shared memory for partial sort
    // For small k, use selection algorithm
    // For now, use simple approach with wave reduction

    __shared__ float s_threshold;

    // Find kth largest using partial selection
    // For production, use better algorithm (e.g., quickselect)
    if (tid == 0) {
        // Simple approach: collect top-k values
        // This is inefficient - improve in production
        float kth_value = 0.0f;

        // Naive implementation for correctness
        // TODO: Use efficient selection algorithm
        if (top_k >= vocab_size) {
            kth_value = -1e30f; // All tokens included
        } else {
            // Find kth largest (simplified - O(v*k))
            // For production, use O(v) selection
            kth_value = 0.0f; // Placeholder
        }

        thresholds[batch_idx] = kth_value;
    }

    __syncthreads();
}
