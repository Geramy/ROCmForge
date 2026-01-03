/**
 * mask.hip - Causal masking kernel for attention scores
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 */

#include <hip/hip_runtime.h>

/**
 * Apply causal mask to attention scores
 *
 * Sets scores to -infinity where mask is -infinity.
 * This prevents attending to future tokens in autoregressive decoding.
 *
 * @param scores  Flattened [batch_size * seq_len * seq_len] attention scores (row-major)
 * @param mask    Flattened [batch_size * seq_len * seq_len] mask values (row-major)
 * @param batch_size Number of batches
 * @param seq_len   Sequence length
 */
extern "C" __global__ void mask_kernel(
    float* __restrict__ scores,
    const float* __restrict__ mask,
    const int batch_size,
    const int seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * seq_len;

    if (idx < total) {
        float m = mask[idx];
        // Check for -inf (use bit pattern comparison for robustness)
        if (m < -1e30f) {
            scores[idx] = m;
        }
    }
}
