/**
 * scale.hip - Element-wise scaling kernel for attention scores
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 */

#include <hip/hip_runtime.h>

/**
 * Scale every element in the attention scores matrix
 *
 * @param scores  Flattened [batch_size * seq_len * seq_len] attention scores (row-major)
 * @param scale   Multiplicative scaling factor (typically 1.0 / sqrt(head_dim))
 * @param batch_size Number of batches
 * @param seq_len   Sequence length
 */
extern "C" __global__ void scale_kernel(
    float* __restrict__ scores,
    const float scale,
    const int batch_size,
    const int seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * seq_len;

    if (idx < total) {
        scores[idx] *= scale;
    }
}
