/**
 * swiglu.hip - SwiGLU activation kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * SwiGLU activation: output = gate * swish(up)
 * where swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * This is an element-wise operation on [seq_len, intermediate_size] tensors.
 *
 * Layout: row-major [seq_len, intermediate_size]
 * Each element: output[i] = gate[i] * swish(up[i])
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * SwiGLU activation kernel
 *
 * Computes: output[i] = gate[i] * swish(up[i])
 * where swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Grid: (ceil(total_elements / BLOCK_SIZE), 1, 1)
 * Block: BLOCK_SIZE threads
 *
 * @param gate    Gate projection tensor [seq_len, intermediate_size]
 * @param up      Up projection tensor [seq_len, intermediate_size]
 * @param output  Output tensor [seq_len, intermediate_size]
 * @param seq_len            Sequence length (rows)
 * @param intermediate_size Intermediate size (columns)
 */
extern "C" __global__ void swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    const int seq_len,
    const int intermediate_size
) {
    // Linear index into 2D tensor [seq_len, intermediate_size]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = seq_len * intermediate_size;

    if (idx >= total_elements) {
        return;
    }

    // Load values
    const float g = gate[idx];
    const float u = up[idx];

    // Swish activation: swish(x) = x * sigmoid(x)
    // sigmoid(x) = 1 / (1 + exp(-x))
    const float sigmoid_up = 1.0f / (1.0f + expf(-u));
    const float swish_up = u * sigmoid_up;

    // SwiGLU: gate * swish(up)
    output[idx] = g * swish_up;
}
