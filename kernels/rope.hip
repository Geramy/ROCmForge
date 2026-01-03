/**
 * rope.hip - Rotary Positional Embedding (RoPE) kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * RoPE kernel - applies rotary position embeddings to input tensor
 *
 * Rotates pairs of dimensions in head_dim using precomputed cos/sin:
 * x0' = x0 * cos - x1 * sin
 * x1' = x0 * sin + x1 * cos
 *
 * The cos/sin arrays should contain pre-extracted values for each token position,
 * arranged as [seq_len, head_dim/2] in row-major order.
 *
 * @param input     Input tensor [seq_len, num_heads, head_dim] (row-major, modified in-place)
 * @param cos       Precomputed cosine values [seq_len, head_dim/2] - one row per token position
 * @param sin       Precomputed sine values [seq_len, head_dim/2] - one row per token position
 * @param seq_len   Sequence length (number of tokens)
 * @param num_heads Number of attention heads
 * @param head_dim  Dimension per head (must be even)
 */
extern "C" __global__ void rope_kernel(
    float* __restrict__ input,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    // Each thread handles one pair in one head for one token
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_pair = threadIdx.x;

    const int half_dim = head_dim / 2;

    // Boundary checks
    if (token_idx >= seq_len || head_idx >= num_heads || dim_pair >= half_dim) {
        return;
    }

    // Linear index into input tensor [seq_len, num_heads, head_dim]
    const int base = (token_idx * num_heads + head_idx) * head_dim;

    // Pair indices: (dim_pair, dim_pair + half_dim)
    const int i0 = base + dim_pair;
    const int i1 = base + dim_pair + half_dim;

    // cos/sin index: [token_idx, dim_pair] - cos/sin are pre-extracted per token
    const int cos_idx = token_idx * half_dim + dim_pair;

    // Load cos/sin values
    const float c = cos[cos_idx];
    const float s = sin[cos_idx];

    // Load input values
    const float x0 = input[i0];
    const float x1 = input[i1];

    // Apply rotation: x_rotated = x * cos + x_flipped * sin
    input[i0] = x0 * c - x1 * s;
    input[i1] = x0 * s + x1 * c;
}
