/**
 * causal_mask.hip - Causal mask kernel for FlashAttention
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 32 threads (1 wave of 32) - exact wavefront size
 *
 * Creates causal mask where mask[query_pos, key_pos] = -inf if key_pos > query_pos
 * Layout: [batch, heads, seq_q, seq_k] explicit for auditable indexing
 *
 * Contract:
 * - Input: None (generates mask in-place)
 * - Output: mask tensor where mask[i, j] = -inf if j > i (key > query)
 * - CPU reference: create_causal_mask from src/attention/mask.rs
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * Causal mask kernel
 *
 * Each block handles one (batch, head, query_pos) triple:
 * - Fills one row of the mask tensor for a specific query position
 * - For each key position: set -inf if key > query, else 0
 *
 * Grid: (seq_q, num_heads, batch_size) blocks
 * Block: WARP_SIZE threads (exactly one wavefront, no partial waves)
 *
 * @param mask      Output mask tensor [batch, heads, seq_q, seq_k]
 * @param batch_size Number of batches
 * @param seq_len   Sequence length (seq_q = seq_k for causal)
 * @param num_heads Number of attention heads
 */
extern "C" __global__ void causal_mask_kernel(
    float* __restrict__ mask,
    const int batch_size,
    const int seq_len,
    const int num_heads
) {
    // Block indexing: each block handles one (query_pos, head, batch) triple
    const int query_pos = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;

    // Bounds check
    if (batch_idx >= batch_size || head_idx >= num_heads || query_pos >= seq_len) {
        return;
    }

    // === Layout: [batch, heads, seq_q, seq_k] ===
    const int batch_head_offset = batch_idx * num_heads * seq_len * seq_len
                                + head_idx * seq_len * seq_len;
    const int row_offset = batch_head_offset + query_pos * seq_len;

    // Fill mask row for this query position
    // Each thread handles a subset of key positions
    for (int key_pos = tid; key_pos < seq_len; key_pos += WARP_SIZE) {
        // Causal: mask out future positions
        // query_pos can only attend to key_pos <= query_pos
        if (key_pos > query_pos) {
            mask[row_offset + key_pos] = -(__builtin_inff());  // -inf
        } else {
            mask[row_offset + key_pos] = 0.0f;
        }
    }
}
