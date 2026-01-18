/**
 * fused_rope_kvappend.hip - Fused RoPE + KV Cache Append Kernel
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32 threads)
 *
 * This kernel fuses Rotary Positional Embedding (RoPE) with KV cache append to:
 * 1. Reduce memory bandwidth by writing directly to KV cache
 * 2. Reduce kernel launch overhead by combining operations
 * 3. Eliminate intermediate buffer allocations
 *
 * RoPE rotation formula:
 *   x0' = x0 * cos - x1 * sin
 *   x1' = x0 * sin + x1 * cos
 *
 * where (x0, x1) are pairs of dimensions in head_dim, and cos/sin are
 * precomputed per-token position values.
 *
 * KV Cache Layout:
 * - Key cache: [num_layers, num_kv_heads, max_seq_len, head_dim]
 * - Value cache: [num_layers, num_kv_heads, max_seq_len, head_dim]
 * - Each token position is appended at the current sequence length
 *
 * Memory Bandwidth Analysis:
 * - Unfused: Read K/V (2 * hidden_size) + Write rotated K/V (2 * hidden_size) +
 *            Read K/V for append (2 * hidden_size) + Write to cache (2 * hidden_size) = 8 * hidden_size
 * - Fused: Read K/V (2 * hidden_size) + Read cos/sin (hidden_size) + Write to cache (2 * hidden_size) = 5 * hidden_size
 * - Bandwidth reduction: ~1.6x for memory traffic
 * - Kernel launch reduction: 2 -> 1 (50% reduction)
 *
 * Reference: Based on rope.hip pattern and KV cache append operations
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * Fused RoPE + KV cache append kernel for Keys
 *
 * For each new token:
 * 1. Apply RoPE rotation to key tensor
 * 2. Write rotated keys directly to KV cache at current position
 *
 * Grid: (num_heads, 1, 1) - one block per head
 * Block: BLOCK_SIZE threads (handles head_dim/2 pairs)
 *
 * @param keys       Input key tensor [num_heads, head_dim] (rotated in-place if needed)
 * @param cos        Precomputed cosine values [seq_len, head_dim/2]
 * @param sin        Precomputed sine values [seq_len, head_dim/2]
 * @param k_cache    Key cache [num_layers, num_kv_heads, max_seq_len, head_dim]
 * @param layer_idx  Layer index for KV cache offset
 * @param token_idx  Token position for KV cache append
 * @param num_heads  Number of attention heads
 * @param head_dim   Dimension per head (must be even)
 * @param max_seq_len Maximum sequence length (cache size)
 */
extern "C" __global__ void fused_rope_k_cache_append_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ k_cache,
    const int layer_idx,
    const int token_idx,
    const int num_heads,
    const int head_dim,
    const int max_seq_len
) {
    // Each block handles one head
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= num_heads) {
        return;
    }

    const int half_dim = head_dim / 2;

    // Boundary check
    if (tid >= half_dim) {
        return;
    }

    // cos/sin index: [token_idx, tid] - one row per token position
    const int cos_idx = token_idx * half_dim + tid;

    // Load cos/sin values
    const float c = cos[cos_idx];
    const float s = sin[cos_idx];

    // Load key values for this head
    const int key_base = head_idx * head_dim;
    const int i0 = key_base + tid;
    const int i1 = key_base + tid + half_dim;

    const float x0 = keys[i0];
    const float x1 = keys[i1];

    // Apply RoPE rotation
    const float rot0 = x0 * c - x1 * s;
    const float rot1 = x0 * s + x1 * c;

    // Write directly to KV cache
    // Cache layout: [layer_idx, head_idx, token_idx, head_dim]
    const int cache_base = (layer_idx * num_heads + head_idx) * max_seq_len + token_idx;
    k_cache[cache_base * head_dim + tid] = rot0;
    k_cache[cache_base * head_dim + tid + half_dim] = rot1;
}

/**
 * Fused RoPE + KV cache append kernel for Values
 *
 * Values don't need RoPE, but we still benefit from fusing the append
 * operation to reduce kernel launches.
 *
 * Grid: (num_heads, 1, 1) - one block per head
 * Block: BLOCK_SIZE threads
 *
 * @param values     Input value tensor [num_heads, head_dim]
 * @param v_cache    Value cache [num_layers, num_kv_heads, max_seq_len, head_dim]
 * @param layer_idx  Layer index for KV cache offset
 * @param token_idx  Token position for KV cache append
 * @param num_heads  Number of attention heads
 * @param head_dim   Dimension per head
 * @param max_seq_len Maximum sequence length (cache size)
 */
extern "C" __global__ void fused_v_cache_append_kernel(
    const float* __restrict__ values,
    float* __restrict__ v_cache,
    const int layer_idx,
    const int token_idx,
    const int num_heads,
    const int head_dim,
    const int max_seq_len
) {
    // Each block handles one head
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= num_heads) {
        return;
    }

    // Boundary check
    if (tid >= head_dim) {
        return;
    }

    // Load value
    const int value_base = head_idx * head_dim;
    const float val = values[value_base + tid];

    // Write directly to KV cache
    // Cache layout: [layer_idx, head_idx, token_idx, head_dim]
    const int cache_base = (layer_idx * num_heads + head_idx) * max_seq_len + token_idx;
    v_cache[cache_base * head_dim + tid] = val;
}

/**
 * Combined fused RoPE + KV cache append kernel (K and V together)
 *
 * Fuses both key rotation/append and value append into a single kernel launch.
 * This is optimal for single-token generation where we process one K/V pair.
 *
 * Grid: (num_heads, 1, 1) - one block per head
 * Block: BLOCK_SIZE threads
 *
 * @param keys       Input key tensor [num_heads, head_dim]
 * @param values     Input value tensor [num_heads, head_dim]
 * @param cos        Precomputed cosine values [seq_len, head_dim/2]
 * @param sin        Precomputed sine values [seq_len, head_dim/2]
 * @param k_cache    Key cache [num_layers, num_kv_heads, max_seq_len, head_dim]
 * @param v_cache    Value cache [num_layers, num_kv_heads, max_seq_len, head_dim]
 * @param layer_idx  Layer index for KV cache offset
 * @param token_idx  Token position for KV cache append
 * @param num_heads  Number of attention heads
 * @param head_dim   Dimension per head (must be even)
 * @param max_seq_len Maximum sequence length (cache size)
 */
extern "C" __global__ void fused_rope_kv_cache_append_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int layer_idx,
    const int token_idx,
    const int num_heads,
    const int head_dim,
    const int max_seq_len
) {
    // Each block handles one head
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= num_heads) {
        return;
    }

    const int half_dim = head_dim / 2;

    // Cache base indices for this head and token
    const int cache_base = (layer_idx * num_heads + head_idx) * max_seq_len + token_idx;
    const int k_cache_offset = cache_base * head_dim;
    const int v_cache_offset = cache_base * head_dim;

    // Process key with RoPE rotation
    if (tid < half_dim) {
        // cos/sin index
        const int cos_idx = token_idx * half_dim + tid;

        // Load cos/sin
        const float c = cos[cos_idx];
        const float s = sin[cos_idx];

        // Load key values
        const int key_base = head_idx * head_dim;
        const float x0 = keys[key_base + tid];
        const float x1 = keys[key_base + tid + half_dim];

        // Apply rotation and write to K cache
        k_cache[k_cache_offset + tid] = x0 * c - x1 * s;
        k_cache[k_cache_offset + tid + half_dim] = x0 * s + x1 * c;
    }

    // Process value (no rotation needed)
    if (tid < head_dim) {
        const int value_base = head_idx * head_dim;
        v_cache[v_cache_offset + tid] = values[value_base + tid];
    }
}

/**
 * Batch fused RoPE + KV cache append kernel for prompt processing
 *
 * Optimized for processing multiple tokens at once (prompt phase).
 * Each block handles one head for all tokens in the batch.
 *
 * Grid: (num_heads, 1, 1) - one block per head
 * Block: BLOCK_SIZE threads
 *
 * @param keys       Input key tensor [num_tokens, num_heads, head_dim]
 * @param values     Input value tensor [num_tokens, num_heads, head_dim]
 * @param cos        Precomputed cosine values [num_tokens, head_dim/2]
 * @param sin        Precomputed sine values [num_tokens, head_dim/2]
 * @param k_cache    Key cache [num_layers, num_kv_heads, max_seq_len, head_dim]
 * @param v_cache    Value cache [num_layers, num_kv_heads, max_seq_len, head_dim]
 * @param layer_idx  Layer index for KV cache offset
 * @param start_idx  Starting token position in KV cache
 * @param num_tokens Number of tokens to process
 * @param num_heads  Number of attention heads
 * @param head_dim   Dimension per head (must be even)
 * @param max_seq_len Maximum sequence length (cache size)
 */
extern "C" __global__ void fused_rope_kv_cache_append_batch_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int layer_idx,
    const int start_idx,
    const int num_tokens,
    const int num_heads,
    const int head_dim,
    const int max_seq_len
) {
    // Each block handles one head
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= num_heads) {
        return;
    }

    const int half_dim = head_dim / 2;

    // Process all tokens for this head
    for (int token_offset = 0; token_offset < num_tokens; ++token_offset) {
        const int token_idx = start_idx + token_offset;

        // Cache base indices
        const int cache_base = (layer_idx * num_heads + head_idx) * max_seq_len + token_idx;
        const int k_cache_offset = cache_base * head_dim;
        const int v_cache_offset = cache_base * head_dim;

        // Input base indices for this token and head
        const int key_base = (token_offset * num_heads + head_idx) * head_dim;
        const int value_base = (token_offset * num_heads + head_idx) * head_dim;
        const int cos_base = token_offset * half_dim;

        // Process key with RoPE rotation
        if (tid < half_dim) {
            const float c = cos[cos_base + tid];
            const float s = sin[cos_base + tid];

            const float x0 = keys[key_base + tid];
            const float x1 = keys[key_base + tid + half_dim];

            k_cache[k_cache_offset + tid] = x0 * c - x1 * s;
            k_cache[k_cache_offset + tid + half_dim] = x0 * s + x1 * c;
        }

        // Process value
        if (tid < head_dim) {
            v_cache[v_cache_offset + tid] = values[value_base + tid];
        }

        __syncthreads();
    }
}
