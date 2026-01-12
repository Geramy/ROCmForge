/**
 * mqa_kv_replicate.hip - KV head replication kernel for MQA/GQA
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Purpose: Replicate K and V tensors from num_kv_heads to num_q_heads
 * for Multi-Query Attention (MQA) and Grouped-Query Attention (GQA).
 *
 * Input layouts:
 *   K: [batch_size, seq_len, num_kv_heads, head_dim]
 *   V: [batch_size, seq_len, num_kv_heads, head_dim]
 *
 * Output layouts:
 *   K_expanded: [batch_size, seq_len, num_q_heads, head_dim]
 *   V_expanded: [batch_size, seq_len, num_q_heads, head_dim]
 *
 * Replication pattern:
 *   heads_per_kv = num_q_heads / num_kv_heads
 *   For each KV head i, replicate to query heads [i*heads_per_kv, (i+1)*heads_per_kv)
 *
 * Example: num_kv_heads=2, num_q_heads=8, heads_per_kv=4
 *   KV head 0 -> Query heads [0, 1, 2, 3]
 *   KV head 1 -> Query heads [4, 5, 6, 7]
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

/**
 * KV replication kernel for K tensor
 *
 * Each thread handles one element (batch, seq, head, dim).
 * Thread mapping: 1D grid over total elements in output tensor.
 *
 * @param K            Input K tensor [batch_size, seq_len, num_kv_heads, head_dim]
 * @param K_expanded   Output K tensor [batch_size, seq_len, num_q_heads, head_dim]
 * @param batch_size   Number of batches
 * @param seq_len      Sequence length
 * @param num_kv_heads Number of KV heads
 * @param num_q_heads  Number of query heads (must be multiple of num_kv_heads)
 * @param head_dim     Dimension per head
 */
extern "C" __global__ void mqa_kv_replicate_k_kernel(
    const float* __restrict__ K,
    float* __restrict__ K_expanded,
    const int batch_size,
    const int seq_len,
    const int num_kv_heads,
    const int num_q_heads,
    const int head_dim
) {
    // Calculate replication factor
    const int heads_per_kv = num_q_heads / num_kv_heads;

    // 1D thread index over total output elements
    const int total_elements = batch_size * seq_len * num_q_heads * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    // Decode linear index to (batch, seq, q_head, dim)
    // Layout: [batch_size, seq_len, num_q_heads, head_dim] (row-major)
    const int batch_idx = idx / (seq_len * num_q_heads * head_dim);
    const int seq_and_heads = idx % (seq_len * num_q_heads * head_dim);
    const int seq_idx = seq_and_heads / (num_q_heads * head_dim);
    const int head_and_dim = seq_and_heads % (num_q_heads * head_dim);
    const int q_head_idx = head_and_dim / head_dim;
    const int dim_idx = head_and_dim % head_dim;

    // Map query head back to source KV head
    // KV head groups: [0,0,0,0, 1,1,1,1, ...] where each group has 'heads_per_kv' entries
    const int kv_head_idx = q_head_idx / heads_per_kv;

    // Source index in K tensor: [batch_size, seq_len, num_kv_heads, head_dim]
    const int src_idx = batch_idx * seq_len * num_kv_heads * head_dim
                      + seq_idx * num_kv_heads * head_dim
                      + kv_head_idx * head_dim
                      + dim_idx;

    // Copy from source KV head to expanded query head
    K_expanded[idx] = K[src_idx];
}

/**
 * KV replication kernel for V tensor
 *
 * Same logic as K kernel, but for V tensor.
 * Each thread handles one element (batch, seq, head, dim).
 *
 * @param V            Input V tensor [batch_size, seq_len, num_kv_heads, head_dim]
 * @param V_expanded   Output V tensor [batch_size, seq_len, num_q_heads, head_dim]
 * @param batch_size   Number of batches
 * @param seq_len      Sequence length
 * @param num_kv_heads Number of KV heads
 * @param num_q_heads  Number of query heads (must be multiple of num_kv_heads)
 * @param head_dim     Dimension per head
 */
extern "C" __global__ void mqa_kv_replicate_v_kernel(
    const float* __restrict__ V,
    float* __restrict__ V_expanded,
    const int batch_size,
    const int seq_len,
    const int num_kv_heads,
    const int num_q_heads,
    const int head_dim
) {
    // Calculate replication factor
    const int heads_per_kv = num_q_heads / num_kv_heads;

    // 1D thread index over total output elements
    const int total_elements = batch_size * seq_len * num_q_heads * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    // Decode linear index to (batch, seq, q_head, dim)
    // Layout: [batch_size, seq_len, num_q_heads, head_dim] (row-major)
    const int batch_idx = idx / (seq_len * num_q_heads * head_dim);
    const int seq_and_heads = idx % (seq_len * num_q_heads * head_dim);
    const int seq_idx = seq_and_heads / (num_q_heads * head_dim);
    const int head_and_dim = seq_and_heads % (num_q_heads * head_dim);
    const int q_head_idx = head_and_dim / head_dim;
    const int dim_idx = head_and_dim % head_dim;

    // Map query head back to source KV head
    // KV head groups: [0,0,0,0, 1,1,1,1, ...] where each group has 'heads_per_kv' entries
    const int kv_head_idx = q_head_idx / heads_per_kv;

    // Source index in V tensor: [batch_size, seq_len, num_kv_heads, head_dim]
    const int src_idx = batch_idx * seq_len * num_kv_heads * head_dim
                      + seq_idx * num_kv_heads * head_dim
                      + kv_head_idx * head_dim
                      + dim_idx;

    // Copy from source KV head to expanded query head
    V_expanded[idx] = V[src_idx];
}

/**
 * Fused KV replication kernel - replicates both K and V in a single launch
 *
 * This kernel is more efficient than launching two separate kernels when
 * you need to replicate both K and V tensors. It handles both in one
 * kernel launch, reducing kernel launch overhead.
 *
 * @param K            Input K tensor [batch_size, seq_len, num_kv_heads, head_dim]
 * @param V            Input V tensor [batch_size, seq_len, num_kv_heads, head_dim]
 * @param K_expanded   Output K tensor [batch_size, seq_len, num_q_heads, head_dim]
 * @param V_expanded   Output V tensor [batch_size, seq_len, num_q_heads, head_dim]
 * @param batch_size   Number of batches
 * @param seq_len      Sequence length
 * @param num_kv_heads Number of KV heads
 * @param num_q_heads  Number of query heads (must be multiple of num_kv_heads)
 * @param head_dim     Dimension per head
 */
extern "C" __global__ void mqa_kv_replicate_fused_kernel(
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ K_expanded,
    float* __restrict__ V_expanded,
    const int batch_size,
    const int seq_len,
    const int num_kv_heads,
    const int num_q_heads,
    const int head_dim
) {
    // Calculate replication factor
    const int heads_per_kv = num_q_heads / num_kv_heads;

    // 1D thread index over total output elements
    const int total_elements = batch_size * seq_len * num_q_heads * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    // Decode linear index to (batch, seq, q_head, dim)
    // Layout: [batch_size, seq_len, num_q_heads, head_dim] (row-major)
    const int batch_idx = idx / (seq_len * num_q_heads * head_dim);
    const int seq_and_heads = idx % (seq_len * num_q_heads * head_dim);
    const int seq_idx = seq_and_heads / (num_q_heads * head_dim);
    const int head_and_dim = seq_and_heads % (num_q_heads * head_dim);
    const int q_head_idx = head_and_dim / head_dim;
    const int dim_idx = head_and_dim % head_dim;

    // Map query head back to source KV head
    // KV head groups: [0,0,0,0, 1,1,1,1, ...] where each group has 'heads_per_kv' entries
    const int kv_head_idx = q_head_idx / heads_per_kv;

    // Source index in K/V tensors: [batch_size, seq_len, num_kv_heads, head_dim]
    const int src_idx = batch_idx * seq_len * num_kv_heads * head_dim
                      + seq_idx * num_kv_heads * head_dim
                      + kv_head_idx * head_dim
                      + dim_idx;

    // Copy both K and V from source KV head to expanded query head
    K_expanded[idx] = K[src_idx];
    V_expanded[idx] = V[src_idx];
}
