//! HIP Kernel for QK^T Computation - Phase E
//!
//! Computes Q @ K^T for scaled dot-product attention.
//! Each thread computes one element of the attention score matrix.
//!
//! Input shapes:
//! - Q: [seq_q, num_heads, head_dim] 
//! - K: [seq_k, num_heads, head_dim]
//! Output: [seq_q, seq_k]
//!
//! Each output[i][j] = sum_h,d(Q[i][h][d] * K[j][h][d])

extern "C" {

/// QK^T kernel for attention computation
/// 
/// Computes attention scores by multiplying Q with K^T.
/// Uses one thread per output element for maximum parallelism.
__global__ void hip_flash_attention_qk_kernel(
    const float* q,           // Q tensor [seq_q, num_heads, head_dim]
    const float* k,           // K tensor [seq_k, num_heads, head_dim]  
    float* output,             // Output [seq_q, seq_k]
    const int seq_q,           // Query sequence length
    const int seq_k,           // Key sequence length
    const int num_heads,       // Number of attention heads
    const int head_dim          // Dimension per head
) {
    // Thread indices
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // seq_q index
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // seq_k index
    
    // Bounds check
    if (i >= seq_q || j >= seq_k) {
        return;
    }
    
    // Compute dot product for Q[i,:] @ K[j,:]
    float sum = 0.0f;
    
    // Loop over heads and head_dim
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            // Q[i][h][d] = q[i * num_heads * head_dim + h * head_dim + d]
            const int q_idx = i * num_heads * head_dim + h * head_dim + d;
            
            // K[j][h][d] = k[j * num_heads * head_dim + h * head_dim + d]
            const int k_idx = j * num_heads * head_dim + h * head_dim + d;
            
            sum += q[q_idx] * k[k_idx];
        }
    }
    
    // Store result
    // output[i][j] = sum
    const int output_idx = i * seq_k + j;
    output[output_idx] = sum;
}

/// QK^T kernel optimized for larger matrices
/// 
/// Uses shared memory for better performance on larger attention matrices.
__global__ void hip_flash_attention_qk_kernel_optimized(
    const float* q,           // Q tensor [seq_q, num_heads, head_dim]
    const float* k,           // K tensor [seq_k, num_heads, head_dim]
    float* output,             // Output [seq_q, seq_k]
    const int seq_q,           // Query sequence length
    const int seq_k,           // Key sequence length
    const int num_heads,       // Number of attention heads
    const int head_dim          // Dimension per head
) {
    // Shared memory for tiles
    extern __shared__ float q_tile[32][32];
    extern __shared__ float k_tile[32][32];
    
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Global indices
    const int i = by * 32 + ty;  // seq_q index
    const int j = bx * 32 + tx;  // seq_k index
    
    // Bounds check
    if (i >= seq_q || j >= seq_k) {
        return;
    }
    
    // Load tiles into shared memory
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            if (ty * 32 + tx < head_dim) {
                const int q_idx = i * num_heads * head_dim + h * head_dim + ty * 32 + tx;
                const int k_idx = j * num_heads * head_dim + h * head_dim + ty * 32 + tx;
                
                if (i < seq_q && j < seq_k && 
                    ty * 32 + tx < head_dim && 
                    (h * head_dim + ty * 32 + tx) < (num_heads * head_dim)) {
                    q_tile[ty][tx] += q[q_idx];
                    k_tile[ty][tx] += k[k_idx];
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute dot product
    float sum = 0.0f;
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            sum += q_tile[h][d] * k_tile[h][d];
        }
    }
    
    // Store result
    const int output_idx = i * seq_k + j;
    output[output_idx] = sum;
}

} // extern "C"