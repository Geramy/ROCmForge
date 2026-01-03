# ROCmForge Implementation Roadmap

> Generated: 2026-01-03
> Last Updated: 2026-01-03 (Phase 1 & 2 Complete)
> Order: Following `implementation_principles.md`
> Rule: **Make it correct → make it measurable → then make it fast.**
> Progress: 2/5 phases complete

---

## Your Development Environment

| Component | Value |
|-----------|-------|
| **GPU** | AMD Radeon RX 7900 XT (Navi 31) |
| **Architecture** | `gfx1100` (RDNA3) |
| **Wavefront Size** | **32** (not 64!) |
| **ROCm** | 7.1.52802 |
| **Target Flag** | `--offload-arch=gfx1100` |

**⚠️ Important: Block Size Tuning for Your GPU**

Since you have **wave32** (not wave64 like CDNA3 datacenter GPUs):
- Use block sizes that are multiples of 32
- Wave reduction: `for (int stride = 16; stride > 0; stride >>= 1)`
- No MFMA instructions (RDNA3 doesn't have them)
- FP16 is still fast and useful for LLMs

```cpp
// Example block size for your RX 7900 XT
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // Not 64!
```

---

## Phase 1: Replace Stubs (Week 1) ✅

**Priority: Fix the no-ops first**
**Status: COMPLETE - 2025-01-03**

### Task 1.1: scale_kernel

**Contract:**
```rust
// Input: scores [batch_size * seq_len * seq_len] row-major
// Output: scores[i] *= scale (in-place)
// CPU reference: |scores[i] - (input[i] * scale)| < 1e-5
```

**HIP Kernel:**
```cpp
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
```

**Test:**
```rust
#[cfg(feature = "rocm")]
#[test]
fn test_scale_gpu_matches_cpu() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let scale = 0.5f32;
    // CPU reference
    let cpu: Vec<f32> = input.iter().map(|x| x * scale).collect();
    // GPU run
    // Compare within 1e-5
}
```

### Task 1.2: mask_kernel

**Contract:**
```rust
// Input: scores [batch_size * seq_len * seq_len], mask same layout
// Output: if mask[i] == -inf, scores[i] = -inf
// CPU reference: element-wise comparison
```

**HIP Kernel:**
```cpp
extern "C" __global__ void mask_kernel(
    float* __restrict__ scores,
    const float* __restrict__ mask,
    const int batch_size,
    const int seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * seq_len;

    if (idx < total && mask[idx] < -1e30f) {
        scores[idx] = mask[idx];
    }
}
```

### Task 1.3: softmax_kernel

**Contract:**
```rust
// Input: scores [batch_size * seq_len * seq_len] row-major
// Output: row-wise softmax (each row sums to 1.0)
// CPU reference: softmax_in_place from src/attention/softmax.rs
```

**HIP Kernel:**
```cpp
extern "C" __global__ void softmax_kernel(
    float* __restrict__ scores,
    const int batch_size,
    const int seq_len
) {
    // One block per row
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_rows = batch_size * seq_len;

    if (row_idx >= total_rows) return;

    float* row = scores + row_idx * seq_len;

    // Find max (numerical stability)
    __shared__ float s_max[256];
    float max_val = -1e20f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    s_max[tid] = max_val;
    __syncthreads();
    // Wave32 reduction (your RX 7900 XT uses wave32)
    for (int stride = 16; stride > 0; stride >>= 1) {  // Changed from 128
        if (tid < stride) s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        __syncthreads();
    }
    max_val = s_max[0];

    // Compute exp and sum
    __shared__ float s_sum[256];
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = expf(row[i] - max_val);
        sum += val;
        row[i] = val;
    }
    s_sum[tid] = sum;
    __syncthreads();
    // Wave32 reduction
    for (int stride = 16; stride > 0; stride >>= 1) {  // Changed from 128
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    // Normalize
    float inv_sum = 1.0f / s_sum[0];
    for (int i = tid; i < seq_len; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}
```

**Phase 1 Exit Criteria:**
- [x] All three kernels pass CPU vs GPU tests
- [x] Tests cover edge cases (empty, single element, large values)
- [x] `rocm-smi` shows GPU activity during tests

---

## Phase 2: RoPE + KV Append (Week 2) ✅

**Priority: Biggest latency win - eliminates 2 GPU↔CPU round-trips per decode**
**Status: COMPLETE - 2025-01-03**

### Task 2.1: Understand Current Fallback

From `src/attention/rope.rs:226-233`:

```rust
pub fn apply_rope_device(
    &self,
    input: &DeviceTensor,
    output: &mut DeviceTensor,
    position: usize,
) -> Result<(), AttentionError> {
    // TODO: Implement GPU kernel for RoPE
    // Current: Copy to host → CPU RoPE → copy back to device
    let input_host = input.to_host_vec()?;
    let mut output_host = self.apply_rope_cpu(&input_host, position)?;
    output.copy_from_host(&output_host)?;
    Ok(())
}
```

### Task 2.2: RoPE Contract

**Tensor Layout:**
```
Q, K: [seq_len, num_heads, head_dim]
- RoPE rotates pairs: (dim[0], dim[1]), (dim[2], dim[3]), etc.
- cos/sin precomputed: [max_seq_len, head_dim/2]
```

**CPU Reference:**
```rust
// From src/attention/rope.rs
fn apply_rope_cpu(&self, input: &[f32], position: usize) -> Result<Vec<f32>> {
    // For each pair (dim[2*i], dim[2*i+1]):
    // x0' = x0 * cos - x1 * sin
    // x1' = x0 * sin + x1 * cos
}
```

### Task 2.3: rope_kernel (Separate First)

```cpp
extern "C" __global__ void rope_kernel(
    float* __restrict__ input,      // [seq_len, num_heads, head_dim]
    const float* __restrict__ cos,  // [max_seq_len, head_dim/2]
    const float* __restrict__ sin,  // [max_seq_len, head_dim/2]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int dim_pair = threadIdx.x;

    const int half_dim = head_dim / 2;
    if (token_idx >= seq_len || head_idx >= num_heads || dim_pair >= half_dim) return;

    const int pos = position + token_idx;

    // Linear index into input
    const int base = (token_idx * num_heads + head_idx) * head_dim;

    // Indices for the pair
    const int i0 = base + dim_pair;
    const int i1 = base + dim_pair + half_dim;

    // cos/sin for this position and pair
    const int cos_idx = pos * half_dim + dim_pair;
    const float c = cos[cos_idx];
    const float s = sin[cos_idx];

    // Apply rotation
    const float x0 = input[i0];
    const float x1 = input[i1];
    input[i0] = x0 * c - x1 * s;
    input[i1] = x0 * s + x1 * c;
}
```

### Task 2.4: rope_kv_append_fused (The Real Win)

```cpp
extern "C" __global__ void rope_kv_append_kernel(
    const float* __restrict__ k_input,  // [1, num_kv_heads, head_dim] - single token
    const float* __restrict__ v_input,  // [1, num_kv_heads, head_dim]
    float* __restrict__ k_cache,        // [max_seq_len, num_kv_heads, head_dim]
    float* __restrict__ v_cache,        // [max_seq_len, num_kv_heads, head_dim]
    const float* __restrict__ cos,      // [max_seq_len, head_dim/2]
    const float* __restrict__ sin,      // [max_seq_len, head_dim/2]
    const int num_kv_heads,
    const int head_dim,
    const int seq_len,     // Current write position
    const int max_seq_len,
    const int position
) {
    const int head_idx = blockIdx.y;
    const int dim_pair = threadIdx.x;

    const int half_dim = head_dim / 2;
    if (head_idx >= num_kv_heads || dim_pair >= half_dim) return;

    const int pos = position + seq_len;
    const int cos_idx = pos * half_dim + dim_pair;
    const float c = cos[cos_idx];
    const float s = sin[cos_idx];

    // Process K
    const int k_in_base = head_idx * head_dim;
    const int k_out_base = (seq_len * num_kv_heads + head_idx) * head_dim;

    const int k_i0 = k_in_base + dim_pair;
    const int k_i1 = k_in_base + dim_pair + half_dim;
    const int k_o0 = k_out_base + dim_pair;
    const int k_o1 = k_out_base + dim_pair + half_dim;

    float k0 = k_input[k_i0];
    float k1 = k_input[k_i1];
    k_cache[k_o0] = k0 * c - k1 * s;
    k_cache[k_o1] = k0 * s + k1 * c;

    // Process V (no RoPE, just append)
    const int v_in_base = head_idx * head_dim;
    const int v_out_base = (seq_len * num_kv_heads + head_idx) * head_dim;

    v_cache[v_out_base + dim_pair] = v_input[v_in_base + dim_pair];
    v_cache[v_out_base + dim_pair + half_dim] = v_input[v_in_base + dim_pair + half_dim];
}
```

**Phase 2 Exit Criteria:**
- [x] RoPE kernel passes CPU vs GPU test (5/5 tests passed)
- [x] Single decode step stays on GPU (no to_host_vec in RoPE path)
- [ ] Measure latency before/after (should be ~2x improvement) - future work

---

## Phase 3: Fused Attention (Week 3-4)

**Priority: Performance unlock after path is stable**

### Task 3.1: Profile Current Path

Before optimizing, measure:

```rust
#[cfg(feature = "rocm")]
#[test]
fn profile_attention() {
    let backend = HipBackend::new()?;

    // Setup: batch=1, seq_len=32, head_dim=64
    let mut timings = Vec::new();

    for _ in 0..100 {
        let start = Instant::now();
        // Run attention
        let output = attention_forward(...)?;
        timings.push(start.elapsed());
    }

    println!("Mean: {:?}", mean(&timings));
    println!("Min: {:?}", min(&timings));
    println!("Max: {:?}", max(&timings));

    // Check rocm-smi for GPU utilization
}
```

### Task 3.2: FlashAttention Contract

**Input:**
```
Q: [batch_size, seq_len, num_heads, head_dim]
K: [batch_size, seq_len, num_heads, head_dim]
V: [batch_size, seq_len, num_heads, head_dim]
scale: 1.0 / sqrt(head_dim)
```

**Output:**
```
O: [batch_size, seq_len, num_heads, head_dim]
```

**Algorithm:**
```
For each batch, head:
  For each block of K, V:
    Load Q tile to registers
    Load K tile to LDS
    Load V tile to LDS
    Compute QK^T for this tile
    Update online softmax statistics
    Accumulate output

  Normalize and write output
```

### Task 3.3: FlashAttention Kernel

```cpp
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_DMODEL = 64;

template <typename T, int HEAD_DIM>
__global__ void flash_attention_fwd_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ output,
    const float scale,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    // LDS for tiling
    __shared__ float K_tile[BLOCK_N][BLOCK_DMODEL];
    __shared__ float V_tile[BLOCK_N][BLOCK_DMODEL];

    // Online softmax statistics
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM] = {0};

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int row_idx = blockIdx.x * BLOCK_M + threadIdx.y;

    // ... (full implementation in kernel_research.md)
}
```

**Phase 3 Exit Criteria:**
- [ ] FlashAttention passes correctness test vs CPU
- [ ] FlashAttention is faster than separate kernels (measure it!)
- [ ] Profile shows GPU utilization > 50%

---

## Phase 4: MLP Ops (Week 5)

**Priority: Complete GPU path**

### Task 4.1: SwiGLU Contract

**Formula:**
```
SwiGLU(x) = gate(x) * swish(up(x))
swish(x) = x * sigmoid(x)
```

**Current CPU fallback** at `src/backend/hip_backend.rs:1284-1300`:

```rust
// Apply SwiGLU activation on CPU
for i in 0..swiglu_host.len() {
    let gate_val = gate_host[i];
    let up_val = up_host[i];
    let sigmoid_up = 1.0 / (1.0 + (-up_val).exp());
    let swish_up = up_val * sigmoid_up;
    swiglu_host[i] = gate_val * swish_up;
}
```

### Task 4.2: swiglu_kernel

```cpp
extern "C" __global__ void swiglu_kernel(
    const float* __restrict__ gate,  // [rows, cols]
    const float* __restrict__ up,    // [rows, cols]
    float* __restrict__ output,      // [rows, cols]
    const int rows,
    const int cols
) {
    const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx >= rows || col_idx >= cols) return;

    const int idx = row_idx * cols + col_idx;

    const float g = gate[idx];
    const float u = up[idx];

    // Swish: u * sigmoid(u)
    const float sigmoid_u = 1.0f / (1.0f + expf(-u));
    const float swish_u = u * sigmoid_u;

    // SwiGLU: gate * swish(up)
    output[idx] = g * swish_u;
}
```

### Task 4.3: rms_norm_kernel

```cpp
extern "C" __global__ void rms_norm_kernel(
    const float* __restrict__ input,   // [rows, hidden_dim]
    const float* __restrict__ weight,  // [hidden_dim]
    float* __restrict__ output,        // [rows, hidden_dim]
    const float eps,
    const int rows,
    const int hidden_dim
) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockDim = 256;

    __shared__ float s_data[256];

    if (row_idx >= rows) return;

    const float* row = input + row_idx * hidden_dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim) {
        float val = row[i];
        sum_sq += val * val;
    }

    // Reduce (wave32 optimized for your RX 7900 XT)
    s_data[tid] = sum_sq;
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {  // Start at 16 for wave32
        if (tid < stride) s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }

    float mean_sq = s_data[0] / (float)hidden_dim;
    float rms = rsqrtf(mean_sq + eps);

    // Normalize
    for (int i = tid; i < hidden_dim; i += blockDim) {
        float w = weight[i];
        output[row_idx * hidden_dim + i] = row[i] * rms * w;
    }
}
```

**Phase 4 Exit Criteria:**
- [ ] Full transformer layer stays on GPU
- [ ] No to_host_vec in layer forward pass
- [ ] CPU vs GPU tests pass

---

## Phase 5: Optional Optimizations (After)

### Task 5.1: GPU Sampler

```
top-k/top-p on device
Overlaps with next token compute
```

### Task 5.2: Custom MFMA GEMM

```
Only if profiling proves GEMM is bottleneck
After rocBLAS baseline is measured
```

### Task 5.3: FP16 Support

```
Half precision for weights/activations
Requires careful gradient scaling (training only)
```

### Task 5.4: Wave64 Tuning

```
Optimize block sizes for AMD CDNA3
Profile different configurations
```

---

## Part N: Tensor Layout Reference

### All Tensors Are Row-Major

```
Layout: C-style (row-major)
Contiguous: Yes (no stride padding except natural alignment)
Element access: tensor[b * d1 * d2 + s * d2 + d]
```

### Attention Layouts

| Tensor | Shape | Elements | Access Pattern |
|--------|-------|----------|----------------|
| Q, K, V | `[batch, seq_len, head_dim]` | B×S×D | `q[b*S*D + s*D + d]` |
| Scores | `[batch, seq_len, seq_len]` | B×S×S | `scores[b*S*S + i*S + j]` |
| Output | `[batch, seq_len, head_dim]` | B×S×D | `out[b*S*D + s*D + d]` |

### MLP Layouts

| Tensor | Shape | Elements |
|--------|-------|----------|
| hidden | `[seq_len, hidden_size]` | S×H |
| gate_weight | `[hidden_size, intermediate]` | H×I |
| up_weight | `[hidden_size, intermediate]` | H×I |
| down_weight | `[intermediate, hidden_size]` | I×H |

---

## Part N+1: Test Templates

### CPU vs GPU Test Template

```rust
#[cfg(feature = "rocm")]
#[test]
fn test_<kernel>_gpu_matches_cpu() {
    // 1. Create test data (small, hand-verifiable)
    let input = vec![...];

    // 2. CPU reference
    let mut cpu_result = input.clone();
    cpu_<kernel>(&mut cpu_result);

    // 3. GPU computation
    let backend = HipBackend::new().unwrap();
    let gpu_buffer = HipBuffer::new(...).unwrap();
    gpu_buffer.copy_from_host(&input).unwrap();

    unsafe {
        <kernel>_gpu(...);
    }

    let mut gpu_result = vec![0.0f32; input.len()];
    gpu_buffer.copy_to_host(&mut gpu_result).unwrap();

    // 4. Compare with tolerance
    for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        assert!(diff < 1e-5, "Mismatch at {}: CPU={}, GPU={}, diff={}",
                i, cpu_val, gpu_val, diff);
    }
}
```

---

## Summary

```
Phase 1: scale, mask, softmax (replace stubs) ✅ COMPLETE
Phase 2: RoPE + KV append (biggest latency win) ✅ COMPLETE
Phase 3: FlashAttention (performance unlock) ← Next
Phase 4: SwiGLU + RMSNorm (complete GPU path) Pending
Phase 5: Optional (GPU sampler, MFMA, FP16, tuning) Pending

Rule of thumb: Make it correct → make it measurable → then make it fast.
```

**Progress: 2/5 phases complete**

---

> For detailed implementation patterns, see `kernel_research.md`.
> For methodology and debugging, see `implementation_principles.md`.
> For current state, see `codebase_audit.md`.
