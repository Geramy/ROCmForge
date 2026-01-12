# Phase 6: GPU Sampler Implementation Plan

**Date**: 2026-01-11
**Status**: Planning Phase
**Goal**: Implement GPU-accelerated top-k and top-p sampling for token generation

---

## Executive Summary

Current token sampling runs entirely on CPU using sorting-based approach. This is a significant bottleneck during LLM inference. Phase 6 will implement GPU-accelerated sampling using sorting-free rejection sampling algorithm from FlashInfer.

### Current State
- **CPU-only sampling**: All token selection happens on host
- **Sorting-based**: O(v log v) complexity where v = vocab_size
- **CPU-GPU sync overhead**: Probabilities copied to CPU for each token generated
- **Bottleneck impact**: Sampling can take 10-30% of total inference time

### Target State
- **GPU-based sampling**: Entire pipeline on device
- **Sorting-free**: O(log v) complexity using rejection sampling
- **Zero CPU round-trip**: Probabilities stay on GPU
- **Expected speedup**: 2-5x faster sampling (based on FlashInfer benchmarks)

---

## Part 1: Research Findings

### Algorithm: Dual Pivot Rejection Sampling

From [FlashInfer blog post](https://flashinfer.ai/2025/03/10/sampling.html):

**Key Innovation**: Avoid expensive sorting by using rejection sampling with dual pivots.

**Algorithm Steps**:
1. Initialize `low = 0`, `high = max(probabilities)`
2. Sample using inverse transform sampling over `(low, ∞)`
3. Let sampled token have probability `pivot1`, compute `pivot2 = (pivot1 + high) / 2`
4. Three cases:
   - If `f(pivot1) = 1`: Accept token (meets filtering criteria)
   - If `f(pivot1) = 0, f(pivot2) = 1`: Set `low = pivot1`, `high = pivot2`
   - If `f(pivot1) = 0, f(pivot2) = 0`: Set `low = pivot2`
5. Repeat until success

**Where `f(x)` checks validity**:
- **Top-k**: `f(x) = 1` if at least k tokens have probability ≥ x
- **Top-p**: `f(x) = 1` if cumulative probability of tokens ≥ x exceeds threshold p
- **Fused top-k+top-p**: Both conditions must be satisfied

**Theoretical Guarantee**: O(log ε) rounds where ε is minimum float value.

### vLLM Implementation Reference

From [vLLM topk_topp_sampler documentation](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/):

**ROCm/HIP Path** (`forward_hip`):
```python
def forward_hip(self, logits, generators, k, p):
    if (k is None and p is None) or generators:
        return self.forward_native(logits, generators, k, p)

    # Joint k+p path
    if use_top_p and use_top_k:
        probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
        next_token_ids = self.aiter_ops.top_k_top_p_sampling_from_probs(
            probs, None, k, p, deterministic=True
        )
        return next_token_ids.view(-1)

    # Top-p only path
    elif use_top_p:
        probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
        next_token_ids = self.aiter_ops.top_p_sampling_from_probs(
            probs, None, p, deterministic=True
        )
        return next_token_ids.view(-1)

    # Top-k only path
    elif use_top_k:
        probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
        renorm_probs = self.aiter_ops.top_k_renorm_probs(probs, k)
        return torch.multinomial(renorm_probs, num_samples=1).view(-1)
```

**Key Patterns**:
- Contiguous probability tensors (no strided views)
- Deterministic sampling (reproducible results)
- Fused kernel for top-k+top-p
- Separate kernels for top-k-only and top-p-only

### ROCmForge Kernel Patterns

From existing kernels (`kernels/weighted_matmul.hip`, etc.):

**Block Configuration**:
- RDNA3 wavefront size: 32 threads
- Shared memory: Wave-level reduction using `__shared__` arrays
- Grid: `(batch, heads, seq)` for batched operations

**FFI Pattern** (from `src/backend/hip_backend.rs`):
```rust
extern "C" {
    fn hipModuleLaunchKernel(
        func: *mut c_void,
        gridDimX, gridDimY, gridDimZ: u32,
        blockDimX, blockDimY, blockDimZ: u32,
        sharedMemBytes: u32,
        stream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}
```

---

## Part 2: Implementation Strategy

### Phase 6.1: Core Sampling Primitives (2-3 hours)

**Goal**: Build reusable GPU primitives for sampling operations.

**Files**:
- `kernels/sampling_utils.hip` - NEW: Utility functions for sampling
- `src/sampler/gpu_utils.rs` - NEW: Rust wrappers for GPU utils

**Primitives to Implement**:

1. **Softmax kernel** (`softmax_kernel.hip`)
   - Input: `[batch_size, vocab_size]` logits
   - Output: `[batch_size, vocab_size]` probabilities
   - Numerical stability: Find max per row, subtract before exp

2. **Prefix sum kernel** (`prefix_sum_kernel.hip`)
   - Input: `[batch_size, vocab_size]` probabilities
   - Output: `[batch_size, vocab_size]` cumulative probabilities
   - Use wave-level reduction for efficiency

3. **Random number generation**
   - Use HIP's `hiprand` library or XORWOW PRNG
   - One random number per batch element per sampling round

### Phase 6.2: Top-P Sampling Kernel (4-6 hours)

**Goal**: Implement nucleus sampling on GPU.

**Files**:
- `kernels/topp_sampling.hip` - NEW
- `src/sampler/gpu_topp.rs` - NEW
- `tests/topp_gpu_tests.rs` - NEW

**Algorithm** (Rejection Sampling):
```
For each batch element:
1. Initialize pivot = 0, cumulative = 0
2. Loop:
   a. Sample token using inverse transform sampling (ignoring < pivot)
   b. Get sampled token's probability p_sample
   c. Compute cumulative prob of tokens >= pivot
   d. If cumulative >= top_p: Accept token
   e. Else: Set pivot = p_sample, repeat
```

**Kernel Signature**:
```cpp
extern "C" __global__ void topp_sampling_kernel(
    const float* __restrict__ probabilities,  // [batch_size, vocab_size]
    const float* __restrict__ random_values,  // [batch_size] pre-generated
    uint32_t* __restrict__ output,            // [batch_size]
    const float top_p,                        // Threshold (0.9, etc.)
    const int batch_size,
    const int vocab_size
);
```

**Block Configuration**:
- Grid: `batch_size` blocks (one per batch element)
- Block: `min(256, vocab_size)` threads
- Shared memory: For prefix sum computation

### Phase 6.3: Top-K Sampling Kernel (4-6 hours)

**Goal**: Implement top-k sampling on GPU.

**Files**:
- `kernels/topk_sampling.hip` - NEW
- `src/sampler/gpu_topk.rs` - NEW
- `tests/topk_gpu_tests.rs` - NEW

**Algorithm Options**:

**Option A: Rejection Sampling** (preferred, matches FlashInfer)
```
For each batch element:
1. Find top-k threshold value (kth largest probability)
2. Use rejection sampling to sample from tokens >= threshold
3. Renormalize and return
```

**Option B: Partial Sort + Sample** (simpler, slower)
```
1. Find top-k values per batch (wave reduction)
2. Compute renormalized probabilities
3. Sample from k values
```

**Kernel Signature**:
```cpp
extern "C" __global__ void topk_sampling_kernel(
    const float* __restrict__ probabilities,  // [batch_size, vocab_size]
    const float* __restrict__ random_values,  // [batch_size]
    uint32_t* __restrict__ output,            // [batch_size]
    const int top_k,                          // K value (50, etc.)
    const int batch_size,
    const int vocab_size
);
```

### Phase 6.4: Fused Top-K+Top-P Kernel (4-6 hours)

**Goal**: Combined filtering for maximum performance.

**Files**:
- `kernels/topk_topp_sampling.hip` - NEW
- `src/sampler/gpu_fused.rs` - NEW
- `tests/fused_gpu_tests.rs` - NEW

**Algorithm**: Apply both filters in single pass

```
For each batch element:
1. Initialize pivot = 0
2. Loop:
   a. Sample token using inverse transform (ignoring < pivot)
   b. Check if token is in top-k AND cumulative >= top_p
   c. If both conditions met: Accept
   d. Else: Update pivot, repeat
```

**Kernel Signature**:
```cpp
extern "C" __global__ void topk_topp_sampling_kernel(
    const float* __restrict__ probabilities,  // [batch_size, vocab_size]
    const float* __restrict__ random_values,  // [batch_size]
    uint32_t* __restrict__ output,            // [batch_size]
    const int top_k,
    const float top_p,
    const int batch_size,
    const int vocab_size
);
```

### Phase 6.5: Integration with Existing Sampler (2-3 hours)

**Files**:
- `src/sampler/sampler.rs` - UPDATE
- `src/sampler/mod.rs` - UPDATE
- `src/engine.rs` - UPDATE

**Changes**:

1. Add GPU path to `Sampler::sample()`:
```rust
#[cfg(feature = "rocm")]
pub fn sample_gpu(
    &mut self,
    logits: &DeviceTensor,
) -> SamplerResult<u32> {
    // Use GPU kernels for top-k/top-p sampling
}
```

2. Auto-detect GPU availability:
```rust
pub fn sample(&mut self, logits: &[f32]) -> SamplerResult<u32> {
    #[cfg(feature = "rocm")]
    {
        if self.use_gpu {
            return self.sample_gpu_device(logits);
        }
    }
    self.sample_cpu(logits)
}
```

3. Fallback to CPU on error

---

## Part 3: Test Strategy

### Unit Tests

1. **Correctness Tests**
   - Compare GPU output vs CPU output for identical inputs
   - Test deterministic behavior with fixed seeds
   - Validate top-k and top-p filtering correctness

2. **Numerical Precision Tests**
   - Verify probabilities sum to 1.0
   - Check tolerance: < 1e-5 difference from CPU
   - Edge cases: All zeros, one dominant token, uniform distribution

3. **Performance Tests**
   - Benchmark GPU vs CPU sampling speed
   - Measure latency vs batch size
   - Profile memory usage

### Integration Tests

```rust
#[test]
fn test_gpu_sampler_matches_cpu() {
    let config = SamplingConfig::new(0.8, 50, 0.9).unwrap();
    let mut cpu_sampler = Sampler::new(config.clone());
    let mut gpu_sampler = GpuSampler::new(config).unwrap();

    let logits = generate_test_logits(vocab_size);

    let cpu_token = cpu_sampler.sample(&logits).unwrap();
    let gpu_token = gpu_sampler.sample(&logits).unwrap();

    // Should match with same random seed
    assert_eq!(cpu_token, gpu_token);
}
```

---

## Part 4: Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Rejection sampling convergence issues | Medium | High | Implement max iterations, fallback to CPU |
| Random number generation quality | Low | Medium | Use HIPRAND, validate statistical properties |
| Numerical precision in prefix sum | Medium | Medium | Use compensated summation (Kahan) |
| Thread safety in parallel sampling | Low | High | One threadblock per batch element |
| HIPRAND availability | Low | Medium | Fallback to XORWOW PRNG |

---

## Part 5: Success Criteria

### Functional Requirements
- [ ] Top-p sampling works correctly on GPU
- [ ] Top-k sampling works correctly on GPU
- [ ] Fused top-k+top-p sampling works correctly on GPU
- [ ] Results match CPU implementation (within tolerance)
- [ ] Fallback to CPU on errors

### Performance Requirements
- [ ] GPU sampling > 2x faster than CPU
- [ ] End-to-end token generation shows measurable improvement
- [ ] Memory overhead < 100MB additional

### Quality Requirements
- [ ] All tests passing (including new GPU tests)
- [ ] No new compiler warnings
- [ ] Documentation complete

---

## Part 6: Implementation Roadmap

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Phase 6.1: Core primitives | `sampling_utils.hip`, `gpu_utils.rs` |
| 2 | Phase 6.2: Top-p kernel | `topp_sampling.hip`, tests passing |
| 3 | Phase 6.3: Top-k kernel | `topk_sampling.hip`, tests passing |
| 4 | Phase 6.4: Fused kernel | `topk_topp_sampling.hip`, tests passing |
| 5 | Phase 6.5: Integration | Sampler uses GPU, fallback works |

---

## Sources

### Documentation
- [FlashInfer Sampling Blog](https://flashinfer.ai/2025/03/10/sampling.html) - Algorithm explanation
- [vLLM topk_topp_sampler](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/) - API reference
- [ROCm HIP Documentation](https://rocm.docs.amd.com/) - HIP programming guide

### Reference Implementations
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer) - CUDA kernels (convertible to HIP)
- [vLLM GitHub](https://github.com/vllm-project/vllm) - Production implementation

### Papers
- FlashInfer: "Sorting-Free GPU Kernels for LLM Sampling" (2025)

---

**Next Step**: Create TODO.md with task breakdown, then trigger implementation subagent.
