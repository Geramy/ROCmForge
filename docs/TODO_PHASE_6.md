# Phase 6: GPU Sampler Implementation TODO

**Date**: 2026-01-11
**Reference**: docs/PLAN_PHASE_6_GPU_SAMPLER.md
**Status**: READY FOR IMPLEMENTATION

---

## Priority Legend

- **P0**: Critical - blocks functionality
- **P1**: High - important for quality/correctness
- **P2**: Medium - performance/features
- **P3**: Low - nice to have

---

## SECTION 1: CORE SAMPLING PRIMITIVES (P1)

### Task 1.1: Create Sampling Utilities Module
**Priority**: P1
**Estimated**: 1-2 hours
**Status**: ⬜ TODO

**Files**:
- `kernels/sampling_utils.hip` (NEW)
- `src/sampler/gpu_utils.rs` (NEW)

**Deliverables**:
1. Softmax kernel for probability computation
2. Prefix sum kernel for CDF computation
3. Random number generation utilities (HIPRAND or XORWOW)

**Acceptance**:
- [ ] Kernels compile with hipcc
- [ ] Rust FFI bindings work
- [ ] Unit tests for each primitive

**Code Template**:
```cpp
// kernels/sampling_utils.hip
#pragma once

#include <hip/hip_runtime.h>

// Row-wise softmax with numerical stability
extern "C" __global__ void softmax_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probabilities,
    const int batch_size,
    const int vocab_size
);

// Exclusive prefix sum per row
extern "C" __global__ void prefix_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int vocab_size
);
```

---

### Task 1.2: Implement Softmax Kernel
**Priority**: P1
**Estimated**: 1-2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 1.1

**Algorithm**:
```
For each row (batch element):
1. Find max value in row (for numerical stability)
2. Compute exp(x - max) for each element
3. Compute sum of exp values
4. Divide each exp by sum
```

**Kernel Signature**:
```cpp
extern "C" __global__ void softmax_kernel(
    const float* __restrict__ logits,      // [batch_size, vocab_size]
    float* __restrict__ probabilities,     // [batch_size, vocab_size]
    const int batch_size,
    const int vocab_size
);
```

**Block Configuration**:
- Grid: `batch_size` blocks
- Block: `min(256, vocab_size)` threads

**Acceptance**:
- [ ] Output probabilities sum to 1.0 (within 1e-6)
- [ ] Numerically stable (no overflow/underflow)
- [ ] Tests pass vs PyTorch softmax

---

### Task 1.3: Implement Prefix Sum Kernel
**Priority**: P1
**Estimated**: 2-3 hours
**Status**: ⬜ TODO
**Dependencies**: Task 1.1

**Algorithm**: Wave-level inclusive scan using shared memory

**Kernel Signature**:
```cpp
extern "C" __global__ void prefix_sum_kernel(
    const float* __restrict__ input,        // [batch_size, vocab_size]
    float* __restrict__ output,             // [batch_size, vocab_size]
    const int batch_size,
    const int vocab_size
);
```

**Acceptance**:
- [ ] Output[i] = sum(input[0..i])
- [ ] Last element equals total sum
- [ ] Tests pass vs CPU prefix sum

---

## SECTION 2: TOP-P SAMPLING (P1)

### Task 2.1: Implement Top-P Sampling Kernel
**Priority**: P1
**Estimated**: 4-6 hours
**Status**: ⬜ TODO
**Dependencies**: Task 1.2, Task 1.3

**Files**:
- `kernels/topp_sampling.hip` (NEW)
- `src/sampler/gpu_topp.rs` (NEW)
- `tests/topp_gpu_tests.rs` (NEW)

**Algorithm**: Rejection sampling with pivot

```
For each batch element:
1. Initialize pivot = 0.0
2. Loop (max 10 iterations):
   a. Sample token using inverse transform (ignore tokens < pivot)
   b. Let p_sample = probability[sampled_token]
   c. Compute cumulative probability of tokens >= pivot
   d. If cumulative >= top_p:
        Return sampled_token
   e. Else:
        pivot = p_sample
        Continue loop
3. If max iterations reached, return argmax
```

**Kernel Signature**:
```cpp
extern "C" __global__ void topp_sampling_kernel(
    const float* __restrict__ probabilities,  // [batch_size, vocab_size]
    const float* __restrict__ random_values,  // [batch_size] pre-generated
    uint32_t* __restrict__ output,            // [batch_size]
    const float top_p,
    const int batch_size,
    const int vocab_size
);
```

**Acceptance**:
- [ ] Returns token ID in valid range [0, vocab_size)
- [ ] Filtered tokens have cumulative probability >= top_p
- [ ] Deterministic with same random seed
- [ ] Tests match CPU implementation

---

### Task 2.2: Create Top-P Rust Wrapper
**Priority**: P1
**Estimated**: 1-2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 2.1

**Files**:
- `src/sampler/gpu_topp.rs` (NEW)

**API**:
```rust
pub struct GpuTopPSampler {
    backend: HipBackend,
    top_p: f32,
}

impl GpuTopPSampler {
    pub fn new(backend: HipBackend, top_p: f32) -> Self;

    pub fn sample(
        &mut self,
        probabilities: &DeviceTensor,
    ) -> SamplerResult<Vec<u32>>;
}
```

**Acceptance**:
- [ ] Compiles without errors
- [ ] Loads kernel successfully
- [ ] Handles errors gracefully

---

### Task 2.3: Top-P Unit Tests
**Priority**: P1
**Estimated**: 2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 2.1, Task 2.2

**Test Cases**:
```rust
#[test]
fn test_topp_sampling_correctness() {
    // Compare GPU vs CPU output
}

#[test]
fn test_topp_edge_cases() {
    // All zeros, one dominant, uniform
}

#[test]
fn test_topp_deterministic() {
    // Same seed produces same output
}
```

**Acceptance**:
- [ ] All tests passing
- [ ] No test flakiness
- [ ] Results match CPU within tolerance

---

## SECTION 3: TOP-K SAMPLING (P1)

### Task 3.1: Implement Top-K Sampling Kernel
**Priority**: P1
**Estimated**: 4-6 hours
**Status**: ⬜ TODO
**Dependencies**: Task 1.2

**Files**:
- `kernels/topk_sampling.hip` (NEW)
- `src/sampler/gpu_topk.rs` (NEW)
- `tests/topk_gpu_tests.rs` (NEW)

**Algorithm Options**:

**Option A**: Rejection sampling (preferred)
```
1. Find kth largest probability as threshold
2. Sample from tokens >= threshold using rejection sampling
3. Return sampled token
```

**Option B**: Partial sort + sample
```
1. Use wave reduction to find top-k values
2. Renormalize top-k probabilities
3. Sample from k values
```

**Kernel Signature**:
```cpp
extern "C" __global__ void topk_sampling_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ random_values,
    uint32_t* __restrict__ output,
    const int top_k,
    const int batch_size,
    const int vocab_size
);
```

**Acceptance**:
- [ ] Returns token from top-k probabilities
- [ ] Correctly handles k > vocab_size
- [ ] Tests match CPU implementation

---

### Task 3.2: Create Top-K Rust Wrapper
**Priority**: P1
**Estimated**: 1-2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 3.1

**Files**:
- `src/sampler/gpu_topk.rs` (NEW)

**API**:
```rust
pub struct GpuTopKSampler {
    backend: HipBackend,
    top_k: usize,
}

impl GpuTopKSampler {
    pub fn new(backend: HipBackend, top_k: usize) -> Self;

    pub fn sample(
        &mut self,
        probabilities: &DeviceTensor,
    ) -> SamplerResult<Vec<u32>>;
}
```

---

### Task 3.3: Top-K Unit Tests
**Priority**: P1
**Estimated**: 2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 3.1, Task 3.2

---

## SECTION 4: FUSED TOP-K+TOP-P (P1)

### Task 4.1: Implement Fused Sampling Kernel
**Priority**: P1
**Estimated**: 4-6 hours
**Status**: ⬜ TODO
**Dependencies**: Task 2.1, Task 3.1

**Files**:
- `kernels/topk_topp_sampling.hip` (NEW)
- `src/sampler/gpu_fused.rs` (NEW)
- `tests/fused_gpu_tests.rs` (NEW)

**Algorithm**: Apply both filters in single rejection sampling loop

```
For each batch element:
1. Initialize pivot = 0.0
2. Loop (max 10 iterations):
   a. Sample token using inverse transform (ignore < pivot)
   b. Let p_sample = probability[sampled_token]
   c. Count tokens >= pivot (should be >= top_k)
   d. Compute cumulative probability of tokens >= pivot
   e. If count >= top_k AND cumulative >= top_p:
        Return sampled_token
   f. Else:
        pivot = p_sample
        Continue loop
```

**Kernel Signature**:
```cpp
extern "C" __global__ void topk_topp_sampling_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ random_values,
    uint32_t* __restrict__ output,
    const int top_k,
    const float top_p,
    const int batch_size,
    const int vocab_size
);
```

**Acceptance**:
- [ ] Returns token meeting both filters
- [ ] Correctly handles edge cases (k=1, p=1.0)
- [ ] Tests match CPU implementation

---

### Task 4.2: Create Fused Rust Wrapper
**Priority**: P1
**Estimated**: 1-2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 4.1

---

### Task 4.3: Fused Unit Tests
**Priority**: P1
**Estimated**: 2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 4.1, Task 4.2

---

## SECTION 5: INTEGRATION (P1)

### Task 5.1: Integrate GPU Sampler into Pipeline
**Priority**: P1
**Estimated**: 2-3 hours
**Status**: ⬜ TODO
**Dependencies**: Task 2.2, Task 3.2, Task 4.2

**Files**:
- `src/sampler/sampler.rs` (UPDATE)
- `src/sampler/mod.rs` (UPDATE)
- `src/engine.rs` (UPDATE)

**Changes**:

1. Add GPU sampling method:
```rust
#[cfg(feature = "rocm")]
impl Sampler {
    pub fn sample_gpu(
        &mut self,
        logits: &DeviceTensor,
    ) -> SamplerResult<u32> {
        // Route to appropriate GPU kernel
    }
}
```

2. Auto-detect GPU path:
```rust
pub fn sample(&mut self, logits: &[f32]) -> SamplerResult<u32> {
    #[cfg(feature = "rocm")]
    if self.use_gpu {
        return self.sample_gpu_from_host(logits);
    }
    self.sample_cpu(logits)
}
```

**Acceptance**:
- [ ] GPU path used when available
- [ ] CPU fallback on error
- [ ] No breaking API changes

---

### Task 5.2: CLI Integration
**Priority**: P1
**Estimated**: 1 hour
**Status**: ⬜ TODO
**Dependencies**: Task 5.1

**Files**:
- `src/bin/rocmforge_cli.rs` (UPDATE)

**Changes**:
Add `--gpu-sampler` flag to enable GPU sampling

**Acceptance**:
- [ ] Flag works correctly
- [ ] Help text updated

---

## SECTION 6: VALIDATION & TESTING (P1)

### Task 6.1: Full Test Suite
**Priority**: P1
**Estimated**: 2 hours
**Status**: ⬜ TODO
**Dependencies**: All implementation tasks

**Command**:
```bash
cargo test --features rocm --lib
```

**Expected Result**:
```
All tests passing (220+ tests)
No new failures
```

**Acceptance**:
- [ ] All existing tests still pass
- [ ] New GPU sampler tests pass
- [ ] Single-threaded: 100% pass rate

---

### Task 6.2: Integration Testing with Real Model
**Priority**: P1
**Estimated**: 2-3 hours
**Status**: ⬜ TODO
**Dependencies**: Task 6.1

**Test**:
```bash
RUST_LOG=info ./target/release/rocmforge_cli generate \
  --gguf ~/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "Hello world" \
  --max-tokens 50 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --gpu-sampler
```

**Acceptance**:
- [ ] Generates 50 tokens without crash
- [ ] Output is reasonable text
- [ ] No GPU errors in logs
- [ ] Faster than CPU sampler

---

### Task 6.3: Performance Benchmarking
**Priority**: P2
**Estimated**: 2 hours
**Status**: ⬜ TODO
**Dependencies**: Task 6.2

**Metrics**:
- Tokens/second with GPU sampler
- Tokens/second with CPU sampler
- Speedup ratio
- Memory usage

**Acceptance**:
- [ ] GPU sampler > 2x faster than CPU
- [ ] Memory overhead reasonable (< 100MB)
- [ ] Results documented

---

## SECTION 7: DOCUMENTATION (P2)

### Task 7.1: Update README.md
**Priority**: P2
**Estimated**: 30 minutes
**Status**: ⬜ TODO
**Dependencies**: Task 6.1

**Changes**:
- Add GPU sampler to features list
- Update performance benchmarks
- Add usage example

**Acceptance**:
- [ ] No "production-ready" language
- [ ] GPU sampler mentioned
- [ ] Examples tested

---

### Task 7.2: Create GPU Sampler Documentation
**Priority**: P2
**Estimated**: 1 hour
**Status**: ⬜ TODO
**Dependencies**: Task 6.1

**Files**:
- `docs/GPU_SAMPLER.md` (NEW)

**Content**:
1. Algorithm explanation
2. API reference
3. Performance characteristics
4. Usage examples

**Acceptance**:
- [ ] Documentation complete
- [ ] Examples tested
- [ ] Cross-references correct

---

### Task 7.3: Update CHANGELOG.md
**Priority**: P2
**Estimated**: 30 minutes
**Status**: ⬜ TODO
**Dependencies**: Task 7.1, Task 7.2

**Content**:
- Phase 6 entry with date
- List of implemented features
- Test results
- Performance metrics

**Acceptance**:
- [ ] Format consistent
- [ ] No "production-ready" language
- [ ] All features listed

---

## CHECKLIST SUMMARY

### Must Complete (P1)
- [ ] Task 1.1: Sampling utilities module
- [ ] Task 1.2: Softmax kernel
- [ ] Task 1.3: Prefix sum kernel
- [ ] Task 2.1: Top-P sampling kernel
- [ ] Task 2.2: Top-P Rust wrapper
- [ ] Task 2.3: Top-P unit tests
- [ ] Task 3.1: Top-K sampling kernel
- [ ] Task 3.2: Top-K Rust wrapper
- [ ] Task 4.1: Fused kernel
- [ ] Task 4.2: Fused Rust wrapper
- [ ] Task 5.1: Integration
- [ ] Task 5.2: CLI integration
- [ ] Task 6.1: Full test suite
- [ ] Task 6.2: Integration testing

### Nice to Have (P2)
- [ ] Task 6.3: Performance benchmarking
- [ ] Task 7.1: Update README
- [ ] Task 7.2: GPU sampler documentation
- [ ] Task 7.3: Update CHANGELOG

---

## TOTAL ESTIMATED TIME

| Priority | Tasks | Time |
|----------|-------|------|
| P1 | 14 tasks | 30-40 hours |
| P2 | 4 tasks | 4 hours |
| **Total** | **18 tasks** | **34-44 hours** |

---

## NEXT STEPS

1. ✅ Plan created: `docs/PLAN_PHASE_6_GPU_SAMPLER.md`
2. ✅ TODO created: `docs/TODO_PHASE_6.md` (this file)
3. ⏳ Trigger implementation subagent
4. ⏳ Trigger cross-check subagent
5. ⏳ Trigger documentation update subagent

---

**Status**: READY FOR IMPLEMENTATION
**Last Updated**: 2026-01-11

## Sources

Research sources consulted:
- [FlashInfer Sampling Blog](https://flashinfer.ai/2025/03/10/sampling.html)
- [vLLM topk_topp_sampler API](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/)
- [ROCm Examples](https://github.com/ROCm/rocm-examples)
- [ROCm 7.0 Documentation](https://rocm.docs.amd.com/en/docs-7.0.0/)
