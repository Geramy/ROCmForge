# Phase 3: Fused Attention (FlashAttention) - Progress & Retrospective

> Start: 2025-01-03
> GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
> Goal: Fuse attention operations into single GPU kernel, eliminate CPU round-trips
> Status: ⚠️ Partial - Tests pass at small sizes, scope was too wide
> **New Plan**: Divide & Conquer - 12 atomic sub-tasks for Phase 3a

---

## Divide & Conquer Plan (2025-01-03)

Based on retrospective analysis, Phase 3 is re-scoped using **divide and conquer**:

```
Attention = 5 atomic operations
├─ Op 1: QK^T matmul       → 5 sub-tasks (test, kernel, build, wrapper, verify)
├─ Op 2: Scale by 1/√d      → 2 sub-tasks (test, kernel)
├─ Op 3: Softmax           → 1 sub-task (verify existing)
├─ Op 4: Weighted × V      → 2 sub-tasks (test, kernel)
└─ Op 5: Fused non-causal  → 2 sub-tasks (test, kernel)
```

**Total: 12 atomic sub-tasks**

### Key Changes from Original Phase 3

| Aspect | Before | After |
|--------|--------|-------|
| Scope | All 5 ops in one kernel | One operation at a time |
| Layout | `[batch, seq_len, head_dim]` (ambiguous) | `[batch, seq, heads, dim]` (explicit) |
| Causal mask | Included by default | Deferred to Phase 3b |
| LDS | Oversized buffers | Simple, unoptimized |
| Test sizes | 4×4, 8×8, 16×16, 32×32, 64×64 | Start small, verify each step |

### Tensor Layout Change

**Before (ambiguous)**:
```cpp
const int batch_offset = batch_idx * seq_len * head_dim;
// Where is num_heads? Is seq_len == head_dim?
```

**After (explicit)**:
```cpp
// Q, K, V: [batch, seq, heads, dim]
const int q_offset = batch_idx * seq_len * num_heads * head_dim
                   + seq_idx * num_heads * head_dim
                   + head_idx * head_dim;
// Each dimension contributes visibly to the offset
```

### Sub-task Breakdown

| ID | Operation | Sub-tasks | Files |
|----|-----------|-----------|-------|
| 3a.1 | QK^T matmul | 5 | `qkt_matmul.{hip,rs}`, `build.rs`, `kernels.rs` |
| 3a.2 | Scale | 2 | `scale_scores.{hip,rs}` |
| 3a.3 | Softmax | 1 | Verify existing `softmax.hip` |
| 3a.4 | Weighted×V | 2 | `weighted_matmul.{hip,rs}` |
| 3a.5 | Fused | 2 | `flash_attention_nocausal.{hip,rs}` |

---

## Executive Summary

Phase 3 achieved **1419× speedup** with correct results at 32×32, but the approach had fundamental issues:
- Scope was too wide (testing 5 operations at once)
- Tensor layout was ambiguous ([batch, seq_len, head_dim] collapses dimensions)
- No isolation tests for individual operations
- Large size (64×64) correctness is unproven

**Recommendation**: Re-scope into Phase 3a (non-causal) and Phase 3b (causal), testing one semantic operation at a time.

---

## What Phase 3 Is Really Testing

> **This is not about attention speed yet.**
>
> It is testing whether your entire engineering model works under stress:
> - Kernel contracts are strong enough
> - Tests are small enough to localize bugs
> - Tooling catches lies instead of hiding them
> - You can reason about wave32 + LDS + reduction without guessing

### What Went Right (Good Engineering Behavior)

1. ✅ **Localization**: Found the `s_partial` vs `s_scores` shared memory corruption bug
2. ✅ **Narrowing Hypotheses**: Tested at multiple sizes (16×16 → 32×32 → 64×64) to isolate issues
3. ✅ **No Blaming Game**: Didn't conclude "ROCm is broken" when tests failed
4. ✅ **TDD Approach**: Wrote tests before implementing kernel

### What Went Wrong (Scope Issues)

Phase 3 attempted to verify **all of these at once** in a single kernel:

| Operation | Lines in kernel | Semantic complexity |
|-----------|-----------------|----------------------|
| QK^T matmul | ~40 | Matrix multiply with transpose |
| Scaling by 1/√d | ~5 | Element-wise multiply |
| Causal masking | ~8 | Conditional branch with mask tensor |
| Softmax | ~60 | Two-pass reduction (max, then sum) |
| softmax × V | ~40 | Matrix multiply |

**When a test failed, we couldn't isolate which operation was wrong.**

---

## Actual Test Results

### Benchmark (Real Output)

```bash
$ cargo test --lib benchmark_flash_attention_vs_separate --features rocm -- --nocapture --test-threads=1

running 1 test
test attention::flash_attention_tests::phase3_flash_attention_tests::benchmark_flash_attention_vs_separate ...

Device 0: AMD Radeon RX 7900 XT - 4194304MB VRAM
CPU (separate kernels) ×10: 15.53ms
GPU (FlashAttention fused) ×10: 10.94µs
Speedup: 1419.58x
Max difference CPU vs GPU: 0.0000014305115
ok

test result: ok. 1 passed; 0 failed; 0 ignored
```

### Correctness Results by Size

| Size | Status | Max Diff | Notes |
|------|--------|----------|-------|
| 4×4 | ✅ Pass | ~1e-6 | Minimal test |
| 8×8 | ✅ Pass | ~1e-6 | Small |
| 16×16 | ✅ Pass | 7.15e-7 | CPU: 1.98ms, GPU: 10.48µs |
| 32×32 | ✅ Pass | 1.43e-6 | CPU: 14.42ms, GPU: 10.53µs |
| 64×64 | ⚠️ Fail | 0.21 | Floating-point accumulation error |

---

## Files Created/Modified

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `kernels/flash_attention.hip` | 252 | Fused attention kernel | ⚠️ Works but scope too wide |
| `src/attention/kernels.rs` | ~400 | Kernel wrapper, HSACO loading | ✅ Working |
| `src/attention/flash_attention_tests.rs` | 550 | Tests + benchmark | ✅ Tests pass at 32×32 |
| `build.rs` | ~150 | HIP compilation for flash_attention.hip | ✅ Working |

---

## Bugs Found and Fixed

### Bug 1: Shared Memory Corruption in QK^T Reduction

**Symptom**: GPU output differed significantly from CPU

**Root Cause**: Using same `s_scores` buffer for both reduction intermediate values AND final attention scores. During wave reduction, intermediate values corrupted the score storage.

**Location**: `kernels/flash_attention.hip:113-158`

**Fix**: Introduced separate `s_partial[BLOCK_SIZE]` buffer for reduction operations, preserving `s_scores[]` for final attention scores only.

```cpp
// Before (wrong):
__shared__ float s_scores[256];  // Used for BOTH reduction and scores
s_scores[tid] = partial_score;   // Corrupts scores during reduction
for (int stride = 16; stride > 0; stride >>= 1) {
    s_scores[tid] += s_scores[tid + stride];  // Corrupting s_scores!
}

// After (correct):
__shared__ float s_partial[BLOCK_SIZE];  // For reduction only
__shared__ float s_scores[256];          // For final scores only
s_partial[tid] = partial_score;
for (int stride = 16; stride > 0; stride >>= 1) {
    s_partial[tid] += s_partial[tid + stride];  // Separate buffer
}
```

### Bug 2: Softmax Reduction Corruption

**Symptom**: Same buffer used for both max-reduction and sum-reduction

**Root Cause**: Two-pass softmax needs to preserve max value while computing sum.

**Fix**: Same fix - use `s_partial` for all reductions, never corrupt `s_scores` until softmax values are final.

---

## What's Missing (The Gaps)

### 1. Test Isolation

**Problem**: No standalone tests for individual operations

| Operation | Has standalone test? |
|-----------|---------------------|
| QK^T matmul | ❌ No |
| Scaling | ✅ Yes (Phase 1) |
| Causal mask | ✅ Yes (Phase 1) |
| Softmax | ✅ Yes (Phase 1) |
| softmax × V | ❌ No |

### 2. Tensor Layout Clarity

**Current Layout** (ambiguous):
```cpp
// In flash_attention.hip:88-93
const int batch_offset = batch_idx * seq_len * head_dim;
// Q, K, V are stored as [batch, seq_len, head_dim]
// But what if seq_len != head_dim? What about num_heads?
```

**The Problem**:
- `seq_len * head_dim` suggests seq_len is collapsed with something
- `num_heads` is a separate parameter but not in the offset calculation
- When reading index math, you have to mentally track: "is this seq or heads?"

**Proposed Layout** (explicit):
```cpp
// Q, K, V: [batch, seq, heads, dim]
const int q_offset = batch_idx * seq_len * num_heads * head_dim
                   + seq_idx * num_heads * head_dim
                   + head_idx * head_dim;
```

**Benefits**:
- Index math is **auditable** - you can see each dimension's contribution
- Matches FlashAttention papers (they use [B, S, H, D])
- Removes "is seq == dim here?" ambiguity
- Can always re-pack/flattened later for performance

### 3. Non-Causal Baseline

**Problem**: We never tested simple attention without masking

The kernel accepts `mask` parameter:
```cpp
const float* __restrict__ mask,  // Optional, use nullptr for no mask
```

But we only tested with `mask = nullptr`. We never verified:
- That the mask branch doesn't execute when `mask == nullptr`
- That causal masking actually works correctly
- That non-causal and causal results differ correctly

### 4. Causal Mask Independent Test

**Problem**: Mask logic exists but was never independently verified

The mask application code (`flash_attention.hip:146-153`):
```cpp
if (mask != nullptr) {
    const int mask_idx = batch_idx * seq_len * seq_len + query_pos * seq_len + key_pos;
    float mask_val = mask[mask_idx];
    if (mask_val < -1e30f) {
        score = mask_val;
    }
}
```

This branch was never tested! We need:
1. Test with `mask != nullptr` to verify branch is taken
2. Verify causal mask actually zeros out future tokens
3. Verify mask doesn't corrupt non-causal computation

### 5. Large Size Correctness

**Problem**: 64×64 fails with floating-point accumulation error

**Suspected causes**:
- Reduction order differences between CPU and GPU
- Floating-point associativity: `(a + b) + c ≠ a + (b + c)`
- At larger sizes, accumulation order matters more

**Why this matters**:
- We can't claim correctness if it only works at 32×32
- Real models use seq_len = 2048 or more
- Need to either fix accumulation order or document tolerance

---

## Revised Order of Attack

### Phase 3a: Non-Causal FlashAttention (Re-Scoped)

**Goal**: Test ONE semantic operation at a time

**Prerequisite**: None

| Task | File | Exit Criteria |
|------|------|----------------|
| 3a.1 QK^T matmul (standalone) | `kernels/qkt_matmul.hip` | Test passes vs CPU |
| 3a.2 Softmax (standalone) | Already exists | Test passes vs CPU |
| 3a.3 Weighted×V (standalone) | `kernels/weighted_matmul.hip` | Test passes vs CPU |
| 3a.4 Fused non-causal | `kernels/flash_attention_nocausal.hip` | Test passes at 64×64 |

**Constraints**:
- NO mask parameter
- NO mask branching
- Layout: `[batch, seq, heads, dim]` (explicit)
- Simple LDS (don't optimize yet)

### Phase 3b: Causal Masking (Sequential)

**Prerequisite**: Phase 3a complete

| Task | File | Exit Criteria |
|------|------|----------------|
| 3b.1 Causal mask (standalone) | `kernels/causal_mask.hip` | Test passes vs CPU |
| 3b.2 Fused causal attention | `kernels/flash_attention_causal.hip` | Test passes at 64×64 |

**Verification**:
- Mask branch doesn't corrupt non-causal path
- Causal and non-causal results differ correctly
- Mask index math is auditable

---

## Additional Guardrails (Recommended)

### 1. Deterministic Test Seeds + Fixed Inputs

**Problem**: Non-deterministic test runs make debugging harder

**Solution**: Use fixed inputs with deterministic seeds

```rust
// flash_attention_tests.rs
fn deterministic_test_data() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Use deterministic values, not random()
    let mut q = vec![0.0f32; n];
    for i in 0..n {
        q[i] = ((i % 257) as f32) / 100.0;  // Deterministic
    }
    // Same for K, V
}
```

**Benefits**:
- Diffs are stable across runs
- Can reproducibly debug issues
- Can add snapshot testing

### 2. Instrumentation Mode

**Problem**: When kernel produces wrong output, no visibility into internals

**Solution**: Add instrumentation mode for small sizes

```cpp
// For S=8, dump intermediate scalars for one block
#ifdef INSTRUMENT
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("After QK^T: max=%.6f, sum=%.6f\n", s_max, s_sum);
    printf("First 4 scores: %.6f %.6f %.6f %.6f\n",
           s_scores[0], s_scores[1], s_scores[2], s_scores[3]);
}
#endif
```

**How to use**:
1. Run with `INSTRUMENT=1` flag
2. Compare CPU vs GPU intermediate values
3. Localize which operation diverges first

---

## Tensor Layout Analysis

### Current Layout

```rust
// In tests and kernel
Q, K, V: [batch_size, seq_len, head_dim]
// Actually stored as: batch_size * seq_len * head_dim
// Note: num_heads is handled separately (not in tensor)
```

**Access pattern in kernel**:
```cpp
const int batch_offset = batch_idx * seq_len * head_dim;
const float* Q_batch = Q + batch_offset;
const int q_row_offset = query_pos * head_dim;  // Assumes seq_len = num_heads * ???
```

**Ambiguities**:
- Where is `num_heads` in the offset?
- Is `seq_len` supposed to equal `head_dim`?
- What happens when `seq_len != head_dim`?

### Proposed Layout (for Phase 3a)

```rust
// Explicit 4D tensor
Q, K, V: [batch, seq, heads, dim]

// Clear index math
const int q_offset = batch_idx * seq_len * num_heads * head_dim
                   + seq_idx * num_heads * head_dim
                   + head_idx * head_dim;
```

**Benefits**:
1. **Auditable**: Each dimension's contribution is explicit
2. **Matches literature**: FlashAttention papers use this notation
3. **No ambiguity**: No "is this seq or heads?" confusion
4. **Easier debugging**: Can verify each index calculation

**Tradeoff**:
- More complex offset calculation (minor)
- Can repack/flattened later for performance (after correctness is proven)

---

## LDS Usage Recommendations

### Current State

```cpp
__shared__ float s_partial[BLOCK_SIZE];  // 256 floats = 1024 bytes
__shared__ float s_scores[256];          // 256 floats = 1024 bytes
__shared__ float s_max;                  // 4 bytes
__shared__ float s_sum;                  // 4 bytes
// Total: ~2052 bytes per block
```

### Issue: Oversized for Small seq_len

For `seq_len = 32`, we only need 32 scores, not 256:
- `s_scores[256]` wastes 224 floats (896 bytes)
- This is fine for correctness, but not optimal

### Recommendation: Keep LDS Simple (For Now)

**Don't optimize yet because**:
1. Early bugs are usually indexing errors, not occupancy issues
2. Aggressive tiling increases surface area for bugs
3. Can optimize occupancy after correctness is locked

**Simple approach for Phase 3a**:
```cpp
constexpr int MAX_SEQ_LEN = 128;  // Fixed max for testing
__shared__ float s_scores[MAX_SEQ_LEN];  // Exact size needed
```

**Optimize later** (Phase 5 or after correctness is proven):
- Use `seq_len` as template parameter
- Ring buffers for online softmax
- Tile K and V for better cache utilization

---

## Reduction Pattern Analysis

### Current Wave32 Reduction

```cpp
// RDNA3 wavefront size = 32
for (int stride = 16; stride > 0; stride >>= 1) {
    if (tid < stride) {
        s_partial[tid] += s_partial[tid + stride];
    }
    __syncthreads();
}
```

**Correct for wave32**:
- Start at stride=16 (half of wave32)
- Reduces 256 threads in 5 steps: 16→8→4→2→1
- Each step halves the active thread count

**NOT correct for wave64** (CDNA3):
- Would start at stride=32
- Would use `__shfl_down_sync` or similar

**Verification**:
- Sum of all threads' partials ends up in `s_partial[0]`
- Each thread contributes exactly once
- No double-counting, no missing contributions

---

## Index Math Verification

### QK^T Index Calculation

**CPU version** (`compute.rs:26-27`):
```rust
let k_idx = l * seq_len + j;  // Transposed access: K[l, j] not K[j, l]
```

**GPU version** (`flash_attention.hip:126`):
```cpp
partial_score += q_row[i] * K_batch[i * seq_len + key_pos];
```

**Verification**:
- Both access K as transposed
- `i * seq_len + key_pos` ≡ `l * seq_len + j` ✓
- This is correct

### What Could Still Be Wrong

1. **Batch offset**: Does `batch_offset = batch_idx * seq_len * head_dim` match CPU?
2. **Row offset**: Does `query_pos * head_dim` match CPU row indexing?
3. **Head indexing**: Where is `head_idx` in the offset? (Currently implicit)

---

## Floating-Point Accumulation Issues

### Problem at 64×64

```
Size: 64×64
Max diff: 0.21 (unacceptable)
Tolerance: 1e-4
```

### Likely Cause: Reduction Order

**CPU reduction** (likely in `matmul_cpu`):
```cpp
// Sequential accumulation
float sum = 0.0f;
for (int i = 0; i < head_dim; i++) {
    sum += q_row[i] * k_row[i];
}
```

**GPU reduction** (parallel):
```cpp
// Tree reduction: different order
s_partial[tid] = partial_score;  // Each thread's partial
for (int stride = 16; stride > 0; stride >>= 1) {
    s_partial[tid] += s_partial[tid + stride];  // Different order!
}
```

**Why order matters**:
```
CPU:  (((a + b) + c) + d) + e)  // Left associative
GPU:  ((a + b) + (c + d)) + e  // Tree reduction
```

**Floating-point is not associative**: `(a + b) + c ≠ a + (b + c)` for some values.

### Possible Fixes

1. **Kahan summation** (more accurate, slower)
2. **Fixed reduction order** (sort before reduce)
3. **Accept tolerance** (document expected error)
4. **Use double precision** (not an option for RDNA3?)

---

## Checklist for Phase 3a

Before starting Phase 3a, verify:

- [ ] Test inputs use deterministic values (no random)
- [ ] Test layout is explicit `[batch, seq, heads, dim]`
- [ ] Each operation has standalone test
- [ ] Tests pass at 64×64, not just 32×32
- [ ] Instrumentation mode exists for debugging
- [ ] Reduction pattern is verified correct
- [ ] Index math is auditable (can trace each offset)

---

## Summary

### What We Achieved
- ✅ 1419× speedup with correct results at 32×32
- ✅ Found and fixed shared memory corruption bug
- ✅ Tests pass at small sizes with ~1e-6 accuracy

### What We Need to Fix
- ❌ Scope was too wide (test one operation at a time)
- ❌ Tensor layout is ambiguous (make it explicit)
- ❌ No isolation tests for QK^T and softmax×V
- ❌ Large size (64×64) correctness unproven
- ❌ Causal masking never independently tested

### Next Steps
1. Phase 3a: Non-causal attention with explicit layout
2. Phase 3b: Causal masking (after 3a works)
3. Only then: Optimize LDS usage and occupancy

---

## References

- Original Phase 3 spec: `docs/implementation_roadmap.md` lines 321-420
- Kernel research: `docs/kernel_research.md` lines 1-150
- Implementation principles: `docs/implementation_principles.md`
- Codebase audit: `docs/codebase_audit.md`
