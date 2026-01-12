# Phase 18: GPU Kernel Fixes & GPU Sampler Implementation Plan

**Date**: 2026-01-11
**Status**: Planning Phase
**Goal**: Fix 6 failing GPU tests and implement GPU sampler (Phase 6)

---

## Executive Summary

After comprehensive audit and research, ROCmForge has **6 failing GPU tests** with errors FAR beyond acceptable tolerance (49.6, 5.9 instead of 0.001-0.002). This indicates **kernel bugs**, not precision issues. Additionally, Phase 6 (GPU Sampler) is completely unimplemented.

### Current State
- **213/220 tests passing** (96.8% health)
- **6 tests failing** with massive numerical errors
- **Code compiles** cleanly after P0 fix
- **GPU Sampler**: Not implemented (CPU-only)

### Research Sources Consulted
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) - Official implementations
- [vLLM Documentation](https://docs.vllm.ai) - GPU sampler patterns
- [FlashInfer Sampling API](https://github.com/flashinfer-ai/flashinfer) - top-k/top-p kernels
- [ROCm HIP Documentation](https://rocm.docs.amd.com) - Floating point precision
- [NVIDIA FlashAttention Issues #366](https://github.com/Dao-AILab/flash-attention/issues/366) - Numerical discrepancies
- [ROCm Optimization Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/)

---

## Part 1: Failing GPU Tests - Root Cause Analysis

### Test Failures Summary

| Test | Error | Tolerance | Actual Error | Status |
|------|-------|-----------|--------------|--------|
| `test_weighted_matmul_matches_cpu_small` | Mismatch | 1e-4 | >0.001 | FAIL |
| `test_weighted_matmul_matches_cpu_32x32` | Mismatch | 1e-3 | Likely >1.0 | FAIL |
| `test_weighted_matmul_non_square_sequences` | Mismatch | 1e-3 | **5.9** | FAIL |
| `test_flash_nocausal_matches_cpu_32x32` | Mismatch | 2e-3 | **49.6** | FAIL |
| `benchmark_flash_attention_vs_separate` | Deviation | 0.154 | Too high | FAIL |
| `test_hip_buffer_copy` | Unknown | - | - | FAIL |

### Key Finding
Errors of **5.9** and **49.6** are NOT precision issues - they indicate **logic bugs** in GPU kernels. Possible causes:
1. **Indexing errors** - Reading/writing wrong memory locations
2. **Shared memory corruption** - Race conditions in reduction
3. **Block/grid configuration** - Wrong thread/block counts
4. **Memory layout mismatch** - CPU vs GPU tensor layouts differ

---

## Part 2: Investigation Strategy

### Step 1: Add Debug Output to Failing Tests

Modify failing tests to print:
- First 10 CPU values vs GPU values
- Min/max/mean of both outputs
- Pattern analysis (e.g., "all zeros", "garbage data")

```rust
// DEBUG: Print first 10 values
println!("DEBUG: First 10 CPU values: {:?}", &cpu_result[..10.min(cpu_result.len())]);
println!("DEBUG: First 10 GPU values: {:?}", &gpu_result[..10.min(gpu_result.len())]);
println!("DEBUG: CPU min/max: {}/{}", cpu_result.iter().cloned().fold(f32::INFINITY, f32::min), cpu_result.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
println!("DEBUG: GPU min/max: {}/{}", gpu_result.iter().cloned().fold(f32::INFINITY, f32::min), gpu_result.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
```

### Step 2: Verify Kernel Launch Parameters

Check grid/block configuration matches problem size:

**For seq_k=32, WARP_SIZE=32:**
- Expected threads per block: 32 (tid 0-31)
- Expected blocks: `seq_q * num_heads * batch_size`
- Each block handles one (query_pos, head, batch) triple

### Step 3: Memory Layout Verification

Verify CPU and GPU use SAME layout:
```rust
// Layout: [batch, heads, seq_q, dim]
// Index = batch * heads * seq_q * dim + head * seq_q * dim + seq_q * dim + d
```

### Step 4: Compare with Working Implementations

Reference implementations from research:
- [FlashAttention test_flash_attn.py](https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py)
- [FlashInfer sampling kernels](https://github.com/flashinfer-ai/flashinfer)

---

## Part 3: GPU Sampler Implementation (Phase 6)

### Overview

Currently, sampling (top-k, top-p, temperature) runs on **CPU only**. This is a performance bottleneck during token generation.

### Requirements

1. **Top-k sampling**: Select k highest probability tokens, sample from them
2. **Top-p (nucleus) sampling**: Select tokens covering cumulative probability p, sample from them
3. **Temperature scaling**: Scale logits before sampling
4. **GPU execution**: Entire pipeline on device, no CPU round-trip

### Reference Implementations

**FlashInfer API** ([docs](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html)):
```python
# Fused top-k + top-p sampling entirely on GPU
def top_k_top_p_sampling_from_probs(
    probs: Tensor,      # [batch_size, vocab_size]
    top_k: int,
    top_p: float,
    uniform_samples: Tensor  # [batch_size, num_samples]
) -> Tensor:
    # Returns sampled token IDs
```

**vLLM Sampler** ([docs](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/)):
- Custom CUDA kernels for top-k/top-p
- Memory-efficient (no explicit sorting)
- Per-batch configurable parameters

### Implementation Plan

#### Phase 6.1: Design (1-2 hours)
1. Design GPU kernel interface
2. Define memory layout for probabilities
3. Plan thread block configuration

#### Phase 6.2: Top-k Kernel (4-6 hours)
1. Implement `topk_kernel.hip`
2. Each block finds k largest values
3. Use shared memory for reduction
4. Add tests

#### Phase 6.3: Top-p Kernel (4-6 hours)
1. Implement `topp_kernel.hip`
2. Prefix sum for cumulative probability
3. Binary search for threshold
4. Add tests

#### Phase 6.4: Fused Top-k+Top-p (4-6 hours)
1. Implement `topk_topp_fused.hip`
2. Combine both filters efficiently
3. Add comprehensive tests

#### Phase 6.5: Integration (2-3 hours)
1. Wire up kernels to sampler module
2. Update inference pipeline
3. Performance benchmarks

---

## Part 4: Tolerance Standards (Research Based)

### Industry Standards

From [FlashAttention Issue #366](https://github.com/Dao-AILab/flash-attention/issues/366) and [PyTorch testing](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html):

| Precision | rtol | atol | Use Case |
|-----------|------|------|----------|
| FP32 | 1e-5 | 1e-8 | Exact comparison |
| FP32 (GPU) | 1e-3 to 1e-4 | 1e-5 to 1e-6 | GPU vs CPU |
| BF16/FP16 | 1e-2 to 1e-3 | 1e-4 to 1e-5 | Mixed precision |

### ROCm-Specific Considerations

From [ROCm precision documentation](https://rocm.docs.amd.com/en/latest/reference/precision-support.html):
- AMD GPUs may have different precision than NVIDIA
- FMA operations can cause larger divergence
- Consider using `-ffp-contract=off` for testing

### Recommended Tolerances for ROCmForge

```rust
// For direct kernel comparison
const TEST_TOLERANCE_FP32: f32 = 1e-4;      // 0.0001 - Strict FP32
const TEST_TOLERANCE_FP32_RELAXED: f32 = 2e-3; // 0.002 - For complex kernels
const TEST_TOLERANCE_FP16: f32 = 1e-2;       // 0.01 - Half precision

// For attention-specific (more FP operations)
const TEST_TOLERANCE_ATTENTION: f32 = 1e-3;   // 0.001 - Attention mechanisms
```

**Current tolerances are reasonable** - the problem is NOT tolerance but kernel bugs.

---

## Part 5: Implementation Roadmap

### Week 1: GPU Kernel Fixes

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Debug output + root cause analysis | Error patterns identified |
| 2 | Fix weighted_matmul kernel | 3 tests passing |
| 3 | Fix flash_attention_nocausal kernel | 2 tests passing |
| 4 | Fix hip_buffer_copy test | 1 test passing |
| 5 | Integration testing | 220/220 tests passing |

### Week 2: GPU Sampler (Phase 6)

| Day | Task | Deliverable |
|-----|------|-------------|
| 6 | Design + topk kernel | topk_kernel.hip + tests |
| 7 | topp kernel | topp_kernel.hip + tests |
| 8 | Fused topk_topp | topk_topp_fused.hip + tests |
| 9 | Integration | Sampler using GPU |
| 10 | Benchmarking | Performance metrics |

---

## Part 6: Success Criteria

### GPU Kernel Fixes
- [ ] All 6 failing tests now pass
- [ ] Max diff < tolerance for all tests
- [ ] No regression in existing tests
- [ ] Target: **220/220 tests passing (100%)**

### GPU Sampler (Phase 6)
- [ ] Top-k sampling works on GPU
- [ ] Top-p sampling works on GPU
- [ ] Fused top-k+top-p works on GPU
- [ ] Performance improvement > 2x over CPU
- [ ] Tests verify correctness vs CPU

### Documentation
- [ ] Updated test count in all docs
- [ ] Phase status accurately reflected
- [ ] Known issues documented

---

## Part 7: Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Kernel bugs harder than expected | Medium | High | Add more debug output |
| GPU sampler requires redesign | Low | Medium | Follow FlashInfer patterns |
| ROCm-specific issues | Medium | Medium | Test on RDNA3 hardware |
| Performance regression | Low | Medium | Benchmark before/after |

---

## Sources

### Documentation
- [ROCm HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [ROCm Precision Support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)
- [NVIDIA Matrix Multiplication Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

### Reference Implementations
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - Official FlashAttention
- [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) - LLM GPU kernels
- [vllm-project/vllm](https://github.com/vllm-project/vllm) - GPU sampler implementation

### Research Papers
- [FlashAttention-3 Paper](https://arxiv.org/html/2407.08608v2) - Latest FlashAttention
- [Finding Numerical Differences NVIDIA vs AMD](https://arxiv.org/html/2410.09172v1) - Cross-vendor precision

---

**Next Step**: Create TODO.md with task breakdown, then trigger implementation subagent.
