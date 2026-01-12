# Phase 6: GPU Kernel Hang Investigation

**Date:** 2026-01-11
**Status:** Investigating GPU kernel hang during top-p sampling

## Problem

When calling `GpuTopPSampler::sample()`, the test times out after 15 seconds. The GPU kernel appears to hang during execution.

## What Works

1. ✅ Kernel cache initializes successfully
2. ✅ HSACO files compile successfully
3. ✅ Kernel loads from HSACO
4. ✅ GPU buffer allocation succeeds
5. ✅ Data copy to GPU succeeds
6. ❌ GPU kernel launch or synchronization hangs

## Code Path

The execution flow in `src/sampler/gpu.rs:try_gpu_sample()`:

1. ✅ `get_or_init_sampling_cache()` - OK
2. ✅ `HipBuffer::new()` - OK
3. ✅ `copy_from_host()` - OK
4. ❌ `topp_sampling_kernel()` - HANG
5. ❌ `backend.synchronize()` - Never reached

## Kernel Signature

```hip
extern "C" __global__ void topp_sampling_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ random_values,
    uint32_t* __restrict__ output,
    const float top_p,
    const int batch_size,
    const int vocab_size
);
```

## Launch Configuration

```rust
let grid_dim = (batch_size, 1, 1);  // e.g., (2, 1, 1)
let block_dim = (BLOCK_SIZE, 1, 1);  // (256, 1, 1)
```

## Potential Issues

### 1. Thread 0 Execution Pattern
The kernel uses `if (tid == 0)` pattern, meaning only thread 0 does work:
```cpp
if (tid == 0) {
    // ... all the work here
}
```
This is inefficient and may cause issues if thread 0 is on a faulty wavefront.

### 2. Infinite Loop in Rejection Sampling
The kernel has a `MAX_ITERATIONS = 10` loop but may still hang if:
- `found` is never set to true
- Probabilities don't sum correctly
- Numerical issues with float comparison

### 3. Memory Access Issues
- Reading/writing out of bounds
- Uninitialized memory reads
- GPU memory faults

### 4. Synchronization Issues
- Missing `__syncthreads()` calls
- Race conditions in shared memory

## Next Steps

Following the TDD methodology with 3 subagents:

### Subagent 1: Research (Context7 + Web)
Research:
- AMD ROCm/HIP kernel debugging best practices
- Common causes of GPU kernel hangs
- How to use `rocprof` or `RGP` for debugging
- Proper thread synchronization patterns

### Subagent 2: Minimal Reproduction
Create a minimal HIP kernel that:
- Just returns without doing work
- Prints debug info
- Uses simpler data patterns

### Subagent 3: Fix Implementation
Based on findings:
- Fix kernel code if issue found
- Add proper error handling
- Improve thread utilization

## References

- HIP Programming Guide: https://rocm.docs.amd.com/
- GPU Debugging: https://rocm.docs.amd.com/_contents/Debugging.html
- FlashInfer sampling: https://flashinfer.ai/2025/03/10/sampling.html
