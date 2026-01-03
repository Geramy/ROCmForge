# Phase 2: RoPE + KV Append - Progress

> Start: 2025-01-03
> GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
> Goal: Eliminate GPU↔CPU round-trips in RoPE path

## Exit Criteria
- [x] RoPE kernel passes CPU vs GPU test (5/5 tests passed)
- [x] Single decode step stays on GPU (no `to_host_vec` in RoPE path)
- [ ] Measure latency before/after (future work)

## Tasks

### Task 2.1: Understand Current Fallback ✅
- **File**: `src/attention/rope.rs:226-246` (original)
- **Symbol**: `apply_rope_device`
- **Current behavior**: Copy to host → CPU RoPE → copy back to device
- **CPU reference**: `apply_rope` at `src/attention/rope.rs:137-200`

### Task 2.2: Write Failing Test (TDD) ✅
- **File**: `src/attention/rope_gpu_tests.rs` (created)
- **Tests**: 5 tests for CPU vs GPU comparison
- **Result**: All 5 tests pass with GPU implementation

### Task 2.3: Implement rope_kernel HIP ✅
- **File**: `kernels/rope.hip` (created)
- **Kernel**: `rope_kernel`
- **Contract**:
  - Input: `[seq_len, num_heads, head_dim]` row-major
  - cos/sin: `[seq_len, head_dim/2]` pre-extracted per token
  - Rotates pairs: x0' = x0*cos - x1*sin, x1' = x0*sin + x1*cos
  - Grid: `(seq_len, num_heads, 1)` - one block per token per head
  - Block: `(256, 1, 1)` - RDNA3 optimized (8 waves of 32)

### Task 2.4: Integrate GPU Kernel ✅
- **File**: `src/attention/kernels.rs`
- **Add**: `rope_gpu_kernel` function, `rope_module`, `rope_kernel` to KernelCache
- **File**: `build.rs`
- **Add**: rope.hip to kernel compilation list
- **File**: `src/attention/rope.rs`
- **Modify**: `apply_rope_device` to use GPU kernel instead of CPU fallback

### Task 2.5: Verify No CPU Round-trip ✅
- **Test**: `grep to_host_vec src/attention/rope.rs` returns no matches
- **Result**: GPU path has zero CPU round-trips

## Changes Log

| Date | Change | File |
|------|--------|------|
| 2025-01-03 | Phase 2 start | - |
| 2025-01-03 | Created rope_gpu_tests.rs with 5 CPU vs GPU tests | src/attention/rope_gpu_tests.rs |
| 2025-01-03 | Created rope.hip kernel for RDNA3 | kernels/rope.hip |
| 2025-01-03 | Added rope_module/rope_kernel to KernelCache | src/attention/kernels.rs |
| 2025-01-03 | Added ROPE_HSACO loading in get_or_init_cache() | src/attention/kernels.rs |
| 2025-01-03 | Implemented rope_gpu_kernel() wrapper function | src/attention/kernels.rs |
| 2025-01-03 | Replaced CPU fallback with GPU kernel in apply_rope_device | src/attention/rope.rs |
| 2025-01-03 | Added rope.hip to build.rs kernel list | build.rs |
| 2025-01-03 | All 5 RoPE GPU tests passing | - |

## Test Results
```
cargo test --features rocm --lib rope_gpu
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 97 filtered out
```
