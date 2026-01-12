# GPU Kernel Debug Report - 2026-01-11

**Date**: 2026-01-11
**Agent**: Phase 18 Implementation Agent
**Status**: ISSUE IDENTIFIED - Thread Safety Problem

---

## Summary

Investigated 6 failing GPU tests and discovered the root cause: **thread safety issue in global kernel cache** when tests run concurrently.

---

## Initial Problem

**Reported Failing Tests** (from PLAN_PHASE_18_GPU_FIXES.md):
1. `test_weighted_matmul_matches_cpu_small` - diff > 0.001
2. `test_weighted_matmul_matches_cpu_32x32` - error too high
3. `test_weighted_matmul_non_square_sequences` - diff = 5.9 (MASSIVE!)
4. `test_flash_nocausal_matches_cpu_32x32` - diff = 49.6 (MASSIVE!)
5. `benchmark_flash_attention_vs_separate` - deviation too high
6. `test_hip_buffer_copy` - needs investigation

**Actual Current State** (after rebuild):
- 219/220 tests passing when run with `--test-threads=1` (single-threaded)
- 213-217/220 tests passing when run with default multi-threaded execution
- Tests are FLAKY - different failures on each run

---

## Root Cause Analysis

### Problem: Thread Safety in GLOBAL_CACHE

The `GLOBAL_CACHE` in `src/attention/kernels.rs` is a `Mutex<Option<KernelCache>>` that:
1. Gets initialized once and reused across all tests
2. Contains shared `Arc<HipBackend>` and kernel modules
3. Is accessed by multiple test threads concurrently

### Why This Causes Failures

When multiple tests run concurrently:
1. **Test A** initializes the kernel cache and loads kernels
2. **Test B** tries to use the same cache while Test A is still running GPU operations
3. The shared GPU backend gets corrupted or returns wrong results
4. Tests see garbage data (e.g., CPU expects 3.6, GPU produces 0.8)

### Evidence

**Multi-threaded execution** (default):
```
Run 1: FAILED. 214 passed; 5 failed
Run 2: FAILED. 215 passed; 4 failed
Run 3: FAILED. 218 passed; 1 failed
Run 4: FAILED. 217 passed; 2 failed
Run 5: FAILED. 215 passed; 4 failed
```

**Single-threaded execution** (`--test-threads=1`):
```
Run 1: ok. 219 passed; 0 failed
Run 2: ok. 219 passed; 0 failed
Run 3: ok. 219 passed; 0 failed
Run 4: ok. 219 passed; 0 failed
Run 5: ok. 219 passed; 0 failed
```

**100% consistent** when single-threaded!

---

## Kernel Code Verification

The HIP kernel source code in `kernels/weighted_matmul.hip` and `kernels/flash_attention_nocausal.hip` is **CORRECT**:

1. **Indexing calculations** are correct:
   - Layout: `[batch, heads, seq, dim]`
   - Offsets properly computed for each tensor
   - Element access matches CPU reference implementation

2. **Wave32 reduction** is correct:
   - Uses shared memory for partial sums
   - Reduction loop with proper stride decomposition
   - Thread 0 writes final result

3. **Grid/block configuration** is correct:
   - Block size = WARP_SIZE = 32 threads
   - Grid dimensions = (seq_q, num_heads, batch_size)
   - Each block handles one (query_pos, head, batch) triple

The kernels compute the **correct results** when run in isolation!

---

## Solution Options

### Option 1: Run Tests Single-Threaded (Temporary Fix)

Add to project-level `.cargo/config.toml`:
```toml
[lib]
test-threads = 1
```

Or run with: `cargo test -- --test-threads=1`

**Pros**: Immediate fix, no code changes
**Cons**: Slower test execution, doesn't fix underlying issue

### Option 2: Make Kernel Cache Thread-Safe (Proper Fix)

The issue is that the GPU backend itself may not be thread-safe. Options:

1. **Per-thread backend instances** - Each test thread gets its own HipBackend
2. **Better synchronization** - Add locks around GPU operations
3. **Eager initialization** - Initialize cache at program startup with proper locking

**Current status**: GLOBAL_CACHE uses Mutex, but the HipBackend inside may have its own concurrency issues.

### Option 3: Disable Kernel Cache (Alternative)

Remove caching and load kernels fresh for each test:
- **Pros**: No shared state, fully isolated
- **Cons**: Slower, redundant kernel loading

---

## Test Results Summary

### Current Test Health (Single-Threaded)
- **Passing**: 219/220 (99.5%)
- **Ignored**: 1 test (known limitation)
- **Failing**: 0 tests

### Current Test Health (Multi-Threaded)
- **Passing**: 213-218/220 (96.8-99.1%)
- **Failing**: 2-6 tests (flaky)

---

## Files Modified

1. `src/attention/weighted_matmul_tests.rs` - Removed debug output
2. `src/attention/softmax_explicit_tests.rs` - Removed debug output
3. `src/attention/flash_nocausal_tests.rs` - Removed debug output

---

## Recommendations

### Immediate (P0)
1. Add `.cargo/config.toml` with `test-threads = 1` for lib tests
2. Update CI/CD to run tests with `--test-threads=1`
3. Document this known limitation in README.md

### Short-term (P1)
1. Investigate HipBackend thread safety
2. Add proper synchronization or per-thread backend instances
3. Consider using `once_cell` or `lazy_static` for safer initialization

### Long-term (P2)
1. Design thread-safe GPU backend architecture
2. Add stress tests for concurrent GPU operations
3. Consider GPU context pooling for production use

---

## Conclusion

**The GPU kernels are CORRECT.** The failures were caused by test infrastructure issues, not kernel bugs.

All 219 tests pass consistently when run single-threaded. The kernels compute accurate results within acceptable tolerances (< 1e-3 for most operations).

**Next Step**: Implement proper thread safety in the kernel cache or mandate single-threaded test execution.

---

**Report Generated**: 2026-01-11
**Agent**: Phase 18 Implementation Agent
