# Option B: Async GPU Loading Implementation Report

**Project:** ROCmForge - AMD GPU Inference Engine for LLMs
**Date:** 2026-01-11
**Status:** PENDING IMPLEMENTATION
**Implementation Guide:** [OPTION_B_ASYNC_GPU_LOADING_GUIDE.md](./OPTION_B_ASYNC_GPU_LOADING_GUIDE.md)

---

## Executive Summary

**Implementation Status:** NOT YET IMPLEMENTED

Option B: Async GPU Loading is a comprehensive optimization strategy to reduce model loading time from **45-60 seconds to 10-20 seconds** (65-75% improvement) through:

1. **Multi-threaded CPU dequantization** using Rayon (7x speedup)
2. **Concurrent GPU uploads** using multiple HIP streams (3x speedup)
3. **Event-based synchronization** for correctness
4. **Pinned memory** for faster transfers (future optimization)

### Current State Analysis

| Component | Current Status | Required Status | Gap |
|-----------|---------------|-----------------|-----|
| HIP Events | Not implemented | Fully implemented | **Missing** |
| Rayon dependency | Not in Cargo.toml | Added to dependencies | **Missing** |
| Async loader module | Does not exist | `src/loader/async_loader.rs` | **Missing** |
| Parallel dequantization | Single-threaded | Multi-threaded (8 threads) | **Missing** |
| Concurrent uploads | Synchronous | 4-8 HIP streams | **Missing** |
| GPU cache integration | Basic `RwLock` | Thread-safe with events | Partial |

**Conclusion:** This feature requires significant new development work.

---

## Implementation Plan

This document outlines what needs to be implemented according to the comprehensive guide at [OPTION_B_ASYNC_GPU_LOADING_GUIDE.md](./OPTION_B_ASYNC_GPU_LOADING_GUIDE.md).

---

## Phase 1: Add HIP Event Support

### File: `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs`

**Current State:** Lines 1-54 contain FFI bindings, but no Event API

**Required Changes:**

#### 1.1 Add Event FFI Bindings (after line 53)

```rust
// Add after hipMemset binding (line 53)
extern "C" {
    // HIP Event API
    fn hipEventCreate(event: *mut *mut c_void) -> i32;
    fn hipEventDestroy(event: *mut c_void) -> i32;
    fn hipEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn hipEventSynchronize(event: *mut c_void) -> i32;
    fn hipEventQuery(event: *mut c_void) -> i32;
    fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
}
```

**Purpose:** Enable event-based synchronization for async GPU operations.

**Lines to add:** ~10 lines

#### 1.2 Implement HipEvent Wrapper (after line 230)

```rust
// Add after HipStream impl (after line 230)
#[repr(C)]
#[derive(Debug)]
pub struct HipEvent {
    event: *mut c_void,
}

impl HipEvent {
    pub fn new() -> HipResult<Self> {
        let mut event: *mut c_void = ptr::null_mut();
        let result = unsafe { hipEventCreate(&mut event) };

        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Failed to create HIP event: {}",
                result
            )));
        }

        if event.is_null() {
            return Err(HipError::DeviceError(
                "hipEventCreate returned null pointer".to_string(),
            ));
        }

        Ok(HipEvent { event })
    }

    pub fn record(&self, stream: &HipStream) -> HipResult<()> {
        let result = unsafe { hipEventRecord(self.event, stream.as_ptr()) };
        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Event record failed: {}",
                result
            )));
        }
        Ok(())
    }

    pub fn synchronize(&self) -> HipResult<()> {
        let result = unsafe { hipEventSynchronize(self.event) };
        if result != HIP_SUCCESS {
            return Err(HipError::DeviceError(format!(
                "Event synchronization failed: {}",
                result
            )));
        }
        Ok(())
    }

    pub fn query(&self) -> HipResult<bool> {
        let result = unsafe { hipEventQuery(self.event) };
        match result {
            HIP_SUCCESS => Ok(true),
            1 => Ok(false),  // hipErrorNotReady
            _ => Err(HipError::DeviceError(format!(
                "Event query failed: {}",
                result
            )))
        }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.event
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                hipEventDestroy(self.event);
            }
        }
    }
}

// SAFETY: HipEvent is Send+Sync because it only contains a raw pointer
// and we ensure thread-safe access through proper synchronization
unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}
```

**Purpose:** RAII wrapper for HIP events with automatic cleanup.

**Lines to add:** ~90 lines

---

## Phase 2: Add Rayon Dependency

### File: `/home/feanor/Projects/ROCmForge/Cargo.toml`

**Current State:** No Rayon dependency (lines 9-84)

**Required Changes:**

```toml
# Add after line 54 (bytemuck dependency)
rayon = "1.10"
```

**Purpose:** Enable parallel CPU dequantization using work-stealing thread pool.

**Lines to add:** 1 line

---

## Phase 3: Implement AsyncModelLoader

### File: `/home/feanor/Projects/ROCmForge/src/loader/async_loader.rs` (NEW FILE)

**Current State:** File does not exist

**Required Implementation:** ~850 lines (see detailed implementation in guide)

**Key Components:**

#### 3.1 AsyncModelLoader Struct

```rust
pub struct AsyncModelLoader {
    backend: Arc<HipBackend>,
    upload_streams: Vec<HipStream>,        // Multiple HIP streams
    completion_events: Mutex<HashMap<String, HipEvent>>,  // Track completion
    num_cpu_threads: usize,
}
```

**Purpose:** Orchestrates concurrent CPU dequantization and GPU uploads.

#### 3.2 Key Methods

| Method | Purpose | Lines |
|--------|---------|-------|
| `new()` | Create loader with N streams and M threads | ~30 |
| `load_to_gpu_async()` | Main entry point for async loading | ~50 |
| `dequantize_tensors_parallel()` | Parallel CPU dequantization | ~60 |
| `upload_tensors_concurrent()` | Concurrent GPU uploads | ~40 |
| `upload_tensor_async()` | Upload single tensor async | ~50 |
| `synchronize_all_uploads()` | Wait for all events | ~20 |

**Total lines:** ~850 (full implementation)

---

## Phase 4: Update GgufLoader

### File: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`

**Current State:** Lines 722-879 contain `load_tensor_to_gpu()` method

**Required Changes:**

#### 4.1 Add GPU Cache Access Method (after line 880)

```rust
/// Get cached GPU tensor (for async loader integration)
pub fn get_gpu_tensor_cached(&self, name: &str) -> Result<DeviceTensor> {
    let cache = self.gpu_cache.read()
        .map_err(|e| anyhow!("GPU cache read lock poisoned: {}", e))?;

    let tensor_arc = cache.get(name)
        .ok_or_else(|| anyhow!("Tensor '{}' not found in GPU cache", name))?;

    // Clone Arc to get new DeviceTensor handle
    Ok(DeviceTensor::clone(tensor_arc))
}
```

**Purpose:** Allow async loader to retrieve cached tensors after upload.

**Lines to add:** ~10 lines

---

## Phase 5: Integration Points

### File: `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs` or wherever `load_to_gpu` is called

**Current State:** Uses synchronous `GgufLoader::load_to_gpu()`

**Required Changes:**

```rust
use crate::loader::async_loader::AsyncModelLoader;

// Replace synchronous loading:
// let tensors = loader.load_to_gpu(&backend)?;

// With async loading:
let async_loader = AsyncModelLoader::new(
    Arc::new(backend.clone()),
    4,  // 4 concurrent upload streams
    8,  // 8 CPU threads for dequantization
)?;
let tensors = async_loader.load_to_gpu_async(&loader)?;
```

**Purpose:** Switch from synchronous to async loading path.

---

## Test Coverage Required

### Unit Tests

| Test | Purpose | Location |
|------|---------|----------|
| `test_hip_event_creation()` | Event FFI bindings work | `src/backend/hip_backend.rs` |
| `test_hip_event_record_sync()` | Event recording and synchronization | `src/backend/hip_backend.rs` |
| `test_parallel_dequantization_correctness()` | Rayon matches sequential | `src/loader/gguf.rs` |
| `test_async_upload_matches_sync()` | Async upload correctness | `tests/async_loader_tests.rs` |

**New test file:** `tests/async_loader_tests.rs` (~300 lines)

### Integration Tests

| Test | Purpose | Lines |
|------|---------|-------|
| `test_async_model_loading()` | End-to-end async loading | ~50 |
| `test_async_loading_memory_usage()` | No memory leaks | ~40 |
| `test_concurrent_uploads_no_corruption()` | Data integrity | ~60 |

**Total test lines:** ~450

### Performance Benchmarks

| Benchmark | Metric | Target |
|-----------|--------|--------|
| `benchmark_loading_time()` | Total load time | < 20s for 7B |
| `benchmark_upload_throughput()` | GPU bandwidth | > 8 GB/s |
| `benchmark_cpu_dequantization()` | CPU utilization | > 80% |

**New benchmark file:** `benches/async_loading_bench.rs` (~200 lines)

---

## API Changes

### New Public Functions

```rust
// In hip_backend.rs
impl HipEvent {
    pub fn new() -> HipResult<Self>;
    pub fn record(&self, stream: &HipStream) -> HipResult<()>;
    pub fn synchronize(&self) -> HipResult<()>;
    pub fn query(&self) -> HipResult<bool>;
}

// In async_loader.rs (new module)
impl AsyncModelLoader {
    pub fn new(backend: Arc<HipBackend>, num_upload_streams: usize, num_cpu_threads: usize) -> Result<Self>;
    pub fn load_to_gpu_async(&self, loader: &GgufLoader) -> Result<HashMap<String, DeviceTensor>>;
}

// In gguf.rs
impl GgufLoader {
    pub fn get_gpu_tensor_cached(&self, name: &str) -> Result<DeviceTensor>;
}
```

### Modified Signatures

**Before:**
```rust
pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>>
```

**After (optional - for async path):**
```rust
pub fn load_to_gpu_async(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>>
```

**Breaking Changes:** None (async loading is opt-in via new API)

---

## Performance Analysis

### Expected Performance Breakdown

**Current (Synchronous):**
```
Operation                    Time        % of Total
--------------------------  --------    ----------
GGUF parsing (mmap)         0.5s        1%
CPU dequantization (1T)     35s         70%
GPU uploads (sync)          12s         24%
Synchronization overhead    2.5s        5%
--------------------------  --------    ----------
TOTAL                       50s         100%
```

**Target (Async with 4 streams, 8 threads):**
```
Operation                    Time        % of Total
--------------------------  --------    ----------
GGUF parsing (mmap)         0.5s        5%
CPU dequantization (8T)     5s          50%
GPU uploads (4 streams)     4s          40%
Synchronization overhead    0.5s        5%
--------------------------  --------    ----------
TOTAL                       10s         100%
```

**Key Improvements:**
- CPU dequantization: 35s → 5s (7x faster, 8 threads)
- GPU uploads: 12s → 4s (3x faster, 4 concurrent streams)
- **Total: 50s → 10s (5x faster)**

### Scalability Analysis

**CPU Scaling (Dequantization):**
```
Threads    Dequant Time    Speedup
-------    -------------    -------
1          35s             1.0x   (current)
2          18s             1.9x
4          9s              3.9x
8          5s              7.0x   (target)
16         3s              11.7x  (diminishing returns)
```

**GPU Stream Scaling (Uploads):**
```
Streams    Upload Time    Speedup
-------    -----------    -------
1          12s            1.0x   (current)
2          6s             2.0x
4          4s             3.0x   (target)
8          4s             3.0x   (PCIe bandwidth saturated)
```

---

## Code Drift Assessment

### Deviations from Established Patterns

| Pattern | Current | Proposed | Justification |
|---------|---------|----------|---------------|
| Error handling | `Result<T>` | `Result<T>` | **No drift** - maintains consistency |
| Thread safety | `Arc<RwLock<T>>` | `Arc<RwLock<T>>` | **No drift** - GPU cache already thread-safe |
| Resource cleanup | `Drop` trait | `Drop` trait | **No drift** - RAII pattern maintained |
| Async operations | Not used | Events + streams | **New pattern** - requires documentation |
| Parallel processing | Single-threaded | Rayon parallel iterators | **New pattern** - industry standard |

**Acceptance:** All deviations are justified and follow industry best practices.

---

## Risk Assessment

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Race conditions in GPU cache** | HIGH | Use `Arc<DeviceTensor>` with `Mutex` for cache writes |
| **Out of memory (GPU)** | MEDIUM | Pre-calculate total memory before upload |
| **Stream starvation** | LOW | Use 4-8 streams (balance concurrency vs overhead) |
| **Event leaks** | MEDIUM | Implement `Drop` for `HipEvent` (RAII) |
| **Thread pool exhaustion** | LOW | Limit Rayon threads to CPU count |

### Implementation Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Incorrect FFI bindings** | HIGH | Test with small tensors first (1MB) |
| **HIP API version mismatch** | MEDIUM | Check ROCm version (`rocm-smi --showallinfo`) |
| **Synchronization bugs** | HIGH | Add extensive logging for each event/sync point |

### Performance Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **CPU bottleneck** | LOW | Profile dequantization, optimize hot loops |
| **PCIe bandwidth saturation** | MEDIUM | Use 2-4 streams (not 16+) |
| **Memory copy overhead** | MEDIUM | Use pinned memory (future optimization) |

---

## Known Issues and Limitations

### Current Limitations (Pre-Implementation)

1. **No HIP Event support** - Cannot track async operation completion
2. **Single-threaded dequantization** - CPU underutilized
3. **Synchronous GPU uploads** - No overlap between CPU and GPU work
4. **No pinned memory** - Slower host-to-device transfers

### Post-Implementation Limitations

1. **Pinned memory not implemented** - Future optimization (Phase 5)
2. **No automatic stream count tuning** - Fixed to 4 streams (user can adjust)
3. **No adaptive thread pool sizing** - Fixed to 8 threads (user can adjust)
4. **MXFP dequantization not parallelized** - Only Q4_0/Q8_0/F16/F32 supported

---

## Migration Guide

### For Users

**Before (Synchronous):**
```rust
let loader = GgufLoader::new("model.gguf")?;
let backend = HipBackend::new()?;
let tensors = loader.load_to_gpu(&backend)?;
```

**After (Async):**
```rust
let loader = GgufLoader::new("model.gguf")?;
let backend = HipBackend::new()?;

// Option 1: Use async loader (recommended for large models)
let async_loader = AsyncModelLoader::new(
    Arc::new(backend.clone()),
    4,  // 4 concurrent upload streams
    8,  // 8 CPU threads for dequantization
)?;
let tensors = async_loader.load_to_gpu_async(&loader)?;

// Option 2: Use synchronous loader (unchanged, for backward compatibility)
let tensors = loader.load_to_gpu(&backend)?;
```

**Performance Expectations:**
- Small models (< 1B params): Negligible difference (overhead dominates)
- Medium models (1-7B params): 3-5x speedup
- Large models (7B+ params): 4-5x speedup

---

## Testing Strategy

### Phase 1: Unit Tests (Week 1)

```bash
# Test HIP event creation and lifecycle
cargo test test_hip_event_creation

# Test event recording and synchronization
cargo test test_hip_event_record_sync

# Test parallel dequantization correctness
cargo test test_parallel_dequantization_correctness
```

**Pass criteria:** All tests pass, bit-exact match with sequential version.

### Phase 2: Integration Tests (Week 2)

```bash
# Test end-to-end async loading
cargo test test_async_model_loading -- --nocapture

# Test memory usage validation
cargo test test_async_loading_memory_usage

# Test concurrent upload correctness
cargo test test_concurrent_uploads_no_corruption
```

**Pass criteria:** All tests pass, no memory leaks, correct data integrity.

### Phase 3: Performance Benchmarks (Week 3)

```bash
# Run performance benchmarks
cargo bench --bench async_loading_bench
```

**Pass criteria:**
- Loading time < 20s for 7B model
- GPU upload bandwidth > 8 GB/s
- CPU utilization > 80% during dequantization

---

## Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Add HIP event FFI bindings to `hip_backend.rs`
- [ ] Implement `HipEvent` wrapper with RAII
- [ ] Add unit tests for event creation/recording/synchronization
- [ ] Add `rayon` dependency to `Cargo.toml`

### Phase 2: CPU Parallelization (Week 1)
- [ ] Implement parallel dequantization with Rayon
- [ ] Add correctness tests (sequential vs parallel comparison)
- [ ] Profile dequantization hotspots
- [ ] Optimize bit-packing loops

### Phase 3: Async Uploads (Week 2)
- [ ] Implement `AsyncModelLoader` struct
- [ ] Add multi-stream upload support
- [ ] Integrate with existing `GgufLoader` cache
- [ ] Add async upload correctness tests

### Phase 4: Integration (Week 2)
- [ ] Update model loading path to use async loader
- [ ] Add performance benchmarks (criterion)
- [ ] Implement memory usage validation
- [ ] Add stress tests (repeated loading)

### Phase 5: Optimization (Week 3)
- [ ] Add pinned memory support (`hipHostMalloc`)
- [ ] Optimize thread pool sizing
- [ ] Tune number of streams vs throughput
- [ ] Profile and optimize hotspots

### Phase 6: Validation (Week 3)
- [ ] End-to-end integration tests
- [ ] Performance regression tests
- [ ] Memory leak detection
- [ ] Production readiness review

---

## References

### Code References

**Current Implementation:**
1. `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs:20-26` - `hipMemcpyAsync` FFI binding
2. `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs:172-230` - `HipStream` implementation
3. `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs:362-417` - `copy_from_host_with_stream()` implementation
4. `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs:1368-1379` - `DeviceTensor::from_host_vec()`
5. `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:898-912` - `load_to_gpu()` entry point
6. `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:862-879` - `load_tensor_to_gpu()`
7. `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:1667-1710` - `dequantize_q4_0()` implementation
8. `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:1628-1664` - `dequantize_q8_0()` implementation
9. `/home/feanor/Projects/ROCmForge/Cargo.toml:1-85` - Dependencies

**Documentation References:**
10. `/home/feanor/Projects/ROCmForge/docs/OPTION_B_ASYNC_GPU_LOADING_GUIDE.md` - Complete implementation guide
11. `/home/feanor/Projects/ROCmForge/docs/CLI_HANG_INVESTIGATION.md` - Stream synchronization fixes

### External Documentation

**ROCm/HIP Documentation:**
- ROCm Documentation: https://rocm.docs.amd.com/
- HIP Programming Guide: https://github.com/ROCm/HIP
- HIP API Reference: https://rocm.docs.amd.com/projects/HIP/en/docs-5.0/

**Performance Optimization:**
- CUDA/HIP Streams: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- Multi-stream Optimization: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleMultiCopy

**Rust Parallelization:**
- Rayon Documentation: https://docs.rs/rayon/
- Parallel Iterators: https://docs.rs/rayon/latest/rayon/prelude/index.html

---

## Conclusion

**Implementation Status:** PENDING

Option B: Async GPU Loading is a comprehensive optimization requiring:

1. **New FFI bindings** for HIP Events (~10 lines)
2. **New wrapper type** `HipEvent` (~90 lines)
3. **New dependency** Rayon (1 line)
4. **New module** `AsyncModelLoader` (~850 lines)
5. **Test coverage** (~450 lines)
6. **Benchmark suite** (~200 lines)

**Total effort:** ~3 weeks (according to guide)

**Expected impact:** 65-75% reduction in model loading time (50s → 10-20s)

**Next steps:**
1. Review and approve implementation plan
2. Begin Phase 1: HIP Event support
3. Follow implementation guide section-by-section
4. Complete each phase with testing and validation

---

**Document Version:** 1.0
**Last Updated:** 2026-01-11
**Status:** PENDING IMPLEMENTATION
