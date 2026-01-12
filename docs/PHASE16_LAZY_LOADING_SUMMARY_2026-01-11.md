# Phase 16: Lazy GGUF Loading Implementation Summary

**Date:** 2026-01-11
**Status:** PENDING - Implementation not started
**Phase:** 16 (following Phase 15)
**Implementation Plan:** `docs/PHASE1_MINIMAL_PATCH_PLAN_2026-01-11.md`

---

## Overview

Phase 16 implements lazy loading for GGUF model files, dramatically reducing model loading time from 60+ seconds to <5 seconds by deferring tensor uploads to GPU until they are actually needed during inference.

**Key Innovation:** Memory-mapped GGUF files with on-demand tensor loading, following the proven pattern used by llama.cpp.

---

## Implementation Status

**Status:** NOT STARTED - Implementation pending

### Progress

| Task | Status | Notes |
|------|--------|-------|
| Memory-mapped file wrapper | NOT STARTED | `src/loader/mmap.rs` - 84 lines planned |
| Lazy tensor handles | NOT STARTED | `src/loader/lazy_tensor.rs` - 131 lines planned |
| GGUF loader modifications | NOT STARTED | Lazy loading integration |
| Unit tests | NOT STARTED | Metadata, on-demand, compatibility tests |
| Performance benchmarks | NOT STARTED | Before/after timing comparison |

---

## Files to Create

### 1. `src/loader/mmap.rs`

**Purpose:** Memory-mapped file access for zero-copy GGUF reading

**Planned LOC:** ~84 lines

**Key Functions:**
- `MmapGguf::open(path: &Path) -> Result<Self>` - Open and mmap GGUF file
- `get_slice(offset: u64, size: usize) -> Result<&[u8]>` - Zero-copy byte slice
- `as_bytes() -> &[u8]` - Full file access

**Dependencies:**
- `memmap2` crate for memory mapping
- `anyhow` for error handling

---

### 2. `src/loader/lazy_tensor.rs`

**Purpose:** Lazy-loaded tensor handles with on-demand GPU upload

**Planned LOC:** ~131 lines

**Key Types:**
```rust
pub enum LazyTensor {
    Unloaded {
        name: String,
        offset: u64,
        size: usize,
        shape: Vec<usize>,
    },
    Gpu {
        name: String,
        tensor: DeviceTensor,
    },
}
```

**Key Methods:**
- `LazyTensor::unloaded()` - Create unloaded tensor handle
- `name() -> &str` - Get tensor name
- `is_gpu_loaded() -> bool` - Check load status

---

## Files to Modify

### 1. `src/loader/gguf.rs`

**Purpose:** Integrate lazy loading into existing GGUF loader

**Planned Changes:**

1. **Add new fields to `GgufLoader`:**
   ```rust
   pub struct GgufLoader {
       // Existing fields...
       mmap: Option<MmapGguf>,              // NEW: Memory-mapped file
       lazy_tensors: HashMap<String, LazyTensor>,  // NEW: Lazy handles
       gpu_cache: Arc<RwLock<HashMap<String, DeviceTensor>>>,  // NEW: GPU cache
   }
   ```

2. **Modify `GgufLoader::new()`:**
   - Parse metadata (fast)
   - Create mmap for lazy access
   - Create lazy tensor handles (no GPU upload)

3. **Add `load_tensor_to_gpu()`:**
   - Load single tensor on-demand
   - Cache in GPU memory
   - Return cached tensor if already loaded

4. **Preserve `load_to_gpu()` API:**
   - Backward compatible with existing code
   - Now uses lazy loading internally

**Estimated LOC:** ~125 lines of new code

---

## Unit Tests

### Test 1: Metadata-Only Loading

**File:** `src/loader/gguf.rs` (inline tests)

**Purpose:** Verify fast metadata parsing without tensor upload

```rust
#[test]
fn test_lazy_load_metadata_only() {
    let start = std::time::Instant::now();
    let loader = GgufLoader::new("tests/fixtures/tiny.gguf").unwrap();
    let elapsed = start.elapsed();

    // Should load metadata in < 100ms
    assert!(elapsed < std::time::Duration::from_millis(100));
    assert!(!loader.lazy_tensors.is_empty());
}
```

**Success Criteria:** Loads in <100ms, no GPU upload

---

### Test 2: On-Demand Tensor Loading

**Purpose:** Verify single tensor loads on-demand and caches

```rust
#[test]
fn test_on_demand_tensor_load() {
    let loader = GgufLoader::new("tests/fixtures/tiny.gguf").unwrap();
    let backend = HipBackend::new()?;

    // Tensor not cached initially
    assert!(!loader.gpu_cache.read().unwrap().contains_key("blk.0.attn_q.weight"));

    // Load on demand
    let tensor = loader.load_tensor_to_gpu("blk.0.attn_q.weight", &backend).unwrap();

    // Now cached
    assert!(loader.gpu_cache.read().unwrap().contains_key("blk.0.attn_q.weight"));
}
```

**Success Criteria:** Tensor loads and caches correctly

---

### Test 3: Backward Compatibility

**Purpose:** Verify existing API still works

```rust
#[test]
fn test_backward_compatibility() {
    let loader = GgufLoader::new("tests/fixtures/tiny.gguf").unwrap();
    let backend = HipBackend::new()?;

    // Old API should still work
    let tensors = loader.load_to_gpu(&backend).unwrap();
    assert!(!tensors.is_empty());
}
```

**Success Criteria:** Old `load_to_gpu()` API passes

---

## Performance Goals

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Model load time | >60s | <5s | **12x faster** |
| hipMalloc calls | ~1000 | <10 (initial) | **99% reduction** |
| Initial memory usage | 100% tensors | ~0 tensors | **Deferred allocation** |
| API compatibility | N/A | 100% | **Zero breaking changes** |

---

## Risk Mitigation

### Risk 1: Breaking Existing API

**Probability:** LOW
**Impact:** HIGH

**Mitigation:**
- Preserve all existing public API signatures
- Add lazy loading as internal implementation detail
- Comprehensive backward compatibility tests

### Risk 2: mmap Permission Issues

**Probability:** LOW
**Impact:** MEDIUM

**Mitigation:**
- Fallback to regular file read if mmap fails
- Clear error messages for permission errors
- Test on various file systems

### Risk 3: GPU Cache Bugs

**Probability:** LOW
**Impact:** MEDIUM

**Mitigation:**
- Extensive unit tests for cache behavior
- RwLock for thread safety
- Cache invalidation on error

---

## Success Criteria

### Must Have (P0)

- [ ] Model loading time reduced to <5 seconds (from 60s)
- [ ] All existing tests pass (zero regressions)
- [ ] Backward compatibility maintained (100%)
- [ ] Memory mapping works on Linux (ROCm platform)

### Should Have (P1)

- [ ] On-demand tensor loading functional
- [ ] GPU cache working correctly
- [ ] Unit tests for lazy loading path
- [ ] Performance benchmarks documented

### Nice to Have (P2)

- [ ] Progress bar during first load
- [ ] Fallback to regular file I/O if mmap fails
- [ ] Metrics for cache hit/miss rates

---

## Known Issues / Limitations

### Pre-Implementation (Expected)

1. **First inference slower:** First forward pass will load all needed tensors
   - **Mitigation:** Acceptable trade-off for 12x faster startup

2. **Memory usage:** Model file remains memory-mapped
   - **Mitigation:** OS page cache manages this efficiently

3. **No tensor offloading:** Once loaded, tensors stay in GPU memory
   - **Future work:** Implement LRU eviction for large models

---

## API Compatibility

### Preserved APIs

All existing public APIs remain unchanged:

```rust
// These continue to work exactly as before
pub fn GgufLoader::new(path: &str) -> Result<Self>
pub fn GgufLoader::load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>>
pub fn GgufLoader::metadata(&self) -> &GgufMetadata
```

### New APIs (Internal)

```rust
// NEW: On-demand tensor loading (internal use)
pub fn GgufLoader::load_tensor_to_gpu(&self, name: &str, backend: &HipBackend) -> Result<DeviceTensor>

// NEW: Query load status
pub fn GgufLoader::is_tensor_loaded(&self, name: &str) -> bool
```

---

## Testing Strategy

### Unit Tests

- **Metadata-only loading test** - Verify fast startup
- **On-demand tensor loading test** - Verify lazy path
- **Backward compatibility test** - Verify old API works
- **GPU cache behavior test** - Verify caching works

### Integration Tests

- **End-to-end model loading** - Time before/after
- **Inference correctness** - Verify outputs match
- **Memory profiling** - Verify reduced allocations

### Performance Benchmarks

```bash
# Before patch
time ./target/release/rocmforge_cli generate --gguf model.gguf --prompt "Hi" --max-tokens 1

# After patch (should be much faster)
time ./target/release/rocmforge_cli generate --gguf model.gguf --prompt "Hi" --max-tokens 1
```

---

## Rollback Plan

If issues arise during implementation:

1. **Revert changes** to `src/loader/gguf.rs`
2. **Delete new files** (`mmap.rs`, `lazy_tensor.rs`)
3. **Verify tests pass** with original implementation

**Safety:** Changes are isolated to loader module, public API preserved

---

## Implementation Timeline

| Step | Task | Estimated Time | Status |
|------|------|----------------|--------|
| 1 | Create `src/loader/mmap.rs` | 30 min | NOT STARTED |
| 2 | Create `src/loader/lazy_tensor.rs` | 30 min | NOT STARTED |
| 3 | Modify `src/loader/gguf.rs` | 2 hours | NOT STARTED |
| 4 | Add unit tests | 1 hour | NOT STARTED |
| 5 | Run existing tests (verify compatibility) | 30 min | NOT STARTED |
| 6 | Benchmark model loading time | 15 min | NOT STARTED |
| 7 | Add progress bar (optional) | 30 min | NOT STARTED |

**Total Estimated Time:** ~5 hours for first working version

---

## Next Steps After Phase 16

Once Phase 16 is verified working:

1. **Add progress bar** for better UX during first tensor load
2. **Implement Phase 2:** Prompt vs Generation optimization
3. **Implement Phase 3:** Memory Pooling integration
4. **Implement Phase 4:** Computation graph optimization

---

## References

- **Implementation Plan:** `docs/PHASE1_MINIMAL_PATCH_PLAN_2026-01-11.md`
- **GGUF Format:** GGUF specification
- **llama.cpp:** Reference implementation of lazy loading
- **memmap2 crate:** https://docs.rs/memmap2/

---

## Implementation Checklist

When implementation begins:

- [ ] Read `docs/PHASE1_MINIMAL_PATCH_PLAN_2026-01-11.md` thoroughly
- [ ] Create `src/loader/mmap.rs` with memory-mapped file wrapper
- [ ] Create `src/loader/lazy_tensor.rs` with lazy tensor enum
- [ ] Modify `src/loader/gguf.rs` to integrate lazy loading
- [ ] Add unit tests (metadata, on-demand, compatibility)
- [ ] Run `cargo test --features rocm` to verify no regressions
- [ ] Benchmark model loading time (before/after)
- [ ] Update this document with actual results
- [ ] Update `docs/CHANGELOG.md` with Phase 16 completion
- [ ] Update `docs/TODO.md` to mark Phase 16 complete

---

**Last Updated:** 2026-01-11
**Status:** PENDING IMPLEMENTATION
**Owner:** TBD (assigned to implementation agent)
