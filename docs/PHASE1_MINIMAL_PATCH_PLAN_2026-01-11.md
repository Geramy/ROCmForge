# Phase 1 Minimal Patch Plan: Lazy GGUF Loading

**Date:** 2026-01-11
**Status:** COMPLETE (Infrastructure Only - See Phase 2 for Speed)
**Updated:** 2026-01-11

## Summary

Phase 1 is **COMPLETE** but **does NOT achieve the original speed goal**. The lazy loading infrastructure is implemented, but the **total model loading time is still ~60s** (not <5s as originally planned).

**What Phase 1 Actually Delivers:**
- ✅ Faster `GgufLoader::new()` (~5s vs ~60s) - metadata-only initialization
- ✅ Lower RAM usage (~5GB vs ~15GB) - no upfront tensor data loading
- ✅ On-demand tensor loading via `load_tensor_to_gpu()` - can load specific tensors
- ✅ Memory-mapped file access - zero-copy reads
- ❌ NO improvement in total loading time (still ~60s due to `ExecutionPlan::from_gguf()`)

**Why Speed Goal Not Achieved:**

The `ExecutionPlan::from_gguf()` method still calls `load_to_gpu()` which loads ALL ~300 tensors to GPU before inference can start. This is a **Phase 2 architectural change** - `ExecutionPlan` must be redesigned to store `LazyTensor` handles instead of `DeviceTensor`.

---

## Original Plan (Preserved for Reference)

---

## Root Problem

```
Current ROCmForge:
1. Open GGUF file
2. Parse metadata
3. For EACH of ~300 tensors:
   - Read tensor data into RAM
   - Allocate GPU buffer (hipMalloc)
   - Copy to GPU
4. Start inference

Result: 60+ seconds loading time, thousands of allocations

llama.cpp approach:
1. Memory-map GGUF file (zero-copy)
2. Parse metadata only
3. Create tensor "handles" (not loaded)
4. Start inference
5. Load tensors on-demand when accessed
6. Cache loaded tensors

Result: <5 seconds loading, minimal allocations
```

---

## Implementation Plan (Minimal, Safe)

### Files to Create

#### 1. `src/loader/mmap.rs`

**Purpose:** Memory-mapped file access (zero-copy)

```rust
//! Memory-mapped GGUF file for zero-copy access

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use anyhow::Result;

/// Memory-mapped GGUF file
pub struct MmapGguf {
    _file: File,
    mmap: Mmap,
}

impl MmapGguf {
    /// Open and memory-map GGUF file
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { _file: file, mmap })
    }

    /// Get slice of file bytes without copying
    pub fn get_slice(&self, offset: u64, size: usize) -> Result<&[u8]> {
        let start = offset as usize;
        let end = start.saturating_add(size);

        if end > self.mmap.len() {
            anyhow::bail!("Slice out of bounds: {}..{} (file size: {})",
                start, end, self.mmap.len());
        }

        Ok(&self.mmap[start..end])
    }

    /// Get full file bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}
```

#### 2. `src/loader/lazy_tensor.rs`

**Purpose:** Lazy-loaded tensor handles

```rust
//! Lazy-loaded tensor with on-demand fetching

use std::sync::Arc;
use crate::backend::DeviceTensor;

/// Tensor that may not be loaded yet
pub enum LazyTensor {
    /// Metadata only, data not loaded
    Unloaded {
        name: String,
        offset: u64,
        size: usize,
        shape: Vec<usize>,
    },
    /// Loaded to GPU
    Gpu {
        name: String,
        tensor: DeviceTensor,
    },
}

impl LazyTensor {
    /// Create unloaded tensor handle
    pub fn unloaded(name: String, offset: u64, size: usize, shape: Vec<usize>) -> Self {
        Self::Unloaded { name, offset, size, shape }
    }

    /// Get tensor name
    pub fn name(&self) -> &str {
        match self {
            Self::Unloaded { name, .. } => name,
            Self::Gpu { name, .. } => name,
        }
    }

    /// Check if loaded to GPU
    pub fn is_gpu_loaded(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }
}
```

### Files to Modify

#### 3. `src/loader/gguf.rs`

**Changes:**
- Add lazy loading mode
- Preserve existing API (backward compatible)
- Add on-demand tensor loading

```rust
// Add to existing imports and use statements

use crate::loader::mmap::MmapGguf;
use crate::loader::lazy_tensor::LazyTensor;
use std::collections::HashMap;

// Modify GgufLoader struct
pub struct GgufLoader {
    path: String,
    metadata: GgufMetadata,
    tensors: HashMap<String, GgufTensor>,  // Keep for backward compatibility

    // NEW: Lazy loading fields
    mmap: Option<MmapGguf>,
    lazy_tensors: HashMap<String, LazyTensor>,
    gpu_cache: Arc<std::sync::RwLock<HashMap<String, DeviceTensor>>>,
}

impl GgufLoader {
    // EXISTING: Keep for compatibility (but implement lazily)
    pub fn new(path: &str) -> Result<Self> {
        // Parse metadata (fast)
        let metadata = Self::parse_metadata(path)?;

        // Create mmap for lazy access
        let mmap = MmapGguf::open(std::path::Path::new(path))?;

        // Create lazy tensor handles
        let mut lazy_tensors = HashMap::new();
        for tensor in &metadata.tensors {
            lazy_tensors.insert(
                tensor.name.clone(),
                LazyTensor::unloaded(
                    tensor.name.clone(),
                    tensor.offset,
                    tensor.nbytes(),
                    tensor.shape.clone(),
                ),
            );
        }

        Ok(Self {
            path: path.to_string(),
            metadata,
            tensors: HashMap::new(),  // Will populate on-demand
            mmap: Some(mmap),
            lazy_tensors,
            gpu_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
        })
    }

    // NEW: Load single tensor to GPU on-demand
    pub fn load_tensor_to_gpu(
        &self,
        name: &str,
        backend: &HipBackend,
    ) -> Result<DeviceTensor> {
        // Check cache first
        {
            let cache = self.gpu_cache.read().unwrap();
            if let Some(tensor) = cache.get(name) {
                return Ok(tensor.clone());
            }
        }

        // Get lazy tensor info
        let lazy = self.lazy_tensors.get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;

        let (offset, size, shape) = match lazy {
            LazyTensor::Unloaded { offset, size, shape, .. } => {
                (*offset, *size, shape.clone())
            }
            LazyTensor::Gpu { .. } => {
                return Err(anyhow::anyhow!("Tensor already loaded: {}", name));
            }
        };

        // Load bytes from mmap
        let mmap = self.mmap.as_ref().unwrap();
        let bytes = mmap.get_slice(offset, size)?;

        // Upload to GPU
        let tensor = DeviceTensor::from_bytes(bytes, shape.clone(), backend)?;

        // Cache it
        {
            let mut cache = self.gpu_cache.write().unwrap();
            cache.insert(name.to_string(), tensor.clone());
        }

        Ok(tensor)
    }

    // EXISTING API: Preserved - now uses lazy loading internally
    pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
        let mut result = HashMap::new();

        for name in self.lazy_tensors.keys() {
            let tensor = self.load_tensor_to_gpu(name, backend)?;
            result.insert(name.clone(), tensor);
        }

        // Also populate old-style tensors map for compatibility
        self.tensors = result.clone().into_iter()
            .map(|(k, v)| (k, self.device_tensor_to_gguf(v)))
            .collect();

        Ok(result)
    }
}
```

---

## Progress Indicator (Bonus)

### File: `src/bin/rocmforge_cli.rs`

Add visible progress during model loading:

```rust
//! Progress reporting for model loading

use indicatif::{ProgressBar, ProgressStyle};

fn create_load_bar(total_tensors: usize) -> ProgressBar {
    let bar = ProgressBar::new(total_tensors);
    bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("##-"));
    bar
}

// In model loading code:
let bar = create_load_bar(loader.metadata().tensor_count);

for (i, name) in loader.tensor_names().enumerate() {
    bar.set_message(format!("Loading: {}", name));
    loader.load_tensor_to_gpu(&name, &backend)?;
    bar.inc(1);
}

bar.finish_with_message("Model loaded");
```

**Add dependency:**
```toml
indicatif = "0.17"  # Progress bars
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_load_metadata_only() {
        let start = std::time::Instant::now();
        let loader = GgufLoader::new("tests/fixtures/tiny.gguf").unwrap();
        let elapsed = start.elapsed();

        // Should load metadata in < 100ms
        assert!(elapsed < std::time::Duration::from_millis(100));
        assert!(!loader.lazy_tensors.is_empty());
    }

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

    #[test]
    fn test_backward_compatibility() {
        let loader = GgufLoader::new("tests/fixtures/tiny.gguf").unwrap();
        let backend = HipBackend::new()?;

        // Old API should still work
        let tensors = loader.load_to_gpu(&backend).unwrap();
        assert!(!tensors.is_empty());
    }
}
```

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|-------|------------|
| Breaking existing API | Low | High | Preserve all public signatures |
| mmap permission issues | Low | Medium | Fallback to regular file read |
| GPU cache bugs | Low | Medium | Extensive unit tests |
| Performance regression | Low | High | Benchmark before/after |

---

## Success Criteria (Actual Results)

| Metric | Before | After (Phase 1) | Goal | Status |
|--------|--------|-----------------|------|--------|
| `GgufLoader::new()` time | ~60s | ~5s | <5s | ✅ PASS |
| RAM usage (during load) | ~15GB | ~5GB | <10GB | ✅ PASS |
| Total model load time | ~60s | ~60s | <5s | ❌ FAIL |
| hipMalloc calls | ~1000 | ~1000 | <10 | ❌ FAIL |
| API compatibility | N/A | 100% | 100% | ✅ PASS |

**Notes:**
- Total model load time is unchanged because `ExecutionPlan::from_gguf()` still eagerly loads all tensors
- hipMalloc count is unchanged because GPU upload still happens upfront
- RAM savings achieved by not loading tensor data into CPU RAM
- Phase 2 required to achieve <5s total loading time (requires ExecutionPlan redesign)

---

## Implementation Steps

1. **Step 1:** Create `src/loader/mmap.rs` (30 min)
2. **Step 2:** Create `src/loader/lazy_tensor.rs` (30 min)
3. **Step 3:** Modify `src/loader/gguf.rs` (2 hours)
4. **Step 4:** Add unit tests (1 hour)
5. **Step 5:** Run existing tests to verify compatibility (30 min)
6. **Step 6:** Benchmark model loading time (15 min)
7. **Step 7:** Add progress bar (optional, 30 min)

**Total time:** ~5 hours for first working version

---

## Rollback Plan

If issues arise:
1. Revert `src/loader/gguf.rs` to previous version
2. Delete new files `mmap.rs` and `lazy_tensor.rs`
3. All existing tests should pass

**Why safe:** Changes are internal to loader module, public API preserved

---

## Next Steps (After Phase 1)

Once Phase 1 is verified working:

1. Add progress bar for better UX
2. Implement Phase 2 (Prompt vs Generation)
3. Implement Phase 3 (Memory Pooling)
4. Implement Phase 4 (Computation Graph)

---

## Verification Command

```bash
# Before patch
time ./target/release/rocmforge_cli generate --gguf model.gguf --prompt "Hi" --max-tokens 1

# After patch (should be much faster)
time ./target/release/rocmforge_cli generate --gguf model.gguf --prompt "Hi" --max-tokens 1
```

**Expected:** Model loading time drops from >60s to <5s
