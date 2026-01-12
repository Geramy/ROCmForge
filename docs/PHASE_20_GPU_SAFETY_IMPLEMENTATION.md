# Phase 20: GPU Testing Safety - Concrete Implementation Plan

**Date**: 2026-01-11
**Status**: Ready for Implementation
**Priority**: P0 - BLOCKS ALL GPU TESTING

---

## Overview

This document provides **concrete implementation** designed specifically for ROCmForge's existing API structure. Unlike the generic safety guide, this plan works with your actual code:

- Your existing `HipBackend::new()` singleton pattern
- Your existing `DeviceTensor::from_host_vec()` API
- Your existing `get_memory_info()` method
- Your existing `HipStream::synchronize()` (which is already safe!)

---

## Current API Analysis

### What You Already Have (Good!)

```rust
// src/backend/hip_backend.rs

impl HipBackend {
    // ✅ Already returns Arc<HipBackend> - singleton pattern exists
    pub fn new() -> HipResult<Arc<Self>> { ... }

    // ✅ Already implemented - use this!
    pub fn get_memory_info(&self) -> HipResult<(usize, usize)> { ... }

    // ✅ Already uses hipStreamSynchronize - this is SAFE!
    pub fn synchronize(&self) -> HipResult<()> {
        self.stream.synchronize()
    }
}

impl HipStream {
    // ✅ Already uses hipStreamSynchronize - SAFE!
    pub fn synchronize(&self) -> HipResult<()> {
        unsafe { hipStreamSynchronize(self.stream) }
    }
}
```

### What Needs Fixing

| Location | Current | Problem | Fix |
|----------|---------|---------|-----|
| `hip_backend.rs:612` | `hipDeviceSynchronize()` | Hangs if desktop using GPU | Use `self.stream.as_ptr()` |
| All test files | `HipBackend::new()` in each test | Multiple backends, no cleanup | Shared fixture |
| `allocate_buffer()` | No safety check | Can exhaust VRAM | Add conservative check |

---

## Phase 20.1: GPU Availability Check

### Implementation

**File**: `src/backend/hip_backend.rs`

**Add to `impl HipBackend`** (after line 883, before `pub fn new()`):

```rust
impl HipBackend {
    /// Check if GPU is available WITHOUT initializing HIP
    /// This is a static check that can be called before creating backend
    ///
    /// Returns false if:
    /// - No GPU device present
    /// - HIP runtime not installed
    /// - hipInit() fails
    pub fn gpu_available() -> bool {
        use std::sync::atomic::{AtomicBool, Ordering, Once};

        static CHECKED: AtomicBool = AtomicBool::new(false);
        static AVAILABLE: AtomicBool = AtomicBool::new(false);
        static INIT: Once = Once::new();

        INIT.call_once(|| {
            // Use catch_unwind to prevent panics from propagating
            let result = std::panic::catch_unwind(|| {
                unsafe {
                    // Try to initialize HIP
                    let init_result = hipInit(0);
                    if init_result != HIP_SUCCESS {
                        return false;
                    }

                    // Try to get device count
                    let mut count: i32 = 0;
                    let count_result = hipGetDeviceCount(&mut count);
                    count_result == HIP_SUCCESS && count > 0
                }
            }).unwrap_or(false);

            AVAILABLE.store(result, Ordering::Release);
            CHECKED.store(true, Ordering::Release);
        });

        AVAILABLE.load(Ordering::Acquire)
    }

    /// Create backend only if GPU is available
    /// Returns a clear error if GPU is not present
    pub fn new_checked() -> HipResult<Arc<Self>> {
        if !Self::gpu_available() {
            return Err(HipError::DeviceNotFound);
        }
        Self::new()
    }
}
```

**No changes needed to tests yet** - this is just an addition.

---

## Phase 20.2: Conservative Memory Allocation

### Implementation

**File**: `src/backend/hip_backend.rs`

**Add to `impl HipBackend`** (after `get_memory_info()`, around line 1012):

```rust
impl HipBackend {
    // ... existing methods ...

    /// Check if an allocation of given size is safe
    /// Returns true if size < 70% of currently free GPU memory
    pub fn can_allocate(&self, size: usize) -> HipResult<bool> {
        let (free, _total) = self.get_memory_info()?;

        // Safety margin: use only 70% of free memory
        // Leave 30% for desktop/compositor/driver overhead
        let safe_threshold = (free * 7) / 10;

        Ok(size <= safe_threshold)
    }

    /// Allocate buffer with conservative memory check
    /// Returns error if requested size exceeds 70% of free GPU memory
    pub fn allocate_buffer_safe(&self, size: usize) -> HipResult<HipBuffer> {
        // First check if allocation is safe
        if !self.can_allocate(size)? {
            // Get details for error message
            let (free, total) = self.get_memory_info()?;
            let safe_threshold = (free * 7) / 10;

            return Err(HipError::MemoryAllocationFailed(format!(
                "Requested {} bytes exceeds safe threshold {} bytes (free={}, total={})\n\
                 This prevents GPU memory exhaustion which would crash the desktop compositor.",
                size, safe_threshold, free, total
            )));
        }

        // Use existing allocate_buffer method
        self.allocate_buffer(size)
    }

    /// Get safe allocation size for testing
    /// Returns 70% of currently free GPU memory
    pub fn safe_alloc_size(&self) -> HipResult<usize> {
        let (free, _) = self.get_memory_info()?;
        Ok((free * 7) / 10)
    }
}
```

**Update `DeviceTensor::empty()` to use safe allocation**:

**File**: `src/backend/hip_backend.rs` (around line 1521)

```rust
impl DeviceTensor {
    // ... existing methods ...

    /// Create empty tensor with conservative memory allocation
    /// This is the SAFE version that won't exhaust GPU memory
    pub fn empty_safe(backend: &HipBackend, shape: TensorShape) -> HipResult<Self> {
        let total_bytes = shape.total_elements() * std::mem::size_of::<f32>();

        // Check if allocation is safe first
        if !backend.can_allocate(total_bytes)? {
            return Err(HipError::MemoryAllocationFailed(format!(
                "Cannot allocate tensor with shape {:?}: {} bytes exceeds safe limit",
                shape.dims(), total_bytes
            )));
        }

        let buffer = backend.allocate_buffer(total_bytes)?;
        Ok(DeviceTensor { buffer, shape })
    }
}
```

---

## Phase 20.3: Fix Dangerous Synchronize

### The Problem

**File**: `src/backend/hip_backend.rs:612`

```rust
// DANGEROUS - This is in HipBuffer::copy_to_host()
let sync_result = unsafe { hipDeviceSynchronize() };
```

### The Fix

**File**: `src/backend/hip_backend.rs`

**Update `HipBuffer::copy_to_host()` method** (around line 594):

```rust
impl HipBuffer {
    // ... existing methods ...

    /// Copy data from device to host
    ///
    /// NOTE: This is the SAFE version that uses stream-aware synchronization
    /// The old version used hipDeviceSynchronize() which could hang if desktop
    /// was using the GPU. This version uses hipStreamSynchronize() which only
    /// waits for our application's stream.
    ///
    /// IMPORTANT: This requires a stream pointer. For code that doesn't have
    /// access to a stream, use HipBackend::copy_from_device() instead.
    pub fn copy_to_host_with_stream_sync<T>(
        &self,
        data: &mut [T],
        stream_ptr: *mut c_void,
    ) -> HipResult<()> {
        let byte_size = std::mem::size_of_val(data);
        if byte_size > self.size() {
            return Err(HipError::MemoryCopyFailed(format!(
                "Destination buffer too small: {} > {}",
                byte_size, self.size()
            )));
        }

        // Synchronize OUR stream first (not the whole device!)
        let sync_result = unsafe { hipStreamSynchronize(stream_ptr) };
        if sync_result != HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "Stream synchronization failed with code {} before D2H copy",
                sync_result
            )));
        }

        let ptr = self.ptr();

        // Use hipMemcpy to copy from device to host
        let result = unsafe {
            hipMemcpy(
                data.as_mut_ptr() as *mut c_void,
                ptr,
                byte_size,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };

        if result != HIP_SUCCESS {
            return Err(HipError::MemoryCopyFailed(format!(
                "hipMemcpyDtoH failed with code {}",
                result
            )));
        }

        Ok(())
    }
}
```

**Note**: You already have `copy_to_host_with_stream()` at line 662 which is safe!
The fix is to **stop using `copy_to_host()`** and use `copy_to_host_with_stream()` instead.

**Update `HipBackend::copy_from_device()`** (around line 1085) to use the safe version:

```rust
impl HipBackend {
    // ... existing methods ...

    /// Copy from GPU to host (safe version with stream sync)
    /// This is the preferred method - it doesn't use hipDeviceSynchronize()
    pub fn copy_from_device_safe<T>(&self, gpu_buffer: &HipBuffer, host_data: &mut [T]) -> HipResult<()> {
        gpu_buffer.copy_to_host_with_stream_sync(host_data, self.stream.as_ptr())
    }
}
```

---

## Phase 20.4: GPU Test Fixture

### Implementation

**Create new file**: `tests/common/mod.rs`

```rust
//! Common test utilities for GPU testing
//!
//! This module provides shared fixtures for GPU tests that:
//! - Check GPU availability before running
//! - Use a single shared backend (no multiple allocations)
//! - Check for memory leaks after each test

use once_cell::sync::Lazy;
use rocmforge::backend::HipBackend;

/// Global GPU test fixture
///
/// This is initialized ONCE for all tests and shared across them.
/// If GPU is not available, tests will skip gracefully.
pub static GPU_FIXTURE: Lazy<Option<GpuTestFixture>> = Lazy::new(|| {
    if !HipBackend::gpu_available() {
        eprintln!("⚠️  WARNING: GPU not available - skipping GPU tests");
        eprintln!("To enable GPU tests, ensure:");
        eprintln!("  1. AMD GPU is present");
        eprintln!("  2. ROCm is installed (check with rocm-smi)");
        eprintln!("  3. amdhip64 library is in LD_LIBRARY_PATH");
        return None;
    }

    match GpuTestFixture::new() {
        Ok(fixture) => {
            eprintln!("✅ GPU Test Fixture initialized");
            eprintln!("   Device: {}", fixture.device_name());
            eprintln!("   Total Memory: {} MB", fixture.total_memory_mb());
            eprintln!("   Free Memory: {} MB", fixture.free_memory_mb());
            eprintln!("   Safe Alloc Limit: {} MB", fixture.safe_alloc_mb());
            Some(fixture)
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to initialize GPU test fixture: {}", e);
            eprintln!("   GPU tests will be skipped");
            None
        }
    }
});

pub struct GpuTestFixture {
    backend: std::sync::Arc<HipBackend>,
    initial_free_mb: usize,
    initial_total_mb: usize,
    device_name: String,
}

impl GpuTestFixture {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let backend = HipBackend::new_checked()?;  // Use new_checked!
        let (free, total) = backend.get_memory_info()?;
        let device = backend.device();

        Ok(Self {
            backend,
            initial_free_mb: free / 1024 / 1024,
            initial_total_mb: total / 1024 / 1024,
            device_name: device.name.clone(),
        })
    }

    /// Get the shared backend
    pub fn backend(&self) -> &std::sync::Arc<HipBackend> {
        &self.backend
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get total GPU memory in MB
    pub fn total_memory_mb(&self) -> usize {
        self.initial_total_mb
    }

    /// Get initial free memory in MB
    pub fn free_memory_mb(&self) -> usize {
        self.initial_free_mb
    }

    /// Get safe allocation limit in MB (70% of initial free)
    pub fn safe_alloc_mb(&self) -> usize {
        (self.initial_free_mb * 7) / 10
    }

    /// Check for memory leaks after test
    ///
    /// Tolerance is in percent (e.g., 5 = allow 5% variance)
    /// This accounts for memory fragmentation and driver overhead
    pub fn assert_no_leak(&self, tolerance_percent: usize) {
        let (free, _total) = self.backend.get_memory_info()
            .expect("Failed to query GPU memory");

        let free_mb = free / 1024 / 1024;
        let leaked_mb = self.initial_free_mb.saturating_sub(free_mb);
        let tolerance_mb = (self.initial_total_mb * tolerance_percent) / 100;

        if leaked_mb > tolerance_mb {
            panic!(
                "🚨 GPU memory leak detected!\n\
                 Initial free: {} MB\n\
                 Current free: {} MB\n\
                 Leaked: {} MB\n\
                 Tolerance: {} MB ({}%)\n\
                 💡 Tip: Make sure DeviceTensors are dropped before end of test",
                self.initial_free_mb, free_mb, leaked_mb, tolerance_mb, tolerance_percent
            );
        }
    }

    /// Get current memory usage stats
    pub fn memory_stats(&self) -> (usize, usize) {
        match self.backend.get_memory_info() {
            Ok((free, total)) => (free / 1024 / 1024, total / 1024 / 1024),
            Err(_) => (0, 0),
        }
    }
}
```

**Update `tests/lib.rs` or create it**:

```rust
//! Test library common module

#[cfg(test)]
pub mod common;
```

---

## Phase 20.5: Update Test Pattern

### Before (Current Pattern)

```rust
#[test]
fn test_kv_replication_mqa() {
    let backend = HipBackend::new().expect("Failed to create HIP backend");
    // ... test code ...
}
```

### After (Safe Pattern)

```rust
#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use crate::attention::multi_query::{MultiQueryAttention, MultiQueryConfig};
    use crate::backend::DeviceTensor;
    use crate::loader::TensorShape;
    use crate::tests::common::GPU_FIXTURE;  // Import the fixture

    /// Test MQA: 32 query heads, 1 KV head
    #[test]
    fn test_kv_replication_mqa() {
        // Get shared fixture - returns early if GPU unavailable
        let fixture = GPU_FIXTURE.as_ref()
            .expect("GPU not available - test skipped");

        let backend = fixture.backend();
        let config = MultiQueryConfig::new(32, 128);
        let mqa = MultiQueryAttention::new(config).expect("Failed to create MQA");

        let batch_size = 1;
        let seq_len = 16;
        let num_kv_heads = 1;
        let num_q_heads = 32;
        let head_dim = 128;

        let k_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let v_shape = TensorShape::from_dims(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let q_shape = TensorShape::from_dims(&[batch_size, seq_len, num_q_heads, head_dim]);

        let mut k_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut v_host = vec![0.0f32; batch_size * seq_len * num_kv_heads * head_dim];
        let mut q_host = vec![0.0f32; batch_size * seq_len * num_q_heads * head_dim];

        for (i, val) in k_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.1;
        }
        for (i, val) in v_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.2;
        }
        for (i, val) in q_host.iter_mut().enumerate() {
            *val = (i as f32) * 0.3;
        }

        let k_device = DeviceTensor::from_host_vec(backend, k_host.clone(), k_shape)
            .expect("Failed to create K tensor");
        let v_device = DeviceTensor::from_host_vec(backend, v_host.clone(), v_shape)
            .expect("Failed to create V tensor");
        let q_device = DeviceTensor::from_host_vec(backend, q_host.clone(), q_shape.clone())
            .expect("Failed to create Q tensor");

        // Execute forward_device
        let output_device = mqa.forward_device(&q_device, &k_device, &v_device, None, None)
            .expect("forward_device failed");

        // Verify output shape matches input
        assert_eq!(output_device.shape().dims(), q_shape.dims());

        // Verify output is not all zeros
        let output_host = output_device.to_host_vec()
            .expect("Failed to copy output to host");
        let non_zero_count = output_host.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero_count > 0, "Output should not be all zeros");

        // 💡 NEW: Check for memory leak (5% tolerance)
        drop(k_device);  // Explicit drop before leak check
        drop(v_device);
        drop(q_device);
        drop(output_device);
        fixture.assert_no_leak(5);
    }
}
```

### Minimal Change Pattern

If you want to change tests with minimal edits:

```rust
#[test]
fn test_something() {
    // Just add these 3 lines at the start:
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available - test skipped");
    let backend = fixture.backend();

    // Replace: let backend = HipBackend::new()...
    // with nothing (backend from above)

    // ... existing test code ...

    // Add at the end:
    fixture.assert_no_leak(5);
}
```

---

## Cargo.toml Changes

**Add to `Cargo.toml`** (dev-dependencies section):

```toml
[dev-dependencies]
# ... existing deps ...
once_cell = "1.19"
serial_test = "3.0"
```

**Note**: `once_cell` may already be in your workspace. Check `Cargo.toml` first.

---

## Implementation Checklist

- [ ] Phase 20.1: Add `gpu_available()` and `new_checked()` to `HipBackend`
- [ ] Phase 20.2: Add `can_allocate()`, `allocate_buffer_safe()`, `safe_alloc_size()` to `HipBackend`
- [ ] Phase 20.2: Add `empty_safe()` to `DeviceTensor`
- [ ] Phase 20.3: Add `copy_to_host_with_stream_sync()` to `HipBuffer`
- [ ] Phase 20.3: Add `copy_from_device_safe()` to `HipBackend`
- [ ] Phase 20.4: Create `tests/common/mod.rs` with `GPU_FIXTURE`
- [ ] Phase 20.5: Update tests to use `GPU_FIXTURE` pattern
- [ ] Phase 20.5: Add `#[serial]` to all GPU tests (requires serial_test crate)
- [ ] Run `cargo test --features rocm --lib -- --test-threads=1` to verify

---

## Testing the Implementation

After implementing each phase:

```bash
# Phase 20.1: Verify GPU detection
cargo test --features rocm --lib hip_backend::tests::test_gpu_available

# Phase 20.2: Verify conservative allocation
cargo test --features rocm --lib hip_backend::tests::test_conservative_alloc

# Phase 20.3: Verify safe sync doesn't hang
cargo test --features rocm --lib hip_backend::tests::test_safe_sync

# Phase 20.4-20.5: Run updated tests
cargo test --features rocm --lib -- --test-threads=1
```

---

## Migration Guide for Existing Tests

### Quick Find & Replace

**In each test file**, add the import:

```rust
use crate::tests::common::GPU_FIXTURE;
```

**Replace**:
```rust
let backend = HipBackend::new().expect("...");
```

**With**:
```rust
let fixture = GPU_FIXTURE.as_ref().expect("GPU not available - test skipped");
let backend = fixture.backend();
```

**Add at end of each test**:
```rust
fixture.assert_no_leak(5);
```

---

## Summary

This implementation plan is designed specifically for ROCmForge's existing API:

| Phase | What | Fits Your API |
|-------|------|---------------|
| 20.1 | `gpu_available()` static check | Works with existing singleton |
| 20.2 | `can_allocate()`, `allocate_buffer_safe()` | Uses existing `get_memory_info()` |
| 20.3 | `copy_to_host_with_stream_sync()` | Avoids dangerous `hipDeviceSynchronize()` |
| 20.4 | `GPU_FIXTURE` in `tests/common/` | Uses existing `HipBackend::new_checked()` |
| 20.5 | Test pattern update | Minimal changes to existing tests |

**Key Insight**: Your codebase already has most of the right patterns (singleton, `get_memory_info()`, stream-based sync). We just need to add safety checks and a shared test fixture.

---

**Author**: Implementation Plan (Phase 20)
**Date**: 2026-01-11
**Status**: Ready for Implementation
