# Phase 23: Fix hipDeviceSynchronize Desktop Hang

**Date**: 2026-01-12
**Status**: IN PROGRESS
**Priority**: P0 - Critical (Desktop crashes)
**Related**: Phase 20 (GPU Testing Safety), Phase 22 (E2E Tests)

---

## Executive Summary

**CRITICAL BUG**: `synchronize_device()` uses `hipDeviceSynchronize()` which waits for ALL GPU streams including the desktop compositor, causing desktop freezes/hangs.

**Root Cause**: `src/backend/hip_backend.rs:2627` - `synchronize_device()` function calls `hipDeviceSynchronize()` instead of `hipStreamSynchronize()`.

**Impact**: Any test or code that calls attention GPU operations triggers desktop hang.

**Fix**: Modify `synchronize_device()` to use the global backend's stream-aware synchronization.

---

## Problem Analysis

### What's Happening

```
┌─────────────────────────────────────────────────────────────┐
│                  GPU Device (Single GPU)                     │
├─────────────────────────────────────────────────────────────┤
│  Stream 1: ROCmForge test kernels                           │
│  Stream 2: Desktop compositor (Wayland/X11)                 │
│  Stream 3: Browser/video decode, etc.                       │
└─────────────────────────────────────────────────────────────┘

hipDeviceSynchronize() called from ROCmForge test
  ↓
Waits for ALL streams (1, 2, 3, ...)
  ↓
Desktop compositor blocks waiting for test to finish
  ↓
Test blocks waiting for desktop to release GPU
  ↓
DEADLOCK → Desktop freezes/hangs
```

### Code Path

```
src/attention/gpu.rs:158 → synchronize_device()
  → hipDeviceSynchronize() (DANGEROUS - waits for ALL streams)

src/attention/gpu.rs:213 → synchronize_device()
  → hipDeviceSynchronize() (DANGEROUS)

src/attention/gpu.rs:256 → synchronize_device()
  → hipDeviceSynchronize() (DANGEROUS)
```

### Current Implementation (WRONG)

**File**: `src/backend/hip_backend.rs:2627`

```rust
/// Synchronize device globally
pub fn synchronize_device() -> HipResult<()> {
    let result = unsafe { hipDeviceSynchronize() };  // ❌ DANGEROUS!
    // ...
}
```

### Why This Wasn't Caught

- Documentation claimed Phase 20 fixed `hipDeviceSynchronize()`
- Fix was only applied to `HipBuffer::copy_to_host()` (deprecated)
- `synchronize_device()` function was left unchanged
- No test validates that stream-aware sync is used

---

## Solution Design

### Correct Implementation

```rust
/// Synchronize device using stream-aware synchronization
///
/// This is a SAFE alternative to hipDeviceSynchronize() that only
/// waits for our application's stream, not the desktop compositor.
pub fn synchronize_device() -> HipResult<()> {
    // Get the global backend (singleton)
    let backend = GLOBAL_BACKEND.lock()
        .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e)))?
        .as_ref()
        .map(Arc::clone)
        .ok_or(HipError::DeviceError("HIP backend not initialized".to_string()))?;

    // Use stream-aware synchronization (SAFE - only waits for our stream)
    backend.stream.synchronize()
}
```

### Why This Works

```
┌─────────────────────────────────────────────────────────────┐
│                  GPU Device (Single GPU)                     │
├─────────────────────────────────────────────────────────────┤
│  Stream 1: ROCmForge test kernels  ← We synchronize THIS ONE
│  Stream 2: Desktop compositor (Wayland/X11)                 │
│  Stream 3: Browser/video decode, etc.                       │
└─────────────────────────────────────────────────────────────┘

hipStreamSynchronize(backend.stream) called from ROCmForge test
  ↓
Waits ONLY for Stream 1 (our kernels)
  ↓
Desktop compositor continues normally on Stream 2
  ↓
No deadlock → Desktop remains stable
```

---

## Implementation Plan (TDD)

### Step 1: Write Failing Test (TDD Rule #3)

Create test that verifies `synchronize_device()` doesn't call `hipDeviceSynchronize()`:

```rust
#[test]
#[serial]
fn test_synchronize_device_is_stream_aware() {
    // This test validates that synchronize_device uses stream-aware sync
    // and doesn't cause desktop hangs
    let fixture = GPU_FIXTURE.as_ref()
        .expect("GPU not available");

    // Call synchronize_device - should NOT hang desktop
    crate::backend::hip_backend::synchronize_device()
        .expect("Synchronization should succeed");

    fixture.assert_no_leak(5);
}
```

### Step 2: Fix Implementation

Update `synchronize_device()` in `src/backend/hip_backend.rs`:

```rust
pub fn synchronize_device() -> HipResult<()> {
    let backend = GLOBAL_BACKEND.lock()
        .map_err(|e| HipError::LockPoisoned(format!("GLOBAL_BACKEND lock poisoned: {}", e)))?
        .as_ref()
        .map(Arc::clone)
        .ok_or(HipError::DeviceError("HIP backend not initialized".to_string()))?;

    backend.stream.synchronize()
}
```

### Step 3: Verify Test Passes

Run the test to confirm the fix works.

### Step 4: Scan for Remaining Issues

Search entire codebase for any remaining `hipDeviceSynchronize` usage.

### Step 5: Update Documentation

Update Phase 20 completion status and document the fix.

---

## Files Modified

1. `src/backend/hip_backend.rs` - Fix `synchronize_device()` function
2. `tests/hip_backend_sync_tests.rs` (NEW) - TDD test for stream-aware sync
3. `docs/PHASE_23_HIP_DEVICE_SYNC_FIX.md` - This document
4. `docs/TODO.md` - Update Phase 23 status
5. `docs/GPU_TESTING_SAFETY_GUIDE.md` - Mark `hipDeviceSynchronize` fix as complete

---

## Success Criteria

- [ ] Test `test_synchronize_device_is_stream_aware` passes
- [ ] Zero `hipDeviceSynchronize` calls remain in codebase
- [ ] All attention GPU tests pass without desktop hangs
- [ ] Documentation updated

---

## Related Issues

- Phase 20: GPU Testing Safety Infrastructure (claimed complete, but this was missed)
- Phase 22: E2E Integration Tests (may have been failing due to this bug)

---

## References

- `src/backend/hip_backend.rs:2627` - `synchronize_device()` function
- `src/backend/hip_backend.rs:221` - `HipStream::synchronize()` (SAFE version)
- `src/attention/gpu.rs:158, 213, 256` - Callers of `synchronize_device()`
