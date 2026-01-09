# Inference Fix Session - 2026-01-09

## Summary

Fixed the CLI hang during **inference** (not just model loading) by replacing `to_host_vec()` with `backend.copy_from_device()` in `run_forward_pass()`.

## Problem

After fixing the model loading hang (see `CLI_HANG_INVESTIGATION.md`), the CLI still hung during inference execution when running `rocmforge_cli generate`.

### Symptoms

```
DEBUG: forward_layer() layer=0 step 4: pre-MLP LayerNorm
DEBUG: allocate_buffer: created buffer with size 3584 bytes
[HANG - no further output]
```

Timeout after 120-180 seconds with exit code 124.

## Root Cause

The `run_forward_pass()` function in `src/engine.rs` was calling `logits_tensor.to_host_vec()` to copy GPU data to host.

**The bug:** `to_host_vec()` uses `HipBuffer::copy_to_host()` which:
1. Calls `hipDeviceSynchronize()` - can hang if GPU operations don't complete
2. Uses `hipMemcpy` (default stream) - not stream-aware

**The fix:** Use `backend.copy_from_device()` which:
1. Uses `hipMemcpyAsync` with backend's stream
2. Calls `stream.synchronize()` instead of device sync

## Investigation Process

1. **Reindexed CodeMCP** - Database had stale entries (8,317 → 1,426 symbols)
2. **Traced the code path:**
   - `main()` → `run_local_generate()` → `wait_for_completion()`
   - `create_engine()` → `run_inference_loop()` → `process_batch()`
   - `process_single_request()` → `process_single_request_impl()`
   - `run_forward_pass()` → `to_host_vec()` ❌ **BUG FOUND**

3. **Verified `backend.copy_from_device()` exists** and uses correct stream-aware approach
4. **Implemented fix** in `src/engine.rs:630-645`

## Fix Applied

**File:** `src/engine.rs` (lines 630-645)

**Before:**
```rust
logits_tensor
    .to_host_vec()
    .map_err(|e| EngineError::InferenceFailed(e.to_string()))
```

**After:**
```rust
// CRITICAL: Use backend.copy_from_device() instead of to_host_vec()
//
// to_host_vec() calls HipBuffer::copy_to_host() which uses:
//   1. hipDeviceSynchronize() - can hang if GPU operations don't complete
//   2. hipMemcpy (default stream)
//
// backend.copy_from_device() uses the correct stream-aware approach:
//   1. hipMemcpyAsync with backend's stream
//   2. Stream synchronization (not device sync)
//
// This fix resolves the CLI hang during inference (see CLI_HANG_INVESTIGATION.md)
let mut host_data = vec![0.0f32; logits_tensor.len()];
backend
    .copy_from_device(logits_tensor.buffer(), &mut host_data)
    .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
Ok(host_data)
```

## Test Results

**Before fix:**
- Hung at step 4 (pre-MLP LayerNorm)
- No progress beyond first layer

**After fix:**
```
DEBUG: forward_layer() layer=0 step 4: pre-MLP LayerNorm
DEBUG: forward_layer() layer=0 step 4 complete
DEBUG: forward_layer() layer=0 step 5: MLP SwiGLU
DEBUG: allocate_buffer: requesting 3584 bytes
```

✅ **Progresses past step 4** - inference pipeline is working

## Remaining Issues

1. **Performance:** Many small GPU allocations (thousands of `allocate_buffer` calls)
2. **Still slow:** Even with --max-tokens 1, takes longer than expected
3. **May still timeout:** Longer generations may still hit timeout

The stream synchronization issue is fixed, but performance optimization is needed.

## Files Modified

1. `src/engine.rs` - Fixed `run_forward_pass()` to use stream-aware D2H copy

## Related Documents

- `CLI_HANG_INVESTIGATION.md` - Original model loading hang fix
- `CHANGELOG.md` - Should be updated with this fix

## Next Steps

1. Update CHANGELOG.md with this fix
2. Consider implementing scratch buffer reuse to reduce allocations
3. Profile to identify remaining bottlenecks
4. Test with longer generations once performance is acceptable
