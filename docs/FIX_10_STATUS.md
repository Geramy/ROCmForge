# FIX-10 Status: KV Cache State Tracking

**Date**: 2026-01-11
**Status**: NEEDS WORK - NOT IMPLEMENTED
**Priority**: CRITICAL
**Code Review Report**: [CODE_REVIEW_FIX_10_KV_CACHE_STATE_TRACKING_2026-01-11.md](CODE_REVIEW_FIX_10_KV_CACHE_STATE_TRACKING_2026-01-11.md)

---

## Quick Summary

FIX-10 has **NOT been implemented**. The codebase has a working `remove_sequence()` method, but it is **only called on explicit cancel**, not when requests complete normally. This causes:

1. **Memory Leak**: KV cache entries never cleaned up after request completion
2. **Metadata Leak**: `completed_requests` HashMap grows unbounded
3. **No LRU**: Cache fails immediately when full instead of evicting stale entries

---

## Critical Gap

### What Works
- `cancel_request()` → calls `kv_cache.remove_sequence()` ✓
- `remove_sequence()` method works correctly ✓
- Scheduler properly tracks completion state ✓

### What's Missing
- Request completes normally → KV cache **NOT** cleaned up ✗
- `completed_requests` HashMap → never purged ✗
- Sequence lifetime tracking → not implemented ✗
- LRU eviction → not implemented ✗

---

## Root Cause

**Location**: `/home/feanor/Projects/ROCmForge/src/engine.rs:501-503`

**Current Code**:
```rust
// Notify completed requests
for completed_req in _completed {
    self.notify_request(completed_req.request_id).await;
}
// ← MISSING: KV cache cleanup
```

**Should Be**:
```rust
// Notify completed requests
for completed_req in _completed {
    self.notify_request(completed_req.request_id).await;

    // CRITICAL: Cleanup KV cache for completed requests
    let mut kv_cache = self.kv_cache.write().await;
    let _ = kv_cache.remove_sequence(completed_req.request_id);
}
```

---

## Implementation Checklist

### Phase 1: Critical Fixes (P0) - 6 hours

- [ ] **FIX-10.1**: Add KV cache cleanup in `process_batch()` (1 hour)
  - File: `src/engine.rs`
  - Add cleanup loop after line 502
  - Test: Run engine with multiple requests, verify cache stats

- [ ] **FIX-10.2**: Add lifetime tracking to `SequenceCache` (2 hours)
  - File: `src/kv_cache/kv_cache.rs`
  - Add fields: `created_at`, `last_access`, `access_count`
  - Update `SequenceCache::new()` to initialize timestamps
  - Add `record_access()` method

- [ ] **FIX-10.3**: Add purge method for `completed_requests` (2 hours)
  - File: `src/scheduler/scheduler.rs`
  - Add `purge_completed_requests(Duration)` method
  - Call from engine periodically (every N requests)

- [ ] **FIX-10.4**: Integration test for auto-cleanup (1 hour)
  - File: `tests/engine_integration_tests.rs` (NEW)
  - Test: Complete request → verify cache cleaned up

### Phase 2: LRU Eviction (P1) - 4 hours

- [ ] **FIX-10.5**: Implement LRU sequence finder (1 hour)
  - Add `find_lru_sequence()` to `KvCache`
  - Returns sequence with oldest `last_access`

- [ ] **FIX-10.6**: Implement eviction in `allocate_page()` (2 hours)
  - When cache full: evict LRU sequence
  - Retry allocation after eviction
  - Test: Fill cache, verify eviction works

- [ ] **FIX-10.7**: Add eviction metrics (1 hour)
  - Track eviction count in `CacheStats`
  - Log eviction events

---

## Test Plan

### Unit Tests
```rust
// Test auto-cleanup on completion
#[tokio::test]
async fn test_kv_cache_auto_cleanup() {
    // Submit request with max_tokens=2
    // Wait for completion
    // Verify cache.active_sequences == 0
}

// Test lifetime tracking
#[test]
fn test_sequence_lifetime_tracking() {
    // Create sequence
    // Check created_at is set
    // Access sequence
    // Check last_access updated
}

// Test LRU eviction
#[test]
fn test_lru_eviction_when_full() {
    // Fill cache to max_pages
    // Try to allocate one more
    // Verify LRU sequence evicted
    // Verify allocation succeeds
}
```

### Integration Tests
```rust
// Test long-running server
#[tokio::test]
async fn test_no_memory_leak_long_running() {
    // Process 1000 requests sequentially
    // Monitor cache size
    // Verify cache doesn't grow unbounded
}

// Test completed_requests purge
#[tokio::test]
async fn test_completed_requests_purge() {
    // Complete 100 requests
    // Call purge_completed_requests()
    // Verify old requests removed
}
```

---

## Impact Analysis

### Before FIX-10
- Request completes → KV cache entry persists forever
- Long-running server → OOM from accumulated KV cache entries
- No recovery mechanism except restart

### After FIX-10
- Request completes → KV cache entry cleaned up
- Long-running server → stable memory usage
- LRU eviction prevents cache full failures

---

## Estimated Effort

| Task | Time | Priority |
|------|------|----------|
| Add auto-cleanup in process_batch | 1 hour | P0 |
| Add lifetime tracking | 2 hours | P0 |
| Add completed_requests purge | 2 hours | P0 |
| Write integration tests | 1 hour | P0 |
| Implement LRU eviction | 4 hours | P1 |
| **Total** | **10 hours** | - |

---

## References

- Full Code Review: [`CODE_REVIEW_FIX_10_KV_CACHE_STATE_TRACKING_2026-01-11.md`](CODE_REVIEW_FIX_10_KV_CACHE_STATE_TRACKING_2026-01-11.md)
- Remediation Plan: `REMEDIATION_PLAN_2026-01-11.md:621-643`
- Implementation Files:
  - `src/engine.rs:445-506` (process_batch)
  - `src/kv_cache/kv_cache.rs:280-302` (SequenceCache)
  - `src/scheduler/scheduler.rs:300-598` (Scheduler)

---

## Next Steps

1. Review this document and code review report
2. Decide on priority (CRITICAL - causes memory leak)
3. Assign implementation (estimated 10 hours)
4. Update CHANGELOG when complete
