# Code Review Report: FIX-10 (KV Cache State Tracking)

**Date**: 2026-01-11
**Reviewer**: code-reviewer
**Scope**: FIX-10 - KV Cache State Tracking (MODEL-2)
**Status**: NEEDS WORK - NOT IMPLEMENTED

---

## Executive Summary

**Overall Assessment**: NEEDS WORK

FIX-10 has **NOT been implemented**. While the codebase has a functional `remove_sequence()` method in `KvCache`, the critical missing piece is **automatic cleanup when requests complete**. The current implementation only cleans up KV cache entries when `cancel_request()` is explicitly called, but **NOT when requests complete normally**.

**Grade**: D (4/10)
- **Sequence tracking**: PARTIAL (sequences tracked but no lifetime metadata)
- **Auto-cleanup**: MISSING (only manual cleanup on cancel)
- **LRU eviction**: NOT IMPLEMENTED
- **Test coverage**: GOOD (existing tests cover manual removal)

---

## What I Found

### 1. Current Implementation State

#### What EXISTS:
1. **KvCache::remove_sequence()** - Manual cleanup method works correctly
   - Location: `/home/feanor/Projects/ROCmForge/src/kv_cache/kv_cache.rs:453-469`
   - Properly removes sequences from the cache
   - Frees pages back to the free list

2. **Scheduler completion tracking** - Requests are tracked through completion
   - Location: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:503-598`
   - Completed requests moved to `completed_requests` HashMap
   - State transitions work correctly (Pending → Processing → Completed)

3. **Manual cleanup on cancel** - Works when explicitly called
   - Location: `/home/feanor/Projects/ROCmForge/src/engine.rs:297-320`
   - `cancel_request()` properly calls `kv_cache.remove_sequence()`

#### What is MISSING:
1. **Automatic cleanup on completion** - CRITICAL GAP
   - When requests complete normally, KV cache is NOT cleaned up
   - Completed requests accumulate in `scheduler.completed_requests` forever
   - No trigger to call `remove_sequence()` on normal completion

2. **Sequence lifetime tracking** - NOT IMPLEMENTED
   - No `last_access` timestamp
   - No metadata tracking when sequence was added
   - No way to identify "stale" sequences

3. **LRU eviction** - NOT IMPLEMENTED
   - No eviction policy when cache is full
   - No automatic cleanup of old completed requests

---

## Detailed Findings

### Critical Issue #1: No Automatic Cleanup on Completion

**Severity**: CRITICAL
**Impact**: Memory leak - completed requests never removed from KV cache

**Evidence from code**:

In `/home/feanor/Projects/ROCmForge/src/engine.rs`:
- Lines 297-320: `cancel_request()` calls `kv_cache.remove_sequence()` ✓
- Lines 445-506: `process_batch()` handles completion but does NOT call cleanup ✗

In `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs`:
- Lines 555-598: `update_iteration_batch()` returns completed requests
- Completed requests are moved to `completed_requests` HashMap
- No callback or trigger to cleanup KV cache

**Problem Flow**:
```
1. Request completes → scheduler.move_to_completed()
2. InferenceEngine notified of completion
3. Request stored in scheduler.completed_requests
4. KV cache entry REMAINS (memory leak!)
```

**Expected Flow**:
```
1. Request completes → scheduler.move_to_completed()
2. InferenceEngine notified
3. Engine calls kv_cache.remove_sequence(request_id)
4. KV cache memory freed
```

---

### Critical Issue #2: completed_requests HashMap Grows Unbounded

**Severity**: CRITICAL
**Impact**: Unbounded memory growth for completed request metadata

**Evidence**:

Location: `/home/feanor/Projects/ROCmForge/src/scheduler/scheduler.rs:305`

```rust
pub struct Scheduler {
    config: SchedulerConfig,
    pending_queue: VecDeque<GenerationRequest>,
    processing_requests: HashMap<u32, GenerationRequest>,
    completed_requests: HashMap<u32, GenerationRequest>,  // ← NEVER CLEANED
    next_batch_id: u32,
    next_request_id: u32,
}
```

**Problem**:
- `completed_requests` HashMap is never cleared
- Every completed request is stored indefinitely
- No limit on size
- No TTL or expiry mechanism

**Memory Impact**:
- Each `GenerationRequest` stores `prompt_tokens` + `generated_tokens`
- For long-running inference sessions, this can be GBs of data
- No way to query or purge old completed requests

---

### High Priority Issue #3: No Sequence Lifetime Tracking

**Severity**: HIGH
**Impact**: Cannot implement LRU or time-based eviction

**Evidence**:

Location: `/home/feanor/Projects/ROCmForge/src/kv_cache/kv_cache.rs:280-293`

```rust
#[derive(Debug)]
pub struct SequenceCache {
    pub sequence_id: u32,
    pub pages: Vec<u32>,
    pub total_tokens: usize,
}
```

**Missing**:
- No `created_at: Instant` timestamp
- No `last_access: Instant` timestamp
- No `access_count: usize` counter
- No way to identify "stale" or "unused" sequences

**Why This Matters**:
- Cannot implement LRU eviction (need last_access time)
- Cannot implement TTL-based cleanup (need created_at time)
- Cannot identify "cold" vs "hot" sequences

---

### Medium Priority Issue #4: No LRU Eviction

**Severity**: MEDIUM
**Impact**: Cache will fill up and reject new requests

**Evidence**:

Location: `/home/feanor/Projects/ROCmForge/src/kv_cache/kv_cache.rs:347-370`

```rust
pub fn allocate_page(&mut self, sequence_id: u32) -> KvCacheResult<u32> {
    let page_id = if let Some(free_id) = self.free_pages.write().unwrap().pop() {
        free_id
    } else if self.pages.read().unwrap().len() >= self.config.max_pages {
        return Err(KvCacheError::CapacityExceeded);  // ← FAILS IMMEDIATELY
    } else {
        // allocate new page
    };
}
```

**Problem**:
- When cache is full, immediately returns error
- No eviction of old/stale sequences
- No fallback to free memory from completed requests

**Expected Behavior** (from vLLM):
1. Try to allocate from free pool
2. If full, evict least-recently-used sequence
3. Reclaim memory
4. Retry allocation

---

## Test Coverage Analysis

### Existing Tests (GOOD)

Location: `/home/feanor/Projects/ROCmForge/src/kv_cache/kv_cache.rs:714-732`

```rust
#[test]
fn test_sequence_removal() {
    let backend = HipBackend::new().unwrap();
    let config = CacheConfig::new(4, 10, 32, 128, 24).unwrap();
    let mut cache = KvCache::new(config, backend).unwrap();

    cache.allocate_page(1).unwrap();
    cache.append_token(1, 42).unwrap();

    let stats_before = cache.get_cache_stats();
    assert_eq!(stats_before.active_sequences, 1);
    assert_eq!(stats_before.free_pages, 0);

    cache.remove_sequence(1).unwrap();

    let stats_after = cache.get_cache_stats();
    assert_eq!(stats_after.active_sequences, 0);
    assert_eq!(stats_after.free_pages, 1);
}
```

**Assessment**: Test covers the manual `remove_sequence()` call. GOOD.

### Missing Tests (CRITICAL GAP)

No tests for:
1. Automatic cleanup on request completion
2. LRU eviction behavior
3. Sequence lifetime tracking
4. Cache full → eviction → retry flow

---

## FIX-10 Requirements vs. Implementation

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **1. Sequence lifetime tracking** | ❌ NOT IMPLEMENTED | No timestamps in `SequenceCache` |
| **2. Auto-cleanup completed sequences** | ❌ NOT IMPLEMENTED | Only cleanup on `cancel_request()`, not on normal completion |
| **3. LRU eviction** | ❌ NOT IMPLEMENTED | `allocate_page()` fails immediately when full |
| **4. Test coverage** | ⚠️ PARTIAL | Manual removal tested, auto-cleanup NOT tested |

---

## Code Locations Analyzed

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `src/kv_cache/kv_cache.rs` | 280-302 | `SequenceCache` struct | ❌ Missing lifetime fields |
| `src/kv_cache/kv_cache.rs` | 453-469 | `remove_sequence()` | ✓ Works correctly |
| `src/scheduler/scheduler.rs` | 305 | `completed_requests` HashMap | ❌ Never cleaned |
| `src/scheduler/scheduler.rs` | 503-598 | `update_iteration_batch()` | ⚠️ Returns completed but no cleanup trigger |
| `src/engine.rs` | 297-320 | `cancel_request()` | ✓ Calls cleanup |
| `src/engine.rs` | 445-506 | `process_batch()` | ❌ Missing cleanup call |

---

## Recommendations

### Immediate Actions (P0 - CRITICAL)

1. **Add automatic cleanup in `process_batch()`**

   Location: `/home/feanor/Projects/ROCmForge/src/engine.rs:501-503`

   **Current code**:
   ```rust
   // Notify completed requests
   for completed_req in _completed {
       self.notify_request(completed_req.request_id).await;
   }

   Ok(())
   ```

   **Recommended fix**:
   ```rust
   // Notify completed requests
   for completed_req in _completed {
       self.notify_request(completed_req.request_id).await;

       // CRITICAL: Cleanup KV cache for completed requests
       let mut kv_cache = self.kv_cache.write().await;
       if let Err(err) = kv_cache.remove_sequence(completed_req.request_id) {
           if !matches!(err, crate::kv_cache::KvCacheError::InvalidSequenceId(_)) {
               warn!(
                   "Failed to remove completed request {} from KV cache: {}",
                   completed_req.request_id, err
               );
           }
       }
   }

   Ok(())
   ```

2. **Add sequence lifetime tracking**

   Location: `/home/feanor/Projects/ROCmForge/src/kv_cache/kv_cache.rs:280-293`

   **Current code**:
   ```rust
   #[derive(Debug)]
   pub struct SequenceCache {
       pub sequence_id: u32,
       pub pages: Vec<u32>,
       pub total_tokens: usize,
   }
   ```

   **Recommended fix**:
   ```rust
   use std::time::Instant;

   #[derive(Debug)]
   pub struct SequenceCache {
       pub sequence_id: u32,
       pub pages: Vec<u32>,
       pub total_tokens: usize,
       pub created_at: Instant,
       pub last_access: Instant,
       pub access_count: usize,
   }

   impl SequenceCache {
       pub fn new(sequence_id: u32) -> Self {
           let now = Instant::now();
           SequenceCache {
               sequence_id,
               pages: Vec::new(),
               total_tokens: 0,
               created_at: now,
               last_access: now,
               access_count: 0,
           }
       }

       pub fn record_access(&mut self) {
           self.last_access = Instant::now();
           self.access_count += 1;
       }
   }
   ```

3. **Add cleanup for completed_requests HashMap**

   Add method to purge old completed requests:
   ```rust
   impl Scheduler {
       /// Remove completed requests older than specified duration
       pub fn purge_completed_requests(&mut self, older_than: Duration) -> usize {
           let now = Instant::now();
           let to_remove: Vec<u32> = self.completed_requests
               .iter()
               .filter(|(_, req)| {
                   req.completed_at
                       .map(|at| now.duration_since(at) > older_than)
                       .unwrap_or(false)
               })
               .map(|(id, _)| *id)
               .collect();

           for id in to_remove {
               self.completed_requests.remove(&id);
           }

           to_remove.len()
       }
   }
   ```

### Secondary Actions (P1 - HIGH)

4. **Implement LRU eviction**

   Add to `KvCache`:
   ```rust
   impl KvCache {
       /// Find least-recently-used sequence for eviction
       pub fn find_lru_sequence(&self) -> Option<u32> {
           let sequences = self.sequences.read().ok()?;
           sequences
               .values()
               .min_by_key(|s| s.last_access)
               .map(|s| s.sequence_id)
       }

       /// Evict LRU sequence when cache is full
       pub fn evict_lru_sequence(&mut self) -> KvCacheResult<u32> {
           let lru_id = self.find_lru_sequence()
               .ok_or(KvCacheError::InvalidConfiguration)?;
           self.remove_sequence(lru_id)?;
           Ok(lru_id)
       }
   ```

5. **Add integration test for auto-cleanup**

   ```rust
   #[tokio::test]
   async fn test_kv_cache_auto_cleanup_on_completion() {
       let config = EngineConfig::default();
       let engine = InferenceEngine::new(config).unwrap();

       // Load model (required for inference)
       // ... setup code ...

       let request_id = engine
           .submit_request(vec![1, 2, 3], 2, 0.8, 50, 0.9)
           .await
           .unwrap();

       // Wait for completion
       // ... wait for request to complete ...

       // Verify KV cache was cleaned up
       let stats = engine.get_engine_stats().await;
       assert_eq!(stats.cache_stats.active_sequences, 0,
                   "KV cache should be cleaned up after request completion");
   }
   ```

---

## Test Results

**Status**: NOT TESTED - Compilation errors prevent test execution

**Error**: `hipDeviceSynchronize` is private FFI function
- Location: `src/attention/gpu.rs:159, 217, 263`
- Blocking: Cannot compile and run tests

**Workaround**: Tests would need FFI to be public or use different sync mechanism

**Note**: This compilation error is separate from FIX-10 implementation and should be fixed separately.

---

## Positive Findings

1. **Thread safety**: `KvCache` properly uses `RwLock` for all mutable state
2. **Manual cleanup works**: `remove_sequence()` is correctly implemented
3. **Good test structure**: Existing tests are well-organized
4. **Clean architecture**: Separation of concerns (scheduler, cache, engine)

---

## Severity Breakdown

| Issue | Severity | Impact | Effort to Fix |
|-------|----------|--------|---------------|
| No auto-cleanup on completion | CRITICAL | Memory leak (unbounded) | 1 hour |
| completed_requests unbounded | CRITICAL | Metadata leak (unbounded) | 2 hours |
| No lifetime tracking | HIGH | Cannot implement LRU | 3 hours |
| No LRU eviction | MEDIUM | Cache fills up, rejects requests | 4 hours |

**Total Estimated Effort**: 10 hours (1-2 days)

---

## Conclusion

**FIX-10 Status**: NOT IMPLEMENTED

The codebase has the building blocks for KV cache state tracking, but **automatic cleanup is missing**. The critical gap is that:

1. Requests complete and are moved to `scheduler.completed_requests`
2. KV cache entries are **NOT** cleaned up on normal completion
3. Memory grows unbounded

**Recommendation**: Implement the immediate actions (P0) above before marking FIX-10 as complete.

**Priority**: CRITICAL - This is a memory leak that will cause production issues.

---

## References

- Remediation Plan: `/home/feanor/Projects/ROCmForge/docs/REMEDIATION_PLAN_2026-01-11.md:621-643`
- Comprehensive Code Review: `/home/feanor/Projects/ROCmForge/docs/COMPREHENSIVE_CODE_REVIEW_SUMMARY_2026-01-10.md:114`
- vLLM KV Cache Management: https://github.com/vllm-project/vllm/blob/main/vllm/attention/selector.py
