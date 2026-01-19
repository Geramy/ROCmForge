# Task 10-17: Replace RwLock unwrap() in prompt/cache.rs

**Phase:** 10 (Production Hardening - Gap Closure)
**Task:** 10-17
**Status:** Complete
**Date:** 2026-01-19

## Issue

17 `unwrap()` calls on RwLock operations in `src/prompt/cache.rs`:
- Lines 106, 109: `get()` method - hits/misses counters
- Lines 123, 131, 132, 139, 140: `insert()` method - cache and memory tracking
- Lines 176-179: `stats()` method - all lock operations
- Lines 201-202: `clear()` method - cache and memory tracking
- Lines 209-210: `evict_lru()` method - cache and memory tracking

## Changes Made

### 1. Modified `src/prompt/cache.rs`

#### `get()` method (lines 99-119)
- Changed lock operations on `hits` and `misses` counters from `.unwrap()` to `if let Ok(...)` pattern
- Lock poisoning is now handled gracefully - counter increments are skipped but cached entry is still returned
- Kept `Option` return type as specified in requirements

#### `insert()` method (lines 122-181)
- Changed all 6 RwLock operations from `.unwrap()` to `.map_err(|_| CacheError::InvalidEntry)?`
- `Result<(), CacheError>` return type already existed

#### `stats()` method (lines 195-229)
- Changed all 4 RwLock operations from `.unwrap()` to `.map_err(|_| CacheError::InvalidEntry)?`
- Changed return type from `CacheStats` to `Result<CacheStats, CacheError>`

#### `clear()` method (lines 232-242)
- Changed from two separate `.unwrap()` calls to `if let (Ok(...), Ok(...))` tuple pattern
- Lock poisoning results in silent failure - acceptable for cache clear operation

#### `evict_lru()` method (lines 245-272)
- Changed both RwLock operations from `.unwrap()` to `.map_err(|_| CacheError::InvalidEntry)?`
- `Result<(), CacheError>` return type already existed

#### Updated tests
- Updated test code to unwrap `Result` from `stats()` method calls
- All test code `unwrap()` calls remain (acceptable for tests)

### 2. Fixed unrelated compilation errors

#### `src/metrics.rs`
- Fixed tracing macro calls in retry methods to use proper key-value syntax
- `operation = operation` instead of bare `operation`

#### `src/engine.rs`
- Fixed jitter calculation: `nanos % jitter_range` where both are `u64`
- Fixed tracing macro call syntax

### 3. Fixed struct initialization in `src/metrics.rs`
- Added initialization for new retry metric fields that were missing

## Acceptance Criteria

- [x] All 17 RwLock unwrap() calls in production code replaced
- [x] Compilation succeeds
- [x] Tests pass (6/6 tests in prompt::cache module)
- [x] No unwrap() in production code paths (only in test code)

## Test Results

```
running 6 tests
test prompt::cache::tests::test_kv_cache_entry_memory ... ok
test prompt::cache::tests::test_prefix_cache_basic ... ok
test prompt::cache::tests::test_prefix_cache_stats ... ok
test prompt::cache::tests::test_find_longest_prefix ... ok
test prompt::cache::tests::test_cache_clear ... ok
test prompt::cache::tests::test_early_exit_detector ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 553 filtered out
```

## Files Modified

- `/home/feanor/Projects/ROCmForge/src/prompt/cache.rs` - Main task
- `/home/feanor/Projects/ROCmForge/src/metrics.rs` - Fixed compilation errors
- `/home/feanor/Projects/ROCmForge/src/engine.rs` - Fixed compilation errors
