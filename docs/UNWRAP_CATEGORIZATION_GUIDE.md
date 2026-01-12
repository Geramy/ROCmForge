# unwrap() and expect() Categorization Guide

**Purpose**: Guide for categorizing and handling unwrap() and expect() calls in ROCmForge codebase

**Last Updated**: 2026-01-11
**Phase**: 13 Complete - P0 fixes applied

---

## Quick Reference: Decision Tree

```
Is this unwrap() in production code (not tests)?
├─ No: ✅ ACCEPTABLE (test assertion)
│
├─ Yes: Is it guarded by explicit check (if let Some(...))?
│  ├─ Yes: ✅ ACCEPTABLE (guarded unwrap)
│  └─ No: Is it on external data (user input, FFI, GPU, network)?
│     ├─ Yes: 🔴 MUST FIX (P0 - critical)
│     └─ No: Is it on validated/controlled data?
│        ├─ Yes: 🟡 CONSIDER (P1 - improve error messages)
│        └─ No: 🔴 MUST FIX (P0 - unsafe)
```

---

## Categories

### P0 - Critical (Must Fix)

**Definition**: Can crash production inference server

**Examples**:

1. **Lock poisoning vulnerability**
   ```rust
   // ❌ BAD
   let cache = GLOBAL_CACHE.lock().unwrap();

   // ✅ GOOD
   let cache = GLOBAL_CACHE.lock()
       .map_err(|e| MyError::LockPoisoned(e.to_string()))?;
   ```

2. **FFI result unwrap**
   ```rust
   // ❌ BAD
   let ptr = unsafe { hipMalloc(size) }.unwrap();

   // ✅ GOOD
   let ptr = unsafe { hipMalloc(size) }
       .ok_or(HipError::AllocationFailed)?;
   ```

3. **Floating-point NaN panic**
   ```rust
   // ❌ BAD
   values.sort_by(|a, b| a.partial_cmp(b).unwrap());

   // ✅ GOOD
   values.sort_by(|a, b| a.total_cmp(b));
   ```

4. **User input unwrap**
   ```rust
   // ❌ BAD
   let id = request.get("id").unwrap();

   // ✅ GOOD
   let id = request.get("id")
       .ok_or(RequestError::MissingField("id"))?;
   ```

**Fix Timeline**: Immediately (blocks deployment)

---

### P1 - High (Should Fix)

**Definition**: Affects initialization or user-facing errors

**Examples**:

1. **Config file parsing**
   ```rust
   // ⚠️ IMPROVE
   let max_tokens = config.get("max_tokens").unwrap();

   // ✅ BETTER
   let max_tokens = config.get("max_tokens")
       .unwrap_or(DEFAULT_MAX_TOKENS);

   // ✅ BEST
   let max_tokens = config.get("max_tokens")
       .ok_or(ConfigError::MissingField("max_tokens"))?;
   ```

2. **Model loading**
   ```rust
   // ⚠️ IMPROVE
   let vocab_size = metadata.get("vocab_size").unwrap();

   // ✅ BETTER
   let vocab_size = metadata.get("vocab_size")
       .ok_or(ModelError::InvalidMetadata("vocab_size"))?;
   ```

**Fix Timeline**: Next sprint

---

### P2 - Medium (Nice to Have)

**Definition**: Edge cases, internal operations

**Examples**:

1. **Internal state access**
   ```rust
   // Acceptable if documented
   let current = self.current_state
       .expect("State initialized in constructor");
   ```

2. **Known invariants**
   ```rust
   // Acceptable if invariant is documented
   let head_dim = self.config.head_dim
       .expect("head_dim set in all supported architectures");
   ```

**Fix Timeline**: When convenient

---

### Acceptable (No Fix Needed)

**Definition**: Safe by design, test code, or guarded access

**Examples**:

1. **Test assertions**
   ```rust
   #[test]
   fn test_something() {
       let result = compute().unwrap();  // ✅ OK - test assertion
       assert_eq!(result, expected);
   }
   ```

2. **Guarded unwrap**
   ```rust
   if let Some(value) = optional {
       let value = value.unwrap();  // ✅ OK - guaranteed Some
       // ... use value
   }
   ```

3. **expect() with clear message**
   ```rust
   let lock = self.lock.read()
       .expect("Lock initialized in constructor and never dropped");  // ✅ OK
   ```

4. **Borrow checker workarounds**
   ```rust
   if self.cache.is_some() {
       let value = self.cache.as_ref().unwrap();  // ✅ OK - checked above
       // ... use value
   }
   ```

---

## Fix Patterns

### Pattern 1: Lock Operations

**Scenario**: Accessing Mutex/RwLock

```rust
// BEFORE
let data = mutex.lock().unwrap();

// AFTER (Result-returning function)
let data = mutex.lock()
    .map_err(|e| MyError::LockPoisoned(e.to_string()))?;

// AFTER (void function)
let data = match mutex.lock() {
    Ok(guard) => guard,
    Err(_) => return Err(MyError::LockPoisoned),
};
```

---

### Pattern 2: Option to Result

**Scenario**: Converting None to error

```rust
// BEFORE
let value = optional.unwrap();

// AFTER
let value = optional
    .ok_or_else(|| MyError::NotFound("value".to_string()))?;

// AFTER (with context)
let value = optional
    .ok_or_else(|| MyError::NotFound(format!("{} not found in {}", name, location)))?;
```

---

### Pattern 3: Floating-Point Comparison

**Scenario**: Sorting/comparing f32/f64

```rust
// BEFORE
items.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

// AFTER
items.sort_by(|a, b| a.value.total_cmp(&b.value));

// AFTER (custom ordering)
items.sort_by(|a, b| {
    a.value.partial_cmp(&b.value)
        .unwrap_or(Ordering::Equal)  // or Ordering::Less/Greater
});
```

---

### Pattern 4: FFI Error Handling

**Scenario**: Checking FFI return values

```rust
// BEFORE
let result = unsafe { hipFunctionLaunch() };
result.unwrap();

// AFTER
let result = unsafe { hipFunctionLaunch() }
    .ok_or(HipError::KernelLaunchFailed)?;

// AFTER (with error code)
let result = unsafe { hipFunctionLaunch() };
if result != hipError_t::hipSuccess {
    return Err(HipError::KernelLaunchFailed);
}
```

---

### Pattern 5: Default Values

**Scenario**: Providing defaults for missing config

```rust
// BEFORE
let timeout = config.get("timeout").unwrap();

// AFTER (with default)
let timeout = config.get("timeout")
    .unwrap_or(DEFAULT_TIMEOUT);

// AFTER (with validation)
let timeout = config.get("timeout")
    .unwrap_or_else(|| validate_timeout(DEFAULT_TIMEOUT)?);
```

---

## expect() vs unwrap()

### When to use expect()

**Use expect() when**:
- The invariant is guaranteed by design
- You can provide a clear, helpful error message
- The failure indicates a bug in the code, not external data

**Good expect() usage**:
```rust
let value = self.config.value
    .expect("Config validated in constructor, bug if missing");

let iterator = self.data.iter()
    .expect("Data initialized in constructor");
```

### When to avoid expect()

**Avoid expect() when**:
- Processing external data (user input, files, network)
- The operation can legitimately fail (resources, permissions)
- Better error handling is possible

**Bad expect() usage**:
```rust
// ❌ BAD - external data
let vocab_size = metadata.get("vocab_size")
    .expect("GGUF file must have vocab_size");  // Should use Result

// ❌ BAD - resource allocation
let buffer = HipBuffer::alloc(size)
    .expect("GPU allocation should succeed");  // Should return Result
```

---

## Code Review Checklist

When reviewing code with unwrap() or expect():

- [ ] Is this in production code (not tests)?
- [ ] Is it guarded by an explicit check?
- [ ] Is it on external/untrusted data?
- [ ] Is it a lock operation (can poison)?
- [ ] Is it a floating-point comparison?
- [ ] Is the error message descriptive (if using expect)?
- [ ] Is there a test for the error path?

---

## Testing Strategy

### Unit Tests for Error Paths

```rust
#[test]
fn test_lock_poisoning_recovery() {
    // Arrange: Create scenario where lock might be poisoned
    let cache = Arc::new(Mutex::new(KernelCache::new()));

    // Act: Simulate lock poisoning
    let _ = catch_unwind(|| {
        let _lock = cache.lock().unwrap();
        panic!();  // Poison the lock
    });

    // Assert: Next access should handle poisoning gracefully
    let result = cache.get_kernel("test");
    assert!(result.is_err());
}

#[test]
fn test_nan_handling_in_sampling() {
    // Arrange: Create logits with NaN
    let mut logits = vec![1.0, 2.0, f32::NAN, 0.5];

    // Act: Sort using total_cmp
    logits.sort_by(|a, b| a.total_cmp(b));

    // Assert: NaN should be last (or first, depending on implementation)
    assert!(logits[3].is_nan());
}
```

---

## Common Mistakes

### Mistake 1: Using unwrap() on FFI results

```rust
// ❌ WRONG
let device = unsafe { hipDeviceGet(0) }.unwrap();

// ✅ RIGHT
let device = unsafe { hipDeviceGet(0) }
    .ok_or(HipError::DeviceNotFound)?;
```

### Mistake 2: Not documenting expect() invariants

```rust
// ❌ WRONG - why won't this be None?
let config = self.config.expect("config not set");

// ✅ RIGHT - document the invariant
let config = self.config.expect(
    "Config initialized in constructor and never changed after"
);
```

### Mistake 3: Using unwrap() in hot paths

```rust
// ❌ WRONG - every GPU inference call hits this
let kernel = self.cache.get(&name).unwrap();

// ✅ RIGHT - cache lookup should handle errors
let kernel = self.cache.get(&name)
    .ok_or_else(|| KernelError::NotFound(name))?;
```

---

## Metrics Tracking

Track these metrics to monitor progress:

| Metric | Current | Target | Trend |
|--------|---------|--------|-------|
| P0 unwrap() calls | 0 | 0 | ✅ |
| P1 unwrap() calls | 0 | <10 | ✅ |
| Production expect() | 28 | <50 | ✅ |
| Test unwrap() | 285 | N/A | - |
| Tests passing | 158/158 | 100% | ✅ |

---

## Related Documentation

- [Phase 13 Progress](./UNWRAP_HELL_PROGRESS.md) - Detailed implementation tracking
- [Phase 13 Fix Report](./UNWRAP_HELL_FIX_REPORT.md) - Technical implementation details
- [Rust Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html) - Official Rust guide

---

**Last Updated**: 2026-01-11 (Phase 13 complete)
**Next Review**: Phase 13B (optional P1/P2 cleanup)
