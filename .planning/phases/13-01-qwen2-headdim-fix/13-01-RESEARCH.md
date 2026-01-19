# Phase 13-01: Qwen2 head_dim Fix - Research

**Researched:** 2026-01-19
**Domain:** GGUF metadata parsing, Rust error handling
**Confidence:** HIGH

## Summary

This phase fixes a critical bug where Qwen2 models fail to load with `buffer_size=0` errors. The root cause is `head_dim = 0` when the `qwen2.rope.dimension_count` key is missing from GGUF files. The fix follows the established llama.cpp pattern: calculate `head_dim = hidden_size / num_heads` **before** GGUF parsing, then optionally override from GGUF if the key exists.

**Primary recommendation:** Add a `calculate_default_head_dim()` method to `GgufMetadata`, call it after parsing `num_heads` and `hidden_size`, and modify the `.unwrap_or(0)` pattern to silently ignore parse failures while preserving the calculated value.

## Standard Stack

No external libraries required. This is a logic fix using existing Rust patterns.

### Core Dependencies
| Component | Version | Purpose | Notes |
|-----------|---------|---------|-------|
| `GgufMetadata` | existing | Metadata storage | Located in `src/loader/metadata.rs` |
| `GgufLoader` | existing | GGUF file parser | Located in `src/loader/gguf.rs` |

### Affected Files
| File | Lines | Change Type |
|------|-------|-------------|
| `src/loader/metadata.rs` | 44-191 | Add method, modify `update_from_kv` |
| `src/loader/gguf.rs` | 1530-1611 | Modify `update_metadata` |
| `src/loader/metadata.rs` | 193-361 | Add unit tests |

## Architecture Patterns

### Recommended Fix Structure

```
┌─────────────────────────────────────────────────────────────┐
│ GgufLoader::new() or metadata_from_file()                    │
│                                                              │
│  1. Create loader with GgufMetadata::default()              │
│     (head_dim = 0, num_heads = 0, hidden_size = 0)          │
│                                                              │
│  2. load_from_disk() -> parse_kv_pairs()                    │
│     - Parse num_heads, hidden_size from GGUF                │
│     - Parse rope.dimension_count if present                  │
│                                                              │
│  3. NEW: calculate_default_head_dim()                        │
│     - If head_dim == 0 && num_heads > 0 && hidden_size > 0  │
│     - head_dim = hidden_size / num_heads                     │
│                                                              │
│  4. Use metadata for model loading                           │
└─────────────────────────────────────────────────────────────┘
```

### Pattern 1: Default Calculation with Optional Override

**What:** Calculate a sensible default before parsing optional metadata.

**When to use:** When a metadata value can be derived from other required values but may be explicitly specified.

**Example (from llama.cpp):**
```cpp
// File: llama.cpp/src/llama-model.cpp:598-607
// First, calculate default from known values
hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();

// Then, try to override from GGUF (optional, doesn't change if key missing)
ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

// Use n_embd_head_k as n_rot
hparams.n_rot = hparams.n_embd_head_k;

// Optional override for rope dimension count
ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);
```

**Rust equivalent:**
```rust
impl GgufMetadata {
    /// Calculate head_dim from hidden_size and num_heads.
    /// Call this after parsing num_heads and hidden_size but before using head_dim.
    pub fn calculate_default_head_dim(&mut self) {
        if self.num_heads > 0 && self.hidden_size > 0 && self.head_dim == 0 {
            self.head_dim = self.hidden_size / self.num_heads;
        }
    }
}
```

### Pattern 2: Safe Metadata Key Parsing

**What:** Replace `.unwrap_or(0)` with safe parse that ignores invalid values.

**When to use:** When parsing optional numeric metadata where missing/invalid should use a calculated default.

**Example:**
```rust
// WRONG (current buggy code):
"qwen2.rope.dimension_count" => self.head_dim = value.parse().unwrap_or(0),

// CORRECT (llama.cpp pattern):
"qwen2.rope.dimension_count" => {
    if let Ok(dim) = value.parse::<usize>() {
        if dim > 0 {
            self.head_dim = dim;
        }
    }
    // If parse fails or key doesn't exist, keep calculated value
}
```

### Anti-Patterns to Avoid

- **Panic on parse failure:** `unwrap_or_else(|_| panic!("..."))` - Users shouldn't see panics for metadata issues
- **Returning Option<usize> for head_dim:** Would require extensive refactoring; `head_dim` is used as `usize` throughout
- **Silent acceptance of zero:** If `num_heads` or `hidden_size` is 0 after parsing, fail explicitly with clear error

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Optional metadata parsing | Custom `unwrap_or` logic | `if let Ok(...) = ...` pattern | Rust's built-in pattern handling is idiomatic and clear |
| Metadata validation | Ad-hoc checks | Explicit `validate()` method | Centralized validation is easier to test and maintain |

**Key insight:** The bug exists because we used `.unwrap_or(0)` which silently sets `head_dim = 0` when the key is missing. The fix is to calculate a sensible default **before** parsing, then only override if the GGUF value is valid.

## Common Pitfalls

### Pitfall 1: Calculation Before Required Values Are Parsed

**What goes wrong:** Calling `calculate_default_head_dim()` before `num_heads` and `hidden_size` are parsed results in division by zero or incorrect calculation.

**Why it happens:** The metadata parsing happens incrementally as KV pairs are read. If we calculate too early, the values are still 0.

**How to avoid:** Call `calculate_default_head_dim()` AFTER `parse_kv_pairs()` completes, not during.

**Warning signs:** `head_dim` remains 0 after calculation, or panic from division by zero.

**Location to add call:**
```rust
// In src/loader/gguf.rs, fn load_from_disk(), after line 1285
fn load_from_disk(&mut self, load_tensors: bool) -> Result<()> {
    // ... parse magic, version, counts ...

    // Parse KV pairs (metadata)
    self.parse_kv_pairs(&mut file, kv_count)?;

    // NEW: Calculate default head_dim if not set by GGUF
    self.metadata.calculate_default_head_dim();

    if load_tensors {
        // ...
    }
    Ok(())
}
```

### Pitfall 2: GGUF Value Overrides Calculated Value Even When Zero

**What goes wrong:** If GGUF contains `qwen2.rope.dimension_count = 0`, we should keep the calculated value, not use 0.

**Why it happens:** The `.unwrap_or(0)` pattern doesn't distinguish between "key missing" and "key present with value 0".

**How to avoid:** Check `dim > 0` before accepting the GGUF override.

### Pitfall 3: Forgetting to Fix Both Locations

**What goes wrong:** The bug exists in TWO files: `src/loader/metadata.rs` line 85 and `src/loader/gguf.rs` line 1568. Fixing only one leaves the bug.

**Why it happens:** Both files have similar `update_from_kv` / `update_metadata` methods.

**How to avoid:** Search for all instances of `rope.dimension_count` before committing.

### Pitfall 4: Breaking LLaMA/Mistral Models

**What goes wrong:** LLaMA models have `llama.rope.dimension_count` key which should still work as override.

**Why it happens:** The fix must preserve the override behavior for models where the key IS present.

**How to avoid:** Unit tests verify GGUF override works when key present.

## Code Examples

### Example 1: Adding calculate_default_head_dim to GgufMetadata

```rust
// File: src/loader/metadata.rs
// Add to impl GgufMetadata block, after update_from_kv method

impl GgufMetadata {
    // ... existing methods ...

    /// Calculate head_dim from hidden_size and num_heads.
    ///
    /// This should be called AFTER parsing num_heads and hidden_size
    /// but BEFORE using head_dim for any calculations.
    ///
    /// Per llama.cpp pattern: calculate default first, then allow
    /// GGUF metadata to override if key is present and valid.
    pub fn calculate_default_head_dim(&mut self) {
        if self.num_heads > 0 && self.hidden_size > 0 && self.head_dim == 0 {
            self.head_dim = self.hidden_size / self.num_heads;
        }
    }
}
```

### Example 2: Fixing qwen2.rope.dimension_count Parsing

```rust
// File: src/loader/metadata.rs, line 85
// BEFORE (buggy):
"qwen2.rope.dimension_count" => self.head_dim = value.parse().unwrap_or(0),

// AFTER (fixed):
"qwen2.rope.dimension_count" => {
    // Optional override: only set if value is valid
    if let Ok(dim) = value.parse::<usize>() {
        if dim > 0 {
            self.head_dim = dim;
        }
    }
    // If key missing or value invalid, keep calculated default
},

// Same fix needed for llama.rope.dimension_count at line 98-100
"llama.rope.dimension_count" => {
    // Usually head_dim = hidden_size / num_heads, but this gives rope dimensions
    if let Ok(dim) = value.parse::<usize>() {
        if dim > 0 {
            self.head_dim = dim;
        }
    }
}
```

### Example 3: Calling calculate_default_head_dim After Parsing

```rust
// File: src/loader/gguf.rs, in load_from_disk method

fn load_from_disk(&mut self, load_tensors: bool) -> Result<()> {
    // ... existing code ...

    // Parse KV pairs (metadata)
    self.parse_kv_pairs(&mut file, kv_count)?;

    // NEW: Calculate default head_dim if not set by GGUF
    self.metadata.calculate_default_head_dim();

    if load_tensors {
        // Parse tensor info (creates LazyTensor handles)
        self.parse_tensor_info(&mut file, tensor_count)?;
        // ...
    }

    Ok(())
}
```

### Example 4: Unit Test for head_dim Calculation

```rust
// File: src/loader/metadata.rs, add to #[cfg(test)] mod tests

#[test]
fn test_qwen2_head_dim_default_calculation() {
    let mut meta = GgufMetadata::default();
    meta.num_heads = 14;
    meta.hidden_size = 896;
    // head_dim starts at 0 (from Default)

    meta.calculate_default_head_dim();

    assert_eq!(meta.head_dim, 64, "head_dim should be 896 / 14 = 64");
}

#[test]
fn test_head_dim_gguf_override() {
    let mut meta = GgufMetadata::default();
    meta.num_heads = 14;
    meta.hidden_size = 896;

    // Calculate default
    meta.calculate_default_head_dim();
    assert_eq!(meta.head_dim, 64);

    // GGUF override with valid value
    meta.update_from_kv("qwen2.rope.dimension_count", "96");
    assert_eq!(meta.head_dim, 96, "GGUF value should override calculated default");
}

#[test]
fn test_head_dim_gguf_invalid_ignored() {
    let mut meta = GgufMetadata::default();
    meta.num_heads = 14;
    meta.hidden_size = 896;

    // Calculate default
    meta.calculate_default_head_dim();
    assert_eq!(meta.head_dim, 64);

    // GGUF override with invalid value (should be ignored)
    meta.update_from_kv("qwen2.rope.dimension_count", "not_a_number");
    assert_eq!(meta.head_dim, 64, "Invalid GGUF value should keep calculated default");
}

#[test]
fn test_head_dim_gguf_zero_ignored() {
    let mut meta = GgufMetadata::default();
    meta.num_heads = 14;
    meta.hidden_size = 896;

    // Calculate default
    meta.calculate_default_head_dim();
    assert_eq!(meta.head_dim, 64);

    // GGUF override with zero (should be ignored)
    meta.update_from_kv("qwen2.rope.dimension_count", "0");
    assert_eq!(meta.head_dim, 64, "Zero GGUF value should keep calculated default");
}

#[test]
fn test_llama_head_dim_calculation() {
    let mut meta = GgufMetadata::default();
    meta.num_heads = 32;
    meta.hidden_size = 4096;

    meta.calculate_default_head_dim();

    assert_eq!(meta.head_dim, 128, "head_dim should be 4096 / 32 = 128");
}
```

## State of the Art

| Old Approach | Current Approach (llama.cpp) | When Changed | Impact |
|--------------|------------------------------|--------------|--------|
| `.unwrap_or(0)` for optional keys | Calculate default, then optional override | 2023-07 (llama.cpp) | Missing keys no longer cause silent failures |

**Deprecated/outdated:**
- `unwrap_or(0)` for optional metadata: Replaced with explicit default calculation
- Assuming all GGUF files have all keys: Different quantizers produce different key sets

## Open Questions

1. **GQA models with different num_kv_heads:**
   - What we know: Some models have `num_kv_heads < num_heads` (Grouped Query Attention)
   - What's unclear: Should `head_dim` calculation use `num_heads` or `num_kv_heads`?
   - Recommendation: Use `num_heads` (query heads) per llama.cpp pattern. The `n_embd_head_k` in llama.cpp uses `n_head()` (query heads).

2. **Models with non-divisible hidden_size:**
   - What we know: Real models always have `hidden_size % num_heads == 0`
   - What's unclear: How to handle malformed GGUF with non-divisible values
   - Recommendation: Accept integer division truncation; malformed GGUF should fail elsewhere with clear error.

## Sources

### Primary (HIGH confidence)
- `/home/feanor/Projects/ROCmForge/docs/QWEN2_HEAD_DIM_BUG_INVESTIGATION.md` - Complete root cause analysis with verified bug locations
- `/home/feanor/Projects/ROCmForge/src/loader/metadata.rs` - Verified buggy code at line 85
- `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs` - Verified buggy code at line 1568
- `/home/feanor/Projects/llama.cpp/src/llama-model.cpp:598-607` - Reference implementation showing correct pattern

### Secondary (MEDIUM confidence)
- `/home/feanor/Projects/ROCmForge/.planning/research/STACK.md` - Documents required changes for v1.1
- `/home/feanor/Projects/ROCmForge/.planning/research/FEATURES.md` - Expected behaviors after fix

### Tertiary (LOW confidence)
- None - all findings verified with source code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No external dependencies, existing Rust patterns
- Architecture: HIGH - Verified pattern from llama.cpp reference implementation
- Pitfalls: HIGH - Root cause and fix locations confirmed in source code

**Research date:** 2026-01-19
**Valid until:** 2026-02-28 (stable - logic fix, no external dependencies)
