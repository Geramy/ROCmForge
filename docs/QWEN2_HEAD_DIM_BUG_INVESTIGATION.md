# Qwen2 head_dim Bug Investigation

**Date:** 2026-01-19
**Status:** Root cause identified, fix pending
**Issue:** Inference fails with `buffer_size=0` error on Qwen2 models

---

## Error Message

```
ERROR: Inference failed: Memory allocation failed: GPU memory sub-allocation failed: offset=0 size=3584 > buffer_size=0
```

- `size=3584` = 896 * 4 bytes = `hidden_size` * sizeof(f32)
- `buffer_size=0` indicates a 0-byte GPU buffer was allocated

---

## Root Cause

### The Bug: `head_dim = 0`

When `qwen2.rope.dimension_count` GGUF metadata key doesn't exist, the code defaults `head_dim` to `0`.

**Impact:**
1. Tensor shapes become `[1, num_heads, head_dim]` = `[1, 14, 0]`
2. `element_count() = 1 * 14 * 0 = 0`
3. `byte_size() = 0 * 4 = 0`
4. 0-byte GPU buffer allocated
5. `sub_buffer_view(0, 3584)` on 0-size buffer fails

---

## GGUF Metadata Analysis

### Qwen2 Model (`models/qwen2.5-0.5b.gguf`)

```python
# Actual keys present in GGUF:
qwen2.attention.head_count = 14
qwen2.context_length = 32768
qwen2.embedding_length = 896
qwen2.rope.freq_base = 1000000.0

# Key NOT present:
qwen2.rope.dimension_count = NOT FOUND
```

### llama.cpp Reference Implementation

**File:** `/home/feanor/Projects/llama.cpp/src/llama-model.cpp`

**Lines 598-607:**
```cpp
// First, calculate default from known values
hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();

// Then, try to override from GGUF (optional, doesn't change if key missing)
ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

hparams.n_embd_head_v = hparams.n_embd / hparams.n_head();
ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

// Use n_embd_head_k as n_rot
hparams.n_rot = hparams.n_embd_head_k;

// Optional override for rope dimension count
ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);
```

**Key Pattern:** Calculate `head_dim = hidden_size / num_heads` **first**, then optionally override from GGUF.

---

## Current ROCmForge Code (Buggy)

### File: `src/loader/metadata.rs`

**Line 85:**
```rust
"qwen2.rope.dimension_count" => self.head_dim = value.parse().unwrap_or(0),
```

### File: `src/loader/gguf.rs`

**Line 1568:**
```rust
"qwen2.rope.dimension_count" => self.metadata.head_dim = value.parse().unwrap_or(0),
```

**Problem:** `.unwrap_or(0)` means missing key → `head_dim = 0`

---

## Related Code Locations

### Where 0-size buffer causes failure

**File:** `src/backend/hip_backend/backend.rs`

**Lines 518-520** (Zero-size allocation allowed):
```rust
// Validate allocation size to prevent segfaults
if size == 0 {
    tracing::warn!("HipBuffer::new: Zero-size allocation requested - this may cause issues");
}
// ... continues to allocate
```

**Lines 588-596** (sub_buffer_view error):
```rust
pub fn sub_buffer_view(&self, offset: usize, size: usize) -> HipResult<Self> {
    if offset + size > self.size() {
        return Err(HipError::MemoryAllocationFailed(format!(
            "GPU memory sub-allocation failed: offset={} size={} > buffer_size={}",
            offset,
            size,
            self.size()
        )));
    }
    // ...
}
```

### Where tensor byte_size is calculated

**File:** `src/ggml/tensor.rs`

**Lines 67-85:**
```rust
pub fn element_count(&self) -> usize {
    element_count(&self.shape)  // shape.iter().copied().product()
}

pub fn byte_size(&self) -> usize {
    self.element_count().saturating_mul(self.element_size())
}
```

When `shape = [1, 14, 0]`:
- `element_count() = 1 * 14 * 0 = 0`
- `byte_size() = 0 * 4 = 0`

---

## NOT the ROCm 7.1 D2H Bug

The documented "ROCm 7.1 memory mapper workaround" in `docs/CHANGELOG.md` was **never implemented**:

| Documented (CHANGELOG.md) | Actual (Current Code) |
|---------------------------|----------------------|
| ROCm 7.1 `hipMemcpyDtoH` from sub-buffers fails | Not ROCm-specific - logic bug |
| "Selective Memory Pooling" with `LARGE_TENSOR_THRESHOLD` | **NOT FOUND** in codebase |
| Skip pooling for tensors needing read-back | No such logic exists |

Search results:
```bash
# Search for the documented workaround:
grep -r "LARGE_TENSOR_THRESHOLD" src/
# Result: No matches found

grep -r "selective.*pooling" src/
# Result: No matches found
```

---

## Proposed Fix

### Pattern from llama.cpp:

1. **Calculate default** `head_dim = hidden_size / num_heads` before parsing GGUF
2. **Then override** if GGUF key exists (optional override)

### Fix Location:

**`src/loader/metadata.rs`** - After parsing `num_heads` and `hidden_size`
**`src/loader/gguf.rs`** - After parsing `num_heads` and `hidden_size`

### Before GGUF parsing:
```rust
// Calculate default head_dim from known values
if self.metadata.num_heads > 0 && self.metadata.hidden_size > 0 {
    self.metadata.head_dim = self.metadata.hidden_size / self.metadata.num_heads;
}
```

### During GGUF parsing (optional override):
```rust
"qwen2.rope.dimension_count" => {
    if let Ok(dim) = value.parse::<usize>() {
        if dim > 0 {
            self.metadata.head_dim = dim;
        }
    }
    // If parse fails or key doesn't exist, keep calculated value
}
```

---

## Verification Steps

1. After fix, verify `head_dim` is calculated correctly:
   - `head_dim = hidden_size / num_heads = 896 / 14 = 64`

2. Run inference:
   ```bash
   ./target/release/rocmforge_cli generate --gguf models/qwen2.5-0.5b.gguf --prompt "Hello" --max-tokens 5
   ```

3. Expected: No `buffer_size=0` error, inference completes

---

## Files Referenced

| File | Lines | Description |
|------|-------|-------------|
| `src/loader/metadata.rs` | 85 | Bug: `.unwrap_or(0)` for head_dim |
| `src/loader/gguf.rs` | 1568 | Bug: `.unwrap_or(0)` for head_dim |
| `src/ggml/tensor.rs` | 67-85 | byte_size calculation |
| `src/backend/hip_backend/backend.rs` | 518-520 | Zero-size allocation warning |
| `src/backend/hip_backend/backend.rs` | 588-596 | sub_buffer_view error |
| `/home/feanor/Projects/llama.cpp/src/llama-model.cpp` | 598-607 | llama.cpp correct pattern |

---

## Summary

- **Root Cause:** `head_dim` defaults to 0 when `qwen2.rope.dimension_count` key is missing from GGUF
- **Fix Pattern:** Calculate `head_dim = hidden_size / num_heads` first, then optionally override from GGUF
- **Reference:** llama.cpp implements this correctly in `llama-model.cpp:598-607`
- **NOT the ROCm 7.1 bug** - That documented workaround was never implemented
