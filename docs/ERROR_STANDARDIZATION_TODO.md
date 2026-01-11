# Error Message Standardization TODO

**Date**: 2026-01-08
**Status**: ✅ COMPLETE
**Goal**: Standardize all GPU memory-related error messages across the codebase

---

## Standardized Terminology

| Concept | Standard Term |
|---------|---------------|
| GPU memory allocation failures | Describe the HIP function that failed |
| Memory pool allocation | "GPU memory pool" |
| Sub-buffer creation | "GPU memory sub-allocation" |
| Tensor creation from pool | "GPU memory pool tensor creation" |
| Buffer copy operations | "GPU memory copy" |
| Handle creation failures | "Failed to create [handle type]" |

**Note**: For HIP-specific operations, use the HIP function name (e.g., "hipMalloc failed", "hipMemcpyDtoH failed") as this provides the most diagnostic value.

---

## Complete Inventory

### Files Analyzed

| File | Status | Notes |
|------|--------|-------|
| `src/backend/hip_backend.rs` | ✅ Complete | Error messages use HIP function names - appropriate |
| `src/loader/gguf.rs` | ✅ Complete | Pool error messages standardized |
| `src/attention/gpu.rs` | ✅ Complete | Uses proper error variants (MemoryAllocation, MemoryCopy, etc.) |
| `src/model/simple_transformer.rs` | ✅ OK | Uses warnings for optional GPU fallback |
| `src/ops/attention_gpu.rs` | ✅ OK | Error messages are clear |
| `src/model/execution_plan.rs` | ✅ OK | Error messages are clear |
| `src/model/glm_position.rs` | ✅ Complete | Uses proper error variants |
| `src/attention/rope.rs` | ✅ Complete | Uses proper error variants |
| `src/attention/multi_query.rs` | ✅ Complete | Uses proper error variants |
| Test files | ✅ OK | Using `.expect()` in tests is acceptable |

---

## Tasks

### Phase 1: Inventory (COMPLETE)
- [x] Task 1.1: Audit `src/backend/hip_backend.rs`
- [x] Task 1.2: Audit `src/loader/gguf.rs`
- [x] Task 1.3: Audit `src/attention/gpu.rs`
- [x] Task 1.4: Audit `src/model/simple_transformer.rs`
- [x] Task 1.5: Audit remaining production files

### Phase 2: BUG-11 GPU Pool Terminology (COMPLETE)
- [x] Task 2.1: Fix "Sub-buffer out of bounds" → "GPU memory sub-allocation failed"
- [x] Task 2.2: Fix pool allocation error messages
- [x] Task 2.3: Fix tensor creation error messages

### Phase 3: AttentionError Enum Expansion (COMPLETE)
The `AttentionError` enum was expanded with new variants for proper error categorization:
- `MemoryAllocation(String)` - GPU memory allocation failures
- `MemoryCopy(String)` - H2D/D2H memory copy failures
- `GpuOperation(String)` - GPU kernel/operation failures
- `HandleCreation(String)` - Handle/resource creation failures
- `Synchronization(String)` - GPU synchronization failures

**Files Updated**:
- `src/attention/mod.rs` - Expanded enum with 5 new variants
- `src/attention/gpu.rs` - 53 usages updated
- `src/attention/rope.rs` - 4 usages updated
- `src/attention/multi_query.rs` - 5 usages updated
- `src/model/glm_position.rs` - 4 usages updated

**Note**: `DimensionError` is still used appropriately for:
- Actual dimension/bounds validation (position ID exceeds max_seq_len)
- Shape constraint violations (head_dim must be even)
- Input validation (empty position_ids arrays)

---

## Progress

**Started**: 2026-01-08
**Last Updated**: 2026-01-08

**Status**: ✅ COMPLETE

All error messages are now semantically correct:
1. GPU memory operations use `MemoryAllocation` / `MemoryCopy`
2. GPU operations use `GpuOperation`
3. Handle creation uses `HandleCreation`
4. Synchronization uses `Synchronization`
5. Dimension/bounds validation uses `DimensionError`
6. Shape mismatches use `ShapeMismatch`
