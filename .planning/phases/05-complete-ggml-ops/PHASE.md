# Phase 5: Complete Missing ggml Ops

## Status: In Progress

## Goal

Implement remaining ggml operations for full IR compatibility.

## Missing Ops

1. **Accumulate** - For KV cache writes without Copy + manual offset ✅ COMPLETE
2. **Tensor Pool/Allocator** - Efficient buffer reuse (llama.cpp's `ggml_allocr`) - PENDING
3. **Graph Optimizer** - CSE, dead code elimination, layout optimization - PENDING

## Completed Work

### Accumulate Op (2026-01-14) ✅

Added `Accumulate { offset: usize }` to `Op` enum for in-place tensor accumulation.

**Implementation Details:**
- `src/ggml/op.rs` - Added `Accumulate { offset: usize }` variant
- `src/ggml/hip_backend/ops/accumulate.rs` - CPU-side accumulate implementation
  - Downloads src/dst buffers from GPU
  - Performs element-wise addition: `dst[offset:offset+src_size] += src`
  - Uploads result to output and updates dst in-place
- `src/ggml/hip_backend/mod.rs` - Added execute_op handler (lines 1145-1203)

**Testing:**
- 3 unit tests in accumulate.rs
- All 206 tests passing

**Known Limitations:**
- Currently uses CPU-side computation (GPU kernel TODO)

## Remaining Work

### Tensor Allocator
Create `src/ggml/allocator.rs`:
- Track allocated buffers in pool
- Reuse buffers for same-sized tensors
- Free all at graph completion

### Graph Optimizer
Create `src/ggml/optimizer.rs`:
- Common subexpression elimination
- Dead code elimination
- Layout optimization (RowMajor vs ColMajor)

## Files to Create

- `src/ggml/allocator.rs` - Tensor pool/allocator
- `src/ggml/optimizer.rs` - Graph optimization passes

## Files to Modify

- `src/ggml/executor.rs` - Integrate allocator and optimizer

## Success Criteria

- [x] Accumulate op implemented and tested
- [ ] Tensor allocator reduces allocations by 50%+
- [ ] Graph optimizer eliminates redundant ops
- [ ] All tests pass
