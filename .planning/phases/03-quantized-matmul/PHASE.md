# Phase 3: Quantized MatMul Operations

## Status: ✅ COMPLETE

## Goal

Add Q4_0 and Q8_0 matmul operations for efficient quantized model inference.

## Problem

Previously `src/ggml/op.rs` only had F32 `MatMul`:

```rust
pub enum Op {
    GetRows,
    MatMul,  // Only F32!
    Add,
    // Missing: Quantized matmul variants
}
```

Quantized models must dequantize to F32 before matmul, negating memory/compute savings.

## Solution Implemented

1. Added `MatMulQ4_0`, `MatMulQ8_0` to `Op` enum in `src/ggml/op.rs`
2. Implemented dequantization functions in `src/ggml/hip_backend/ops/quantized_matmul.rs`
3. Added execute_op handlers for quantized matmul in `src/ggml/hip_backend/mod.rs`

**Format Specifications:**
- **Q4_0**: block_size=32, scale (f32, 4 bytes) + 16 bytes of 4-bit packed values = 20 bytes/block
- **Q8_0**: block_size=32, scale (f32, 4 bytes) + 32 bytes of int8 values = 36 bytes/block

## Files Created/Modified

### Created
- `src/ggml/hip_backend/ops/quantized_matmul.rs` - Dequantization and matmul functions with tests

### Modified
- `src/ggml/op.rs` - Added `MatMulQ4_0` and `MatMulQ8_0` variants
- `src/ggml/hip_backend/mod.rs` - Added execute_op match arms for quantized matmul
- `src/ggml/hip_backend/ops/mod.rs` - Added quantized_matmul module

## Implementation Notes

Current implementation uses:
1. GPU-to-host copy of quantized weights
2. CPU-side dequantization
3. Host-to-GPU copy of dequantized weights
4. Standard GPU matmul

This is **not optimal** but enables quantized model support. Future optimization would implement HIP kernels for on-device dequantization during matmul.

## Success Criteria

- [x] Q4_0 and Q8_0 matmul ops implemented
- [x] Unit tests pass (test_dequantize_q4_0_simple, test_dequantize_q8_0_simple)
- [x] All 203 lib tests pass
- [x] Follows llama.cpp quantization format specification

## Completed

2026-01-14
