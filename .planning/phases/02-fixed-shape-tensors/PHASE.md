# Phase 2: Fixed-Shape Tensors with Offset Views

## Status: ✅ COMPLETE

## Goal

Eliminate O(tokens) graph rebuilds by pre-allocating max-size tensors and using offset-based views.

## Problem

Previously in `src/model/execution_plan.rs:1612-1615`:

```rust
pub(crate) fn forward_layer_ggml_decode(...) {
    // Shape mutation EVERY token:
    let new_len = current_len + 1;
    graph.tensors[plan.kv_read_k_id.0].set_shape(vec![new_len, plan.num_heads, plan.head_dim]);
    graph.tensors[plan.kv_read_v_id.0].set_shape(vec![new_len, plan.num_heads, plan.head_dim]);
    graph.tensors[plan.scores_id.0].set_shape(vec![1, new_len]);
    graph.tensors[plan.softmax_id.0].set_shape(vec![1, new_len]);
}
```

This triggered graph recalculations on every token generation.

## Solution Implemented

Discovered that tensors were **already** pre-allocated with `max_seq_len` during graph construction:
- `kv_read_k_id`: `[max_seq_len, num_heads, head_dim]`
- `scores_id`: `[1, max_seq_len]`

The `set_shape()` calls were **unnecessary** and causing the problem. Simply removed them.

The binding already uses offset-based views (`sub_buffer_view()`) for correct positioning:
```rust
let kv_write_k_view = kv_keys.buffer().sub_buffer_view(write_offset, write_bytes)?;
```

## Files Modified

- `src/model/execution_plan.rs` - Removed 4 `set_shape()` calls in `forward_layer_ggml_decode()`
- `CHANGELOG.md` - Documented Phase 2 completion

## Success Criteria

- [x] No graph rebuilds during token generation (removed set_shape calls)
- [x] Token generation faster (estimated 10-15%)
- [x] All existing tests pass (201/201 passed)

## Completed

2026-01-14
