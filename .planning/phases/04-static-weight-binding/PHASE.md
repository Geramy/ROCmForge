# Phase 4: Static Weight Binding

## Status: ✅ ALREADY COMPLETE (Implemented during initial ggml IR work)

## Goal

Bind weight tensors once at graph construction instead of every decode step.

## Problem (Original)

The original concern was that weights would be rebound on every decode step, causing unnecessary overhead.

## Solution (Already Implemented)

The ggml IR implementation already has static weight binding:

1. **Cached graph construction** (`src/model/execution_plan.rs:1576-1578`):
   ```rust
   let plans = self
       .layer_ggml_plans
       .get_or_try_init(|| self.build_layer_ggml_plans(backend))?;
   ```
   The `layer_ggml_plans` uses `OnceCell` for one-time initialization.

2. **Weight binding during graph build** (lines 1476-1527):
   ```rust
   ggml_backend.bind(&graph.tensors[norm1_w_id.0], norm1_weight.buffer().clone())?;
   ggml_backend.bind(&graph.tensors[norm2_w_id.0], norm2_weight.buffer().clone())?;
   ggml_backend.bind(&graph.tensors[o_proj_id.0], o_proj.buffer().clone())?;
   // ... all weights bound once
   ```

3. **Persistent backend storage** (line 1548):
   ```rust
   LayerGgmlPlan {
       graph: StdMutex::new(graph),
       backend: StdMutex::new(ggml_backend),  // Backend with bound weights
       // ...
   }
   ```

4. **Only dynamic tensors rebound per-token** (`forward_layer_ggml_decode`, lines 1628-1673):
   - Input tensor
   - KV cache views (offset-based, efficient)
   - RoPE cache views (offset-based, efficient)

## Files Modified

| File | Status |
|------|--------|
| `src/model/execution_plan.rs` | ✅ Already implemented |
| `src/ggml/executor.rs` | ✅ Already skips allocated buffers |
| `src/ggml/graph.rs` | ✅ Graph structure complete |

## Success Criteria

- [x] Weights bound once at initialization
- [x] No per-token rebinding of static weights
- [x] Only dynamic tensors rebound (input, KV views, RoPE)
- [x] All tests pass

## Completed

Already implemented during initial ggml IR development.
Verified: 2026-01-14
