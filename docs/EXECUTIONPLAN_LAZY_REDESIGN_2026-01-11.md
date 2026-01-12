# ExecutionPlan LazyTensor Integration - Design Document

**Date**: 2026-01-11
**Status**: ⚠️ **NOT IMPLEMENTED** - Architecture proposal only
**Related**: Phase 1 Lazy Loading (COMPLETE), Phase 2 Lazy Loading (PLANNED)

---

## Executive Summary

This document describes the **proposed** redesign of `ExecutionPlan` to use `LazyTensor` handles instead of eagerly-loaded `DeviceTensor` weights. This redesign is **Phase 2** of the lazy loading initiative and is required to achieve the original goal of <5s model loading time.

**Current State (Phase 1 - COMPLETE)**:
- ✅ LazyTensor infrastructure implemented
- ✅ Memory-mapped file access working
- ✅ On-demand tensor loading via `load_tensor_to_gpu()`
- ✅ 67% RAM reduction during model loading (~15GB → ~5GB)
- ❌ NO speed improvement in total loading time (still ~60s)

**Proposed (Phase 2 - NOT IMPLEMENTED)**:
- ⏳ ExecutionPlan stores LazyTensor handles
- ⏳ Weights loaded on-demand during first forward pass
- ⏳ Achieve <5s model initialization time
- ⏳ Progressive layer loading for generation

**Status**: Phase 2 was **REJECTED** after code review found 6 critical flaws. See `docs/CODE_REVIEW_PHASE1_LAZY_LOADING_2026-01-11.md` for details.

---

## Architecture Overview

### Before: Eager Loading (Current Implementation)

```
┌─────────────────────────────────────────────────────────────┐
│ ExecutionPlan::from_gguf(loader, backend)                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Load ALL ~300 tensors to GPU (load_to_gpu())             │
│ 2. Create ExecutionPlan with DeviceTensor fields            │
│ 3. Return fully-loaded plan                                 │
└─────────────────────────────────────────────────────────────┘
           │ (~60 seconds)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ ExecutionPlan {                                             │
│     embedding_weights: DeviceTensor,      // ← Loaded to GPU│
│     lm_head: DeviceTensor,                // ← Loaded to GPU│
│     layers: Vec<LayerPlan>,                              │
│ }                                                          │
│                                                            │
│ LayerPlan {                                                │
│     qkv_weight: DeviceTensor,              // ← Loaded to GPU│
│     o_proj: DeviceTensor,                // ← Loaded to GPU│
│     mlp_gate_proj: DeviceTensor,         // ← Loaded to GPU│
│     mlp_up_proj: DeviceTensor,           // ← Loaded to GPU│
│     mlp_down_proj: DeviceTensor,         // ← Loaded to GPU│
│     ... (8 more DeviceTensor fields)                    │
│ }                                                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼ (ready for inference)
```

**Problem**: ALL tensors must be loaded before inference can start, requiring ~60s.

### After: Lazy Loading (Proposed Phase 2)

```
┌─────────────────────────────────────────────────────────────┐
│ ExecutionPlan::from_gguf(loader, backend)                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Create LazyTensor handles (metadata only)                │
│ 2. Create ExecutionPlan with LazyTensor fields              │
│ 3. Return immediately (<5s)                                 │
└─────────────────────────────────────────────────────────────┘
           │ (<5 seconds)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ ExecutionPlan {                                             │
│     embedding_weights: Arc<LazyTensor>,    // ← Unloaded    │
│     lm_head: Arc<LazyTensor>,              // ← Unloaded    │
│     loader: Arc<GgufLoader>,               // ← Kept alive  │
│     backend: Arc<HipBackend>,              // ← For loading │
│     layers: Vec<LayerPlan>,                               │
│ }                                                          │
│                                                            │
│ LayerPlan {                                                │
│     qkv_weight: Arc<LazyTensor>,           // ← Unloaded    │
│     o_proj: Arc<LazyTensor>,             // ← Unloaded    │
│     mlp_gate_proj: Arc<LazyTensor>,      // ← Unloaded    │
│     mlp_up_proj: Arc<LazyTensor>,        // ← Unloaded    │
│     mlp_down_proj: Arc<LazyTensor>,      // ← Unloaded    │
│     ... (8 more Arc<LazyTensor> fields)                 │
│ }                                                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼ (first forward pass triggers loading)
┌─────────────────────────────────────────────────────────────┐
│ forward_layer()                                             │
├─────────────────────────────────────────────────────────────┤
│ 1. Check if layer weights loaded                            │
│ 2. If not, load from LazyTensor → GPU (on-demand)           │
│ 3. Cache loaded tensors for subsequent passes               │
│ 4. Execute layer computation                                │
└─────────────────────────────────────────────────────────────┘
           │ (~100ms first access, 0ms cached)
           ▼ (ready for inference)
```

**Benefit**: Model initializes in <5s, weights load progressively during first inference pass.

---

## Data Structures

### LazyTensor (Phase 1 - ✅ COMPLETE)

```rust
// src/loader/lazy_tensor.rs

/// Tensor that may not be loaded yet
pub enum LazyTensor {
    /// Metadata only, data not loaded
    Unloaded {
        name: String,
        offset: u64,           // Byte offset in GGUF file
        size: usize,           // Size in bytes
        shape: Vec<usize>,     // Tensor dimensions
        tensor_type: GgufTensorType,  // For dequantization
    },

    /// Loaded to GPU (cached)
    Gpu {
        name: String,
        tensor: Arc<DeviceTensor>,  // GPU tensor with data
    },
}

// SAFETY: LazyTensor is Send+Sync because all fields are Send+Sync
unsafe impl Send for LazyTensor {}
unsafe impl Sync for LazyTensor {}
```

### ExecutionPlan (Proposed Phase 2 - ❌ NOT IMPLEMENTED)

```rust
// src/model/execution_plan.rs (PROPOSED DESIGN)

/// Static execution plan for a transformer model
///
/// Contains lazy weight handles and execution information for all layers.
/// Weights are loaded on-demand during inference, enabling fast initialization.
pub struct ExecutionPlan {
    layers: Vec<LayerPlan>,
    config: ModelConfig,

    // LAZY TENSOR HANDLES (loaded on-demand)
    /// Lazy tensor handle for embedding weights
    embedding_weights: Arc<LazyTensor>,

    /// Lazy tensor handle for LM head
    lm_head: Arc<LazyTensor>,

    /// Position encoding handler for applying RoPE embeddings
    position_handler: Option<GlmPositionHandler>,

    // REFERENCES FOR ON-DEMAND LOADING
    /// GGUF loader reference for lazy loading (kept alive during inference)
    loader: Arc<GgufLoader>,

    /// HIP backend for GPU operations
    backend: Arc<HipBackend>,
}

impl ExecutionPlan {
    /// Get or load embedding weights (lazy loading)
    pub fn get_or_load_embedding(&self) -> HipResult<Arc<DeviceTensor>> {
        match &*self.embedding_weights {
            LazyTensor::Unloaded { .. } => {
                // Load on-demand
                let tensor = self.loader.load_tensor_to_gpu(
                    self.embedding_weights.name(),
                    &self.backend
                )?;
                Ok(Arc::new(tensor))
            }
            LazyTensor::Gpu { tensor, .. } => {
                // Already loaded, return cached
                Ok(tensor.clone())
            }
        }
    }

    /// Get or load layer weights (lazy loading)
    pub fn get_or_load_layer_weights(
        &self,
        layer_idx: usize
    ) -> HipResult<LoadedLayerWeights> {
        let layer_plan = &self.layers[layer_idx];

        // Load all layer weights on-demand
        let qkv_weight = self.get_or_load_tensor(&layer_plan.qkv_weight)?;
        let o_proj = self.get_or_load_tensor(&layer_plan.o_proj)?;
        let mlp_gate = self.get_or_load_tensor(&layer_plan.mlp_gate_proj)?;
        let mlp_up = self.get_or_load_tensor(&layer_plan.mlp_up_proj)?;
        let mlp_down = self.get_or_load_tensor(&layer_plan.mlp_down_proj)?;

        Ok(LoadedLayerWeights {
            qkv_weight,
            o_proj,
            mlp_gate,
            mlp_up,
            mlp_down,
        })
    }

    fn get_or_load_tensor(
        &self,
        lazy: &Arc<LazyTensor>
    ) -> HipResult<Arc<DeviceTensor>> {
        match &**lazy {
            LazyTensor::Unloaded { name, .. } => {
                let tensor = self.loader.load_tensor_to_gpu(name, &self.backend)?;
                Ok(Arc::new(tensor))
            }
            LazyTensor::Gpu { tensor, .. } => Ok(tensor.clone()),
        }
    }
}

/// Loaded layer weights (cached after first access)
pub struct LoadedLayerWeights {
    pub qkv_weight: Arc<DeviceTensor>,
    pub o_proj: Arc<DeviceTensor>,
    pub mlp_gate: Arc<DeviceTensor>,
    pub mlp_up: Arc<DeviceTensor>,
    pub mlp_down: Arc<DeviceTensor>,
}
```

### LayerPlan (Proposed Phase 2 - ❌ NOT IMPLEMENTED)

```rust
// src/model/execution_plan.rs (PROPOSED DESIGN)

/// Execution plan for a single transformer layer
///
/// Contains LAZY handles for all weight tensors.
/// Weights are loaded on-demand during first forward pass.
#[derive(Debug, Clone)]
pub struct LayerPlan {
    /// LAZY handles for all layer weights
    pub qkv_weight: Arc<LazyTensor>,
    pub qkv_bias: Option<Arc<LazyTensor>>,
    pub o_proj: Arc<LazyTensor>,
    pub o_proj_bias: Option<Arc<LazyTensor>>,
    pub mlp_gate_proj: Arc<LazyTensor>,
    pub mlp_up_proj: Arc<LazyTensor>,
    pub mlp_down_proj: Arc<LazyTensor>,
    pub norm1_weight: Arc<LazyTensor>,
    pub norm1_bias: Option<Arc<LazyTensor>>,
    pub norm2_weight: Arc<LazyTensor>,
    pub norm2_bias: Option<Arc<LazyTensor>>,
}
```

---

## API Changes

### Construction

**Before (Current)**:
```rust
// Loads ALL tensors to GPU (~60s)
let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;
```

**After (Proposed)**:
```rust
// Creates LazyTensor handles (<5s)
let execution_plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// Embedding and LM head still unloaded
assert!(!execution_plan.embedding_weights.is_gpu_loaded());
assert!(!execution_plan.lm_head.is_gpu_loaded());

// Layer weights also unloaded
assert!(!execution_plan.layers[0].qkv_weight.is_gpu_loaded());
```

### Forward Pass

**Before (Current)**:
```rust
impl ExecutionPlan {
    pub fn forward_layer(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        layer_plan: &LayerPlan,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        // All weights already loaded in GPU
        self.self_attention(
            backend,
            &normed_hidden,
            &layer_plan.qkv_weight,  // ← DeviceTensor (ready to use)
            layer_plan.qkv_bias.as_ref(),
            &layer_plan.o_proj,       // ← DeviceTensor (ready to use)
            ...
        )
    }
}
```

**After (Proposed)**:
```rust
impl ExecutionPlan {
    pub fn forward_layer(
        &self,
        backend: &HipBackend,
        hidden_states: &DeviceTensor,
        layer_plan: &LayerPlan,
        kv_cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> HipResult<DeviceTensor> {
        // Load layer weights on-demand (first access only)
        let weights = self.get_or_load_layer_weights(layer_idx)?;

        // Use loaded weights
        self.self_attention(
            backend,
            &normed_hidden,
            &weights.qkv_weight,  // ← Loaded on-demand, cached
            weights.qkv_bias.as_ref(),
            &weights.o_proj,       // ← Loaded on-demand, cached
            ...
        )
    }
}
```

---

## Loading Strategy

### Prompt vs Generation Batching

**Prompt Phase** (all tokens at once):
```
Load Strategy: Load all layers upfront (current behavior)
Reason: All layers needed for prompt processing
Time Cost: ~60s one-time cost
Benefit: No latency during prompt processing
```

**Generation Phase** (one token at a time):
```
Load Strategy: Load layers incrementally
Reason: Layers processed sequentially
Time Cost: ~100ms per layer (amortized over generation)
Benefit: Faster time-to-first-token

Example (32-layer model):
  - Layer 0 loaded on first forward pass (~100ms)
  - Layer 1 loaded on second forward pass (~100ms)
  - ...
  - All layers cached after 32nd forward pass
  - Subsequent passes use cached weights (0ms load time)
```

### Progressive Loading

```rust
// Proposed API for progressive loading
impl ExecutionPlan {
    /// Preload specific layers (e.g., for prompt processing)
    pub fn preload_layers(&self, layer_indices: &[usize]) -> HipResult<()> {
        for &idx in layer_indices {
            self.get_or_load_layer_weights(idx)?;
        }
        Ok(())
    }

    /// Preload all layers (eager loading, for prompt phase)
    pub fn preload_all(&self) -> HipResult<()> {
        for idx in 0..self.num_layers() {
            self.get_or_load_layer_weights(idx)?;
        }
        Ok(())
    }
}

// Usage:
let plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// Generation: Lazy load (default)
// - First token: ~5s init + ~100ms per layer for first pass
// - Subsequent tokens: 0ms load time (cached)

// Prompt: Preload all layers
plan.preload_all()?;
// - All layers loaded (~60s one-time cost)
// - Prompt processing proceeds without load latency
```

---

## Memory Management

### Memory Layout

**Phase 1 (Current)**:
```
┌────────────────────────────────────────────┐
│ RAM: ~5GB (metadata only)                 │
│  - GgufLoader: LazyTensor handles         │
│  - MmapGguf: Memory-mapped file           │
│                                            │
│ GPU VRAM: ~0GB (after GgufLoader::new())  │
│  - No tensors loaded yet                  │
│                                            │
│ After ExecutionPlan::from_gguf():          │
│  RAM: ~5GB (unchanged)                    │
│  GPU VRAM: ~60GB (ALL tensors loaded)     │
└────────────────────────────────────────────┘
```

**Phase 2 (Proposed)**:
```
┌────────────────────────────────────────────┐
│ RAM: ~5GB (metadata only)                 │
│  - GgufLoader: LazyTensor handles         │
│  - MmapGguf: Memory-mapped file           │
│  - ExecutionPlan: LazyTensor handles      │
│                                            │
│ GPU VRAM: ~0GB (after ExecutionPlan init) │
│  - No tensors loaded yet                  │
│                                            │
│ After first forward pass (layer 0):        │
│  RAM: ~5GB (unchanged)                    │
│  GPU VRAM: ~2GB (layer 0 weights only)    │
│                                            │
│ After 32 forward passes (all layers):      │
│  RAM: ~5GB (unchanged)                    │
│  GPU VRAM: ~60GB (all layers cached)      │
└────────────────────────────────────────────┘
```

### Caching Strategy

```rust
// GgufLoader maintains GPU cache
pub struct GgufLoader {
    // ... other fields ...

    /// GPU tensor cache (thread-safe)
    gpu_cache: Arc<RwLock<HashMap<String, Arc<DeviceTensor>>>>,
}

impl GgufLoader {
    /// Load single tensor to GPU with caching
    pub fn load_tensor_to_gpu(
        &self,
        name: &str,
        backend: &HipBackend,
    ) -> Result<DeviceTensor> {
        // Check cache first (fast path)
        {
            let cache = self.gpu_cache.read().unwrap();
            if let Some(tensor) = cache.get(name) {
                return Ok(tensor.clone());  // Arc clone (cheap)
            }
        }

        // Cache miss: load from mmap
        let lazy = self.lazy_tensors.get(name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", name))?;

        // Load bytes, dequantize, upload to GPU
        let tensor = self.load_tensor_from_mmap(lazy, backend)?;

        // Cache result (atomic entry API prevents race)
        {
            let mut cache = self.gpu_cache.write().unwrap();
            cache.entry(name.to_string())
                .or_insert_with(|| tensor.clone());
        }

        Ok(tensor)
    }
}
```

---

## Performance Analysis

### Timing Breakdown

**Current (Eager Loading)**:
```
Operation                  Time            Notes
─────────────────────────────────────────────────────────────
GgufLoader::new()         ~5s            Metadata parsing
  ├─ Parse metadata       ~1s            GGUF KV parsing
  ├─ Create LazyTensor    ~4s            ~300 tensor handles
  └─ Open mmap            <1s            Memory-mapped file

ExecutionPlan::from_gguf() ~55s          Load ALL tensors
  ├─ load_to_gpu()       ~55s           Upload ~300 tensors
  │   ├─ Dequantize     ~45s           Q4_0 → FP32 conversion
  │   ├─ hipMemcpy      ~10s           H2D transfers
  │   └─ hipMalloc      ~0s            Memory pooling (Phase 10)

Total: ~60s
```

**Proposed (Lazy Loading)**:
```
Operation                  Time            Notes
─────────────────────────────────────────────────────────────
GgufLoader::new()         ~5s            Metadata parsing
  ├─ Parse metadata       ~1s            GGUF KV parsing
  ├─ Create LazyTensor    ~4s            ~300 tensor handles
  └─ Open mmap            <1s            Memory-mapped file

ExecutionPlan::from_gguf() <1s           Create handles only
  ├─ Create Arc<LazyTensor> <1s         ~300 Arc wrappers
  └─ Store loader/backend <1s           Keep references

First forward pass (layer 0) ~100ms       Load first layer
  ├─ load_tensor_to_gpu() ~100ms         8 tensors × 12ms
  │   ├─ Dequantize     ~90ms           Q4_0 → FP32
  │   ├─ hipMemcpy      ~10ms           H2D transfer
  │   └─ Cache insert   <1ms            HashMap insert
  └─ Forward computation ~10ms           Layer 0 ops

Subsequent passes (cached) ~0ms           No load time
  └─ Forward computation ~10ms           Layer ops only

Time-to-first-token: ~5s (init) + ~100ms (layer 0)
Total initialization: <5s
```

### Memory Access Patterns

**Eager Loading**:
```
Timeline:  0s          60s         60.1s      60.2s
           │           │           │          │
           │ Load ALL  │           │          │
           │ tensors   │           │          │
           ▼           ▼           ▼          ▼
GPU:       [████████████████████████████████]
Memory:    0GB                          60GB

Access:    ALL tensors available immediately
Cost:      60s one-time load, zero subsequent latency
```

**Lazy Loading**:
```
Timeline:  0s    5s    5.1s  5.2s  5.3s  ...  10s
           │    │     │     │     │          │
           │    │     L0    L1    L2         L32
           │    │     ▼     ▼     ▼          ▼
GPU:       [    ]     [█]   [██]  [███]  ... [████████████]
Memory:    0GB  0GB   2GB   4GB   6GB   ...  60GB

Access:    Layers load progressively
Cost:      <5s init, ~100ms per layer (amortized)
```

---

## Migration Guide

### For Code Using ExecutionPlan

**Before**:
```rust
// Create execution plan (loads all tensors)
let plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// Access weights directly
let qkv = &layer_plan.qkv_weight;  // DeviceTensor
```

**After**:
```rust
// Create execution plan (lazy handles)
let plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// Access weights through lazy loading
let qkv = plan.get_or_load_layer_weight(layer_idx, "qkv_weight")?;
// Alternatively: let weights = plan.get_or_load_layer_weights(layer_idx)?;
```

### For Code Creating LayerPlans

**Before**:
```rust
let layer_plan = LayerPlan {
    qkv_weight: gpu_tensors["blk.0.attn_q.weight"].clone(),
    o_proj: gpu_tensors["blk.0.attn_output.weight"].clone(),
    // ... all weights as DeviceTensor
};
```

**After**:
```rust
let layer_plan = LayerPlan {
    qkv_weight: lazy_tensors["blk.0.attn_q.weight"].clone(),
    o_proj: lazy_tensors["blk.0.attn_output.weight"].clone(),
    // ... all weights as Arc<LazyTensor>
};
```

---

## Implementation Checklist

### Phase 2A: ExecutionPlan Redesign (2-3 weeks)

- [ ] **Step 1**: Update ExecutionPlan struct
  - [ ] Replace `DeviceTensor` fields with `Arc<LazyTensor>`
  - [ ] Add `loader: Arc<GgufLoader>` field
  - [ ] Add `backend: Arc<HipBackend>` field
  - [ ] Update struct documentation

- [ ] **Step 2**: Update LayerPlan struct
  - [ ] Replace all `DeviceTensor` fields with `Arc<LazyTensor>`
  - [ ] Update struct documentation
  - [ ] Verify Clone derive still works

- [ ] **Step 3**: Implement lazy loading methods
  - [ ] `get_or_load_tensor(&self, lazy: &Arc<LazyTensor>)`
  - [ ] `get_or_load_embedding(&self) -> Arc<DeviceTensor>`
  - [ ] `get_or_load_lm_head(&self) -> Arc<DeviceTensor>`
  - [ ] `get_or_load_layer_weights(&self, idx) -> LoadedLayerWeights`

- [ ] **Step 4**: Update forward pass
  - [ ] Modify `forward_layer()` to use lazy loading
  - [ ] Add `LoadedLayerWeights` struct for cached weights
  - [ ] Update attention kernel calls
  - [ ] Update MLP kernel calls

- [ ] **Step 5**: Update construction
  - [ ] Modify `ExecutionPlan::from_gguf()` to create LazyTensor handles
  - [ ] Remove eager `load_to_gpu()` call
  - [ ] Verify Arc reference cycles don't occur

- [ ] **Step 6**: Testing
  - [ ] Unit tests for lazy loading
  - [ ] Integration tests with actual GGUF model
  - [ ] Performance benchmarks (init time, first pass time)
  - [ ] Memory profiling (RAM, VRAM usage)

### Phase 2B: Progressive Loading (1-2 weeks)

- [ ] **Step 7**: Implement preloading APIs
  - [ ] `preload_layers(&self, indices: &[usize])`
  - [ ] `preload_all(&self)`
  - [ ] `is_layer_loaded(&self, idx) -> bool`

- [ ] **Step 8**: Generation optimization
  - [ ] Load layers sequentially during generation
  - [ ] Cache loaded layers in ExecutionPlan
  - [ ] Profile amortized load time

- [ ] **Step 9**: Prompt optimization
  - [ ] Add `preload_all()` call for prompt phase
  - [ ] Verify prompt processing latency
  - [ ] Compare eager vs lazy for prompt-only workloads

### Phase 2C: Integration & Polish (1 week)

- [ ] **Step 10**: Engine integration
  - [ ] Update `ModelRuntime::load_from_gguf()` to use lazy plan
  - [ ] Update `InferenceEngine` to handle lazy loading
  - [ ] Update CLI to show loading progress

- [ ] **Step 11**: Documentation
  - [ ] Update API documentation
  - [ ] Add migration guide
  - [ ] Update CHANGELOG.md
  - [ ] Update README.md with performance numbers

- [ ] **Step 12: Final validation
  - [ ] End-to-end testing with real model
  - [ ] Performance regression testing
  - [ ] Memory leak testing
  - [ ] Thread safety testing

---

## Risks and Mitigations

### Risk 1: First-Pass Latency Spike

**Risk**: First forward pass triggers ~100ms load time per layer, causing inconsistent latency.

**Mitigation**:
- Preload first N layers during initialization
- Background loading thread for remaining layers
- Document expected latency pattern

### Risk 2: Memory Leaks

**Risk**: Arc<LazyTensor> creates reference cycles, leaking GPU memory.

**Mitigation**:
- Use weak references for loader/backend links if needed
- Run leak detection tests (valgrind, cuda-memcheck)
- Monitor GPU memory during long-running inference

### Risk 3: Thread Safety

**Risk**: Concurrent lazy loading causes race conditions or deadlocks.

**Mitigation**:
- GgufLoader's RwLock provides thread-safe cache
- Use `HashMap::entry()` API for atomic cache insertion
- Test with multi-threaded inference workloads

### Risk 4: Performance Regression

**Risk**: Lazy loading overhead (Arc dereference, cache checks) slows down inference.

**Mitigation**:
- Benchmark before/after with real models
- Profile hot paths (cache hit should be <1μs)
- Consider eager loading option for latency-sensitive workloads

---

## Alternatives Considered

### Alternative 1: CPU-First Architecture

**Approach**: Keep ExecutionPlan eager, but move to CPU-first architecture where CPU handles most operations.

**Pros**:
- No ExecutionPlan redesign needed
- CPU can handle quantized weights directly from mmap
- Matches llama.cpp architecture (proven at scale)

**Cons**:
- Requires complete rewrite of inference engine
- 2-3 month implementation timeline
- Unclear performance benefit for RDNA3 GPUs

**Status**: Research complete, see `docs/CPU_FIRST_ARCHITECTURE_PLAN_2026-01-11.md`

### Alternative 2: Hybrid Eager/Lazy

**Approach**: Eager load frequently-used tensors (embedding, LM head), lazy load layer weights.

**Pros**:
- Simpler than full lazy loading
- Embedding accessed every pass (should be eager)
- Reduces lazy loading complexity

**Cons**:
- Still requires ExecutionPlan redesign
- Only partial speed improvement
- Adds API complexity (some eager, some lazy)

**Status**: Not pursued in favor of full lazy loading

### Alternative 3: Background Loading

**Approach**: Spawn background thread to load all tensors while initialization returns immediately.

**Pros**:
- No API changes (ExecutionPlan unchanged)
- User sees <5s init time
- All weights loaded before first access

**Cons**:
- First forward pass blocks if loading not complete
- Requires async/sync coordination
- Race condition risk if weights accessed early

**Status**: Interesting alternative, requires async runtime integration

---

## References

### Code Review Findings

- **Code Review Report**: `docs/CODE_REVIEW_PHASE1_LAZY_LOADING_2026-01-11.md`
  - Section 5: "Integration Issue - ExecutionPlan eager loading"
  - Section 6: "Memory Leak Risk"
  - Recommendations for Phase 2 implementation

### Phase 1 Implementation

- **Implementation Report**: `docs/PHASE1_LAZY_GGUF_LOADING_IMPLEMENTATION.md`
  - Files created: `src/loader/mmap.rs`, `src/loader/lazy_tensor.rs`
  - Files modified: `src/loader/gguf.rs`
  - Test results: 150/150 tests passing

### Related Documentation

- **Phase 1 Plan**: `docs/PHASE1_MINIMAL_PATCH_PLAN_2026-01-11.md`
- **CPU-First Research**: `docs/CPU_FIRST_ARCHITECTURE_PLAN_2026-01-11.md`
- **LLaMA.cpp Analysis**: `docs/LLAMACPP_RUST_IMPLEMENTATION_PLAN_2026-01-11.md`

---

## Conclusion

**Phase 1 Status**: ✅ COMPLETE - Infrastructure in place, 67% RAM reduction achieved

**Phase 2 Status**: ❌ NOT IMPLEMENTED - Architecture proposal only, rejected after code review

**Next Steps**:
1. Decide if Phase 2 effort is justified (2-3 weeks implementation)
2. Consider Alternative 1 (CPU-First) for larger performance gains
3. Consider Alternative 3 (Background Loading) for simpler implementation
4. Document decision rationale in CHANGELOG.md

**Performance Potential**:
- Best case: <5s initialization, 100ms amortized load time
- Expected case: <5s initialization, 200-300ms first-pass latency
- Worst case: No improvement (if all layers loaded immediately)

**Recommendation**: Pursue Phase 2 only if CPU-First architecture is not viable. The 2-3 week implementation effort may not justify the modest performance improvement over Phase 1.
