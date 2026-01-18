# Plan 03-01: Split execution_plan.rs into Focused Modules

**Phase**: 03 - Codebase Modularization
**Status**: Pending
**Complexity**: High
**Estimated Time**: 4-5 hours

---

## Problem Statement

`src/model/execution_plan.rs` is 4410 lines - far exceeding the 300 LOC convention. This monolithic file contains multiple distinct concerns:
- Architecture detection (Qwen2, LLaMA, Mistral)
- Lazy tensor management
- GGML graph execution
- Layer plan structures
- RoPE caching
- Embedding/LM head handling

**Current State**:
- 4410 lines in a single file
- Mix of architecture detection, tensor management, and execution logic
- Difficult to navigate and maintain

---

## Analysis

### File Structure Breakdown

Based on code analysis, the file contains these distinct components:

1. **Architecture Detection** (~50 lines)
   - `Architecture` enum (Qwen2, LLaMA, Mistral)
   - Layer prefix patterns

2. **Statistics Types** (~30 lines)
   - `LoadingStats` struct

3. **Core ExecutionPlan** (~400 lines)
   - Main struct with lazy tensors and cached state
   - `ExecutionPlan::new()` - constructor
   - `ExecutionPlan::decode()` - decode path execution

4. **LayerPlan** (~100 lines)
   - Per-layer weight structure
   - Layer validation methods

5. **GGML Execution Types** (~200 lines)
   - `EmbeddingGgmlPlan` - embedding graph
   - `LayerGgmlPlan` - layer decode graphs
   - `RopeCache` - RoPE table caching

6. **Plan Construction** (~800 lines)
   - `ExecutionPlan::from_loader()` - main builder
   - Tensor mapping logic
   - Layer weight discovery

7. **Graph Building** (~1500 lines)
   - `build_embedding_graph()` - embedding computation
   - `build_layer_graph()` - layer computation
   - GGML operation construction

8. **Validation** (~300 lines)
   - Shape validation
   - Missing tensor detection
   - Architecture-specific checks

9. **Execution Methods** (~1000 lines)
   - Embedding lookup
   - Layer execution
   - KV cache integration

---

## Implementation Plan

### Target Structure

```
src/model/execution_plan/
├── mod.rs              # Public exports (LayerPlan, ExecutionPlan, Architecture)
├── architecture.rs     # Architecture enum and detection
├── layer_plan.rs       # LayerPlan struct and validation
├── ggml_plan.rs        # EmbeddingGgmlPlan, LayerGgmlPlan, RopeCache
├── builder.rs          # ExecutionPlan::from_loader() and plan construction
├── graph_builder.rs    # build_embedding_graph(), build_layer_graph()
└── execute.rs          # decode() and execution methods
```

### Task 1: Create Module Directory

```bash
mkdir -p src/model/execution_plan/
touch src/model/execution_plan/{mod.rs,architecture.rs,layer_plan.rs,ggml_plan.rs,builder.rs,graph_builder.rs,execute.rs}
```

### Task 2: Extract Architecture Detection

**File**: `src/model/execution_plan/architecture.rs`

```rust
//! Model architecture detection and naming patterns

use std::fmt;

/// Detected model architecture based on tensor naming patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    LLaMA,
    Mistral,
}

impl Architecture {
    pub fn layer_prefix(&self, layer_idx: usize) -> String { ... }
    pub fn name(&self) -> &'static str { ... }
}
```

### Task 3: Extract LayerPlan

**File**: `src/model/execution_plan/layer_plan.rs`

```rust
//! Per-layer execution plan

use crate::loader::lazy_tensor::LazyTensor;
use std::sync::Arc;

/// Execution plan for a single transformer layer
#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub qkv_weight: Arc<LazyTensor>,
    pub q_weight: Option<Arc<LazyTensor>>,
    // ... all layer fields
}

impl LayerPlan {
    pub fn validate(&self) -> Result<(), Error> { ... }
}
```

### Task 4: Extract GGML Plan Types

**File**: `src/model/execution_plan/ggml_plan.rs`

```rust
//! GGML graph structures for execution

use crate::ggml::{Graph, TensorId};

/// Cached embedding computation graph
#[derive(Debug)]
pub struct EmbeddingGgmlPlan {
    pub graph: Graph,
    pub backend: StdMutex<HipGgmlBackend>,
    // ...
}

/// Cached layer computation graph
#[derive(Debug)]
pub struct LayerGgmlPlan {
    pub graph: StdMutex<Graph>,
    pub input_id: TensorId,
    // ...
}

/// Cached RoPE tables
#[derive(Debug)]
pub struct RopeCache {
    pub cos: DeviceTensor,
    pub sin: DeviceTensor,
    // ...
}
```

### Task 5: Extract Builder Logic

**File**: `src/model/execution_plan/builder.rs`

```rust
//! ExecutionPlan construction from GGUF loader

use super::{ExecutionPlan, LayerPlan, Architecture};

impl ExecutionPlan {
    pub fn from_loader(
        loader: Arc<GgufLoader>,
        backend: Arc<HipBackend>,
        config: ModelConfig,
    ) -> Result<Self, Error> {
        // Detect architecture
        // Build layer plans
        // Cache metadata
    }
}
```

### Task 6: Extract Graph Building

**File**: `src/model/execution_plan/graph_builder.rs`

```rust
//! GGML graph construction for embedding and layer execution

use crate::ggml::{Graph, TensorId};

/// Build embedding lookup graph
pub fn build_embedding_graph(
    tokens: TensorId,
    embedding: Arc<LazyTensor>,
    backend: &mut HipGgmlBackend,
) -> Result<(Graph, TensorId), Error> {
    // ...
}

/// Build transformer layer computation graph
pub fn build_layer_graph(
    input: TensorId,
    layer: &LayerPlan,
    backend: &mut HipGgmlBackend,
) -> Result<(Graph, TensorId, TensorId, TensorId), Error> {
    // Returns (graph, output, k_write, v_write)
}
```

### Task 7: Extract Execution Logic

**File**: `src/model/execution_plan/execute.rs`

```rust
//! Execution methods for ExecutionPlan

use super::{ExecutionPlan, LayerGgmlPlan};

impl ExecutionPlan {
    pub fn decode(
        &self,
        tokens: &[u32],
        position: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<f32>, Error> {
        // Embedding lookup
        // Layer iteration
        // KV cache update
    }

    pub fn embedding_lookup(&self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        // ...
    }
}
```

### Task 8: Create mod.rs Barrel File

**File**: `src/model/execution_plan/mod.rs`

```rust
//! Static execution plan for transformer layers
//!
//! This module contains the execution plan system that describes
//! how each transformer layer executes. Tensors are loaded on-demand
//! during inference.

mod architecture;
mod layer_plan;
mod ggml_plan;
mod builder;
mod graph_builder;
mod execute;

// Public exports
pub use architecture::Architecture;
pub use layer_plan::LayerPlan;
pub use ggml_plan::{EmbeddingGgmlPlan, LayerGgmlPlan, RopeCache};

use crate::loader::lazy_tensor::LazyTensor;
use crate::loader::TensorShape;
// ... other imports

/// Loading statistics for debugging/observability
#[derive(Debug, Clone, PartialEq)]
pub struct LoadingStats {
    pub total_tensors: usize,
    pub loaded_tensors: usize,
    pub unloaded_tensors: usize,
    pub cached_tensors: usize,
}

/// Static execution plan for a transformer model
#[derive(Debug)]
pub struct ExecutionPlan {
    layers: Vec<LayerPlan>,
    config: ModelConfig,
    // ... private fields
}

// Include builder, graph_builder, execute impls here
// Use pub use to re-export methods if needed
```

### Task 9: Update model/mod.rs

**Before**:
```rust
mod execution_plan;
pub use execution_plan::{ExecutionPlan, LayerPlan, Architecture};
```

**After**:
```rust
// execution_plan is now a module directory
pub mod execution_plan;
```

### Task 10: Verify Compilation

```bash
cargo check
cargo test --lib
```

---

## Strategy

1. **Bottom-up extraction**: Start with leaf modules (architecture, layer_plan, ggml_plan)
2. **Maintain imports**: Keep all existing `use` statements, adjust paths
3. **Preserve visibility**: Keep public APIs public, internal details private
4. **Test incrementally**: Run `cargo check` after each file extraction

---

## Dependencies

**No Dependencies**: Can run in parallel with 03-02, 03-03, 03-04

**Affects**:
- `src/model/mod.rs` (module declaration)
- Files that import from `crate::model::execution_plan`

---

## Definition of Done

- [ ] New directory: `src/model/execution_plan/` created
- [ ] 7 module files created (mod.rs, architecture.rs, layer_plan.rs, ggml_plan.rs, builder.rs, graph_builder.rs, execute.rs)
- [ ] Each module <600 LOC
- [ ] All public APIs preserved (ExecutionPlan, LayerPlan, Architecture)
- [ ] `cargo check` passes
- [ ] `cargo test --lib` passes
- [ ] No changes to public behavior (refactor only)

---

## Notes

- This is a pure refactoring - no behavior changes
- Keep all Phase 2 lazy loading functionality intact
- Preserve all documentation comments
- Update module-level docs to reflect new structure

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Max file LOC | 4410 | <600 per file |
| Module count | 1 | 7 |
| Public API | Same | Same |
| Test pass rate | 100% | 100% |

---

*Plan: 03-01*
*Created: 2026-01-18*
