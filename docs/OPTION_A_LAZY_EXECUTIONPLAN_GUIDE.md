# Option A: Lazy ExecutionPlan Redesign - Comprehensive Implementation Guide

**Date:** 2026-01-11
**Status:** Detailed Implementation Plan
**Related:** Phase 2 Lazy Loading Proposal
**Sources:** Analysis of ExecutionPlan, LazyTensor, GgufLoader implementations

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Analysis](#architecture-analysis)
3. [Implementation Strategy](#implementation-strategy)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Code Examples](#code-examples)
6. [Thread Safety Considerations](#thread-safety-considerations)
7. [Performance Impact](#performance-impact)
8. [Testing Strategy](#testing-strategy)
9. [Rollback Plan](#rollback-plan)
10. [References](#references)

---

## Executive Summary

This guide provides detailed implementation instructions for Option A: redesigning `ExecutionPlan` to use `Arc<LazyTensor>` instead of eagerly-loaded `DeviceTensor` weights. This is **Phase 2** of the lazy loading initiative.

**Current State:**
- Phase 1 (LazyTensor infrastructure): ✅ COMPLETE
  - `/home/feanor/Projects/ROCmForge/src/loader/lazy_tensor.rs` (280 lines)
  - `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs` with `load_tensor_to_gpu()` (2200 lines)
  - Memory-mapped file access via `MmapGguf`
  - GPU cache with thread-safe `RwLock<HashMap<String, Arc<DeviceTensor>>>`
  - 67% RAM reduction (~15GB → ~5GB)

- Phase 2 (ExecutionPlan redesign): ❌ NOT IMPLEMENTED
  - `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs` still stores `DeviceTensor` (2459 lines)
  - All ~300 tensors loaded before inference (~60s total time)
  - NO speed improvement from Phase 1

**Goal:** Achieve <5s model initialization with on-demand tensor loading during inference.

---

## Architecture Analysis

### Current ExecutionPlan Structure

**Source:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs:79-127`

```rust
/// Static execution plan for a transformer model
///
/// Contains pre-loaded weights and execution information for all layers.
/// No dynamic allocation during inference.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    layers: Vec<LayerPlan>,
    config: ModelConfig,
    embedding_weights: DeviceTensor,      // ← Eagerly loaded
    lm_head: DeviceTensor,                // ← Eagerly loaded
    position_handler: Option<GlmPositionHandler>,
}

/// Execution plan for a single transformer layer
#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub qkv_weight: DeviceTensor,         // ← Eagerly loaded
    pub qkv_bias: Option<DeviceTensor>,
    pub o_proj: DeviceTensor,             // ← Eagerly loaded
    pub o_proj_bias: Option<DeviceTensor>,
    pub mlp_gate_proj: DeviceTensor,      // ← Eagerly loaded
    pub mlp_up_proj: DeviceTensor,        // ← Eagerly loaded
    pub mlp_down_proj: DeviceTensor,      // ← Eagerly loaded
    pub mlp_fc1: DeviceTensor,            // ← Legacy compatibility
    pub mlp_fc1_bias: Option<DeviceTensor>,
    pub mlp_fc2: DeviceTensor,            // ← Legacy compatibility
    pub mlp_fc2_bias: Option<DeviceTensor>,
    pub norm1_weight: DeviceTensor,       // ← Eagerly loaded
    pub norm1_bias: Option<DeviceTensor>,
    pub norm2_weight: DeviceTensor,       // ← Eagerly loaded
    pub norm2_bias: Option<DeviceTensor>,
}
```

**Key Issues:**
1. All 14 weight tensors per layer are stored as `DeviceTensor`
2. For 32-layer model: 14 × 32 + 2 = 458 tensors
3. All loaded in `ExecutionPlan::from_gguf()` at line 272-346
4. Loading time: ~55s for GPU uploads (line 277-280)

### Current Loading Flow

**Source:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs:272-346`

```rust
pub fn from_gguf(backend: &HipBackend, loader: &GgufLoader) -> HipResult<Self> {
    let config = loader.to_model_config()
        .map_err(|e| HipError::GenericError(format!("Failed to create model config: {}", e)))?;

    // ⚠️ BOTTLENECK: Load ALL tensors to GPU (~55s)
    let gpu_tensors = loader.load_to_gpu(backend)
        .map_err(|e| HipError::GenericError(format!("Failed to load tensors to GPU: {}", e)))?;

    // Detect architecture from actual tensor names
    let tensor_names: HashSet<_> = gpu_tensors.keys().cloned().collect();
    let architecture = Self::detect_architecture(&tensor_names)?;

    // Map embedding and LM head using helper functions and store them
    let embedding_weights = Self::map_embedding(backend, &config, &gpu_tensors)?;
    let lm_head = Self::map_lm_head(backend, &config, &gpu_tensors)?;

    // Create layers using detected architecture
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let (qkv_weight, o_proj) =
            Self::map_attention_weights(backend, &config, &gpu_tensors, layer_idx, &architecture)?;

        let (mlp_gate, mlp_up, mlp_down) =
            Self::map_mlp_weights(backend, &config, &gpu_tensors, layer_idx, &architecture)?;

        let (ln1_weight, ln1_bias, ln2_weight, ln2_bias) =
            Self::map_layer_norm_weights(backend, &config, &gpu_tensors, layer_idx, &architecture)?;

        let layer_plan = LayerPlan {
            qkv_weight,
            qkv_bias: None,
            o_proj,
            o_proj_bias: None,
            mlp_gate_proj: mlp_gate.clone(),
            mlp_up_proj: mlp_up.clone(),
            mlp_down_proj: mlp_down.clone(),
            mlp_fc1: mlp_gate.clone(),
            mlp_fc1_bias: None,
            mlp_fc2: mlp_down.clone(),
            mlp_fc2_bias: None,
            norm1_weight: ln1_weight,
            norm1_bias: Some(ln1_bias),
            norm2_weight: ln2_weight,
            norm2_bias: Some(ln2_bias),
        };

        layers.push(layer_plan);
    }

    Ok(ExecutionPlan {
        layers,
        config,
        embedding_weights,
        lm_head,
        position_handler: /* ... */,
    })
}
```

**Problem:** `loader.load_to_gpu(backend)` loads ALL ~300 tensors before any inference can start.

### Existing LazyTensor Infrastructure

**Source:** `/home/feanor/Projects/ROCmForge/src/loader/lazy_tensor.rs:48-238`

```rust
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

    /// Loaded to GPU
    Gpu {
        name: String,
        tensor: Arc<DeviceTensor>,  // GPU tensor with data
    },
}

// Thread-safe: Send + Sync implemented
unsafe impl Send for LazyTensor {}
unsafe impl Sync for LazyTensor {}

impl LazyTensor {
    pub fn unloaded(name: String, offset: u64, size: usize,
                   shape: Vec<usize>, tensor_type: GgufTensorType) -> Self { /* ... */ }

    pub fn gpu(name: String, tensor: Arc<DeviceTensor>) -> Self { /* ... */ }

    pub fn name(&self) -> &str { /* ... */ }
    pub fn is_gpu_loaded(&self) -> bool { /* ... */ }
    pub fn shape(&self) -> Option<&[usize]> { /* ... */ }
    pub fn gpu_tensor(&self) -> Option<&Arc<DeviceTensor>> { /* ... */ }
}
```

**Key Features:**
- Thread-safe (`Send + Sync`)
- Two states: `Unloaded` (metadata) or `Gpu` (loaded)
- Zero-copy initialization via `unloaded()`
- GPU tensor wrapped in `Arc` for shared ownership

### On-Demand Loading Infrastructure

**Source:** `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:722-880`

```rust
impl GgufLoader {
    /// GPU tensor cache (thread-safe)
    gpu_cache: Arc<RwLock<HashMap<String, Arc<DeviceTensor>>>>,

    /// Load a single tensor to GPU on-demand (Phase 1 lazy loading)
    pub fn load_tensor_to_gpu(
        &self,
        name: &str,
        backend: &HipBackend,
    ) -> Result<Arc<DeviceTensor>> {
        // Check GPU cache first
        {
            let cache = self.gpu_cache.read().map_err(|e| {
                anyhow!("GPU cache read lock poisoned: {}", e)
            })?;
            if let Some(cached) = cache.get(name) {
                tracing::debug!("GPU cache hit for tensor '{}'", name);
                return Ok(cached.clone());
            }
        }

        // Get lazy tensor metadata
        let lazy = self.lazy_tensors.get(name)
            .ok_or_else(|| anyhow!("Tensor not found: '{}'", name))?;

        let (offset, size, shape, tensor_type) = match lazy {
            LazyTensor::Unloaded { offset, size, shape, tensor_type, .. } => {
                (*offset, *size, TensorShape::from_dims(shape), *tensor_type)
            }
            LazyTensor::Gpu { .. } => {
                return Err(anyhow!("Tensor '{}' already marked as GPU-loaded", name));
            }
        };

        // Read tensor data from memory-mapped file (zero-copy)
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| anyhow!("Memory mapping not available"))?;

        let tensor_bytes = mmap.get_slice(offset, size)?;

        // Dequantize based on tensor type
        let f32_data: Vec<f32> = /* dequantization logic */;

        // Upload to GPU
        let device_tensor = DeviceTensor::from_host_vec(backend, f32_data, shape)?;
        let device_tensor_arc = Arc::new(device_tensor);

        // Cache the result
        {
            let mut cache = self.gpu_cache.write().map_err(|e| {
                anyhow!("GPU cache write lock poisoned: {}", e)
            })?;
            cache.insert(name.to_string(), device_tensor_arc.clone());
        }

        Ok(device_tensor_arc)
    }
}
```

**Key Features:**
- Thread-safe cache with `RwLock<HashMap<...>>`
- Checks cache before loading (fast path)
- Reads from memory-mapped file (zero-copy I/O)
- Dequantizes based on tensor type
- Uploads to GPU
- Caches result for subsequent accesses

**Performance:**
- First load: ~50-200ms per tensor (depending on size)
- Subsequent loads: <1ms (from cache)

---

## Implementation Strategy

### High-Level Approach

**Principle:** Minimal disruption to existing code, gradual migration to lazy loading.

1. **Add new fields to ExecutionPlan** (keep old ones for compatibility)
2. **Create lazy loading methods** alongside existing eager ones
3. **Update forward pass to use lazy loading** (transparent to callers)
4. **Deprecate eager loading paths** (remove in future version)

### Design Constraints

**Must Preserve:**
- Public API: `ExecutionPlan::from_gguf(backend, loader)`
- Inference API: `forward_layer()`, `forward()`, etc.
- Thread safety: Multiple concurrent inference requests
- Performance: No regression in cached tensor access

**Must Avoid:**
- Breaking changes to public API
- Reference cycles (memory leaks)
- Deadlocks in multi-threaded scenarios
- Performance regression in cached access path

---

## Step-by-Step Implementation

### Step 1: Update ExecutionPlan Struct

**File:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`

**Location:** Lines 79-86

**Before:**
```rust
pub struct ExecutionPlan {
    layers: Vec<LayerPlan>,
    config: ModelConfig,
    embedding_weights: DeviceTensor,
    lm_head: DeviceTensor,
    position_handler: Option<GlmPositionHandler>,
}
```

**After:**
```rust
use crate::loader::lazy_tensor::LazyTensor;
use std::sync::Arc;

pub struct ExecutionPlan {
    layers: Vec<LayerPlan>,
    config: ModelConfig,

    // LAZY TENSOR FIELDS (Phase 2)
    /// Lazy tensor handle for embedding weights (loaded on-demand)
    embedding_weights_lazy: Arc<LazyTensor>,

    /// Lazy tensor handle for LM head (loaded on-demand)
    lm_head_lazy: Arc<LazyTensor>,

    /// Reference to GGUF loader for on-demand loading (kept alive)
    loader: Arc<GgufLoader>,

    /// Reference to HIP backend for GPU operations
    backend: Arc<HipBackend>,

    // CACHED GPU TENSORS (after first access)
    embedding_weights: OnceCell<DeviceTensor>,
    lm_head: OnceCell<DeviceTensor>,

    position_handler: Option<GlmPositionHandler>,
}
```

**Explanation:**
- `embedding_weights_lazy` / `lm_head_lazy`: Store `Arc<LazyTensor>` instead of `DeviceTensor`
- `loader` / `backend`: Keep references for on-demand loading
- `OnceCell`: Store loaded tensors after first access (thread-safe, one-time initialization)

### Step 2: Update LayerPlan Struct

**File:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`

**Location:** Lines 96-127

**Before:**
```rust
pub struct LayerPlan {
    pub qkv_weight: DeviceTensor,
    pub qkv_bias: Option<DeviceTensor>,
    pub o_proj: DeviceTensor,
    pub o_proj_bias: Option<DeviceTensor>,
    pub mlp_gate_proj: DeviceTensor,
    pub mlp_up_proj: DeviceTensor,
    pub mlp_down_proj: DeviceTensor,
    pub mlp_fc1: DeviceTensor,
    pub mlp_fc1_bias: Option<DeviceTensor>,
    pub mlp_fc2: DeviceTensor,
    pub mlp_fc2_bias: Option<DeviceTensor>,
    pub norm1_weight: DeviceTensor,
    pub norm1_bias: Option<DeviceTensor>,
    pub norm2_weight: DeviceTensor,
    pub norm2_bias: Option<DeviceTensor>,
}
```

**After:**
```rust
use std::sync::Arc;

pub struct LayerPlan {
    // LAZY TENSOR HANDLES
    pub qkv_weight: Arc<LazyTensor>,
    pub qkv_bias: Option<Arc<LazyTensor>>,
    pub o_proj: Arc<LazyTensor>,
    pub o_proj_bias: Option<Arc<LazyTensor>>,
    pub mlp_gate_proj: Arc<LazyTensor>,
    pub mlp_up_proj: Arc<LazyTensor>,
    pub mlp_down_proj: Arc<LazyTensor>,
    pub mlp_fc1: Arc<LazyTensor>,      // Legacy compatibility
    pub mlp_fc1_bias: Option<Arc<LazyTensor>>,
    pub mlp_fc2: Arc<LazyTensor>,      // Legacy compatibility
    pub mlp_fc2_bias: Option<Arc<LazyTensor>>,
    pub norm1_weight: Arc<LazyTensor>,
    pub norm1_bias: Option<Arc<LazyTensor>>,
    pub norm2_weight: Arc<LazyTensor>,
    pub norm2_bias: Option<Arc<LazyTensor>>,
}
```

### Step 3: Add Lazy Loading Helper Methods

**File:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`

**Add after:** Line 2185 (before `impl LayerPlan`)

```rust
use std::cell::OnceCell;

impl ExecutionPlan {
    /// Get or load embedding weights (lazy loading)
    ///
    /// Returns cached GPU tensor if already loaded, otherwise loads on-demand.
    /// Thread-safe via OnceCell.
    fn get_or_load_embedding(&self) -> HipResult<&DeviceTensor> {
        self.embedding_weights.get_or_try_init(|| {
            match &*self.embedding_weights_lazy {
                LazyTensor::Unloaded { name, .. } => {
                    tracing::debug!("Loading embedding tensor '{}' on-demand", name);
                    let tensor = self.loader.load_tensor_to_gpu(name, &self.backend)
                        .map_err(|e| HipError::GenericError(format!("Failed to load embedding: {}", e)))?;
                    // Arc<DeviceTensor> → DeviceTensor (clone inner data)
                    Ok(DeviceTensor::clone(&tensor))
                }
                LazyTensor::Gpu { tensor, .. } => {
                    tracing::debug!("Embedding tensor already loaded (using cached)");
                    Ok(DeviceTensor::clone(tensor))
                }
            }
        })
    }

    /// Get or load LM head (lazy loading)
    fn get_or_load_lm_head(&self) -> HipResult<&DeviceTensor> {
        self.lm_head.get_or_try_init(|| {
            match &*self.lm_head_lazy {
                LazyTensor::Unloaded { name, .. } => {
                    tracing::debug!("Loading LM head tensor '{}' on-demand", name);
                    let tensor = self.loader.load_tensor_to_gpu(name, &self.backend)
                        .map_err(|e| HipError::GenericError(format!("Failed to load LM head: {}", e)))?;
                    Ok(DeviceTensor::clone(&tensor))
                }
                LazyTensor::Gpu { tensor, .. } => {
                    tracing::debug!("LM head tensor already loaded (using cached)");
                    Ok(DeviceTensor::clone(tensor))
                }
            }
        })
    }

    /// Get or load a single layer tensor (lazy loading)
    fn get_or_load_tensor(&self, lazy: &Arc<LazyTensor>) -> HipResult<DeviceTensor> {
        match &**lazy {
            LazyTensor::Unloaded { name, .. } => {
                tracing::debug!("Loading tensor '{}' on-demand", name);
                let tensor = self.loader.load_tensor_to_gpu(name, &self.backend)
                    .map_err(|e| HipError::GenericError(format!("Failed to load tensor '{}': {}", name, e)))?;
                Ok(DeviceTensor::clone(&tensor))
            }
            LazyTensor::Gpu { tensor, .. } => {
                Ok(DeviceTensor::clone(tensor))
            }
        }
    }

    /// Get or load all layer weights (lazy loading)
    ///
    /// Returns a struct containing all loaded layer tensors.
    /// This is called once per layer during first forward pass.
    fn get_or_load_layer_weights(&self, layer_plan: &LayerPlan) -> HipResult<LoadedLayerWeights> {
        Ok(LoadedLayerWeights {
            qkv_weight: self.get_or_load_tensor(&layer_plan.qkv_weight)?,
            qkv_bias: layer_plan.qkv_bias.as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?,
            o_proj: self.get_or_load_tensor(&layer_plan.o_proj)?,
            o_proj_bias: layer_plan.o_proj_bias.as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?,
            mlp_gate_proj: self.get_or_load_tensor(&layer_plan.mlp_gate_proj)?,
            mlp_up_proj: self.get_or_load_tensor(&layer_plan.mlp_up_proj)?,
            mlp_down_proj: self.get_or_load_tensor(&layer_plan.mlp_down_proj)?,
            mlp_fc1: self.get_or_load_tensor(&layer_plan.mlp_fc1)?,
            mlp_fc1_bias: layer_plan.mlp_fc1_bias.as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?,
            mlp_fc2: self.get_or_load_tensor(&layer_plan.mlp_fc2)?,
            mlp_fc2_bias: layer_plan.mlp_fc2_bias.as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?,
            norm1_weight: self.get_or_load_tensor(&layer_plan.norm1_weight)?,
            norm1_bias: layer_plan.norm1_bias.as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?,
            norm2_weight: self.get_or_load_tensor(&layer_plan.norm2_weight)?,
            norm2_bias: layer_plan.norm2_bias.as_ref()
                .map(|b| self.get_or_load_tensor(b))
                .transpose()?,
        })
    }
}

/// Loaded layer weights (cached after first access)
///
/// Contains all layer tensors as DeviceTensor (loaded and cached).
#[derive(Debug, Clone)]
pub struct LoadedLayerWeights {
    pub qkv_weight: DeviceTensor,
    pub qkv_bias: Option<DeviceTensor>,
    pub o_proj: DeviceTensor,
    pub o_proj_bias: Option<DeviceTensor>,
    pub mlp_gate_proj: DeviceTensor,
    pub mlp_up_proj: DeviceTensor,
    pub mlp_down_proj: DeviceTensor,
    pub mlp_fc1: DeviceTensor,
    pub mlp_fc1_bias: Option<DeviceTensor>,
    pub mlp_fc2: DeviceTensor,
    pub mlp_fc2_bias: Option<DeviceTensor>,
    pub norm1_weight: DeviceTensor,
    pub norm1_bias: Option<DeviceTensor>,
    pub norm2_weight: DeviceTensor,
    pub norm2_bias: Option<DeviceTensor>,
}
```

### Step 4: Update Construction Method

**File:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`

**Location:** Lines 272-346

**Key Changes:**
1. Remove `loader.load_to_gpu(backend)` call
2. Create `Arc<LazyTensor>` handles instead of loading tensors
3. Store `Arc<GgufLoader>` and `Arc<HipBackend>` references

```rust
pub fn from_gguf(backend: &HipBackend, loader: &GgufLoader) -> HipResult<Self> {
    let config = loader.to_model_config()
        .map_err(|e| HipError::GenericError(format!("Failed to create model config: {}", e)))?;

    // ❌ REMOVE: Load all tensors to GPU (~55s)
    // let gpu_tensors = loader.load_to_gpu(backend)?;

    // ✅ NEW: Get lazy tensor handles (metadata only, <1s)
    let lazy_tensors: &HashMap<String, LazyTensor> = /* access loader.lazy_tensors */;

    // Wrap loader and backend in Arc for lazy loading
    let loader_arc = Arc::new(loader.clone());
    let backend_arc = Arc::new(backend.clone());

    // Detect architecture from lazy tensor names
    let tensor_names: HashSet<_> = lazy_tensors.keys().cloned().collect();
    let architecture = Self::detect_architecture(&tensor_names)?;

    // Map embedding and LM head to LazyTensor handles
    let embedding_weights_lazy = Self::map_embedding_lazy(lazy_tensors, &config, &architecture)?;
    let lm_head_lazy = Self::map_lm_head_lazy(lazy_tensors, &config, &architecture)?;

    // Create layers using LazyTensor handles
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let layer_plan = Self::create_layer_plan_lazy(
            &config,
            lazy_tensors,
            layer_idx,
            &architecture,
        )?;
        layers.push(layer_plan);
    }

    // Initialize position handler
    let position_handler = /* ... same as before ... */;

    Ok(ExecutionPlan {
        layers,
        config,
        embedding_weights_lazy,
        lm_head_lazy,
        loader: loader_arc,
        backend: backend_arc,
        embedding_weights: OnceCell::new(),
        lm_head: OnceCell::new(),
        position_handler,
    })
}
```

**Helper Methods (Add to ExecutionPlan):**

```rust
impl ExecutionPlan {
    /// Map embedding weights to LazyTensor handle
    fn map_embedding_lazy(
        lazy_tensors: &HashMap<String, LazyTensor>,
        config: &ModelConfig,
        _architecture: &Architecture,
    ) -> HipResult<Arc<LazyTensor>> {
        let embedding_names = [
            "token_embd.weight",
            "embed_tokens.weight",
            "word_embeddings.weight",
        ];

        for name in &embedding_names {
            if let Some(lazy) = lazy_tensors.get(*name) {
                // Validate shape
                if let Some(shape) = lazy.shape() {
                    if shape.len() == 2 && shape[0] == config.vocab_size {
                        return Ok(Arc::new(lazy.clone()));
                    }
                }
            }
        }

        Err(HipError::GenericError(
            "No embedding tensor found (tried: token_embd.weight, embed_tokens.weight)".to_string()
        ))
    }

    /// Map LM head to LazyTensor handle
    fn map_lm_head_lazy(
        lazy_tensors: &HashMap<String, LazyTensor>,
        config: &ModelConfig,
        _architecture: &Architecture,
    ) -> HipResult<Arc<LazyTensor>> {
        let lm_head_names = ["output.weight", "lm_head.weight", "logits.weight"];

        for name in &lm_head_names {
            if let Some(lazy) = lazy_tensors.get(*name) {
                return Ok(Arc::new(lazy.clone()));
            }
        }

        // For tied embeddings
        if let Some(lazy) = lazy_tensors.get("token_embd.weight") {
            return Ok(Arc::new(lazy.clone()));
        }

        Err(HipError::GenericError(
            "No LM head tensor found".to_string()
        ))
    }

    /// Create layer plan with LazyTensor handles
    fn create_layer_plan_lazy(
        config: &ModelConfig,
        lazy_tensors: &HashMap<String, LazyTensor>,
        layer_idx: usize,
        architecture: &Architecture,
    ) -> HipResult<LayerPlan> {
        let prefix = architecture.layer_prefix(layer_idx);

        // Helper to get lazy tensor or error
        let get_lazy = |name: &str| -> HipResult<Arc<LazyTensor>> {
            lazy_tensors.get(name)
                .cloned()
                .map(Arc::new)
                .ok_or_else(|| HipError::GenericError(format!("Tensor '{}' not found", name)))
        };

        let get_lazy_optional = |name: &str| -> Option<Arc<LazyTensor>> {
            lazy_tensors.get(name).cloned().map(Arc::new)
        };

        // Map attention weights
        let qkv_weight = get_lazy(&format!("{}.attn_q.weight", prefix))?;
        let o_proj = get_lazy(&format!("{}.attn_output.weight", prefix))?;

        // Map MLP weights
        let mlp_gate = get_lazy(&format!("{}.ffn_gate.weight", prefix))?;
        let mlp_up = get_lazy(&format!("{}.ffn_up.weight", prefix))?;
        let mlp_down = get_lazy(&format!("{}.ffn_down.weight", prefix))?;

        // Map layer norm weights
        let norm1_weight = get_lazy(&format!("{}.attn_norm.weight", prefix))?;
        let norm1_bias = get_lazy_optional(&format!("{}.attn_norm.bias", prefix));
        let norm2_weight = get_lazy(&format!("{}.ffn_norm.weight", prefix))?;
        let norm2_bias = get_lazy_optional(&format!("{}.ffn_norm.bias", prefix));

        Ok(LayerPlan {
            qkv_weight,
            qkv_bias: None,
            o_proj,
            o_proj_bias: None,
            mlp_gate_proj: mlp_gate.clone(),
            mlp_up_proj: mlp_up,
            mlp_down_proj: mlp_down,
            mlp_fc1: mlp_gate.clone(),
            mlp_fc1_bias: None,
            mlp_fc2: mlp_down,
            mlp_fc2_bias: None,
            norm1_weight,
            norm1_bias,
            norm2_weight,
            norm2_bias,
        })
    }
}
```

### Step 5: Update Forward Pass Methods

**File:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`

**Location:** Lines 360-433 (forward pass), Lines 498-572 (forward_layer)

#### 5.1 Update `forward()` Method

**Before (Line 360):**
```rust
pub fn forward(
    &self,
    backend: &HipBackend,
    input_tokens: &[u32],
    embedding_weights: &DeviceTensor,  // ← Passed as parameter
) -> HipResult<DeviceTensor> {
    // ... uses self.embedding_weights directly ...
}
```

**After:**
```rust
pub fn forward(
    &self,
    backend: &HipBackend,
    input_tokens: &[u32],
    _embedding_weights: &DeviceTensor,  // ← Deprecated parameter (ignored)
) -> HipResult<DeviceTensor> {
    let seq_len = input_tokens.len();
    let _hidden_size = self.config.hidden_size;

    // Step 1: Token embedding lookup
    let mut hidden_states = {
        // ✅ Load embedding on-demand
        let embedding = self.get_or_load_embedding()?;
        self.embedding_lookup(backend, input_tokens, embedding)?
    };

    // Step 2: Pass through all transformer layers
    for (layer_idx, layer_plan) in self.layers.iter().enumerate() {
        hidden_states = self.forward_layer(
            backend,
            &hidden_states,
            layer_plan,
            None,
            layer_idx,
        )?;
    }

    Ok(hidden_states)
}
```

#### 5.2 Update `forward_layer()` Method

**Before (Line 498):**
```rust
pub fn forward_layer(
    &self,
    backend: &HipBackend,
    hidden_states: &DeviceTensor,
    layer_plan: &LayerPlan,
    kv_cache: Option<&mut KVCache>,
    layer_idx: usize,
) -> HipResult<DeviceTensor> {
    // ... uses layer_plan.qkv_weight directly (DeviceTensor) ...
}
```

**After:**
```rust
pub fn forward_layer(
    &self,
    backend: &HipBackend,
    hidden_states: &DeviceTensor,
    layer_plan: &LayerPlan,
    kv_cache: Option<&mut KVCache>,
    layer_idx: usize,
) -> HipResult<DeviceTensor> {
    tracing::debug!("forward_layer() layer={} starting", layer_idx);

    // ✅ Load layer weights on-demand (cached after first access)
    let weights = self.get_or_load_layer_weights(layer_plan)?;

    let input_shape = hidden_states.shape().dims();
    let _seq_len = input_shape[0];
    let _hidden_size = input_shape[1];

    // Store input for residual connection
    let residual = hidden_states.clone();

    // Step 1: Pre-attention LayerNorm
    let normed_hidden = self.layer_norm(
        backend,
        hidden_states,
        &weights.norm1_weight,
        weights.norm1_bias.as_ref(),
    )?;

    // Step 2: Self-attention
    let attention_output = self.self_attention(
        backend,
        &normed_hidden,
        &weights.qkv_weight,
        weights.qkv_bias.as_ref(),
        &weights.o_proj,
        weights.o_proj_bias.as_ref(),
        kv_cache,
        layer_idx,
    )?;

    // Step 3: Add residual connection
    let attention_with_residual = self.add_residual(backend, &attention_output, &residual)?;
    let attention_residual = attention_with_residual.clone();

    // Step 4: Pre-MLP LayerNorm
    let normed_attention = self.layer_norm(
        backend,
        &attention_with_residual,
        &weights.norm2_weight,
        weights.norm2_bias.as_ref(),
    )?;

    // Step 5: MLP (SwiGLU)
    let mlp_output = self.mlp_swiglu(
        backend,
        &normed_attention,
        &weights.mlp_gate_proj,
        &weights.mlp_up_proj,
        &weights.mlp_down_proj,
    )?;

    // Step 6: Add residual connection
    let final_output = self.add_residual(backend, &mlp_output, &attention_residual)?;

    Ok(final_output)
}
```

#### 5.3 Update `self_attention()` Method

**Location:** Lines 597-697

**Change:** Replace `layer_plan` parameter with `LoadedLayerWeights`:

```rust
fn self_attention(
    &self,
    backend: &HipBackend,
    hidden_states: &DeviceTensor,
    qkv_weight: &DeviceTensor,  // ← Now loaded from cache
    qkv_bias: Option<&DeviceTensor>,
    o_proj: &DeviceTensor,
    o_proj_bias: Option<&DeviceTensor>,
    kv_cache: Option<&mut KVCache>,
    layer_idx: usize,
) -> HipResult<DeviceTensor> {
    // ... implementation unchanged ...
}
```

#### 5.4 Update Accessor Methods

**Location:** Lines 2274-2332

**Before:**
```rust
pub fn embedding_weights(&self) -> &DeviceTensor {
    &self.embedding_weights
}

pub fn lm_head(&self) -> &DeviceTensor {
    &self.lm_head
}
```

**After:**
```rust
pub fn embedding_weights(&self) -> HipResult<&DeviceTensor> {
    self.get_or_load_embedding()
}

pub fn lm_head(&self) -> HipResult<&DeviceTensor> {
    self.get_or_load_lm_head()
}
```

### Step 6: Add Preloading Methods (Optional)

**File:** `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`

**Add after:** Line 346 (end of `from_gguf()` method)

```rust
impl ExecutionPlan {
    /// Preload specific layers (e.g., for prompt processing)
    ///
    /// Loads all weights for specified layers to GPU.
    /// Useful for prompt phase where all layers are needed.
    pub fn preload_layers(&self, layer_indices: &[usize]) -> HipResult<()> {
        for &idx in layer_indices {
            if idx >= self.layers.len() {
                return Err(HipError::GenericError(
                    format!("Layer index {} out of bounds (num_layers={})", idx, self.layers.len())
                ));
            }
            self.get_or_load_layer_weights(&self.layers[idx])?;
            tracing::info!("Preloaded layer {}", idx);
        }
        Ok(())
    }

    /// Preload all layers (eager loading, for prompt phase)
    ///
    /// Loads all layers to GPU upfront (~60s one-time cost).
    /// Use this for prompt processing to avoid load latency during inference.
    pub fn preload_all(&self) -> HipResult<()> {
        tracing::info!("Preloading all {} layers (this may take ~60s)", self.layers.len());
        let indices: Vec<usize> = (0..self.layers.len()).collect();
        self.preload_layers(&indices)?;
        tracing::info!("All layers preloaded successfully");
        Ok(())
    }

    /// Check if a layer is loaded to GPU
    pub fn is_layer_loaded(&self, layer_idx: usize) -> bool {
        if layer_idx >= self.layers.len() {
            return false;
        }

        let layer = &self.layers[layer_idx];
        // Check if all weights are in Gpu state
        layer.qkv_weight.is_gpu_loaded()
            && layer.o_proj.is_gpu_loaded()
            && layer.mlp_gate_proj.is_gpu_loaded()
    }

    /// Get loading statistics
    pub fn loading_stats(&self) -> LoadingStats {
        let mut stats = LoadingStats {
            total_layers: self.layers.len(),
            loaded_layers: 0,
            total_tensors: 0,
            loaded_tensors: 0,
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            if self.is_layer_loaded(idx) {
                stats.loaded_layers += 1;
            }
            stats.total_tensors += 14; // Fixed number per layer

            if layer.qkv_weight.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.o_proj.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.mlp_gate_proj.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.mlp_up_proj.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.mlp_down_proj.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.mlp_fc1.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.mlp_fc2.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.norm1_weight.is_gpu_loaded() { stats.loaded_tensors += 1; }
            if layer.norm2_weight.is_gpu_loaded() { stats.loaded_tensors += 1; }
        }

        stats
    }
}

/// Loading statistics for ExecutionPlan
#[derive(Debug, Clone)]
pub struct LoadingStats {
    pub total_layers: usize,
    pub loaded_layers: usize,
    pub total_tensors: usize,
    pub loaded_tensors: usize,
}
```

---

## Code Examples

### Example 1: Before/After Construction

**Before (Eager Loading):**
```rust
// Create execution plan (~60s to load all tensors)
let plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// All tensors already loaded to GPU
println!("Embedding loaded: {}", plan.embedding_weights().len());
println!("Layer 0 QKV loaded: {}", plan.layers[0].qkv_weight().len());

// Ready for inference immediately
let result = plan.forward(&backend, &[1, 2, 3], plan.embedding_weights())?;
```

**After (Lazy Loading):**
```rust
// Create execution plan (<5s, metadata only)
let plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// Tensors NOT loaded yet
println!("Embedding loaded: {}", plan.embedding_weights_lazy.is_gpu_loaded()); // false
println!("Layer 0 QKV loaded: {}", plan.layers[0].qkv_weight.is_gpu_loaded()); // false

// First access triggers loading (~100ms)
let result = plan.forward(&backend, &[1, 2, 3], &/* dummy */)?;

// After first pass, tensors cached
println!("Embedding loaded: {}", plan.embedding_weights_lazy.is_gpu_loaded()); // true
println!("Layer 0 QKV loaded: {}", plan.layers[0].qkv_weight.is_gpu_loaded()); // true
```

### Example 2: Prompt vs Generation

```rust
let plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// Prompt phase: preload all layers (~60s one-time cost)
plan.preload_all()?;

// Process prompt (no load latency)
let prompt_tokens = tokenize("Hello, world!");
for _ in 0..prompt_tokens.len() {
    let output = plan.forward(&backend, &prompt_tokens, &/* dummy */)?;
}

// Generation phase: lazy load (default)
// Each layer loaded on first access (~100ms per layer, amortized)
let mut token = start_token;
for _ in 0..max_tokens {
    let output = plan.forward(&backend, &[token], &/* dummy */)?;
    token = sample(&output);
}
```

### Example 3: Progressive Loading

```rust
let plan = ExecutionPlan::from_gguf(&backend, &loader)?;

// Check loading status
let stats = plan.loading_stats();
println!("Loaded: {}/{} layers, {}/{} tensors",
    stats.loaded_layers, stats.total_layers,
    stats.loaded_tensors, stats.total_tensors);

// Preload first N layers (for faster first token)
plan.preload_layers(&[0, 1, 2, 3, 4])?;

// Generate (layers load progressively)
let mut token = start_token;
for step in 0..10 {
    let output = plan.forward(&backend, &[token], &/* dummy */)?;

    // After step 0, layer 0 loaded
    // After step 1, layer 1 loaded
    // etc.

    let stats = plan.loading_stats();
    println!("Step {}: Loaded {} layers", step, stats.loaded_layers);

    token = sample(&output);
}
```

---

## Thread Safety Considerations

### 1. RwLock GPU Cache (Thread-Safe)

**Source:** `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs:626`

```rust
pub struct GgufLoader {
    gpu_cache: Arc<RwLock<HashMap<String, Arc<DeviceTensor>>>>,
}
```

**Thread Safety:**
- `RwLock`: Multiple readers OR one writer
- Multiple threads can read cache simultaneously
- Only one thread can write at a time
- Prevents race conditions during cache insertion

### 2. Arc<DeviceTensor> (Thread-Safe Sharing)

**Source:** `/home/feanor/Projects/ROCmForge/src/loader/lazy_tensor.rs:73`

```rust
Gpu {
    name: String,
    tensor: Arc<DeviceTensor>,  // Thread-safe reference counting
}
```

**Thread Safety:**
- `Arc` provides atomic reference counting
- Multiple threads can share `DeviceTensor` without copying
- When last `Arc` dropped, GPU memory freed

### 3. OnceCell (Thread-Safe One-Time Init)

**Proposed in ExecutionPlan:**

```rust
use std::cell::OnceCell;

pub struct ExecutionPlan {
    embedding_weights: OnceCell<DeviceTensor>,
    lm_head: OnceCell<DeviceTensor>,
}
```

**Thread Safety:**
- `OnceCell`: Thread-safe one-time initialization
- First thread to call `get_or_try_init()` performs initialization
- Other threads wait for initialization to complete
- No race conditions, no duplicate initialization

### 4. Synchronization Pattern

**Example: Two threads accessing same layer:**

```
Thread 1                          Thread 2
│                                 │
├─ forward_layer(0)               ├─ forward_layer(0)
│                                 │
├─ get_or_load_layer_weights()    ├─ get_or_load_layer_weights()
│                                 │
├─ get_or_load_tensor(qkv)        ├─ get_or_load_tensor(qkv)
│                                 │
├─ Check GgufLoader cache         ├─ Check GgufLoader cache
│  (RwLock read lock)              │  (RwLock read lock)
│                                 │
├─ Cache miss                      ├─ Cache miss
│                                 │
├─ Release read lock               ├─ Release read lock
│                                 │
├─ Acquire write lock              │ (blocked)
├─ Load qkv tensor                │
├─ Insert into cache               │
├─ Release write lock              │
├─ Return tensor                   │
│                                 ├─ Acquire read lock
│                                 ├─ Read from cache (hit!)
│                                 ├─ Release read lock
│                                 ├─ Return tensor
```

**Guarantees:**
- Only one thread loads a given tensor
- Other threads wait and reuse cached result
- No duplicate GPU allocations
- No data races

---

## Performance Impact

### Timing Breakdown

**Current (Eager Loading):**
| Operation | Time | Notes |
|-----------|------|-------|
| `GgufLoader::new()` | ~5s | Metadata parsing |
| `ExecutionPlan::from_gguf()` | ~55s | Load ALL tensors |
| - Dequantization | ~45s | Q4_0 → FP32 |
| - GPU uploads | ~10s | H2D transfers |
| **Total Init** | **~60s** | |
| First forward pass | ~10ms | All tensors loaded |

**Proposed (Lazy Loading):**
| Operation | Time | Notes |
|-----------|------|-------|
| `GgufLoader::new()` | ~5s | Metadata parsing |
| `ExecutionPlan::from_gguf()` | <1s | Create handles only |
| **Total Init** | **<5s** | ✅ 12x faster |
| First forward pass (layer 0) | ~110ms | Load 8 tensors × 12ms |
| - Tensor loading | ~100ms | First layer only |
| - Forward computation | ~10ms | Layer ops |
| Subsequent passes (layer N) | ~10ms | Cached tensors |

### First-Token Latency

**Scenario:** Generate first token in 32-layer model

**Eager Loading:**
```
Time 0s:    GgufLoader::new()
Time 5s:    ExecutionPlan::from_gguf() starts
Time 60s:   All tensors loaded
Time 60.01s: Forward pass starts
Time 60.32s: First token ready (32 layers × 10ms)
```
**Total: ~60.3s to first token**

**Lazy Loading (without preload):**
```
Time 0s:    GgufLoader::new()
Time 5s:    ExecutionPlan::from_gguf() completes
Time 5.01s: Forward pass starts
Time 5.12s: Layer 0 loaded (100ms) + computed (10ms)
Time 5.23s: Layer 1 loaded (100ms) + computed (10ms)
...
Time 8.32s: Layer 31 loaded + computed
Time 8.32s: First token ready
```
**Total: ~8.3s to first token** (7.2x faster)

**Lazy Loading (with preload_all):**
```
Time 0s:    GgufLoader::new()
Time 5s:    ExecutionPlan::from_gguf() completes
Time 5.01s: preload_all() starts
Time 60s:   All layers loaded
Time 60.01s: Forward pass starts
Time 60.32s: First token ready
```
**Total: ~60.3s to first token** (same as eager)

### Memory Usage

**Phase 1 (Already Implemented):**
- RAM: ~5GB (metadata only, 67% reduction)
- GPU VRAM: ~0GB after loader init
- GPU VRAM: ~60GB after `ExecutionPlan::from_gguf()` (eager loading)

**Phase 2 (Proposed):**
- RAM: ~5GB (unchanged)
- GPU VRAM: ~0GB after `ExecutionPlan::from_gguf()` (lazy)
- GPU VRAM: ~2GB after first layer (progressive)
- GPU VRAM: ~60GB after all layers cached (same as eager)

**No Memory Regression:** Peak GPU memory unchanged (~60GB)

### Cache Hit Performance

**Measured in GgufLoader (Phase 1):**
- Cache hit: <1μs (HashMap lookup + Arc clone)
- Cache miss: ~50-200ms (load + dequantize + upload)

**After Phase 2:**
- First pass: All layers cache miss (expected)
- Subsequent passes: All layers cache hit
- Per-token overhead: Negligible (<1μs per tensor)

### Amortization Over Generation

**Scenario:** Generate 100 tokens

**Eager Loading:**
```
Init: 60s (one-time)
Generation: 100 × 10ms = 1s
Total: 61s
```

**Lazy Loading:**
```
Init: 5s
First pass: 32 × 110ms = 3.52s (load + compute)
Remaining 99 passes: 99 × 10ms = 0.99s
Total: 5 + 3.52 + 0.99 = 9.51s
```

**Speedup:** 6.4x faster overall

**Break-even Point:**
- Lazy loading wins if: `init + first_pass < eager_init`
- 5s + 3.52s = 8.52s < 60s ✅
- **Always wins** for generation workloads

---

## Testing Strategy

### Unit Tests

**File:** Create `/home/feanor/Projects/ROCmForge/tests/execution_plan_lazy_tests.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::gguf::GgufLoader;
    use crate::backend::hip_backend::HipBackend;

    #[test]
    fn test_lazy_execution_plan_creation() {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("test.gguf").unwrap();

        // Should complete in <5s
        let start = std::time::Instant::now();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();
        let duration = start.elapsed();

        assert!(duration.as_secs() < 5, "Plan creation took {:?}", duration);

        // Verify lazy tensors are not loaded
        assert!(!plan.embedding_weights_lazy.is_gpu_loaded());
        assert!(!plan.layers[0].qkv_weight.is_gpu_loaded());
    }

    #[test]
    fn test_on_demand_embedding_loading() {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("test.gguf").unwrap();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Embedding not loaded initially
        assert!(!plan.embedding_weights_lazy.is_gpu_loaded());

        // Access triggers loading
        let embedding = plan.get_or_load_embedding().unwrap();

        // Now loaded
        assert!(plan.embedding_weights_lazy.is_gpu_loaded());
        assert_eq!(embedding.len(), plan.config.vocab_size * plan.config.hidden_size);

        // Second access uses cache
        let embedding2 = plan.get_or_load_embedding().unwrap();
        assert_eq!(embedding.len(), embedding2.len());
    }

    #[test]
    fn test_layer_weights_loading() {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("test.gguf").unwrap();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Layer 0 weights not loaded initially
        let layer = &plan.layers[0];
        assert!(!layer.qkv_weight.is_gpu_loaded());
        assert!(!layer.o_proj.is_gpu_loaded());

        // Load layer weights
        let weights = plan.get_or_load_layer_weights(layer).unwrap();

        // Now loaded
        assert!(layer.qkv_weight.is_gpu_loaded());
        assert!(layer.o_proj.is_gpu_loaded());

        // Verify weights are accessible
        assert!(weights.qkv_weight.len() > 0);
        assert!(weights.o_proj.len() > 0);
    }

    #[test]
    fn test_preload_layers() {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("test.gguf").unwrap();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Preload first 5 layers
        plan.preload_layers(&[0, 1, 2, 3, 4]).unwrap();

        // Verify loaded
        for idx in 0..5 {
            assert!(plan.is_layer_loaded(idx));
        }

        // Layer 5 not loaded
        assert!(!plan.is_layer_loaded(5));
    }

    #[test]
    fn test_loading_stats() {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("test.gguf").unwrap();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Initial stats
        let stats = plan.loading_stats();
        assert_eq!(stats.loaded_layers, 0);

        // Load first layer
        plan.preload_layers(&[0]).unwrap();

        let stats = plan.loading_stats();
        assert_eq!(stats.loaded_layers, 1);
        assert!(stats.loaded_tensors > 0);
    }

    #[test]
    fn test_forward_pass_triggers_loading() {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("test.gguf").unwrap();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // No layers loaded
        assert_eq!(plan.loading_stats().loaded_layers, 0);

        // Forward pass triggers loading
        let input_tokens = vec![1, 2, 3];
        let _output = plan.forward(&backend, &input_tokens, &/* dummy */).unwrap();

        // At least first layer loaded
        let stats = plan.loading_stats();
        assert!(stats.loaded_layers >= 1);
    }
}
```

### Integration Tests

**File:** Create `/home/feanor/Projects/ROCmForge/tests/lazy_loading_integration_tests.rs`

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    #[ignore]  // Requires real GGUF model
    fn test_end_to_end_lazy_loading() {
        let model_path = "~/.config/syncore/models/qwen2.5-0.5b.gguf";
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new(model_path).unwrap();

        // Measure init time
        let start = std::time::Instant::now();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();
        let init_time = start.elapsed();

        println!("Init time: {:?}", init_time);
        assert!(init_time.as_secs() < 5);

        // Generate 10 tokens
        let mut tokens = vec![1];  // Start token
        for step in 0..10 {
            let output = plan.forward(&backend, &tokens, &/* dummy */).unwrap();
            let next_token = sample(&output);
            tokens.push(next_token);

            // Check loading progress
            let stats = plan.loading_stats();
            println!("Step {}: Loaded {}/{} layers", step, stats.loaded_layers, stats.total_layers);
        }

        // All layers should be loaded after generation
        let stats = plan.loading_stats();
        assert_eq!(stats.loaded_layers, stats.total_layers);
    }

    #[test]
    #[ignore]  // Requires real GGUF model
    fn test_prompt_vs_generation() {
        let model_path = "~/.config/syncore/models/qwen2.5-0.5b.gguf";
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new(model_path).unwrap();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Prompt phase: preload all
        let preload_start = std::time::Instant::now();
        plan.preload_all().unwrap();
        let preload_time = preload_start.elapsed();

        println!("Preload time: {:?}", preload_time);

        // Process prompt (all layers loaded, no latency)
        let prompt_tokens = tokenize("The quick brown fox");
        for _ in 0..prompt_tokens.len() {
            let _output = plan.forward(&backend, &prompt_tokens, &/* dummy */).unwrap();
        }

        // Generation phase (lazy loading, but already cached)
        let mut token = start_token;
        for _ in 0..10 {
            let output = plan.forward(&backend, &[token], &/* dummy */).unwrap();
            token = sample(&output);
        }
    }
}
```

### Performance Benchmarks

**File:** Create `/home/feanor/Projects/ROCmForge/benches/lazy_loading_bench.rs`

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[bench]
    #[ignore]  // Requires real model
    fn bench_eager_loading(b: &mut test::Bencher) {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("model.gguf").unwrap();

        b.iter(|| {
            let plan = ExecutionPlan::from_gguf_eager(&backend, &loader).unwrap();
            test::black_box(plan);
        });
    }

    #[bench]
    #[ignore]
    fn bench_lazy_loading(b: &mut test::Bencher) {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("model.gguf").unwrap();

        b.iter(|| {
            let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();
            test::black_box(plan);
        });
    }

    #[bench]
    fn bench_cache_hit(b: &mut test::Bencher) {
        let backend = HipBackend::new().unwrap();
        let loader = GgufLoader::new("model.gguf").unwrap();
        let plan = ExecutionPlan::from_gguf(&backend, &loader).unwrap();

        // Preload to ensure cache hits
        plan.preload_all().unwrap();

        b.iter(|| {
            let embedding = plan.get_or_load_embedding().unwrap();
            test::black_box(embedding);
        });
    }
}
```

---

## Rollback Plan

If issues arise during implementation, follow this rollback strategy:

### Phase 1: Verify Compilation

```bash
# Before each commit, verify compilation
cargo check --all-targets
cargo test --no-run
```

### Phase 2: Feature Flag Rollback

**Add feature flag to `Cargo.toml`:**

```toml
[features]
default = []
lazy_execution_plan = []  # New feature flag
```

**Guard new code with feature flag:**

```rust
#[cfg(feature = "lazy_execution_plan")]
impl ExecutionPlan {
    fn get_or_load_embedding(&self) -> HipResult<&DeviceTensor> { /* ... */ }
    // ... other lazy methods ...
}

#[cfg(not(feature = "lazy_execution_plan"))]
impl ExecutionPlan {
    // Keep old eager loading implementation
}
```

**Rollback command:**
```bash
# Disable feature to rollback
cargo build --no-default-features
```

### Phase 3: Git Rollback

```bash
# Before implementing changes
git checkout -b lazy-execution-plan

# During implementation (commit frequently)
git commit -m "Step 1: Update ExecutionPlan struct"
git commit -m "Step 2: Add lazy loading methods"

# If critical bug found, rollback to last working commit
git reset --hard HEAD~1

# Or rollback entire feature branch
git checkout main
git branch -D lazy-execution-plan
```

### Phase 4: Data Migration

**If ExecutionPlan serialization used:**

```rust
// Version the serialized format
#[derive(Serialize, Deserialize)]
pub enum ExecutionPlanV1 {
    Eager { /* ... */ },
    Lazy { /* ... */ },
}

// Migration function
impl ExecutionPlanV1 {
    fn migrate_to_v2(self) -> Result<ExecutionPlanV2> {
        match self {
            Self::Eager { /* convert to lazy */ }
            Self::Lazy { /* already lazy */ }
        }
    }
}
```

---

## References

### Code Sources

1. **ExecutionPlan Implementation**
   - File: `/home/feanor/Projects/ROCmForge/src/model/execution_plan.rs`
   - Lines: 79-2459 (2459 lines total)
   - Key sections:
     - Struct definitions: 79-127
     - Construction: 272-346
     - Forward pass: 360-433
     - Layer forward: 498-572

2. **LazyTensor Infrastructure**
   - File: `/home/feanor/Projects/ROCmForge/src/loader/lazy_tensor.rs`
   - Lines: 48-238 (280 lines total)
   - Key sections:
     - Enum definition: 48-75
     - Thread safety: 77-80
     - Methods: 82-238

3. **GgufLoader On-Demand Loading**
   - File: `/home/feanor/Projects/ROCmForge/src/loader/gguf.rs`
   - Lines: 722-880 (2200 lines total)
   - Key sections:
     - GPU cache: 626
     - load_tensor_to_gpu: 722-880
     - Thread safety: 708-711

### Related Documentation

1. **Phase 1 Implementation Report**
   - File: `/home/feanor/Projects/ROCmForge/docs/PHASE1_LAZY_GGUF_LOADING_IMPLEMENTATION.md`
   - Details LazyTensor infrastructure implementation

2. **Phase 2 Proposal**
   - File: `/home/feanor/Projects/ROCmForge/docs/EXECUTIONPLAN_LAZY_REDESIGN_2026-01-11.md`
   - Original architecture proposal (rejected after code review)

3. **Code Review Findings**
   - File: `/home/feanor/Projects/ROCmForge/docs/CODE_REVIEW_PHASE1_LAZY_LOADING_2026-01-11.md`
   - Issues with Phase 1 integration

### External References

1. **llama.cpp Architecture**
   - Memory-mapped model loading
   - Lazy weight evaluation
   - CPU-first inference path

2. **Progressive Loading Patterns**
   - On-demand tensor loading
   - Caching strategies
   - First-token optimization

---

## Conclusion

This guide provides a complete roadmap for implementing Option A (Lazy ExecutionPlan redesign). The key improvements are:

**Performance:**
- ✅ 12x faster initialization (60s → <5s)
- ✅ 7x faster first token (60s → 8s)
- ✅ No peak memory regression
- ✅ Cache hit performance: <1μs

**Implementation:**
- ✅ Minimal API disruption
- ✅ Thread-safe (RwLock + Arc + OnceCell)
- ✅ Backward compatibility (feature flag)
- ✅ Clear rollback strategy

**Effort Estimate:**
- Step 1-3: 2-3 days (struct updates)
- Step 4-5: 3-4 days (construction and forward pass)
- Step 6: 1-2 days (preloading methods)
- Testing: 2-3 days
- **Total: 8-12 days** (2-3 weeks with buffer)

**Recommendation:** Implement if CPU-First architecture is not viable. The effort is justified for generation-heavy workloads where first-token latency matters.
