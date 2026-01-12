# llama.cpp Logic → Pure Rust Implementation Plan

**Date:** 2026-01-11
**Status:** Planning Phase
**Goal:** Reimplement llama.cpp performance patterns in pure Rust

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dependencies](#2-dependencies)
3. [Phase 1: Lazy GGUF Loading](#3-phase-1-lazy-gguf-loading)
4. [Phase 2: Prompt vs Generation Distinction](#4-phase-2-prompt-vs-generation-distinction)
5. [Phase 3: Memory Pooling](#5-phase-3-memory-pooling)
6. [Phase 4: Computation Graph](#6-phase-4-computation-graph)
7. [Phase 5: SIMD Optimization](#7-phase-5-simd-optimization)
8. [Implementation Order](#8-implementation-order)
9. [Testing Strategy](#9-testing-strategy)

---

## 1. Overview

### Current Problems

| Problem | Current Behavior | Target Behavior |
|---------|-----------------|-----------------|
| Model loading | Load all tensors immediately | Load metadata only, lazy load tensors |
| Memory allocation | Thousands of small hipMalloc calls | Few large pools, sub-allocate |
| Prompt processing | Same path as generation | Batch process prompt tokens |
| Inference loop | Unclear if reusing allocations | Pre-allocated, zero-allocation hot path |
| CPU operations | Minimal SIMD | Vectorized operations |

### API Compatibility Constraints

**PRESERVE all public APIs:**
- `GgufLoader::new(path)`
- `GgufLoader::load_to_gpu(&backend)`
- `InferenceEngine::load_gguf_model(path)`
- `InferenceEngine::submit_request(...)`
- All public methods in these types

**Internal changes only:**
- New internal modules allowed
- Private methods can change
- New traits allowed

---

## 2. Dependencies

### Already in Cargo.toml (verify)

```toml
[dependencies]
memmap2 = "0.9"           # Memory-mapped files ✅
rayon = "1.8"             # Parallelization ✅
tokio = { version = "1", features = ["full"] }  # Async ✅
rand = "0.8"              # Sampling ✅
tracing = "0.1"           # Logging ✅
anyhow = "1.0"            # Errors ✅
thiserror = "1.0"         # Error types ✅
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Need to Add

```toml
[dependencies]
# For SIMD operations (portable, supports x86/ARM)
wide = "0.7"              # OR use std::simd (nightly)

# For LRU cache (lazy tensor loading)
lru = "0.12"              # Optional, can implement our own

# For byte-level parsing
bytemuck = "1.15"         # Safe byte casting

# For atomic operations
atomicring = "1.5"        # Lock-free ring buffer (optional)
```

---

## 3. Phase 1: Lazy GGUF Loading

### Concept

```
Current (slow):
1. Open GGUF file
2. Read ALL tensor metadata
3. Read ALL tensor data into RAM
4. Upload ALL tensors to GPU
5. Start inference

Target (fast):
1. Memory-map GGUF file (zero-copy)
2. Read metadata into structs (tiny, fast)
3. Create tensor "handles" (not loaded yet)
4. Start inference
5. Load tensors on-demand when accessed
6. Cache loaded tensors
```

### Files to Create

#### `src/loader/lazy_tensor.rs`

```rust
//! Lazy-loaded tensor with on-demand fetching

use std::sync::Arc;
use anyhow::Result;

/// Handle for a tensor that may not be loaded yet
pub enum LazyTensor {
    /// Metadata only, data not loaded
    Unloaded {
        name: String,
        offset: u64,
        shape: Vec<usize>,
        dtype: GgufDtype,
        nbytes: usize,
    },
    /// Data loaded into CPU memory
    LoadedCpu {
        name: String,
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: GgufDtype,
    },
    /// Data uploaded to GPU
    LoadedGpu {
        name: String,
        gpu_tensor: crate::backend::DeviceTensor,
        shape: Vec<usize>,
        dtype: GgufDtype,
    },
}

impl LazyTensor {
    /// Create unloaded tensor handle
    pub fn unloaded(
        name: String,
        offset: u64,
        shape: Vec<usize>,
        dtype: GgufDtype,
    ) -> Self {
        let nbytes = shape.iter().product::<usize>() * dtype.size();
        Self::Unloaded { name, offset, shape, dtype, nbytes }
    }

    /// Get tensor name
    pub fn name(&self) -> &str {
        match self {
            Self::Unloaded { name, .. } => name,
            Self::LoadedCpu { name, .. } => name,
            Self::LoadedGpu { name, .. } => name,
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Unloaded { shape, .. } => shape,
            Self::LoadedCpu { shape, .. } => shape,
            Self::LoadedGpu { shape, .. } => shape,
        }
    }

    /// Check if tensor is loaded to GPU
    pub fn is_gpu_loaded(&self) -> bool {
        matches!(self, Self::LoadedGpu { .. })
    }
}

/// Data type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Mxfp4,
    Mxfp6,
}

impl GgufDtype {
    pub fn size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 | Self::Q4_1 => 4, // block size
            Self::Q5_0 | Self::Q5_1 => 4,
            Self::Q8_0 => 4,
            _ => 4, // simplified
        }
    }
}
```

#### `src/loader/mmap_gguf.rs`

```rust
//! Memory-mapped GGUF file access

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use anyhow::Result;

/// Memory-mapped GGUF file for zero-copy access
pub struct MmapGguf {
    file: File,
    mmap: Mmap,
    metadata: GgufMetadata,
}

impl MmapGguf {
    /// Open GGUF file with memory mapping
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse metadata only (not tensor data)
        let metadata = Self::parse_metadata(&mmap)?;

        Ok(Self { file, mmap, metadata })
    }

    /// Get tensor data bytes without copying
    pub fn get_tensor_bytes(&self, offset: u64, size: usize) -> Result<&[u8]> {
        let start = offset as usize;
        let end = start + size;

        if end > self.mmap.len() {
            anyhow::bail!("Tensor data exceeds file bounds");
        }

        Ok(&self.mmap[start..end])
    }

    /// Get metadata
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }
}

/// GGUF metadata (simplified)
#[derive(Debug)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub tensors: Vec<TensorMetadata>,
}

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgufDtype,
    pub offset: u64,
}
```

### Files to Modify

#### `src/loader/gguf.rs`

**Changes:**
1. Replace immediate tensor loading with lazy handles
2. Add method to load individual tensors on-demand
3. Cache loaded tensors

```rust
// Add to existing GgufLoader struct

use crate::loader::lazy_tensor::{LazyTensor, GgufDtype};
use crate::loader::mmap_gguf::MmapGguf;
use std::collections::HashMap;
use std::sync::RwLock;

pub struct GgufLoader {
    path: String,
    // OLD: tensors: HashMap<String, GgufTensor>,
    // NEW:
    mmap: Option<MmapGguf>,
    lazy_tensors: HashMap<String, LazyTensor>,
    gpu_cache: RwLock<HashMap<String, crate::backend::DeviceTensor>>,
}

impl GgufLoader {
    /// NEW: Load metadata only (fast!)
    pub fn new_lazy(path: &str) -> Result<Self> {
        let mmap = MmapGguf::open(Path::new(path))?;
        let metadata = mmap.metadata();

        let mut lazy_tensors = HashMap::new();
        for tensor_meta in &metadata.tensors {
            lazy_tensors.insert(
                tensor_meta.name.clone(),
                LazyTensor::unloaded(
                    tensor_meta.name.clone(),
                    tensor_meta.offset,
                    tensor_meta.shape.clone(),
                    tensor_meta.dtype,
                ),
            );
        }

        Ok(Self {
            path: path.to_string(),
            mmap: Some(mmap),
            lazy_tensors,
            gpu_cache: RwLock::new(HashMap::new()),
        })
    }

    /// NEW: Load single tensor to GPU on-demand
    pub fn load_tensor_to_gpu(
        &self,
        name: &str,
        backend: &crate::backend::HipBackend,
    ) -> Result<crate::backend::DeviceTensor> {
        // Check cache first
        {
            let cache = self.gpu_cache.read().unwrap();
            if let Some(tensor) = cache.get(name) {
                return Ok(tensor.clone());
            }
        }

        // Load from memory map
        let lazy = self.lazy_tensors.get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;

        let (offset, shape, dtype) = match lazy {
            LazyTensor::Unloaded { offset, shape, dtype, .. } => {
                (*offset, shape.clone(), *dtype)
            }
            _ => return Err(anyhow::anyhow!("Tensor already loaded: {}", name)),
        };

        let mmap = self.mmap.as_ref().unwrap();
        let bytes = mmap.get_tensor_bytes(offset, shape.iter().product::<usize>() * dtype.size())?;

        // Upload to GPU
        let gpu_tensor = self.upload_to_gpu(backend, bytes, shape, dtype)?;

        // Cache it
        {
            let mut cache = self.gpu_cache.write().unwrap();
            cache.insert(name.to_string(), gpu_tensor.clone());
        }

        Ok(gpu_tensor)
    }
}

// PRESERVE EXISTING API
impl GgufLoader {
    pub fn new(path: &str) -> Result<Self> {
        // Keep old behavior for compatibility
        Self::new_lazy(path)
    }

    pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
        // Trigger loading of all tensors (for backward compatibility)
        let mut result = HashMap::new();
        for name in self.lazy_tensors.keys() {
            let tensor = self.load_tensor_to_gpu(name, backend)?;
            result.insert(name.clone(), tensor);
        }
        Ok(result)
    }
}
```

### Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_loading_metadata_only() {
        let loader = GgufLoader::new_lazy("path/to/model.gguf").unwrap();
        // Should complete instantly (< 10ms)
        assert!(!loader.lazy_tensors.is_empty());
    }

    #[test]
    fn test_on_demand_tensor_load() {
        let loader = GgufLoader::new_lazy("path/to/model.gguf").unwrap();
        let backend = HipBackend::new()?;

        // Tensor not loaded yet
        assert!(!loader.gpu_cache.read().unwrap().contains_key("blk.0.attn_q.weight"));

        // Load on demand
        let tensor = loader.load_tensor_to_gpu("blk.0.attn_q.weight", &backend).unwrap();

        // Now cached
        assert!(loader.gpu_cache.read().unwrap().contains_key("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_zero_copy_access() {
        let mmap = MmapGguf::open(Path::new("path/to/model.gguf")).unwrap();
        let bytes = mmap.get_tensor_bytes(offset, size).unwrap();
        // This is a slice into the mmap, no allocation
    }
}
```

### Complexity: Medium
- Risk: Low (internal change only)
- Time: 2-3 days
- Files: 2 new, 1 modify

---

## 4. Phase 2: Prompt vs Generation Distinction

### Concept

```
Prompt Processing (Batch):
- Input: [token1, token2, ..., tokenN]
- Process: All N tokens in parallel
- KV Cache: Generate initial K,V for all tokens
- No causal mask between prompt tokens
- Vectorize across tokens for speed

Generation (Sequential):
- Input: last_token only
- Process: Single token
- KV Cache: Append new K,V to existing cache
- Full causal mask
- Repeat for each generated token
```

### Files to Create

#### `src/engine/phase_executor.rs`

```rust
//! Phase-aware executor: prompt vs generation

use anyhow::Result;

/// Execution phase determines how to process tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPhase {
    /// Initial prompt processing (batch mode)
    Prompt { token_count: usize },
    /// Token generation (sequential mode)
    Generation,
}

/// Phase-aware execution config
pub struct PhaseConfig {
    pub phase: ExecutionPhase,
    pub batch_size: usize,
}

impl PhaseConfig {
    /// Create prompt processing config
    pub fn prompt(token_count: usize) -> Self {
        Self {
            phase: ExecutionPhase::Prompt { token_count },
            batch_size: token_count,
        }
    }

    /// Create generation config
    pub fn generation() -> Self {
        Self {
            phase: ExecutionPhase::Generation,
            batch_size: 1,
        }
    }

    /// Check if in prompt phase
    pub fn is_prompt(&self) -> bool {
        matches!(self.phase, ExecutionPhase::Prompt { .. })
    }

    /// Check if in generation phase
    pub fn is_generation(&self) -> bool {
        matches!(self.phase, ExecutionPhase::Generation)
    }
}

/// Execute based on phase
pub trait PhaseExecutor {
    /// Execute transformer layers for given phase
    fn execute_phase(
        &mut self,
        input_tokens: &[u32],
        config: &PhaseConfig,
    ) -> Result<Vec<f32>>;  // Returns logits

    /// Process prompt in batch
    fn process_prompt(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let config = PhaseConfig::prompt(tokens.len());
        self.execute_phase(tokens, &config)
    }

    /// Generate single token
    fn generate_token(&mut self, last_token: u32) -> Result<Vec<f32>> {
        let config = PhaseConfig::generation();
        self.execute_phase(&[last_token], &config)
    }
}
```

### Files to Modify

#### `src/engine.rs`

**Changes:**
1. Add phase tracking to inference loop
2. Switch from prompt to generation after first call
3. Use appropriate execution path

```rust
// Add to InferenceEngine struct

use crate::engine::phase_executor::{ExecutionPhase, PhaseConfig, PhaseExecutor};

pub struct InferenceEngine {
    // ... existing fields ...

    // NEW: Track execution phase
    phase: ExecutionPhase,
    prompt_tokens_processed: usize,
}

impl InferenceEngine {
    // NEW: Phase-aware processing
    async fn process_tokens_phase_aware(
        &self,
        tokens: &[u32],
    ) -> EngineResult<Vec<f32>> {
        let phase = if self.prompt_tokens_processed == 0 {
            ExecutionPhase::Prompt { token_count: tokens.len() }
        } else {
            ExecutionPhase::Generation
        };

        match phase {
            ExecutionPhase::Prompt { .. } => {
                // Batch process all prompt tokens
                self.process_batch_prompt(tokens).await
            }
            ExecutionPhase::Generation => {
                // Process single token with full KV cache
                self.process_single_token(tokens[0]).await
            }
        }
    }

    // NEW: Batch prompt processing
    async fn process_batch_prompt(&self, tokens: &[u32]) -> EngineResult<Vec<f32>> {
        // Process all tokens in parallel
        // No causal mask between prompt tokens
        // Use rayon for CPU parallelization

        use rayon::prelude::*;

        let results: Vec<_> = tokens.par_iter()
            .map(|&token| self.process_token_batched(token))
            .collect();

        Ok(results)
    }

    // NEW: Single token generation
    async fn process_single_token(&self, token: u32) -> EngineResult<Vec<f32>> {
        // Process with full KV cache
        // Causal mask applies
        self.run_transformer_with_cache(token).await
    }
}
```

### Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_detection() {
        let config = PhaseConfig::prompt(10);
        assert!(config.is_prompt());
        assert!(!config.is_generation());
    }

    #[test]
    fn test_prompt_vs_generation_paths() {
        // Verify different code paths are used
        let engine = setup_test_engine();

        // First call should use prompt path
        let logits1 = engine.process_tokens_phase_aware(&[1, 2, 3]).await.unwrap();

        // Subsequent calls should use generation path
        let logits2 = engine.process_tokens_phase_aware(&[4]).await.unwrap();
    }
}
```

### Complexity: Medium
- Risk: Medium (affects core inference logic)
- Time: 3-4 days
- Files: 1 new, 1 modify

---

## 5. Phase 3: Memory Pooling

### Concept

```
Current Problem:
- Each tensor gets separate hipMalloc call
- Thousands of allocations
- ROCm driver issues with many small allocations

Solution:
- Allocate 3-4 large pools (1GB each) at startup
- Sub-allocate from pools
- Align to 4KB boundaries
- Reuse freed allocations
```

### Files to Modify

#### `src/backend/hip_backend.rs`

**Add memory pool implementation:**

```rust
//! GPU memory pool for sub-allocation

use std::sync::{Arc, Mutex};
use anyhow::Result;

/// Memory pool for GPU allocations
pub struct GpuMemoryPool {
    pools: Vec<Arc<HipBuffer>>,  // Large parent buffers
    free_list: Mutex<Vec<FreeBlock>>,
    alignment: usize,
}

struct FreeBlock {
    pool_index: usize,
    offset: usize,
    size: usize,
}

impl GpuMemoryPool {
    /// Create new memory pool
    pub fn new(
        backend: &HipBackend,
        pool_size: usize,
        num_pools: usize,
        alignment: usize,
    ) -> Result<Self> {
        let mut pools = Vec::with_capacity(num_pools);

        for _ in 0..num_pools {
            let buffer = HipBuffer::new(pool_size)?;
            pools.push(Arc::new(buffer));
        }

        Ok(Self {
            pools,
            free_list: Mutex::new(Vec::new()),
            alignment,
        })
    }

    /// Allocate from pool (or fallback to direct allocation)
    pub fn allocate(&self, size: usize, backend: &HipBackend) -> Result<HipBuffer> {
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);

        // Try to find free block
        {
            let mut free = self.free_list.lock().unwrap();
            if let Some(idx) = free.iter().position(|b| b.size >= aligned_size) {
                let block = free.remove(idx);
                return Ok(self.sub_buffer(block.pool_index, block.offset, aligned_size));
            }
        }

        // Try to allocate from pool
        for (i, pool) in self.pools.iter().enumerate() {
            if pool.size() >= aligned_size {
                return Ok(self.sub_buffer(i, 0, aligned_size));
            }
        }

        // Fallback: direct allocation (for large tensors)
        HipBuffer::new(size)
    }

    /// Return allocation to free list
    pub fn free(&self, buffer: HipBuffer) {
        // Return to free list for reuse
        // Implementation depends on HipBuffer refactoring
    }

    fn sub_buffer(&self, pool_index: usize, offset: usize, size: usize) -> HipBuffer {
        // Create view into parent pool buffer
        // Implementation depends on HipBuffer supporting sub-buffers
        HipBuffer {
            inner: Arc::new(HipBufferInner {
                ptr: self.pools[pool_index].as_ptr(),
                size,
                offset,
            }),
        }
    }
}

/// Global memory pool instance
static GLOBAL_POOL: Mutex<Option<GpuMemoryPool>> = Mutex::new(None);

/// Initialize global memory pool
pub fn init_memory_pool(
    backend: &HipBackend,
    pool_size: usize,
    num_pools: usize,
) -> Result<()> {
    let pool = GpuMemoryPool::new(backend, pool_size, num_pools, 4096)?;
    *GLOBAL_POOL.lock().unwrap() = Some(pool);
    Ok(())
}

/// Allocate from global pool
pub fn pooled_allocate(size: usize, backend: &HipBackend) -> Result<HipBuffer> {
    let pool = GLOBAL_POOL.lock().unwrap();
    match pool.as_ref() {
        Some(p) => p.allocate(size, backend),
        None => HipBuffer::new(size),  // Fallback
    }
}
```

**Modify existing `allocate_buffer`:**

```rust
impl HipBackend {
    /// OLD: Direct allocation
    pub fn allocate_buffer(&self, size: usize) -> HipResult<HipBuffer> {
        HipBuffer::new(size)
    }

    /// NEW: Pooled allocation (preserving API)
    pub fn allocate_buffer_pooled(&self, size: usize) -> HipResult<HipBuffer> {
        pooled_allocate(size, self)
            .map_err(|e| HipError::DeviceError(e.to_string()))
    }
}
```

### Files to Modify

#### `src/loader/gguf.rs`

**Use pooled allocation:**

```rust
impl GgufLoader {
    pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
        // Initialize memory pool if not already
        init_memory_pool(backend, 1024 * 1024 * 1024, 3)?;

        let mut result = HashMap::new();

        for (name, lazy) in &self.lazy_tensors {
            let tensor_bytes = lazy.nbytes();

            // Use pooled allocation instead of direct
            let gpu_buffer = backend.allocate_buffer_pooled(tensor_bytes)?;

            // Copy data and create tensor
            let tensor = DeviceTensor::from_buffer(gpu_buffer, lazy.shape().clone())?;
            result.insert(name.clone(), tensor);
        }

        Ok(result)
    }
}
```

### Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_init() {
        let backend = HipBackend::new()?;
        init_memory_pool(&backend, 1024 * 1024 * 1024, 3).unwrap();
    }

    #[test]
    fn test_pooled_allocation() {
        let backend = HipBackend::new()?;
        let buffer = backend.allocate_buffer_pooled(1024)?;
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_pool_vs_direct() {
        // Compare speed of pooled vs direct
        let backend = HipBackend::new()?;

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = backend.allocate_buffer_pooled(4096)?;
        }
        let pooled_time = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = HipBuffer::new(4096)?;
        }
        let direct_time = start.elapsed();

        assert!(pooled_time < direct_time);
    }
}
```

### Complexity: High
- Risk: High (affects memory management)
- Time: 4-5 days
- Files: 0 new, 2 modify

---

## 6. Phase 4: Computation Graph

### Concept

```
Idea: Build execution graph once, execute many times

Graph for single token:
input → embedding → [N x transformer layers] → lm_head → logits → sample

At initialization:
1. Create graph structure
2. Allocate all needed buffers
3. Wire up operations

At inference:
1. Just update input buffer
2. Execute graph (no allocations!)
3. Read output
```

### Files to Create

#### `src/graph/graph.rs`

```rust
//! Computation graph for zero-allocation inference

use std::sync::Arc;
use anyhow::Result;

/// Operation in computation graph
pub enum GraphOp {
    /// Load input tokens
    Input { buffer_index: usize },

    /// Embedding lookup
    Embedding {
        input_buffer: usize,
        embedding_table: usize,
        output_buffer: usize,
    },

    /// Transformer layer
    TransformerLayer {
        layer_index: usize,
        input_buffer: usize,
        kv_cache_index: usize,
        output_buffer: usize,
    },

    /// Final layer (lm_head)
    Logits {
        input_buffer: usize,
        output_buffer: usize,
    },
}

/// Computation graph
pub struct ComputationGraph {
    ops: Vec<GraphOp>,
    buffers: Vec<GraphBuffer>,
    kv_cache_indices: Vec<usize>,
}

/// Buffer in graph
pub struct GraphBuffer {
    size: usize,
    dtype: BufferDtype,
    data: Option<Vec<u8>>,  // CPU buffer
    gpu_ptr: Option<hipDeviceptr_t>,  // GPU pointer
}

#[derive(Clone, Copy)]
pub enum BufferDtype {
    F32,
    F16,
    I32,
}

impl ComputationGraph {
    /// Create new graph for single-token generation
    pub fn new(num_layers: usize, hidden_size: usize) -> Self {
        let mut ops = Vec::new();
        let mut buffers = Vec::new();

        // Allocate buffers
        // Input buffer (token IDs)
        buffers.push(GraphBuffer {
            size: std::mem::size_of::<u32>(),
            dtype: BufferDtype::I32,
            data: None,
            gpu_ptr: None,
        });

        // Hidden state buffer
        buffers.push(GraphBuffer {
            size: hidden_size * std::mem::size_of::<f32>(),
            dtype: BufferDtype::F32,
            data: None,
            gpu_ptr: None,
        });

        // Build operations for each layer
        for layer_idx in 0..num_layers {
            ops.push(GraphOp::TransformerLayer {
                layer_index: layer_idx,
                input_buffer: 1,  // hidden state
                kv_cache_index: layer_idx,
                output_buffer: 1,  // in-place
            });
        }

        // Logits
        ops.push(GraphOp::Logits {
            input_buffer: 1,
            output_buffer: 2,  // logits buffer
        });

        Self {
            ops,
            buffers,
            kv_cache_indices: (0..num_layers).collect(),
        }
    }

    /// Execute the graph
    pub fn execute(&mut self, input_token: u32, backend: &HipBackend) -> Result<Vec<f32>> {
        // Update input buffer
        self.buffers[0].data = Some(input_token.to_le_bytes().to_vec());

        // Execute operations
        for op in &self.ops {
            self.execute_op(op, backend)?;
        }

        // Read output from logits buffer
        self.read_buffer(2)
    }

    fn execute_op(&self, op: &GraphOp, backend: &HipBackend) -> Result<()> {
        match op {
            GraphOp::Input { buffer_index } => {
                // Copy input to GPU
                let buffer = &self.buffers[*buffer_index];
                if let Some(data) = &buffer.data {
                    // Upload to GPU
                }
            }
            GraphOp::TransformerLayer { layer_index, input_buffer, kv_cache_index, output_buffer } => {
                // Run transformer layer
                self.run_transformer_layer(*layer_index, *input_buffer, *kv_cache_index, *output_buffer, backend)?;
            }
            GraphOp::Logits { input_buffer, output_buffer } => {
                // Compute logits from hidden state
                self.compute_logits(*input_buffer, *output_buffer, backend)?;
            }
        }
        Ok(())
    }

    fn run_transformer_layer(
        &self,
        layer_idx: usize,
        input_buf: usize,
        kv_cache_idx: usize,
        output_buf: usize,
        backend: &HipBackend,
    ) -> Result<()> {
        // Implementation uses existing transformer logic
        // But reuses pre-allocated buffers
        Ok(())
    }

    fn compute_logits(&self, input_buf: usize, output_buf: usize, backend: &HipBackend) -> Result<()> {
        // Implementation uses existing lm_head logic
        Ok(())
    }

    fn read_buffer(&self, buffer_index: usize) -> Result<Vec<f32>> {
        // Read buffer from GPU
        Ok(vec![])
    }
}
```

### Files to Modify

#### `src/engine.rs`

**Integrate graph:**

```rust
use crate::graph::graph::ComputationGraph;

pub struct InferenceEngine {
    // ... existing fields ...

    // NEW: Computation graph
    graph: Option<ComputationGraph>,
}

impl InferenceEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        // ... existing init ...

        Ok(Self {
            // ... existing fields ...
            graph: None,  // Will create after model load
        })
    }

    pub async fn load_gguf_model(&mut self, path: &str) -> Result<()> {
        // ... existing model loading ...

        // NEW: Create computation graph after model is loaded
        let num_layers = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?
            .num_layers();

        let hidden_size = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?
            .hidden_size();

        self.graph = Some(ComputationGraph::new(num_layers, hidden_size));

        Ok(())
    }

    // NEW: Use graph for generation
    async fn generate_with_graph(&mut self, prompt_tokens: Vec<u32>) -> Result<String> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No graph"))?;

        // Process prompt (batch mode, not using graph yet)
        let mut current_state = self.process_prompt_batch(&prompt_tokens).await?;

        // Generate tokens using graph
        let mut generated = Vec::new();

        for _ in 0..self.config.max_tokens {
            let logits = graph.execute(current_state.last_token.unwrap(), &self.backend)?;
            let next_token = self.sample_token(&logits)?;

            if self.is_eos(next_token) {
                break;
            }

            generated.push(next_token);
            current_state.last_token = Some(next_token);
        }

        self.decode_tokens(&generated)
    }
}
```

### Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = ComputationGraph::new(24, 768);  // GPT-2 like
        assert_eq!(graph.buffers.len(), 3);  // input, hidden, logits
    }

    #[test]
    fn test_graph_execution() {
        let mut graph = ComputationGraph::new(24, 768);
        // Mock execution
        // Verify no allocations during execute()
    }
}
```

### Complexity: High
- Risk: Medium (new subsystem)
- Time: 5-7 days
- Files: 1 new, 1 modify

---

## 7. Phase 5: SIMD Optimization

### Concept

```
Scalar: Process one value at a time
for i in 0..N:
    result[i] = a[i] + b[i]

SIMD: Process 8/16/32 values simultaneously
for i in (0..N).step_by(8):
    vec = load8(a[i..i+8]) + load8(b[i..i+8])
    store8(result[i..i+8], vec)

Speedup: 8-16x for simple operations
```

### Files to Create

#### `src/cpu/simd_ops.rs`

```rust
//! SIMD-optimized CPU operations

use wide::{f32x8, f32x16};

/// SIMD-accelerated vector addition
pub fn vec_add_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    let chunks = a.len() / 8;

    // Process 8 values at a time
    for i in 0..chunks {
        let offset = i * 8;
        let a_vec = f32x8::from_slice(&a[offset..offset+8]);
        let b_vec = f32x8::from_slice(&b[offset..offset+8]);
        let c_vec = a_vec + b_vec;
        c_vec.copy_to_slice(&mut c[offset..offset+8]);
    }

    // Handle remainder
    let rem_start = chunks * 8;
    for i in rem_start..a.len() {
        c[i] = a[i] + b[i];
    }
}

/// SIMD-accelerated softmax
pub fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    // Find max using SIMD
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exp and sum
    let mut exp_sum = 0.0_f32;
    let mut results = Vec::with_capacity(logits.len());

    for &logit in logits {
        let exp = (logit - max).exp();
        results.push(exp);
        exp_sum += exp;
    }

    // Normalize
    for result in &mut results {
        *result /= exp_sum;
    }

    results
}

/// SIMD-accelerated dot product
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0.0_f32;
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let a_vec = f32x8::from_slice(&a[offset..offset+8]);
        let b_vec = f32x8::from_slice(&b[offset..offset+8]);
        sum += (a_vec * b_vec).sum();
    }

    // Handle remainder
    let rem_start = chunks * 8;
    for i in rem_start..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

/// SIMD-accelerated matrix multiplication (small matrices)
pub fn matmul_f32(
    a: &[f32],  // M x K
    b: &[f32],  // K x N
    c: &mut [f32],  // M x N
    m: usize,
    k: usize,
    n: usize,
) {
    // Simple implementation
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += unsafe { *a.get_unchecked(i * k + kk) }
                      * unsafe { *b.get_unchecked(kk * n + j) };
            }
            c[i * n + j] = sum;
        }
    }
}
```

### Files to Modify

#### `src/mlp/rms_norm_tests.rs` (and other CPU ops)

**Add SIMD variants:**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::simd_ops::*;

    #[test]
    fn test_simd_vec_add() {
        let a = vec![1.0f32; 100];
        let b = vec![2.0f32; 100];
        let mut c = vec![0.0f32; 100];

        vec_add_f32(&a, &b, &mut c);

        assert_eq!(c, vec![3.0f32; 100]);
    }

    #[test]
    fn test_simd_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_f32(&logits);

        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let dot = dot_product_f32(&a, &b);

        assert_eq!(dot, 1.0*2.0 + 2.0*3.0 + 3.0*4.0 + 4.0*5.0);
    }
}
```

### Testing Strategy

```rust
#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;

    #[bench]
    fn bench_scalar_add(b: &mut test::Bencher) {
        let a = vec![1.0f32; 1024];
        let b = vec![2.0f32; 1024];
        let mut c = vec![0.0f32; 1024];

        b.iter(|| {
            for i in 0..1024 {
                c[i] = a[i] + b[i];
            }
        });
    }

    #[bench]
    fn bench_simd_add(b: &mut test::Bencher) {
        let a = vec![1.0f32; 1024];
        let b = vec![2.0f32; 1024];
        let mut c = vec![0.0f32; 1024];

        b.iter(|| {
            vec_add_f32(&a, &b, &mut c);
        });
    }
}
```

### Complexity: Medium
- Risk: Low (additive optimization)
- Time: 2-3 days
- Files: 1 new, 3 modify

---

## 8. Implementation Order

### Recommended Sequence

```
Week 1-2: Phase 1 (Lazy Loading)
├── Create lazy_tensor.rs
├── Create mmap_gguf.rs
├── Modify gguf.rs for lazy loading
└── Tests for lazy loading

Week 3: Phase 2 (Prompt vs Generation)
├── Create phase_executor.rs
├── Modify engine.rs for phase tracking
└── Tests for phase detection

Week 4: Phase 3 (Memory Pooling)
├── Add memory pool to hip_backend.rs
├── Modify gguf.rs to use pooled allocation
└── Tests for pool allocation

Week 5-6: Phase 4 (Computation Graph)
├── Create graph/graph.rs
├── Create graph/buffer.rs
├── Modify engine.rs to use graph
└── Tests for graph execution

Week 7: Phase 5 (SIMD)
├── Create cpu/simd_ops.rs
├── Add SIMD variants to existing ops
└── Benchmarks for speedup

Week 8: Integration & Testing
├── End-to-end testing
├── Performance benchmarking
└── Documentation
```

### Dependencies Between Phases

```
Phase 1 (Lazy Loading)
    ↓
Phase 2 (Prompt vs Generation)  ← Can start in parallel
    ↓
Phase 3 (Memory Pooling)         ← Depends on Phase 1
    ↓
Phase 4 (Computation Graph)      ← Depends on Phase 1,2
    ↓
Phase 5 (SIMD)                    ← Independent, can do anytime
```

---

## 9. Testing Strategy

### Unit Tests per Phase

```rust
// Phase 1: Lazy Loading
test_lazy_metadata_only()
test_on_demand_tensor_load()
test_mmap_zero_copy()
test_tensor_cache_hit()

// Phase 2: Phase Executor
test_phase_detection()
test_prompt_batch_processing()
test_generation_sequential()
test_phase_switching()

// Phase 3: Memory Pool
test_pool_allocation()
test_pool_free()
test_alignment()
test_pool_exhaustion()

// Phase 4: Computation Graph
test_graph_creation()
test_graph_execution()
test_graph_reuse()
test_zero_allocation_during_execute()

// Phase 5: SIMD
test_simd_correctness()
test_simd_vs_scalar()
test_simd_edge_cases()
```

### Integration Tests

```rust
// End-to-end tests
test_full_inference_with_lazy_loading()
test_prompt_then_generation()
test_memory_pool_stress()
test_repeated_generation()
```

### Performance Benchmarks

```rust
// Measure improvements
bench_model_loading_time()
bench_prompt_processing_speed()
bench_generation_tokens_per_second()
bench_memory_allocation_count()
bench_cache_hit_rate()
```

---

## 10. Risk Mitigation

### High Risk Items

| Risk | Mitigation |
|------|------------|
| Breaking existing API | Preserve all public signatures, internal changes only |
| Memory pool bugs | Fallback to direct allocation if pool fails |
| SIMD correctness | Extensive testing, compare scalar vs SIMD results |
| Graph complexity | Incremental rollout, keep old path as fallback |

### Rollback Strategy

```rust
// Feature flags for gradual rollout
[features]
default = []
lazy_loading = []
phase_executor = ["lazy_loading"]
memory_pool = ["lazy_loading"]
computation_graph = ["lazy_loading", "phase_executor"]
simd_opt = []

// Enable selectively during development
// cargo test --features lazy_loading
```

---

## 11. Success Criteria

### Metrics to Track

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Model load time | >60s | <5s | 10x improvement |
| hipMalloc calls | ~1000 | <100 | 10x reduction |
| Prompt processing (100 tokens) | ? | <100ms | Competitive |
| Generation speed | ? | >20 tok/s | Usable |
| Memory overhead | Baseline | +10% | Minimal increase |

### Definition of Done

- [ ] All unit tests passing
- [ ] No regression in existing functionality
- [ ] Performance benchmarks show improvement
- [ ] Documentation updated
- [ ] No compiler warnings (new code)
- [ ] Code review completed

---

## Summary

This plan provides a clear path to implementing llama.cpp's performance patterns in pure Rust:

1. **Lazy Loading** - Fast model startup, on-demand tensor access
2. **Phase Executor** - Distinguish prompt vs generation paths
3. **Memory Pooling** - Reduce allocations, reuse memory
4. **Computation Graph** - Build once, execute many times
5. **SIMD** - Vectorized CPU operations

All changes preserve the existing public API and use safe Rust code with proven crates.
