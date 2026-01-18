# Plan 03-03: Split gguf.rs into Focused Modules

**Phase**: 03 - Codebase Modularization
**Status**: Pending
**Complexity**: High
**Estimated Time**: 4-5 hours

---

## Problem Statement

`src/loader/gguf.rs` is 2832 lines - far exceeding the 300 LOC convention. This monolithic file contains multiple distinct concerns:
- GGUF format types (E8M0, MXFP blocks)
- Tensor type definitions
- Metadata structures
- File parsing and validation
- Quantization/dequantization
- Tensor loading
- GPU memory upload

**Current State**:
- 2832 lines in a single file
- Mix of format definitions, parsing logic, and GPU operations
- Difficult to locate specific quantization formats

---

## Analysis

### File Structure Breakdown

Based on code analysis, the file contains these distinct components:

1. **MXFP Format Types** (~400 lines)
   - `E8M0` - exponent-only format
   - `MxfpBlock` - block-scaled floating point
   - `pack_mxfp4()`, `unpack_mxfp4()`
   - `pack_mxfp6()`, `unpack_mxfp6()`
   - E2M1/E2M3 encoding helpers

2. **Tensor Type Enum** (~200 lines)
   - `GgufTensorType` enum (F16, F32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
   - Type size calculation
   - String representation

3. **F16 Helper** (~50 lines)
   - `F16` struct for half-precision float
   - Conversion to/from f32

4. **GgufMetadata** (~300 lines)
   - Model metadata (architecture, vocab size, etc.)
   - KV cache parameters
   - String/value parsing

5. **GgufTensor** (~200 lines)
   - Tensor descriptor (name, shape, type, offset)
   - Block reading helpers

6. **GgufLoader** Main Struct (~150 lines)
   - File handle, memory mapping
   - Tensor registry
   - GPU cache

7. **File Parsing** (~400 lines)
   - `GgufLoader::new()` - file opening and validation
   - Magic number checking
   - Header parsing
   - Tensor info reading

8. **Quantization/Dequantization** (~600 lines)
   - `dequant_q4_0()`, `dequant_q4_1()`, etc.
   - Block format handling
   - Parallel dequantization (Rayon)

9. **Tensor Loading** (~400 lines)
   - `load_tensor()` - CPU-side loading
   - `load_tensor_to_device()` - GPU upload
   - Lazy tensor integration

10. **Helper Methods** (~132 lines)
    - Metadata accessors
    - Tensor lookup
    - Statistics

---

## Implementation Plan

### Target Structure

```
src/loader/
├── mod.rs              # Public exports
├── gguf.rs             # Main GgufLoader struct (simplified)
├── mxfp.rs             # E8M0, MxfpBlock, MXFP encoding/decoding
├── tensor_type.rs      # GgufTensorType enum
├── metadata.rs         # GgufMetadata struct
├── gguf_tensor.rs      # GgufTensor struct
└── dequant.rs          # Quantization/dequantization functions
```

### Task 1: Create Module Files

```bash
touch src/loader/{mxfp.rs,tensor_type.rs,metadata.rs,gguf_tensor.rs,dequant.rs}
```

### Task 2: Extract MXFP Format Types

**File**: `src/loader/mxfp.rs`

```rust
//! MXFP (block-scaled floating-point) format support
//!
//! Per OCP MX Specification v1.0:
//! - Block size: 32 elements
//! - Scale: E8M0 (1 byte)
//! - Elements: packed 4-bit or 6-bit values

/// E8M0 scale format (8-bit exponent only)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct E8M0 {
    pub exponent: i8,
}

impl E8M0 {
    pub fn to_f32(&self) -> f32 { ... }
    pub fn from_f32(value: f32) -> Self { ... }
}

/// MXFP block (block-scaled floating-point)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MxfpBlock {
    pub scale: E8M0,
    pub elements: Vec<u8>,
}

impl MxfpBlock {
    pub fn new_mxfp4() -> Self { ... }
    pub fn new_mxfp6() -> Self { ... }
    pub fn pack_mxfp4(values: &[f32]) -> Self { ... }
    pub fn unpack_mxfp4(&self) -> Vec<f32> { ... }
    pub fn pack_mxfp6(values: &[f32]) -> Self { ... }
    pub fn unpack_mxfp6(&self) -> Vec<f32> { ... }

    // E2M1/E2M3 encoding helpers
    fn encode_e2m1(value: f32) -> u8 { ... }
    fn decode_e2m1(encoded: u8) -> f32 { ... }
    fn encode_e2m3(value: f32) -> u8 { ... }
    fn decode_e2m3(encoded: u8) -> f32 { ... }
    fn pack_6bit_values(values: &[u8]) -> Vec<u8> { ... }
}
```

### Task 3: Extract Tensor Type Enum

**File**: `src/loader/tensor_type.rs`

```rust
//! GGUF tensor type definitions

use serde::Serialize;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    // ... other types
}

impl GgufTensorType {
    /// Get the type size for this tensor type
    pub fn type_size(&self) -> usize { ... }

    /// Get the block size (for block-quantized types)
    pub fn block_size(&self) -> usize { ... }

    /// Check if this is a quantized type
    pub fn is_quantized(&self) -> bool { ... }
}

impl fmt::Display for GgufTensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { ... }
}

impl TryFrom<u32> for GgufTensorType {
    type Error = String;
    fn try_from(value: u32) -> Result<Self, Self::Error> { ... }
}
```

### Task 4: Extract Metadata

**File**: `src/loader/metadata.rs`

```rust
//! GGUF metadata structures

use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct GgufMetadata {
    // Architecture
    pub architecture: String,
    pub vocab_size: u32,

    // Model dimensions
    pub n_embd: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_kv_head: Option<u32>,

    // Quantization
    pub f16_kv: bool,
    pub q8_kv: bool,

    // RoPE
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub rope_scaling_type: Option<String>,

    // KV cache
    pub kv_cache_type: Option<String>,

    // Additional metadata
    pub extra: HashMap<String, MetadataValue>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Array(Vec<MetadataValue>),
}

impl Default for GgufMetadata {
    fn default() -> Self { ... }
}

impl GgufMetadata {
    pub fn from_kv_pairs(pairs: HashMap<String, MetadataValue>) -> Self { ... }
    pub fn get(&self, key: &str) -> Option<&MetadataValue> { ... }
}
```

### Task 5: Extract GgufTensor

**File**: `src/loader/gguf_tensor.rs`

```rust
//! GGUF tensor descriptor

use super::tensor_type::GgufTensorType;
use crate::loader::TensorShape;

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub shape: TensorShape,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
    pub data: Option<Vec<u8>>,
}

impl GgufTensor {
    pub fn new(
        name: String,
        shape: TensorShape,
        tensor_type: GgufTensorType,
        offset: u64,
    ) -> Self {
        Self {
            name,
            shape,
            tensor_type,
            offset,
            data: None,
        }
    }

    /// Calculate total bytes for this tensor
    pub fn byte_size(&self) -> usize { ... }

    /// Calculate number of elements
    pub fn element_count(&self) -> usize { ... }

    /// Load tensor data from file at current offset
    pub fn load_from_file(&mut self, file: &mut std::fs::File) -> anyhow::Result<()> { ... }
}
```

### Task 6: Extract Dequantization

**File**: `src/loader/dequant.rs`

```rust
//! Quantization and dequantization functions

use super::tensor_type::GgufTensorType;
use rayon::prelude::*;

/// Dequantize Q4_0 block to f32
pub fn dequant_q4_0(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q4_1 block to f32
pub fn dequant_q4_1(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q5_0 block to f32
pub fn dequant_q5_0(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q5_1 block to f32
pub fn dequant_q5_1(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q8_0 block to f32
pub fn dequant_q8_0(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q2_K block to f32
pub fn dequant_q2_k(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q3_K block to f32
pub fn dequant_q3_k(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q4_K block to f32
pub fn dequant_q4_k(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q5_K block to f32
pub fn dequant_q5_k(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Dequantize Q6_K block to f32
pub fn dequant_q6_k(data: &[u8], shape: &[usize]) -> Vec<f32> { ... }

/// Generic dequantization dispatcher
pub fn dequantize(
    data: &[u8],
    tensor_type: GgufTensorType,
    shape: &[usize],
) -> Vec<f32> {
    match tensor_type {
        GgufTensorType::Q4_0 => dequant_q4_0(data, shape),
        GgufTensorType::Q4_1 => dequant_q4_1(data, shape),
        // ... etc
    }
}

/// Parallel dequantization using Rayon
pub fn dequantize_parallel(
    data: &[u8],
    tensor_type: GgufTensorType,
    shape: &[usize],
) -> Vec<f32> { ... }
```

### Task 7: Simplify gguf.rs to Main Loader

**File**: `src/loader/gguf.rs` (simplified)

```rust
//! GGUF (GPT-Generated Unified Format) Loader

mod mxfp;
mod tensor_type;
mod metadata;
mod gguf_tensor;
mod dequant;

// Public exports
pub use mxfp::{E8M0, MxfpBlock};
pub use tensor_type::GgufTensorType;
pub use metadata::{GgufMetadata, MetadataValue};
pub use gguf_tensor::GgufTensor;

use crate::backend::hip_backend::{DeviceTensor, HipBackend, HipBuffer};
use crate::loader::lazy_tensor::LazyTensor;
use crate::loader::mmap::MmapGguf;
use crate::loader::TensorShape;
use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, RwLock};

const GGUF_MAGIC: &[u8] = b"GGUF";

/// F16 helper struct
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct F16 {
    bits: u16,
}

impl F16 {
    pub fn from_f32(value: f32) -> Self { ... }
    pub fn to_f32(self) -> f32 { ... }
}

/// Main GGUF loader
#[derive(Debug)]
pub struct GgufLoader {
    file: Option<MmapGguf>,
    metadata: GgufMetadata,
    tensors: HashMap<String, GgufTensor>,
    gpu_cache: RwLock<HashMap<String, Arc<DeviceTensor>>>,
}

impl GgufLoader {
    pub fn new(path: &str) -> Result<Self> { ... }
    pub fn metadata(&self) -> &GgufMetadata { ... }
    pub fn tensors(&self) -> &HashMap<String, GgufTensor> { ... }
    pub fn get_tensor(&self, name: &str) -> Option<&GgufTensor> { ... }
    pub fn load_tensor(&self, name: &str) -> Result<Vec<f32>> { ... }
    pub fn load_tensor_to_device(&self, name: &str, backend: &HipBackend) -> Result<Arc<DeviceTensor>> { ... }
}
```

### Task 8: Update mod.rs

**File**: `src/loader/mod.rs`

```rust
//! GGUF model loader

pub mod gguf;
pub mod lazy_tensor;
pub mod mmap;
pub mod mmap_loader;

pub use gguf::{GgufLoader, GgufMetadata, GgufTensor, GgufTensorType, E8M0, MxfpBlock};
```

---

## Strategy

1. **Extract format types first**: mxfp.rs, tensor_type.rs (no dependencies)
2. **Extract data structures**: metadata.rs, gguf_tensor.rs (depend on format types)
3. **Extract functions**: dequant.rs (depends on format types)
4. **Simplify main file**: gguf.rs becomes the orchestrator
5. **Test incrementally**: Run `cargo check` after each extraction

---

## Dependencies

**No Dependencies**: Can run in parallel with 03-01, 03-02, 03-04

**Affects**:
- `src/loader/mod.rs` (module declarations)
- All files importing from `crate::loader::gguf`

---

## Definition of Done

- [ ] 6 new module files created (mxfp.rs, tensor_type.rs, metadata.rs, gguf_tensor.rs, dequant.rs, simplified gguf.rs)
- [ ] Each module <500 LOC
- [ ] All public APIs preserved
- [ ] `cargo check` passes
- [ ] `cargo test --lib` passes
- [ ] All quantization formats still supported
- [ ] Phase 2 Rayon parallelization preserved

---

## Notes

- MXFP formats are OCP MX Specification v1.0 compliant
- Keep all quantization/dequantization functions
- Preserve Phase 2 async loading integration
- F16 helper is small enough to stay in gguf.rs or move to tensor_type.rs

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Max file LOC | 2832 | <500 per file |
| Module count | 1 | 6 |
| Public API | Same | Same |
| Test pass rate | 100% | 100% |

---

*Plan: 03-03*
*Created: 2026-01-18*
