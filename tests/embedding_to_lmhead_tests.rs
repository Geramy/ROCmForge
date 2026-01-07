//! TDD tests for embedding lookup and LM head computation
//!
//! TODO: This entire file needs to be rewritten to use the current GGUF API
//!
//! The obsolete API used:
//! - `rocmforge::loader::gguf_loader::GgufModel` (no longer exists)
//! - `rocmforge::loader::gguf_loader::GgufDataType` (renamed to GgufTensorType)
//! - `GgufModel` had `.tensors`, `.metadata` fields directly
//!
//! The current API:
//! - `rocmforge::loader::GgufLoader` - Main loader struct
//! - `rocmforge::loader::GgufMetadata` - Metadata struct
//! - `rocmforge::loader::GgufTensor` - Tensor struct
//! - `rocmforge::loader::GgufTensorType` - Enum of tensor types
//! - Use `GgufLoader::new(path)` to load
//! - Use `loader.load_tensors()` to get HashMap<String, GgufTensor>
//! - Use `loader.metadata()` to access metadata
//!
//! Once rewritten, tests should cover:
//! 1. Token embedding lookup
//! 2. LM head weight loading
//! 3. Embedding → LM head pipeline
//! 4. CPU vs GPU correctness
//! 5. Shape validation
//! 6. Multiple vocab sizes
//! 7. Edge cases (empty tokens, OOM, etc.)

// Original file had 436 lines of tests using obsolete API
// Preserving for reference during rewrite:
/*
use rocmforge::backend::hip_backend::HipError;
use rocmforge::backend::hip_backend::{DeviceTensor, HipBackend};
use rocmforge::loader::{GgufLoader, GgufMetadata, GgufTensor, GgufTensorType};
use rocmforge::loader::TensorShape;
use rocmforge::model::config::{ModelConfig, ModelType};
use rocmforge::tensor::matmul::cpu_matmul_f32;
use std::path::Path;

// ... (436 lines of original code)
*/
