//! Model loaders module

pub mod gguf;
pub mod lazy_tensor;
pub mod mmap;
pub mod mmap_loader;
pub mod onnx_loader;

// GGUF submodules (03-03: modularization)
mod mxfp;
mod tensor_type;
mod metadata;
mod gguf_tensor;
mod dequant;

// Re-export GGUF types
pub use gguf::{GgufLoader, F16};
pub use mxfp::{E8M0, MxfpBlock};
pub use tensor_type::GgufTensorType;
pub use metadata::GgufMetadata;
pub use gguf_tensor::GgufTensor;

pub use lazy_tensor::*;
pub use mmap::*;
pub use mmap_loader::*;
pub use onnx_loader::*;
