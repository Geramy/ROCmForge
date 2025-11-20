//! ROCmForge - AMD GPU Inference Engine
//!
//! A high-performance inference engine for Large Language Models
//! specifically designed for AMD GPUs using ROCm and HIP.

pub mod attention;
pub mod backend;
pub mod engine;
pub mod http;
pub mod kv_cache;
pub mod loader;
pub mod model;
pub mod models;
pub mod ops;
pub mod sampler;
pub mod scheduler;
pub mod tensor;
pub mod tokenizer;

#[cfg(test)]
mod hip_backend_debug_tests;
#[cfg(test)]
mod hip_isolation_test;

pub use attention::Attention;
pub use backend::HipBackend;
pub use engine::InferenceEngine;
pub use kv_cache::KvCache;
pub use sampler::Sampler;
pub use scheduler::Scheduler;
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_imports() {
        // Basic smoke test to ensure all modules compile
        assert!(true);
    }
}
