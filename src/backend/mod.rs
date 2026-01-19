//! ROCm/HIP backend module

// CPU module is always available for CPU feature detection
// SIMD functions are feature-gated within the cpu module
pub mod cpu;
pub mod gpu_executor;
pub mod gpu_test_common;
pub mod hip_backend;
pub mod hip_blas;
pub mod scratch;

// CPU feature detection is always available
pub use cpu::{CpuArch, CpuFeatures};

// SIMD functions require feature flag
#[cfg(feature = "simd")]
pub use cpu::{simd_matmul_f32, simd_matmul_tiled_f32, SimdMatmulError, SimdMatmulResult};

pub use gpu_executor::*;
#[cfg(test)]
pub use gpu_test_common::*;
pub use hip_backend::*;
pub use hip_blas::*;
pub use scratch::*;
