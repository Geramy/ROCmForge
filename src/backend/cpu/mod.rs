//! CPU backend module
//!
//! Provides CPU-based implementations of tensor operations with SIMD acceleration.

// CPU feature detection is always available (needed for runtime dispatch)
pub mod cpu_features;

#[cfg(feature = "simd")]
pub mod simd;

// Re-export SIMD types when available
#[cfg(feature = "simd")]
pub use simd::{simd_matmul_f32, simd_matmul_tiled_f32, SimdMatmulError, SimdMatmulResult};

// CPU feature detection is always available
pub use cpu_features::{CpuArch, CpuFeatures};
