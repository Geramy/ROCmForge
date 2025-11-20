//! Backend abstraction for attention computation

/// Backend types for attention computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    /// CPU backend using standard Rust computation
    Cpu,
    /// GPU backend using ROCm/HIP acceleration
    #[cfg(feature = "rocm")]
    Gpu,
}

impl Default for AttentionBackend {
    fn default() -> Self {
        AttentionBackend::Cpu
    }
}
