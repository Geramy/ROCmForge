//! Attention mechanism for ROCmForge
//! Implements Scaled Dot-Product Attention

pub mod backend;
pub mod compare;
pub mod compute;
pub mod cpu;
pub mod gpu;
pub mod kernels;
pub mod mask;
pub mod multi_query;
pub mod rope;
pub mod softmax;

#[cfg(feature = "rocm")]
use crate::backend::{DeviceTensor, HipBackend};
#[cfg(feature = "rocm")]
use crate::loader::mmap_loader::TensorShape;
pub use backend::AttentionBackend;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Dimension error: {0}")]
    DimensionError(String),
}

pub type AttentionResult<T> = Result<T, AttentionError>;

pub struct Attention {
    pub dim: usize,
    pub backend: AttentionBackend,
}

impl Attention {
    pub fn new(dim: usize) -> Self {
        Attention {
            dim,
            backend: AttentionBackend::default(),
        }
    }

    pub fn with_backend(dim: usize, backend: AttentionBackend) -> Self {
        Attention { dim, backend }
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        dropout: Option<f32>,
    ) -> AttentionResult<Vec<f32>> {
        println!(
            "DEBUG: Attention::forward called with backend: {:?}",
            self.backend
        );
        println!(
            "DEBUG: Input lengths - q: {}, k: {}, v: {}",
            q.len(),
            k.len(),
            v.len()
        );
        match self.backend {
            AttentionBackend::Cpu => {
                println!("DEBUG: Using CPU backend");
                cpu::CpuBackend::forward(self.dim, q, k, v, mask, dropout)
            }
            #[cfg(feature = "rocm")]
            AttentionBackend::Gpu => {
                println!("DEBUG: Using GPU backend");
                gpu::GpuBackend::forward(self.dim, q, k, v, mask, dropout)
            }
        }
    }

    /// Forward pass with DeviceTensor inputs for zero-copy GPU computation
    #[cfg(feature = "rocm")]
    pub fn forward_device(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        mask: Option<&DeviceTensor>,
        dropout: Option<f32>,
    ) -> AttentionResult<DeviceTensor> {
        match self.backend {
            AttentionBackend::Cpu => {
                // Fallback to CPU implementation by copying to host
                let q_host = q.to_host_vec().map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to copy Q to host: {}", e))
                })?;
                let k_host = k.to_host_vec().map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to copy K to host: {}", e))
                })?;
                let v_host = v.to_host_vec().map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to copy V to host: {}", e))
                })?;
                let mask_host = mask
                    .map(|m| {
                        m.to_host_vec().map_err(|e| {
                            AttentionError::DimensionError(format!(
                                "Failed to copy mask to host: {}",
                                e
                            ))
                        })
                    })
                    .transpose()?;

                let output = cpu::CpuBackend::forward(
                    self.dim,
                    &q_host,
                    &k_host,
                    &v_host,
                    mask_host.as_ref().map(|m| m.as_slice()),
                    dropout,
                )?;
                println!(
                    "DEBUG: CPU returned output with {} elements: {:?}",
                    output.len(),
                    output
                );

                // Create output DeviceTensor with same shape as input V tensor
                let backend = HipBackend::new().map_err(|e| {
                    AttentionError::DimensionError(format!("Failed to create HIP backend: {}", e))
                })?;
                let output_shape = v.shape().clone(); // Use same shape as V tensor
                println!(
                    "DEBUG: CPU returned output with {} elements: {:?}",
                    output.len(),
                    output
                );
                println!("DEBUG: About to call DeviceTensor::from_host_vec...");
                let result = DeviceTensor::from_host_vec(&backend, output, output_shape);
                match result {
                    Ok(tensor) => {
                        println!(
                            "DEBUG: Created output tensor: len() = {}, size() = {}, shape = {:?}",
                            tensor.len(),
                            tensor.size(),
                            tensor.shape()
                        );
                        Ok(tensor)
                    }
                    Err(e) => {
                        println!("DEBUG: Failed to create output tensor: {}", e);
                        Err(AttentionError::DimensionError(format!(
                            "Failed to create output tensor: {}",
                            e
                        )))
                    }
                }
            }
            #[cfg(feature = "rocm")]
            AttentionBackend::Gpu => {
                gpu::GpuBackend::forward_device(self.dim, q, k, v, mask, dropout)
            }
        }
    }
}
