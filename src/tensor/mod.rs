//! Tensor operations for ROCmForge
//! Provides matrix operations and linear algebra primitives

pub mod matmul;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Basic tensor structure for ROCmForge operations
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor data stored as f32 values in row-major order
    pub data: Vec<f32>,
}

impl Tensor {
    /// Create a tensor with random values in range [0, 1)
    pub fn random(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
        Self { data }
    }

    /// Create a tensor with random values using a specific seed for reproducibility
    pub fn random_seeded(size: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
        Self { data }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(size: usize) -> Self {
        let data = vec![0.0f32; size];
        Self { data }
    }

    /// Get the number of elements in the tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

pub use matmul::*;
