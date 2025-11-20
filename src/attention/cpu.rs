//! CPU backend implementation for attention computation

use crate::attention::{compute, mask, softmax, AttentionError, AttentionResult};
use crate::tensor::matmul::cpu_matmul_f32;

/// CPU backend for attention computation
pub struct CpuBackend;

impl CpuBackend {
    /// Compute attention using CPU implementation
    pub fn forward(
        dim: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        dropout: Option<f32>,
    ) -> AttentionResult<Vec<f32>> {
        let batch_size = q.len() / (dim * dim);
        let seq_len = dim;

        if q.len() != k.len() || q.len() != v.len() {
            return Err(AttentionError::ShapeMismatch(
                "Q, K, V must have same shape".to_string(),
            ));
        }

        let scale = 1.0 / (dim as f32).sqrt();

        // Compute QK^T
        let mut scores = compute::matmul_cpu(q, k, batch_size, seq_len, seq_len, dim)?;

        // Apply scaling
        for score in &mut scores {
            *score *= scale;
        }

        // Apply mask if provided
        if let Some(mask_data) = mask {
            if mask_data.len() != batch_size * seq_len * seq_len {
                return Err(AttentionError::ShapeMismatch(
                    "Mask shape mismatch".to_string(),
                ));
            }
            for (i, score) in scores.iter_mut().enumerate() {
                if mask_data[i] == f32::NEG_INFINITY {
                    *score = f32::NEG_INFINITY;
                }
            }
        }

        // Apply softmax row-wise
        softmax::softmax_in_place(&mut scores, batch_size, seq_len);

        // Apply dropout if provided
        if let Some(dropout_prob) = dropout {
            compute::apply_dropout(&mut scores, dropout_prob, 42);
        }

        // Compute final output: scores * V
        compute::matmul_cpu(&scores, v, batch_size, seq_len, dim, seq_len)
    }
}
