//! Fused QKV Operations
//!
//! Implements fused QKV projection using a single GEMM operation
//! and splits output into Q, K, V tensors by offset.

#![allow(deprecated)] // TODO: Migrate from to_host_vec() to copy_from_device_safe() (Phase 13-03-02)

use crate::backend::{DeviceTensor, HipBackend, HipError, HipResult};
use crate::loader::TensorShape;

/// Fused QKV projection implementation
///
/// Performs QKV projection in a single GEMM: output = input @ W_qkv^T
/// Then splits output into Q, K, V by offset.
impl HipBackend {
    /// Perform fused QKV projection
    ///
    /// Computes: [seq_len, 3 * hidden_size] = [seq_len, hidden_size] @ [hidden_size, 3 * hidden_size]
    /// Then splits into Q, K, V tensors.
    pub fn fused_qkv(
        &self,
        input: &DeviceTensor,
        qkv_weight: &DeviceTensor,
        qkv_bias: Option<&DeviceTensor>,
    ) -> HipResult<DeviceTensor> {
        // Validate input shapes
        let input_shape = input.shape();
        let qkv_shape = qkv_weight.shape();

        if input_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "Input must be 2D tensor [seq_len, hidden_size]".to_string(),
            ));
        }

        if qkv_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "QKV weight must be 2D tensor [hidden_size, 3 * hidden_size]".to_string(),
            ));
        }

        let seq_len = input_shape.dims()[0];
        let hidden_size = input_shape.dims()[1];
        let qkv_out_dim = qkv_shape.dims()[1];

        if qkv_out_dim != 3 * hidden_size {
            return Err(HipError::GenericError(format!(
                "QKV weight output dimension {} must equal 3 * hidden_size {}",
                qkv_out_dim,
                3 * hidden_size
            )));
        }

        // Perform GEMM: output = input @ qkv_weight^T
        let output_shape = TensorShape::from_dims(&[seq_len, 3 * hidden_size]);
        let mut output = DeviceTensor::empty(self, output_shape)?;

        // Use HIP BLAS for matrix multiplication
        self.gemm(
            input,       // A: [seq_len, hidden_size]
            qkv_weight,  // B: [hidden_size, 3 * hidden_size]
            None,        // No bias for now
            &mut output, // C: [seq_len, 3 * hidden_size]
        )?;

        // Add bias if provided
        if let Some(bias) = qkv_bias {
            self.add_bias(&mut output, bias)?;
        }

        Ok(output)
    }

    /// Split fused QKV output into separate Q, K, V tensors
    ///
    /// Takes [seq_len, 3 * hidden_size] tensor and splits by offset:
    /// - Q: [seq_len, num_heads, head_dim]
    /// - K: [seq_len, num_heads, head_dim]  
    /// - V: [seq_len, num_heads, head_dim]
    pub fn split_qkv(
        &self,
        fused_output: &DeviceTensor,
        num_heads: usize,
        head_dim: usize,
    ) -> HipResult<(DeviceTensor, DeviceTensor, DeviceTensor)> {
        let fused_shape = fused_output.shape();

        if fused_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "Fused output must be 2D tensor [seq_len, 3 * hidden_size]".to_string(),
            ));
        }

        let seq_len = fused_shape.dims()[0];
        let hidden_size = num_heads * head_dim;
        let expected_fused_dim = 3 * hidden_size;

        if fused_shape.dims()[1] != expected_fused_dim {
            return Err(HipError::GenericError(format!(
                "Fused output dimension {} must equal 3 * hidden_size {}",
                fused_shape.dims()[1],
                expected_fused_dim
            )));
        }

        // Create output tensors
        let q_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let k_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);
        let v_shape = TensorShape::from_dims(&[seq_len, num_heads, head_dim]);

        let mut q_tensor = DeviceTensor::empty(self, q_shape)?;
        let mut k_tensor = DeviceTensor::empty(self, k_shape)?;
        let mut v_tensor = DeviceTensor::empty(self, v_shape)?;

        // Copy data by offset
        // Q: offset 0, K: offset hidden_size, V: offset 2 * hidden_size
        self.split_fused_output(
            fused_output,
            &mut q_tensor,
            &mut k_tensor,
            &mut v_tensor,
            seq_len,
            hidden_size,
        )?;

        Ok((q_tensor, k_tensor, v_tensor))
    }

    /// Add bias to tensor (broadcast along sequence dimension)
    fn add_bias(&self, output: &mut DeviceTensor, bias: &DeviceTensor) -> HipResult<()> {
        let output_shape = output.shape();
        let bias_shape = bias.shape();

        if bias_shape.dims().len() != 1 {
            return Err(HipError::GenericError("Bias must be 1D tensor".to_string()));
        }

        if bias_shape.dims()[0] != output_shape.dims()[1] {
            return Err(HipError::GenericError(
                "Bias dimension must match output feature dimension".to_string(),
            ));
        }

        // Simple bias addition: broadcast bias along sequence dimension
        // In a real implementation, this would be a GPU kernel
        let seq_len = output_shape.dims()[0];
        let feature_dim = bias_shape.dims()[0];

        // For now, use a simple approach - this would be optimized with GPU kernels
        let bias_host = bias.to_host_vec()?;
        let mut output_host = output.to_host_vec()?;

        for i in 0..seq_len {
            for j in 0..feature_dim {
                output_host[i * feature_dim + j] += bias_host[j];
            }
        }

        // Copy back to device
        *output = DeviceTensor::from_host_vec(self, output_host, output_shape.clone())?;

        Ok(())
    }

    /// Split fused output into Q, K, V by offset
    fn split_fused_output(
        &self,
        fused_output: &DeviceTensor,
        q_tensor: &mut DeviceTensor,
        k_tensor: &mut DeviceTensor,
        v_tensor: &mut DeviceTensor,
        seq_len: usize,
        hidden_size: usize,
    ) -> HipResult<()> {
        // Copy fused output to host for splitting
        let fused_host = fused_output.to_host_vec()?;

        // Split into Q, K, V
        let mut q_host = vec![0.0f32; seq_len * hidden_size];
        let mut k_host = vec![0.0f32; seq_len * hidden_size];
        let mut v_host = vec![0.0f32; seq_len * hidden_size];

        for i in 0..seq_len {
            for j in 0..hidden_size {
                let fused_idx = i * (3 * hidden_size) + j;
                q_host[i * hidden_size + j] = fused_host[fused_idx];
                k_host[i * hidden_size + j] = fused_host[fused_idx + hidden_size];
                v_host[i * hidden_size + j] = fused_host[fused_idx + 2 * hidden_size];
            }
        }

        // Copy back to device tensors
        let q_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let k_shape = TensorShape::from_dims(&[seq_len, hidden_size]);
        let v_shape = TensorShape::from_dims(&[seq_len, hidden_size]);

        *q_tensor = DeviceTensor::from_host_vec(self, q_host, q_shape)?;
        *k_tensor = DeviceTensor::from_host_vec(self, k_host, k_shape)?;
        *v_tensor = DeviceTensor::from_host_vec(self, v_host, v_shape)?;

        Ok(())
    }

    /// Perform matrix multiplication using HIP BLAS
    fn gemm(
        &self,
        a: &DeviceTensor,
        b: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        c: &mut DeviceTensor,
    ) -> HipResult<()> {
        // This would use hipBLAS for actual GPU matrix multiplication
        // For now, implement a simple CPU fallback for testing

        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        if a_shape.dims().len() != 2 || b_shape.dims().len() != 2 || c_shape.dims().len() != 2 {
            return Err(HipError::GenericError(
                "All tensors must be 2D for GEMM".to_string(),
            ));
        }

        let m = a_shape.dims()[0]; // rows of A
        let k = a_shape.dims()[1]; // cols of A = rows of B
        let n = b_shape.dims()[1]; // cols of B

        if b_shape.dims()[0] != k {
            return Err(HipError::GenericError(
                "Inner dimensions must match for matrix multiplication".to_string(),
            ));
        }

        if c_shape.dims()[0] != m || c_shape.dims()[1] != n {
            return Err(HipError::GenericError(
                "Output tensor shape doesn't match expected result".to_string(),
            ));
        }

        // Perform matrix multiplication on CPU for now
        let a_host = a.to_host_vec()?;
        let b_host = b.to_host_vec()?;
        let mut c_host = c.to_host_vec()?;

        // C = A * B
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_host[i * k + l] * b_host[l * n + j];
                }
                c_host[i * n + j] = sum;
            }
        }

        // Add bias if provided
        if let Some(bias_tensor) = bias {
            let bias_host = bias_tensor.to_host_vec()?;
            if bias_host.len() == n {
                for i in 0..m {
                    for j in 0..n {
                        c_host[i * n + j] += bias_host[j];
                    }
                }
            }
        }

        // Copy result back to device
        *c = DeviceTensor::from_host_vec(self, c_host, c_shape.clone())?;

        Ok(())
    }
}
