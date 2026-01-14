//! Quantized matmul operations for Q4_0 and Q8_0 formats.
//!
//! PHASE 3: Quantized MatMul Operations
//!
//! This module provides matmul operations for quantized weights. Currently uses
//! CPU-side dequantization followed by GPU matmul. Future work will implement
//! native HIP kernels for on-device dequantization.
//!
//! # Format Specifications
//!
//! ## Q4_0
//! - Block size: 32 elements
//! - Per block: scale (f32, 4 bytes) + 16 bytes 4-bit packed values = 20 bytes
//! - Dequantization: value = scale * ((packed & 0x0F) - 8)
//!
//! ## Q8_0
//! - Block size: 32 elements
//! - Per block: scale (f32, 4 bytes) + 32 bytes int8 values = 36 bytes
//! - Dequantization: value = scale * int8_value

use crate::backend::HipBackend;

/// Result type for quantized operations
pub type QuantizedResult<T> = Result<T, String>;

/// Q4_0 block header (scale only)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Q4_0Block {
    /// Scale factor for this block
    pub scale: f32,
    // 16 bytes of packed 4-bit values follow in the actual data
}

/// Q8_0 block header (scale only)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Q8_0Block {
    /// Scale factor for this block
    pub scale: f32,
    // 32 bytes of int8 values follow in the actual data
}

/// Dequantize Q4_0 weights to f32
///
/// # Format
/// - Each block has 32 values packed into 16 bytes
/// - Each value is 4 bits, interpreted as signed: unpacked - 8
/// - Scale applies to all 32 values in the block
pub fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 31) / 32;
    let mut result = vec![0.0f32; n_elements];
    let block_size = 20; // 4 bytes scale + 16 bytes packed data

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * block_size;
        if block_offset + 4 > data.len() {
            break;
        }

        // Read scale
        let scale = f32::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
            data[block_offset + 2],
            data[block_offset + 3],
        ]);

        // Unpack 4-bit values
        let data_start = block_offset + 4;
        let base_idx = block_idx * 32;

        for i in 0..16 {
            if data_start + i >= data.len() {
                break;
            }
            let packed = data[data_start + i];

            // Low nibble
            let low = (packed & 0x0F) as i32 - 8;
            if base_idx + i * 2 < n_elements {
                result[base_idx + i * 2] = scale * low as f32;
            }

            // High nibble
            let high = ((packed >> 4) & 0x0F) as i32 - 8;
            if base_idx + i * 2 + 1 < n_elements {
                result[base_idx + i * 2 + 1] = scale * high as f32;
            }
        }
    }

    result
}

/// Dequantize Q8_0 weights to f32
///
/// # Format
/// - Each block has 32 int8 values
/// - Scale applies to all 32 values in the block
pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 31) / 32;
    let mut result = vec![0.0f32; n_elements];
    let block_size = 36; // 4 bytes scale + 32 bytes int8 data

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * block_size;
        if block_offset + 4 > data.len() {
            break;
        }

        // Read scale
        let scale = f32::from_le_bytes([
            data[block_offset],
            data[block_offset + 1],
            data[block_offset + 2],
            data[block_offset + 3],
        ]);

        // Read int8 values
        let data_start = block_offset + 4;
        let base_idx = block_idx * 32;

        for i in 0..32 {
            let idx = data_start + i;
            if idx >= data.len() {
                break;
            }
            let elem_idx = base_idx + i;
            if elem_idx < n_elements {
                let int8_val = data[idx] as i8;
                result[elem_idx] = scale * int8_val as f32;
            }
        }
    }

    result
}

/// MatMul with Q4_0 quantized weights
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_weights`: Raw Q4_0 quantized weight data
/// - `input`: Input tensor (f32)
/// - `n_rows`: Number of rows in weight matrix
/// - `n_cols`: Number of columns in weight matrix
/// - `output`: Output buffer
///
/// # Note
/// Currently dequantizes on CPU then performs matmul on GPU.
/// TODO: Implement native HIP kernel for on-device dequantization.
pub fn matmul_q4_0(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &crate::backend::HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &crate::backend::HipBuffer,
) -> QuantizedResult<()> {
    // Dequantize weights
    let n_elements = n_rows * n_cols;
    let dequant_weights = dequantize_q4_0(quantized_weights, n_elements);

    // Upload dequantized weights to GPU
    let weight_bytes = n_elements * 4;
    let weight_buffer = backend
        .allocate_buffer(weight_bytes)
        .map_err(|e| format!("Failed to allocate weight buffer: {}", e))?;

    weight_buffer
        .copy_from_host(&dequant_weights)
        .map_err(|e| format!("Failed to upload weights: {}", e))?;

    // Perform matmul using standard matmul op
    let m = 1i32; // Input is typically [1, n_cols]
    let k = n_cols as i32;
    let n = n_rows as i32;

    crate::ggml::hip_backend::ops::matmul::matmul(
        backend,
        input,
        &weight_buffer,
        m,
        n,
        k,
        output,
    )
    .map_err(|e| format!("MatMul failed: {}", e))
}

/// MatMul with Q8_0 quantized weights
///
/// # Parameters
/// - `backend`: HIP backend for GPU operations
/// - `quantized_weights`: Raw Q8_0 quantized weight data
/// - `input`: Input tensor (f32)
/// - `n_rows`: Number of rows in weight matrix
/// - `n_cols`: Number of columns in weight matrix
/// - `output`: Output buffer
///
/// # Note
/// Currently dequantizes on CPU then performs matmul on GPU.
/// TODO: Implement native HIP kernel for on-device dequantization.
pub fn matmul_q8_0(
    backend: &HipBackend,
    quantized_weights: &[u8],
    input: &crate::backend::HipBuffer,
    n_rows: usize,
    n_cols: usize,
    output: &crate::backend::HipBuffer,
) -> QuantizedResult<()> {
    // Dequantize weights
    let n_elements = n_rows * n_cols;
    let dequant_weights = dequantize_q8_0(quantized_weights, n_elements);

    // Upload dequantized weights to GPU
    let weight_bytes = n_elements * 4;
    let weight_buffer = backend
        .allocate_buffer(weight_bytes)
        .map_err(|e| format!("Failed to allocate weight buffer: {}", e))?;

    weight_buffer
        .copy_from_host(&dequant_weights)
        .map_err(|e| format!("Failed to upload weights: {}", e))?;

    // Perform matmul using standard matmul op
    let m = 1i32; // Input is typically [1, n_cols]
    let k = n_cols as i32;
    let n = n_rows as i32;

    crate::ggml::hip_backend::ops::matmul::matmul(
        backend,
        input,
        &weight_buffer,
        m,
        n,
        k,
        output,
    )
    .map_err(|e| format!("MatMul failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q4_0_simple() {
        // Create simple Q4_0 data: 1 block with 32 identical values
        // Q4_0: 4-bit values (0-15), dequantized as (value - 8) * scale
        let mut data = vec![0u8; 20]; // 1 block * 20 bytes

        // Block 0: scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        // Pack 32 values, all stored as 8 (representing 0 after dequantization)
        // Each byte holds two 4-bit values: low nibble + high nibble
        for i in 0..16 {
            data[4 + i] = 0x88; // Both nibbles = 8, representing 0.0
        }

        let result = dequantize_q4_0(&data, 32);

        // All 32 values should be 0.0
        for i in 0..32 {
            assert!((result[i] - 0.0).abs() < 0.01, "result[{}]={}", i, result[i]);
        }
    }

    #[test]
    fn test_dequantize_q8_0_simple() {
        // Create simple Q8_0 data: 1 block
        let mut data = vec![0u8; 36]; // 1 block * 36 bytes

        // Scale = 1.0
        data[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        // Int8 values: [-16, -15, ..., 15]
        for i in 0..32 {
            data[4 + i] = (i as i8 - 16) as u8;
        }

        let result = dequantize_q8_0(&data, 32);

        for i in 0..32 {
            let expected = (i as i8 - 16) as f32;
            assert!((result[i] - expected).abs() < 0.01, "result[{}]={}, expected={}", i, result[i], expected);
        }
    }
}
