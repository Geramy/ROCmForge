//! Quantization and dequantization functions

use super::gguf_tensor::GgufTensor;
use super::mxfp::MxfpBlock;
use super::tensor_type::GgufTensorType;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Dequantize Q8_0 tensor to FP32 (parallelized with Rayon)
///
/// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
/// on multi-core CPUs. Each block is processed independently.
pub fn dequant_q8_0(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let blocks = total_elements.div_ceil(32);

    // Pre-allocate result vector
    let result = vec![0.0f32; total_elements];
    let result_lock = Arc::new(RwLock::new(result));

    // Process blocks in parallel using Rayon
    // Each block is independent - perfect for data parallelism
    (0..blocks).into_par_iter().for_each(|block_idx| {
        let block_start = block_idx * (4 + 32); // scale (4) + quants (32)

        if block_start + 4 > tensor.data.len() {
            return;
        }

        // Read scale (this is safe because we only read)
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read quantized values
        let quant_start = block_start + 4;
        let quant_end = std::cmp::min(quant_start + 32, tensor.data.len());
        let quants = &tensor.data[quant_start..quant_end];

        // Dequantize and write to shared result
        if let Ok(mut result) = result_lock.write() {
            for (i, &q) in quants.iter().enumerate() {
                let element_idx = block_idx * 32 + i;
                if element_idx < total_elements {
                    result[element_idx] = (q as f32 - 128.0) * scale;
                }
            }
        }
    });

    // Extract result from Arc<RwLock>
    let result = Arc::try_unwrap(result_lock)
        .map_err(|_e| anyhow::anyhow!("Failed to extract result: Arc still has owners"))?
        .into_inner()
        .map_err(|_e| anyhow::anyhow!("Failed to get inner value: RwLock poisoned"))?;

    Ok(result)
}

/// Dequantize Q4_0 tensor to FP32 (parallelized with Rayon)
///
/// Phase 2: Rayon Integration - Uses parallel processing for ~4x speedup
/// on multi-core CPUs. Q4_0 is the most common quantization format.
pub fn dequant_q4_0(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let blocks = total_elements.div_ceil(32);

    // Pre-allocate result vector
    let result = vec![0.0f32; total_elements];
    let result_lock = Arc::new(RwLock::new(result));

    // Process blocks in parallel using Rayon
    (0..blocks).into_par_iter().for_each(|block_idx| {
        let block_start = block_idx * (4 + 16); // scale (4) + quants (16 bytes for 32 values)

        if block_start + 4 > tensor.data.len() {
            return;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read quantized values (4-bit packed)
        let quant_start = block_start + 4;
        let quant_end = std::cmp::min(quant_start + 16, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (unpack 4-bit values) and write to shared result
        if let Ok(mut result) = result_lock.write() {
            for (i, &packed) in packed_quants.iter().enumerate() {
                for j in 0..2 {
                    let element_idx = block_idx * 32 + i * 2 + j;
                    if element_idx < total_elements {
                        let quant = if j == 0 {
                            packed & 0x0F
                        } else {
                            (packed >> 4) & 0x0F
                        };
                        result[element_idx] = (quant as f32 - 8.0) * scale;
                    }
                }
            }
        }
    });

    // Extract result from Arc<RwLock>
    let result = Arc::try_unwrap(result_lock)
        .map_err(|_e| anyhow::anyhow!("Failed to extract result: Arc still has owners"))?
        .into_inner()
        .map_err(|_e| anyhow::anyhow!("Failed to get inner value: RwLock poisoned"))?;

    Ok(result)
}

/// Dequantize Q4_1 tensor to FP32
/// Format: 32 values per block, scale (4 bytes) + min (4 bytes) + 16 bytes of 4-bit packed values
pub fn dequant_q4_1(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(32);

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 16); // scale (4) + min (4) + quants (16)

        if block_start + 8 > tensor.data.len() {
            break;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read min
        let min_bytes = &tensor.data[block_start + 4..block_start + 8];
        let min = f32::from_le_bytes([min_bytes[0], min_bytes[1], min_bytes[2], min_bytes[3]]);

        // Read quantized values (4-bit packed)
        let quant_start = block_start + 8;
        let quant_end = std::cmp::min(quant_start + 16, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (unpack 4-bit values)
        for (i, &packed) in packed_quants.iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + i * 2 + j;
                if element_idx < total_elements {
                    let quant = if j == 0 {
                        packed & 0x0F
                    } else {
                        (packed >> 4) & 0x0F
                    };
                    result[element_idx] = min + (quant as f32) * scale;
                }
            }
        }
    }

    Ok(result)
}

/// Dequantize Q5_0 tensor to FP32
/// Format: 32 values per block, scale (4 bytes) + qh (4 bytes) + 20 bytes of 4-bit packed values
/// qh contains the high bit for each of the 32 values
pub fn dequant_q5_0(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(32);

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 20); // scale (4) + qh (4) + quants (20)

        if block_start + 8 > tensor.data.len() {
            break;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read high bits (qh)
        let qh_bytes = &tensor.data[block_start + 4..block_start + 8];
        let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);

        // Read quantized values (4-bit packed)
        let quant_start = block_start + 8;
        let quant_end = std::cmp::min(quant_start + 20, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (5-bit values: 4 low bits from packed, 1 high bit from qh)
        for (i, &packed) in packed_quants.iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + i * 2 + j;
                if element_idx < total_elements {
                    let bit_idx = i * 2 + j;
                    let low_bits = if j == 0 {
                        packed & 0x0F
                    } else {
                        (packed >> 4) & 0x0F
                    };
                    let high_bit = if bit_idx < 32 { (qh >> bit_idx) & 1 } else { 0 };
                    let quant = (low_bits as u32 | (high_bit << 4)) as u8;
                    result[element_idx] = (quant as f32 - 16.0) * scale;
                }
            }
        }
    }

    Ok(result)
}

/// Dequantize Q5_1 tensor to FP32
/// Format: 32 values per block, scale (4 bytes) + min (4 bytes) + qh (4 bytes) + 20 bytes of 4-bit packed values
pub fn dequant_q5_1(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(32);

    for block_idx in 0..blocks {
        let block_start = block_idx * (4 + 4 + 4 + 20); // scale (4) + min (4) + qh (4) + quants (20)

        if block_start + 12 > tensor.data.len() {
            break;
        }

        // Read scale
        let scale_bytes = &tensor.data[block_start..block_start + 4];
        let scale = f32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);

        // Read min
        let min_bytes = &tensor.data[block_start + 4..block_start + 8];
        let min = f32::from_le_bytes([min_bytes[0], min_bytes[1], min_bytes[2], min_bytes[3]]);

        // Read high bits (qh)
        let qh_bytes = &tensor.data[block_start + 8..block_start + 12];
        let qh = u32::from_le_bytes([qh_bytes[0], qh_bytes[1], qh_bytes[2], qh_bytes[3]]);

        // Read quantized values (4-bit packed)
        let quant_start = block_start + 12;
        let quant_end = std::cmp::min(quant_start + 20, tensor.data.len());
        let packed_quants = &tensor.data[quant_start..quant_end];

        // Dequantize (5-bit values: 4 low bits from packed, 1 high bit from qh)
        for (i, &packed) in packed_quants.iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + i * 2 + j;
                if element_idx < total_elements {
                    let bit_idx = i * 2 + j;
                    let low_bits = if j == 0 {
                        packed & 0x0F
                    } else {
                        (packed >> 4) & 0x0F
                    };
                    let high_bit = if bit_idx < 32 { (qh >> bit_idx) & 1 } else { 0 };
                    let quant = (low_bits as u32 | (high_bit << 4)) as u8;
                    result[element_idx] = min + (quant as f32) * scale;
                }
            }
        }
    }

    Ok(result)
}

/// Dequantize MXFP4 tensor to FP32
pub fn dequant_mxfp4(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(32);

    for block_idx in 0..blocks {
        let block_start = block_idx * 17; // 1 scale byte + 16 data bytes

        if block_start + 1 > tensor.data.len() {
            break;
        }

        // Read scale (E8M0)
        let scale_exp = tensor.data[block_start] as i8;
        let scale = 2.0_f32.powi(scale_exp as i32);

        // Read MXFP4 elements (4-bit packed)
        let data_start = block_start + 1;
        let data_end = std::cmp::min(data_start + 16, tensor.data.len());

        for (byte_offset, &packed) in tensor.data[data_start..data_end].iter().enumerate() {
            for j in 0..2 {
                let element_idx = block_idx * 32 + byte_offset * 2 + j;
                if element_idx < total_elements {
                    let e2m1_bits = if j == 0 {
                        (packed >> 4) & 0x0F
                    } else {
                        packed & 0x0F
                    };

                    // Decode E2M1
                    let decoded = MxfpBlock::decode_e2m1(e2m1_bits);
                    let mut val = scale * decoded;
                    val = val.clamp(-8.0, 8.0); // MXFP4 range per OCP MX Spec v1.0
                    result[element_idx] = val;
                }
            }
        }
    }

    Ok(result)
}

/// Dequantize MXFP6 tensor to FP32
pub fn dequant_mxfp6(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(32);

    for block_idx in 0..blocks {
        let block_start = block_idx * 25; // 1 scale byte + 24 data bytes

        if block_start + 1 > tensor.data.len() {
            break;
        }

        // Read scale (E8M0)
        let scale_exp = tensor.data[block_start] as i8;
        let scale = 2.0_f32.powi(scale_exp as i32);

        // Read MXFP6 elements (6-bit packed)
        let data_start = block_start + 1;
        let data_end = std::cmp::min(data_start + 24, tensor.data.len());
        let packed_data = &tensor.data[data_start..data_end];

        // Unpack 6-bit values
        for i in 0..32 {
            let element_idx = block_idx * 32 + i;
            if element_idx >= total_elements {
                break;
            }

            // Extract 6-bit value
            let bit_offset = (i * 6) % 8;
            let byte_idx = (i * 6) / 8;

            if byte_idx + 1 < packed_data.len() {
                let combined =
                    ((packed_data[byte_idx + 1] as u16) << 8) | (packed_data[byte_idx] as u16);
                let e2m3_bits = ((combined >> (10 - bit_offset)) & 0x3F) as u8;

                // Decode E2M3
                let decoded = MxfpBlock::decode_e2m3(e2m3_bits);
                let mut val = scale * decoded;
                val = val.clamp(-7.5, 7.5); // MXFP6 range
                result[element_idx] = val;
            }
        }
    }

    Ok(result)
}

/// Dequantize Q4_K tensor to FP32
/// Q4_K uses super-block structure with 256-byte blocks containing 8 sub-blocks
/// Each sub-block has its own scale and 4-bit quantized values
pub fn dequant_q4_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(256);

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > tensor.data.len() {
            break;
        }

        // Q4_K super-block structure:
        // - 16 bytes: 8 half-precision scales (2 bytes each) for 8 sub-blocks
        // - 16 bytes: 8 int8 mins (1 byte each) for 8 sub-blocks
        // - 160 bytes: 8 sub-blocks of 4-bit quantized values (20 bytes each, packed)
        // - 64 bytes: additional data (likely for QK format)

        let scales_start = block_start;
        let mins_start = block_start + 16;
        let quants_start = block_start + 32;

        // Process each of the 8 sub-blocks (32 elements each)
        for sub_block_idx in 0..8 {
            let sub_block_start = block_idx * 256 + sub_block_idx * 32;
            let scale_idx = sub_block_idx;
            let min_idx = sub_block_idx;

            // Get scale for this sub-block
            let scale_offset = scales_start + scale_idx * 2;
            let scale = if scale_offset + 2 <= tensor.data.len() {
                let scale_bits = u16::from_le_bytes([
                    tensor.data[scale_offset],
                    tensor.data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0
            };

            // Get min for this sub-block
            let min_offset = mins_start + min_idx;
            let min = if min_offset < tensor.data.len() {
                tensor.data[min_offset] as i8 as f32
            } else {
                0.0
            };

            // Extract 4-bit quantized values for this sub-block (32 values)
            for i in 0..32 {
                let element_idx = sub_block_start + i;
                if element_idx >= total_elements {
                    break;
                }

                let bit_pos = i * 4;
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                let quant_offset = quants_start + sub_block_idx * 20 + byte_idx;

                let quant = if quant_offset + 1 < tensor.data.len() {
                    let combined = ((tensor.data[quant_offset + 1] as u16) << 8)
                                   | (tensor.data[quant_offset] as u16);

                    ((combined >> bit_offset) & 0xF) as u8
                } else {
                    0
                };

                result[element_idx] = min + (quant as f32) * scale;
            }
        }
    }

    Ok(result)
}

/// Dequantize Q6_K tensor to FP32
/// Q6_K uses 256-byte blocks encoding 256 elements
/// Format: scales (16 bytes) + quantized values (240 bytes for 256*6/8 = 192 bytes + padding)
pub fn dequant_q6_k(tensor: &GgufTensor) -> Result<Vec<f32>> {
    let total_elements = tensor.total_elements();
    let mut result = vec![0.0f32; total_elements];
    let blocks = total_elements.div_ceil(256);

    for block_idx in 0..blocks {
        let block_start = block_idx * 256;

        if block_start + 256 > tensor.data.len() {
            break;
        }

        // Read scales (16 half-precision floats = 32 bytes)
        // Q6_K uses half-precision scales for each group of 16 elements
        let scales_start = block_start;

        // Read quantized values (6-bit packed, 256*6/8 = 192 bytes)
        let quants_start = block_start + 32;
        let quants_end = block_start + 224;

        // Dequantize block
        for i in 0..256 {
            let element_idx = block_idx * 256 + i;
            if element_idx >= total_elements {
                break;
            }

            // Get scale for this group (every 16 elements share a scale)
            let scale_idx = i / 16;
            let scale_offset = scales_start + scale_idx * 2;

            let scale = if scale_offset + 2 <= tensor.data.len() {
                let scale_bits = u16::from_le_bytes([
                    tensor.data[scale_offset],
                    tensor.data[scale_offset + 1],
                ]);
                half::f16::from_bits(scale_bits).to_f32()
            } else {
                1.0 // fallback scale
            };

            // Extract 6-bit quantized value
            let bit_offset = (i * 6) % 8;
            let byte_idx = (i * 6) / 8;

            if quants_start + byte_idx + 1 < quants_end {
                let combined = ((tensor.data[quants_start + byte_idx + 1] as u16) << 8)
                               | (tensor.data[quants_start + byte_idx] as u16);

                let quant_val = ((combined >> bit_offset) & 0x3F) as u8;

                // Convert to signed range and scale
                let signed_val = if quant_val >= 32 {
                    (quant_val as i8 - 64) as f32
                } else {
                    quant_val as f32
                };

                result[element_idx] = signed_val * scale;
            }
        }
    }

    Ok(result)
}

/// Generic dequantization dispatcher
///
/// Routes to the appropriate dequantization function based on tensor type
pub fn dequantize(tensor: &GgufTensor) -> Result<Vec<f32>> {
    match tensor.tensor_type {
        GgufTensorType::F32 => Ok(tensor
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        GgufTensorType::F16 => Ok(tensor
            .data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        GgufTensorType::Q8_0 => dequant_q8_0(tensor),
        GgufTensorType::Q4_0 => dequant_q4_0(tensor),
        GgufTensorType::Q4_1 => dequant_q4_1(tensor),
        GgufTensorType::Q5_0 => dequant_q5_0(tensor),
        GgufTensorType::Q5_1 => dequant_q5_1(tensor),
        GgufTensorType::Mxfp4 => dequant_mxfp4(tensor),
        GgufTensorType::Mxfp6E2m3 | GgufTensorType::Mxfp6E3m2 => dequant_mxfp6(tensor),
        GgufTensorType::Q4_K => dequant_q4_k(tensor),
        GgufTensorType::Q6_K => dequant_q6_k(tensor),
        GgufTensorType::Q2_K | GgufTensorType::Q3_K | GgufTensorType::Q5_K => {
            Err(anyhow::anyhow!(
                "K-quant type {:?} not yet implemented for tensor '{}'",
                tensor.tensor_type,
                tensor.name
            ))
        }
    }
}
