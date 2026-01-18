//! MXFP (block-scaled floating-point) format support
//!
//! Per OCP MX Specification v1.0:
//! - Block size: 32 elements
//! - Scale: E8M0 (1 byte)
//! - Elements: packed 4-bit or 6-bit values

/// E8M0 scale format (8-bit exponent only)
///
/// Per OCP MX Specification v1.0:
/// - 8-bit signed exponent
/// - Value = 2^exponent
/// - Range: 2^(-127) to 2^(127)
/// - Used as block scale for MXFP4/MXFP6
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct E8M0 {
    pub exponent: i8,
}

impl E8M0 {
    /// Convert E8M0 to f32
    pub fn to_f32(&self) -> f32 {
        2.0_f32.powi(self.exponent as i32)
    }

    /// Create E8M0 from f32
    /// E8M0 represents 2^exponent as a scale factor
    pub fn from_f32(value: f32) -> Self {
        if value == 0.0 || value.is_nan() {
            return E8M0 { exponent: 0 };
        }

        if value.is_infinite() {
            return E8M0 { exponent: 127 };
        }

        // E8M0 scale should be the largest value in the block
        // so we can represent values in [0, scale] range
        let abs_val = value.abs();
        let exp = abs_val.log2().clamp(-127.0, 127.0).round() as i8;
        E8M0 { exponent: exp }
    }
}

/// MXFP block (block-scaled floating-point)
///
/// Per OCP MX Specification v1.0:
/// - Block size: 32 elements
/// - Scale: E8M0 (1 byte)
/// - Elements: packed 4-bit or 6-bit values
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MxfpBlock {
    pub scale: E8M0,
    pub elements: Vec<u8>,
}

impl MxfpBlock {
    /// Create new MXFP4 block (4-bit elements)
    pub fn new_mxfp4() -> Self {
        MxfpBlock {
            scale: E8M0 { exponent: 0 },
            elements: vec![0u8; 16], // 32 elements * 4 bits / 8
        }
    }

    /// Create new MXFP6 block (6-bit elements)
    pub fn new_mxfp6() -> Self {
        MxfpBlock {
            scale: E8M0 { exponent: 0 },
            elements: vec![0u8; 24], // 32 elements * 6 bits / 8
        }
    }

    /// Pack f32 values into MXFP4 block
    pub fn pack_mxfp4(values: &[f32]) -> Self {
        // Find max absolute value for scale
        let max_val = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

        // Handle edge case where all values are zero
        let scale = if max_val == 0.0 {
            E8M0 { exponent: 0 }
        } else {
            // Scale is the max value itself (as power of 2)
            // This normalizes values to [0, 1] range for encoding
            E8M0::from_f32(max_val)
        };

        let scale_f32 = scale.to_f32();

        // Encode values as E2M1 (4-bit)
        let mut packed = vec![0u8; 16];
        for (i, &val) in values.iter().take(32).enumerate() {
            // Normalize value by scale (should now be in range [0, 1])
            let normalized = if scale_f32 > 0.0 {
                val / scale_f32
            } else {
                val
            };

            let encoded = Self::encode_e2m1(normalized);
            let byte_idx = i / 2;
            let nibble = i % 2;

            if nibble == 0 {
                packed[byte_idx] |= encoded << 4;
            } else {
                packed[byte_idx] |= encoded & 0x0F;
            }
        }

        MxfpBlock {
            scale,
            elements: packed,
        }
    }

    /// Unpack MXFP4 block to f32 values
    pub fn unpack_mxfp4(&self) -> Vec<f32> {
        let mut values = vec![0.0f32; 32];
        let scale_f32 = self.scale.to_f32();

        for i in 0..32 {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (self.elements[byte_idx] >> 4) & 0x0F
            } else {
                self.elements[byte_idx] & 0x0F
            };

            let decoded = Self::decode_e2m1(nibble);
            let mut val = scale_f32 * decoded;
            val = val.clamp(-8.0, 8.0); // MXFP4 range per OCP MX Spec v1.0
            values[i] = val;
        }

        values
    }

    /// Pack f32 values into MXFP6 block
    pub fn pack_mxfp6(values: &[f32]) -> Self {
        // Find max absolute value for scale
        let max_val = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

        // Handle edge case where all values are zero
        let scale = if max_val == 0.0 {
            E8M0 { exponent: 0 }
        } else {
            // Scale is the max value itself (as power of 2)
            // This normalizes values to [0, 1] range for encoding
            E8M0::from_f32(max_val)
        };

        let scale_f32 = scale.to_f32();

        // Encode values as E2M3 (6-bit)
        let packed = Self::pack_6bit_values(
            &values
                .iter()
                .take(32)
                .map(|&v| {
                    // Normalize value by scale (should now be in range [0, 1])
                    let normalized = if scale_f32 > 0.0 { v / scale_f32 } else { v };
                    Self::encode_e2m3(normalized)
                })
                .collect::<Vec<u8>>(),
        );

        MxfpBlock {
            scale,
            elements: packed,
        }
    }

    /// Unpack MXFP6 block to f32 values
    pub fn unpack_mxfp6(&self) -> Vec<f32> {
        let unpacked_bits = Self::unpack_6bit_values(&self.elements, 32);
        let scale_f32 = self.scale.to_f32();

        unpacked_bits
            .iter()
            .map(|&bits| {
                let decoded = Self::decode_e2m3(bits);
                let mut val = scale_f32 * decoded;
                val = val.clamp(-7.5, 7.5); // MXFP6 range
                val
            })
            .collect()
    }

    /// Get packed size in bytes
    pub fn packed_size(&self) -> usize {
        1 + self.elements.len() // scale + elements
    }

    /// Encode f32 as E2M1 (4-bit): sign(1) + exp(2) + mant(1)
    /// E2M1 format: value = (-1)^sign * 2^(exp-1) * (1 + mant)
    /// Input should be normalized to approximately [0, 8] range per OCP MX Spec v1.0
    pub fn encode_e2m1(value: f32) -> u8 {
        if value == 0.0 {
            return 0b0000;
        }

        let sign = if value < 0.0 { 0b1000 } else { 0b0000 };
        let abs = value.abs();

        // E2M1 can represent values in [0.5, 8.0] with exp in [0, 3] and mant in [0, 1]
        // For values < 0.5, we encode as 0.5 (minimum positive value)
        let clamped = abs.max(0.5).min(8.0);

        // Try all 4 combinations and pick the closest
        let mut best_encoding = 0u8;
        let mut best_error = f32::MAX;

        for exp_bits in 0..4 {
            for mant_bits in 0..2 {
                let exp = exp_bits as i32 - 1;
                let mant = mant_bits as f32;
                let decoded = (1.0 + mant) * 2_f32.powi(exp);

                let error = (clamped - decoded).abs();
                if error < best_error {
                    best_error = error;
                    best_encoding = (exp_bits << 1) | mant_bits;
                }
            }
        }

        sign | best_encoding
    }

    /// Decode E2M1 (4-bit) to f32
    pub fn decode_e2m1(bits: u8) -> f32 {
        if bits == 0 {
            return 0.0;
        }

        let sign = if bits & 0x08 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 1) & 0x03) as i32 - 1;
        let mant = (bits & 0x01) as f32;

        sign * (1.0 + mant) * 2_f32.powi(exp)
    }

    /// Encode f32 as E2M3 (6-bit): sign(1) + exp(2) + mant(3)
    /// E2M3 format: value = (-1)^sign * 2^(exp-1) * (1 + mant/8)
    /// Input should be normalized to approximately [0, 7.5] range
    pub fn encode_e2m3(value: f32) -> u8 {
        if value == 0.0 {
            return 0b000000;
        }

        let sign = if value < 0.0 { 0b100000 } else { 0b000000 };
        let abs = value.abs();

        // E2M3 can represent values in [0.5, 7.5] with exp in [0, 3] and mant in [0, 7]
        // For values < 0.5, we encode as 0.5 (minimum positive value)
        let clamped = abs.max(0.5).min(7.5);

        // Try all 32 combinations and pick the closest
        let mut best_encoding = 0u8;
        let mut best_error = f32::MAX;

        for exp_bits in 0..4 {
            for mant_bits in 0u8..8 {
                let exp = exp_bits as i32 - 1;
                let mant = mant_bits as f32 / 8.0;
                let decoded = (1.0 + mant) * 2_f32.powi(exp);

                let error = (clamped - decoded).abs();
                if error < best_error {
                    best_error = error;
                    best_encoding = (exp_bits << 3) | mant_bits;
                }
            }
        }

        sign | best_encoding
    }

    /// Decode E2M3 (6-bit) to f32
    pub fn decode_e2m3(bits: u8) -> f32 {
        if bits == 0 {
            return 0.0;
        }

        let sign = if bits & 0x20 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 3) & 0x03) as i32 - 1;
        let mant = ((bits & 0x07) as f32) / 8.0;

        sign * (1.0 + mant) * 2_f32.powi(exp)
    }

    /// Pack 6-bit values into bytes
    /// Packs values in little-endian bit order
    pub fn pack_6bit_values(values: &[u8]) -> Vec<u8> {
        let mut packed = vec![0u8; (values.len() * 6).div_ceil(8)];
        for (i, &val) in values.iter().enumerate() {
            let bit_pos = i * 6;
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            // Mask value to 6 bits
            let val_6bit = val & 0x3F;

            if bit_offset <= 2 {
                // Fits entirely in current byte (with room to spare)
                packed[byte_idx] |= val_6bit << bit_offset;
            } else {
                // Spans across two bytes
                let bits_in_first_byte = 8 - bit_offset;
                let _bits_in_second_byte = 6 - bits_in_first_byte;

                packed[byte_idx] |= val_6bit << bit_offset;
                packed[byte_idx + 1] |= val_6bit >> bits_in_first_byte;
            }
        }
        packed
    }

    /// Unpack 6-bit values from bytes
    /// Unpacks values in little-endian bit order
    pub fn unpack_6bit_values(packed: &[u8], count: usize) -> Vec<u8> {
        let mut values = vec![0u8; count];
        for i in 0..count {
            let bit_pos = i * 6;
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;

            if byte_idx < packed.len() {
                if bit_offset <= 2 {
                    // Value fits entirely in current byte
                    values[i] = (packed[byte_idx] >> bit_offset) & 0x3F;
                } else {
                    // Value spans two bytes
                    let bits_from_first_byte = 8 - bit_offset;
                    let bits_from_second_byte = 6 - bits_from_first_byte;

                    let first_part =
                        (packed[byte_idx] >> bit_offset) & ((1 << bits_from_first_byte) - 1);
                    let second_part = if byte_idx + 1 < packed.len() {
                        packed[byte_idx + 1] & ((1 << bits_from_second_byte) - 1)
                    } else {
                        0
                    };

                    values[i] = first_part | (second_part << bits_from_first_byte);
                }
            }
        }
        values
    }
}

#[cfg(test)]
#[path = "mxfp_tests.rs"]
mod mxfp_tests;
