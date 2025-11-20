//! Causal masking operations for attention

pub fn create_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_pattern() {
        let mask = create_causal_mask(3);

        // Expected pattern:
        // [  0, -inf, -inf]
        // [  0,   0, -inf]
        // [  0,   0,   0]

        assert_eq!(mask[0], 0.0f32);
        assert_eq!(mask[1], f32::NEG_INFINITY);
        assert_eq!(mask[2], f32::NEG_INFINITY);
        assert_eq!(mask[3], 0.0f32);
        assert_eq!(mask[4], 0.0f32);
        assert_eq!(mask[5], f32::NEG_INFINITY);
        assert_eq!(mask[6], 0.0f32);
        assert_eq!(mask[7], 0.0f32);
        assert_eq!(mask[8], 0.0f32);
    }
}
