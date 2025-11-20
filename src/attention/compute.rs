//! Matrix computation operations for attention

use crate::attention::AttentionError;

pub fn matmul_cpu(
    a: &[f32],
    b: &[f32],
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, AttentionError> {
    let a_len = batch_size * m * k;
    let b_len = batch_size * k * n;

    if a.len() != a_len || b.len() != b_len {
        return Err(AttentionError::ShapeMismatch(
            "Matrix dimensions don't match input data".to_string(),
        ));
    }

    let mut c = vec![0.0f32; batch_size * m * n];

    for batch in 0..batch_size {
        let a_offset = batch * m * k;
        let b_offset = batch * k * n;
        let c_offset = batch * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_idx = a_offset + i * k + l;
                    let b_idx = b_offset + l * n + j;
                    sum += a[a_idx] * b[b_idx];
                }
                let c_idx = c_offset + i * n + j;
                c[c_idx] = sum;
            }
        }
    }

    Ok(c)
}

pub fn apply_dropout(data: &mut [f32], dropout_prob: f32, seed: u32) {
    if dropout_prob <= 0.0f32 || dropout_prob >= 1.0f32 {
        return;
    }

    let mut rng_state = seed;
    let scale = 1.0f32 / (1.0f32 - dropout_prob);

    for i in 0..data.len() {
        // Simple LCG for deterministic dropout
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random_val = (rng_state as f32) / (u32::MAX as f32);

        if random_val < dropout_prob {
            data[i] = 0.0f32;
        } else {
            data[i] *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_cpu_basic() {
        let a = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]; // 2x2
        let b = vec![5.0f32, 6.0f32, 7.0f32, 8.0f32]; // 2x2

        let result = matmul_cpu(&a, &b, 1, 2, 2, 2).unwrap();
        let expected = vec![19.0f32, 22.0f32, 43.0f32, 50.0f32];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_dropout_deterministic() {
        let mut data = vec![1.0f32; 100];
        let original = data.clone();

        apply_dropout(&mut data, 0.5f32, 42);

        let mut changed = false;
        for i in 0..100 {
            if data[i] != original[i] {
                changed = true;
                break;
            }
        }
        assert!(changed);

        // Test determinism
        let mut data2 = vec![1.0f32; 100];
        apply_dropout(&mut data2, 0.5f32, 42);
        assert_eq!(data, data2);
    }
}
