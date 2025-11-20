//! Softmax operations for attention mechanism

pub fn softmax_in_place(data: &mut [f32], batch_size: usize, seq_len: usize) {
    let total_rows = batch_size * seq_len;

    for row_idx in 0..total_rows {
        let row_start = row_idx * seq_len;

        if row_start + seq_len > data.len() {
            break; // Avoid out of bounds
        }

        let row_end = row_start + seq_len;

        // Find max for numerical stability
        let max_val = data[row_start..row_end]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let mut sum = 0.0f32;
        for j in row_start..row_end {
            data[j] = (data[j] - max_val).exp();
            sum += data[j];
        }

        // Normalize
        for j in row_start..row_end {
            data[j] /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        // Test with a single batch, single row of attention scores (seq_len=3)
        // This represents a scores[1][3] matrix flattened to [1.0, 2.0, 3.0]
        let mut data = vec![1.0f32, 2.0f32, 3.0f32];
        softmax_in_place(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_stability() {
        // Test with a single batch, single row of attention scores (seq_len=3)
        // This represents a scores[1][3] matrix flattened to [1000.0, 1001.0, 1002.0]
        let mut data = vec![1000.0f32, 1001.0f32, 1002.0f32];
        softmax_in_place(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        for &val in &data {
            assert!(val > 0.0f32 && val <= 1.0f32);
        }
    }
}
