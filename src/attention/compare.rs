//! Utility functions for comparing CPU and GPU attention outputs

/// Compute maximum absolute difference between two tensors
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Tensor lengths must match");

    let mut max_diff = 0.0f32;
    for (&a_val, &b_val) in a.iter().zip(b.iter()) {
        let diff = (a_val - b_val).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}
