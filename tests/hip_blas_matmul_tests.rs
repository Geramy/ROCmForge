//! hipBLAS and matrix multiplication tests for ROCmForge
//! Tests GPU matmul against CPU reference implementation

use rocmforge::backend::hip_backend::HipBuffer;
use rocmforge::backend::hip_blas::HipBlasHandle;
use rocmforge::tensor::matmul::{cpu_matmul_f32, matmul_f32};

/// Validate matrix multiplication dimensions
///
/// Returns Ok(()) if dimensions are valid for matmul C = A @ B
/// where A is (m, k) and B is (k, n), producing C (m, n)
fn validate_matmul_dims(
    m: i32,
    k: i32,
    n: i32,
    a_size: usize,
    b_size: usize,
    c_size: usize,
) -> Result<(), String> {
    // Check A dimensions
    let expected_a_size = (m * k) as usize;
    if a_size != expected_a_size {
        return Err(format!(
            "Matrix A size mismatch: expected {} elements (m={} * k={}), got {}",
            expected_a_size, m, k, a_size
        ));
    }

    // Check B dimensions
    let expected_b_size = (k * n) as usize;
    if b_size != expected_b_size {
        return Err(format!(
            "Matrix B size mismatch: expected {} elements (k={} * n={}), got {}",
            expected_b_size, k, n, b_size
        ));
    }

    // Check C dimensions
    let expected_c_size = (m * n) as usize;
    if c_size != expected_c_size {
        return Err(format!(
            "Matrix C size mismatch: expected {} elements (m={} * n={}), got {}",
            expected_c_size, m, n, c_size
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_blas_handle_creation_and_drop() {
        // Test that we can create and destroy a hipBLAS handle
        let handle = HipBlasHandle::new();

        assert!(handle.is_ok(), "hipBLAS handle creation should succeed");

        let handle = handle.unwrap();
        assert!(!handle.as_ptr().is_null(), "Handle should not be null");

        // Handle should be destroyed when dropped
    }

    #[test]
    fn test_hipblas_sgemm_simple() {
        // Test hipBLAS SGEMM with minimal parameters to check if it's working
        let handle = HipBlasHandle::new().unwrap();

        // Create tiny 1x1 matrices
        let a = vec![2.0f32];
        let b = vec![3.0f32];

        let gpu_a = HipBuffer::new(1 * std::mem::size_of::<f32>()).unwrap();
        let gpu_b = HipBuffer::new(1 * std::mem::size_of::<f32>()).unwrap();

        gpu_a.copy_from_host(&a).unwrap();
        gpu_b.copy_from_host(&b).unwrap();

        // Validate dimensions before matmul
        let m = 1;
        let k = 1;
        let n = 1;
        validate_matmul_dims(m, k, n, a.len(), b.len(), 1).expect("Dimension validation failed");

        // Simple 1x1 * 1x1 = 1x1 matrix multiplication
        let result = matmul_f32(&handle, &gpu_a, &gpu_b, m, k, n);

        assert!(result.is_ok(), "Simple 1x1 matmul should succeed");

        let gpu_c = result.unwrap();
        let mut host_result = vec![0.0f32; 1];
        gpu_c.copy_to_host(&mut host_result).unwrap();

        assert!(
            (host_result[0] - 6.0).abs() < 1e-6,
            "1x1 matmul: 2*3=6, got {}",
            host_result[0]
        );
    }

    #[test]
    fn test_gpu_matmul_matches_cpu_small() {
        // Test 2x2 * 2x2 case
        let m = 2;
        let n = 2;
        let k = 2;

        // Create simple deterministic matrices
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 row-major: [[1,2],[3,4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 row-major: [[5,6],[7,8]]

        // Validate dimensions
        validate_matmul_dims(m, k, n, a.len(), b.len(), (m * n) as usize)
            .expect("Dimension validation failed");

        // Expected result: [[19,22],[43,50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        // Compute CPU reference
        let cpu_result = cpu_matmul_f32(&a, &b, m as usize, n as usize, k as usize);

        // Verify CPU computation
        assert_eq!(cpu_result.len(), expected.len());
        for (i, &val) in cpu_result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "CPU matmul element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }

        // Test GPU matmul (will fail until implemented)
        let handle = HipBlasHandle::new().unwrap();

        let gpu_a = HipBuffer::new((m * k) as usize * std::mem::size_of::<f32>()).unwrap();
        let gpu_b = HipBuffer::new((k * n) as usize * std::mem::size_of::<f32>()).unwrap();

        let gpu_result = matmul_f32(&handle, &gpu_a, &gpu_b, m as i32, n as i32, k as i32);
        // Copy data to GPU
        gpu_a.copy_from_host(&a).unwrap();
        gpu_b.copy_from_host(&b).unwrap();

        // Perform GPU matmul
        let gpu_c = matmul_f32(&handle, &gpu_a, &gpu_b, m as i32, n as i32, k as i32).unwrap();

        // Copy result back from GPU
        let mut gpu_result = vec![0.0f32; (m * n) as usize];
        gpu_c.copy_to_host(&mut gpu_result).unwrap();

        // Compare GPU vs CPU results
        assert_eq!(gpu_result.len(), expected.len());
        for (i, &val) in gpu_result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "GPU matmul element {} mismatch: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    fn test_gpu_matmul_larger_matrix() {
        // Test 4x3 * 3x2 case
        let m = 4;
        let n = 2;
        let k = 3;

        // Create structured deterministic matrices
        let a = vec![
            1.0, 2.0, 3.0, // Row 0
            4.0, 5.0, 6.0, // Row 1
            7.0, 8.0, 9.0, // Row 2
            10.0, 11.0, 12.0, // Row 3
        ];

        let b = vec![
            13.0, 14.0, // Row 0
            15.0, 16.0, // Row 1
            17.0, 18.0, // Row 2
        ];

        // Compute CPU reference
        let cpu_result = cpu_matmul_f32(&a, &b, m as usize, n as usize, k as usize);

        // Verify dimensions
        assert_eq!(cpu_result.len(), (m * n) as usize);

        // Test GPU matmul
        let handle = HipBlasHandle::new().unwrap();

        let gpu_a = HipBuffer::new((m * k) as usize * std::mem::size_of::<f32>()).unwrap();
        let gpu_b = HipBuffer::new((k * n) as usize * std::mem::size_of::<f32>()).unwrap();

        // Copy data to GPU
        gpu_a.copy_from_host(&a).unwrap();
        gpu_b.copy_from_host(&b).unwrap();

        // Perform GPU matmul
        let gpu_c = matmul_f32(&handle, &gpu_a, &gpu_b, m as i32, n as i32, k as i32).unwrap();

        // Copy result back from GPU
        let mut gpu_result = vec![0.0f32; (m * n) as usize];
        gpu_c.copy_to_host(&mut gpu_result).unwrap();

        // Compare GPU vs CPU results
        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (&cpu_val, &gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            assert!(
                (gpu_val - cpu_val).abs() < 1e-6,
                "GPU matmul element {} mismatch: expected {}, got {}",
                i,
                cpu_val,
                gpu_val
            );
        }
    }

    #[test]
    fn test_matmul_invalid_dims_error() {
        // Test dimension mismatch: 2x3 * 4x2 (k=3 vs k'=4)
        let m = 2;
        let n = 2;
        let k = 3;
        let k_prime = 4; // Mismatched inner dimension

        let handle = HipBlasHandle::new().unwrap();

        let gpu_a = HipBuffer::new((m * k) as usize * std::mem::size_of::<f32>()).unwrap();
        let gpu_b = HipBuffer::new((k_prime * n) as usize * std::mem::size_of::<f32>()).unwrap();

        // Validate dimensions - should detect mismatch
        let validation_result = validate_matmul_dims(
            m as i32,
            k as i32,
            n as i32,
            (m * k) as usize,
            (k_prime * n) as usize,
            (m * n) as usize,
        );

        assert!(
            validation_result.is_err(),
            "Dimension validation should detect k dimension mismatch: k={} vs k'={}",
            k,
            k_prime
        );

        // This should fail due to dimension mismatch
        let result = matmul_f32(&handle, &gpu_a, &gpu_b, m as i32, n as i32, k as i32);

        // Expect error due to dimension mismatch
        match result {
            Ok(_) => panic!("Expected dimension mismatch error"),
            Err(_) => (), // Expected
        }
    }
}
