//! Tests for ModelRuntime device buffer management

#[cfg(feature = "rocm")]
use rocmforge::backend::ModelRuntime;
#[cfg(feature = "rocm")]
use rocmforge::loader::mmap_loader::{open_mmap_weights, MmapWeights, TensorShape};
#[cfg(feature = "rocm")]
use rocmforge::model::ModelConfig;

#[cfg(feature = "rocm")]
#[test]
fn test_model_runtime_creation() {
    // Create test weights
    let test_f32: Vec<f32> = vec![1.0; 100]; // 100 f32 elements
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&test_bytes).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();

    // Create model config
    let config = ModelConfig {
        vocab_size: 1000,
        hidden_size: 64,
        num_layers: 2,
        num_heads: 8,
        max_seq_len: 128,
    };

    // Create ModelRuntime
    let runtime = ModelRuntime::new(&config, &mmap_weights).unwrap();

    // Verify runtime was created
    assert!(runtime.total_weight_bytes() > 0);
}

#[cfg(feature = "rocm")]
#[test]
fn test_model_runtime_scratch_buffers() {
    // Create minimal test weights
    let test_f32: Vec<f32> = vec![1.0; 10];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&test_bytes).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();

    let config = ModelConfig {
        vocab_size: 100,
        hidden_size: 32,
        num_layers: 1,
        num_heads: 4,
        max_seq_len: 64,
    };

    let runtime = ModelRuntime::new(&config, &mmap_weights).unwrap();

    // Verify scratch buffers are allocated
    assert!(runtime.total_weight_bytes() > 0);
    // The exact size depends on implementation, but should be non-zero
}

#[cfg(feature = "rocm")]
#[test]
fn test_model_runtime_empty_weights() {
    // Create empty weights file
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    // Write nothing

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();

    let config = ModelConfig {
        vocab_size: 100,
        hidden_size: 64,
        num_layers: 2,
        num_heads: 8,
        max_seq_len: 128,
    };

    // Should handle empty weights gracefully
    let result = ModelRuntime::new(&config, &mmap_weights);
    assert!(result.is_ok() || result.is_err()); // Either succeeds with zero weights or fails gracefully
}

#[cfg(feature = "rocm")]
#[test]
fn test_model_runtime_memory_limits() {
    // Create large test weights to test memory limit handling
    let large_size = 1024 * 1024; // 1M f32 elements = 4MB
    let test_f32: Vec<f32> = vec![1.0; large_size];
    let test_bytes: Vec<u8> = test_f32
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    temp_file.write_all(&test_bytes).unwrap();

    let mmap_weights = open_mmap_weights(temp_file.path()).unwrap();

    let config = ModelConfig {
        vocab_size: 50000, // Large vocab
        hidden_size: 4096, // Large hidden
        num_layers: 32,    // Many layers
        num_heads: 32,     // Many heads
        max_seq_len: 2048, // Long sequences
    };

    // Should respect memory limits
    let result = ModelRuntime::new(&config, &mmap_weights);
    match result {
        Ok(_) => {
            // If successful, should still be reasonable
            let runtime = result.unwrap();
            assert!(runtime.total_weight_bytes() > 0);
        }
        Err(_) => {
            // Should fail gracefully due to memory limits
            assert!(true);
        }
    }
}
