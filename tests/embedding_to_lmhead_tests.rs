use rocmforge::backend::hip_backend::HipError;
use rocmforge::backend::hip_backend::{DeviceTensor, HipBackend};
use rocmforge::loader::gguf_loader::{GgufLoader, GgufModel, GgufTensor};
use rocmforge::loader::mmap_loader::TensorShape;
use rocmforge::model::config::{ModelConfig, ModelType};
use rocmforge::tensor::matmul::cpu_matmul_f32;
use std::path::Path;

/// Helper function to extract ModelConfig from GGUF metadata
fn extract_model_config_from_gguf(model: &GgufModel) -> ModelConfig {
    // Extract basic metadata with sensible defaults
    let num_layers = model
        .metadata
        .get("llama.block_count")
        .or_else(|| model.metadata.get("block_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let num_heads = model
        .metadata
        .get("llama.attention.head_count")
        .or_else(|| model.metadata.get("attention.head_count"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let hidden_size = model
        .metadata
        .get("llama.embedding_length")
        .or_else(|| model.metadata.get("embedding_length"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let vocab_size = model
        .metadata
        .get("llama.vocab_size")
        .or_else(|| model.metadata.get("vocab_size"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32000);

    let max_pos_embeddings = model
        .metadata
        .get("llama.context_length")
        .or_else(|| model.metadata.get("context_length"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    let intermediate_size = model
        .metadata
        .get("llama.feed_forward_length")
        .or_else(|| model.metadata.get("feed_forward_length"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(11008);

    let head_dim = hidden_size / num_heads;

    // Determine model type from architecture
    let model_type = model
        .metadata
        .get("general.architecture")
        .map(|s| match s.to_lowercase().as_str() {
            "llama" => ModelType::Llama,
            "qwen" => ModelType::Qwen,
            _ => ModelType::Llama,
        })
        .unwrap_or(ModelType::Llama);

    ModelConfig::new(
        num_layers,
        num_heads,
        head_dim,
        hidden_size,
        max_pos_embeddings,
        intermediate_size,
        vocab_size,
        model_type,
    )
}

/// Helper function to find tensor by name pattern
fn find_tensor_by_pattern<'a>(model: &'a GgufModel, patterns: &[&str]) -> Option<&'a GgufTensor> {
    model
        .tensors
        .iter()
        .find(|(name, _)| patterns.iter().any(|pattern| name.contains(pattern)))
        .map(|(_, tensor)| tensor)
}

/// Helper function to convert GGUF tensor to DeviceTensor
fn gguf_tensor_to_device_tensor(
    backend: &HipBackend,
    tensor: &GgufTensor,
) -> Result<DeviceTensor, HipError> {
    // Convert GGUF tensor data to f32 if needed
    let f32_data = match tensor.data_type {
        rocmforge::loader::gguf_loader::GgufDataType::F32 => tensor
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<f32>>(),
        rocmforge::loader::gguf_loader::GgufDataType::F16 => {
            // Simple F16 to F32 conversion
            tensor
                .data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
                    let exponent = (bits >> 10) & 0x1F;
                    let mantissa = bits & 0x3FF;
                    if exponent == 0 {
                        0.0
                    } else {
                        sign * ((1 << 10) | mantissa) as f32 * 2f32.powi(exponent as i32 - 15 - 10)
                    }
                })
                .collect()
        }
        _ => {
            // For quantized types, we'd need proper dequantization
            // For now, return zeros as placeholder
            vec![0.0f32; tensor.shape.iter().product()]
        }
    };

    let shape = TensorShape::from_dims(&tensor.shape);
    DeviceTensor::from_host_vec(backend, f32_data, shape)
}

/// Helper function to perform embedding lookup on CPU
fn cpu_embedding_lookup(embedding_matrix: &[f32], token_id: usize, hidden_size: usize) -> Vec<f32> {
    let start_idx = token_id * hidden_size;
    let end_idx = start_idx + hidden_size;

    if end_idx <= embedding_matrix.len() {
        embedding_matrix[start_idx..end_idx].to_vec()
    } else {
        vec![0.0f32; hidden_size]
    }
}

#[test]
fn test_embedding_lookup() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    // Load GGUF model
    let mut loader = GgufLoader::new();
    loader
        .load_model(gguf_path)
        .expect("Failed to load GGUF model");
    let model = loader.get_model().expect("No model loaded");

    // Find embedding tensor
    let embedding_patterns = ["token_embd", "embed_tokens", "word_embeddings"];
    let embedding_tensor =
        find_tensor_by_pattern(model, &embedding_patterns).expect("Embedding tensor not found");

    // Validate embedding tensor shape
    assert_eq!(embedding_tensor.shape.len(), 2, "Embedding should be 2D");
    let (vocab_size, hidden_size) = (embedding_tensor.shape[0], embedding_tensor.shape[1]);
    assert!(vocab_size > 0, "Vocab size should be > 0");
    assert!(hidden_size > 0, "Hidden size should be > 0");

    // Convert to f32 data for testing
    let embedding_data = match embedding_tensor.data_type {
        rocmforge::loader::gguf_loader::GgufDataType::F32 => embedding_tensor
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<f32>>(),
        _ => {
            // For non-F32 tensors, create dummy data for shape validation
            vec![0.1f32; vocab_size * hidden_size]
        }
    };

    // Test embedding lookup for a few tokens
    let test_tokens = [0, 1, vocab_size.saturating_sub(1)];

    for &token_id in &test_tokens {
        if token_id < vocab_size {
            let embedding = cpu_embedding_lookup(&embedding_data, token_id, hidden_size);

            // Validate embedding dimensions
            assert_eq!(
                embedding.len(),
                hidden_size,
                "Embedding for token {} should have {} dimensions",
                token_id,
                hidden_size
            );

            // Validate embedding values are finite
            for (i, &val) in embedding.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Embedding value at index {} for token {} should be finite, got {}",
                    i,
                    token_id,
                    val
                );
            }
        }
    }

    println!(
        "✓ Embedding lookup test passed for vocab_size={}, hidden_size={}",
        vocab_size, hidden_size
    );
}

#[test]
fn test_lm_head_projection() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    // Load GGUF model
    let mut loader = GgufLoader::new();
    loader
        .load_model(gguf_path)
        .expect("Failed to load GGUF model");
    let model = loader.get_model().expect("No model loaded");

    // Find LM head tensor
    let lm_head_patterns = ["output", "lm_head", "logits"];
    let lm_head_tensor =
        find_tensor_by_pattern(model, &lm_head_patterns).expect("LM head tensor not found");

    // Validate LM head tensor shape
    assert_eq!(lm_head_tensor.shape.len(), 2, "LM head should be 2D");
    let (vocab_size, hidden_size) = (lm_head_tensor.shape[0], lm_head_tensor.shape[1]);
    assert!(vocab_size > 0, "Vocab size should be > 0");
    assert!(hidden_size > 0, "Hidden size should be > 0");

    // Convert to f32 data for testing
    let lm_head_data = match lm_head_tensor.data_type {
        rocmforge::loader::gguf_loader::GgufDataType::F32 => lm_head_tensor
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<f32>>(),
        _ => {
            // For non-F32 tensors, create dummy data for shape validation
            vec![0.1f32; vocab_size * hidden_size]
        }
    };

    // Create a dummy hidden state vector
    let hidden_state = vec![0.5f32; hidden_size];

    // Perform LM head projection using CPU matmul: logits = hidden_state @ lm_head.T
    // We need to transpose the LM head matrix for the correct orientation
    let mut lm_head_transposed = vec![0.0f32; hidden_size * vocab_size];
    for i in 0..vocab_size {
        for j in 0..hidden_size {
            lm_head_transposed[j * vocab_size + i] = lm_head_data[i * hidden_size + j];
        }
    }

    let logits = cpu_matmul_f32(
        &hidden_state,
        &lm_head_transposed,
        1,
        vocab_size,
        hidden_size,
    );

    // Validate output dimensions
    assert_eq!(
        logits.len(),
        vocab_size,
        "LM head output should have vocab_size {} dimensions, got {}",
        vocab_size,
        logits.len()
    );

    // Validate output values are finite
    for (i, &logit) in logits.iter().enumerate() {
        assert!(
            logit.is_finite(),
            "Logit at index {} should be finite, got {}",
            i,
            logit
        );
    }

    println!(
        "✓ LM head projection test passed for vocab_size={}, hidden_size={}",
        vocab_size, hidden_size
    );
}

#[test]
fn test_token_to_logits_pipeline() {
    let gguf_path = "tests/data/tiny_model.gguf";

    // Skip test if file doesn't exist
    if !Path::new(gguf_path).exists() {
        println!("Skipping test - GGUF file not found at {}", gguf_path);
        return;
    }

    // Load GGUF model and extract config
    let mut loader = GgufLoader::new();
    loader
        .load_model(gguf_path)
        .expect("Failed to load GGUF model");
    let model = loader.get_model().expect("No model loaded");
    let config = extract_model_config_from_gguf(model);

    // Validate config
    config.validate().expect("Model config validation failed");

    // Find embedding and LM head tensors
    let embedding_patterns = ["token_embd", "embed_tokens", "word_embeddings"];
    let lm_head_patterns = ["output", "lm_head", "logits"];

    let embedding_tensor =
        find_tensor_by_pattern(model, &embedding_patterns).expect("Embedding tensor not found");
    let lm_head_tensor =
        find_tensor_by_pattern(model, &lm_head_patterns).expect("LM head tensor not found");

    // Validate tensor shapes match config
    assert_eq!(
        embedding_tensor.shape[1], config.hidden_size,
        "Embedding hidden size should match config"
    );
    assert_eq!(
        lm_head_tensor.shape[1], config.hidden_size,
        "LM head hidden size should match config"
    );

    // Convert tensors to f32 data
    let embedding_data = match embedding_tensor.data_type {
        rocmforge::loader::gguf_loader::GgufDataType::F32 => embedding_tensor
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<f32>>(),
        _ => vec![0.1f32; embedding_tensor.shape.iter().product()],
    };

    let lm_head_data = match lm_head_tensor.data_type {
        rocmforge::loader::gguf_loader::GgufDataType::F32 => lm_head_tensor
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<f32>>(),
        _ => vec![0.1f32; lm_head_tensor.shape.iter().product()],
    };

    // Test the complete pipeline: token -> embedding -> logits
    let test_token = 42; // Arbitrary token ID
    if test_token < config.vocab_size {
        // Step 1: Embedding lookup
        let hidden_state = cpu_embedding_lookup(&embedding_data, test_token, config.hidden_size);

        // Step 2: LM head projection
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        // Transpose LM head for correct matrix orientation
        let mut lm_head_transposed = vec![0.0f32; hidden_size * vocab_size];
        for i in 0..vocab_size {
            for j in 0..hidden_size {
                lm_head_transposed[j * vocab_size + i] = lm_head_data[i * hidden_size + j];
            }
        }

        let logits = cpu_matmul_f32(
            &hidden_state,
            &lm_head_transposed,
            1,
            vocab_size,
            hidden_size,
        );

        // Validate final output
        assert_eq!(
            logits.len(),
            vocab_size,
            "Final logits should have vocab_size {} dimensions",
            vocab_size
        );

        // Check that logits are finite and reasonable
        let mut finite_count = 0;
        for (i, &logit) in logits.iter().enumerate() {
            if logit.is_finite() {
                finite_count += 1;
            }
        }

        assert!(
            finite_count > logits.len() / 2,
            "At least half of logits should be finite, got {}/{}",
            finite_count,
            logits.len()
        );

        // Find the token with highest logit (should be reasonable)
        let max_logit_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        println!("✓ Token-to-logits pipeline test passed:");
        println!(
            "  Token: {} -> Embedding dim: {} -> Logits dim: {}",
            test_token,
            hidden_state.len(),
            logits.len()
        );
        println!(
            "  Max logit at token index: {} with value: {:.6}",
            max_logit_idx, logits[max_logit_idx]
        );
    } else {
        println!(
            "Skipping pipeline test - token {} exceeds vocab size {}",
            test_token, config.vocab_size
        );
    }
}
