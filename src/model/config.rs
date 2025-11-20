//! Model configuration structures

use serde::{Deserialize, Serialize};

/// Model type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Llama,
    Qwen,
}

/// Configuration for transformer model runtime
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub max_position_embeddings: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub model_type: ModelType,
    pub rms_norm_eps: f32,
    pub use_rotary_embeddings: bool,
}

impl ModelConfig {
    /// Create new model configuration
    pub fn new(
        num_hidden_layers: usize,
        num_attention_heads: usize,
        head_dim: usize,
        hidden_size: usize,
        max_position_embeddings: usize,
        intermediate_size: usize,
        vocab_size: usize,
        model_type: ModelType,
    ) -> Self {
        Self {
            num_hidden_layers,
            num_attention_heads,
            head_dim,
            hidden_size,
            max_position_embeddings,
            intermediate_size,
            vocab_size,
            model_type,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_hidden_layers == 0 {
            return Err("num_hidden_layers must be > 0".to_string());
        }
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".to_string());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".to_string());
        }
        if self.max_position_embeddings == 0 {
            return Err("max_position_embeddings must be > 0".to_string());
        }
        if self.intermediate_size == 0 {
            return Err("intermediate_size must be > 0".to_string());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }
        if self.hidden_size != self.num_attention_heads * self.head_dim {
            return Err(format!(
                "hidden_size ({}) must equal num_attention_heads ({}) * head_dim ({})",
                self.hidden_size, self.num_attention_heads, self.head_dim
            ));
        }
        Ok(())
    }

    /// Create LLaMA 7B configuration
    pub fn llama2_7b() -> Self {
        Self {
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: 128,
            hidden_size: 4096,
            max_position_embeddings: 2048,
            intermediate_size: 11008,
            vocab_size: 32000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        }
    }

    /// Create default LLaMA configuration
    pub fn default_llama() -> Self {
        Self {
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: 128,
            hidden_size: 4096,
            max_position_embeddings: 2048,
            intermediate_size: 11008,
            vocab_size: 32000,
            model_type: ModelType::Llama,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        }
    }

    /// Create default Qwen configuration
    pub fn default_qwen() -> Self {
        Self {
            num_hidden_layers: 24,
            num_attention_heads: 20,
            head_dim: 128,
            hidden_size: 2560,
            max_position_embeddings: 2048,
            intermediate_size: 13312,
            vocab_size: 151936,
            model_type: ModelType::Qwen,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
        }
    }
}
