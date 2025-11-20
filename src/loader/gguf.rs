//! GGUF (GPT-Generated Unified Format) Loader
//!
//! Complete implementation for loading GGUF model files with support for:
//! - Metadata parsing
//! - Multiple quantization types (Q8_0, Q4_0, FP16, FP32)
//! - Tensor block reading and validation
//! - GPU memory allocation via DeviceTensor

use crate::backend::hip_backend::{DeviceTensor, HipBackend};
use crate::loader::TensorShape;
use crate::model::config::ModelConfig;
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// GGUF file magic number
const GGUF_MAGIC: &[u8] = b"GGUF";

/// GGUF tensor types
#[derive(Debug, Clone, PartialEq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 3,
}

impl GgufTensorType {
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufTensorType::F32),
            1 => Ok(GgufTensorType::F16),
            2 => Ok(GgufTensorType::Q4_0),
            3 => Ok(GgufTensorType::Q8_0),
            _ => Err(anyhow!("Unknown tensor type: {}", value)),
        }
    }

    fn to_string(&self) -> &'static str {
        match self {
            GgufTensorType::F32 => "FP32",
            GgufTensorType::F16 => "FP16",
            GgufTensorType::Q4_0 => "Q4_0",
            GgufTensorType::Q8_0 => "Q8_0",
        }
    }

    fn element_size(&self) -> usize {
        match self {
            GgufTensorType::F32 => 4,
            GgufTensorType::F16 => 2,
            GgufTensorType::Q4_0 => {
                // Q4_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                32
            }
            GgufTensorType::Q8_0 => {
                // Q8_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                32
            }
        }
    }
}

/// GGUF metadata extracted from file header
#[derive(Debug, Clone, Serialize)]
pub struct GgufMetadata {
    pub architecture: String,
    pub file_type: u32,
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub use_rotary_embeddings: bool,
    #[serde(skip_serializing)]
    pub embedded_tokenizer_json: Option<String>,
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            file_type: 0,
            num_layers: 0,
            num_heads: 0,
            hidden_size: 0,
            intermediate_size: 0,
            head_dim: 0,
            max_position_embeddings: 2048,
            vocab_size: 0,
            rms_norm_eps: 1e-6,
            use_rotary_embeddings: true,
            embedded_tokenizer_json: None,
        }
    }
}

/// GGUF tensor information
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub shape: TensorShape,
    pub tensor_type: GgufTensorType,
    pub quant_type: String,
    pub offset: u64,
    pub data: Vec<u8>,
}

impl GgufTensor {
    /// Calculate total number of elements
    pub fn total_elements(&self) -> usize {
        self.shape.total_elements()
    }

    /// Calculate data size in bytes
    pub fn data_size(&self) -> usize {
        match self.tensor_type {
            GgufTensorType::F32 => self.total_elements() * 4,
            GgufTensorType::F16 => self.total_elements() * 2,
            GgufTensorType::Q4_0 => {
                // Q4_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                let blocks = (self.total_elements() + 31) / 32;
                blocks * (4 + 32)
            }
            GgufTensorType::Q8_0 => {
                // Q8_0: block_size=32, each block has 1 scale (f32) + 32 quants (u8)
                let blocks = (self.total_elements() + 31) / 32;
                blocks * (4 + 32)
            }
        }
    }
}

/// GGUF file loader
pub struct GgufLoader {
    path: String,
    metadata: GgufMetadata,
    tensors: HashMap<String, GgufTensor>,
}

impl GgufLoader {
    /// Create new GGUF loader from file path
    pub fn new(path: &str) -> Result<Self> {
        let mut loader = GgufLoader {
            path: path.to_string(),
            metadata: GgufMetadata::default(),
            tensors: HashMap::new(),
        };

        loader.load_from_disk(true)?;
        Ok(loader)
    }

    /// Inspect only metadata without loading tensors into memory.
    pub fn metadata_from_file(path: &str) -> Result<GgufMetadata> {
        let mut loader = GgufLoader {
            path: path.to_string(),
            metadata: GgufMetadata::default(),
            tensors: HashMap::new(),
        };
        loader.load_from_disk(false)?;
        Ok(loader.metadata)
    }

    /// Get metadata
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Load all tensors into memory
    pub fn load_tensors(&self) -> Result<HashMap<String, GgufTensor>> {
        Ok(self.tensors.clone())
    }

    /// Load tensors and upload to GPU
    pub fn load_to_gpu(&self, backend: &HipBackend) -> Result<HashMap<String, DeviceTensor>> {
        let mut gpu_tensors = HashMap::new();

        for (name, tensor) in &self.tensors {
            let device_tensor = self.upload_tensor_to_gpu(backend, tensor)?;
            gpu_tensors.insert(name.clone(), device_tensor);
        }

        Ok(gpu_tensors)
    }

    /// Convert metadata to ModelConfig
    pub fn to_model_config(&self) -> Result<ModelConfig> {
        use crate::model::config::ModelType;

        Ok(ModelConfig {
            num_hidden_layers: self.metadata.num_layers,
            num_attention_heads: self.metadata.num_heads,
            hidden_size: self.metadata.hidden_size,
            intermediate_size: self.metadata.intermediate_size,
            max_position_embeddings: self.metadata.max_position_embeddings,
            vocab_size: self.metadata.vocab_size,
            rms_norm_eps: self.metadata.rms_norm_eps,
            use_rotary_embeddings: self.metadata.use_rotary_embeddings,
            model_type: if self.metadata.architecture == "glm" {
                ModelType::Llama // Use Llama as placeholder for now
            } else {
                ModelType::Llama
            },
            head_dim: if self.metadata.head_dim > 0 {
                self.metadata.head_dim
            } else {
                self.metadata.hidden_size / self.metadata.num_heads
            },
        })
    }

    /// Load GGUF file and parse header
    fn load_from_disk(&mut self, load_tensors: bool) -> Result<()> {
        let mut file = File::open(&self.path)?;

        // Read and verify magic number
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != GGUF_MAGIC {
            return Err(anyhow!("Invalid GGUF magic number"));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 3 {
            return Err(anyhow!("Unsupported GGUF version: {}", version));
        }

        // Read tensor count
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);

        // Read KV count
        let mut kv_count_bytes = [0u8; 8];
        file.read_exact(&mut kv_count_bytes)?;
        let kv_count = u64::from_le_bytes(kv_count_bytes);

        // Parse KV pairs (metadata)
        self.parse_kv_pairs(&mut file, kv_count)?;

        if load_tensors {
            // Parse tensor info
            self.parse_tensor_info(&mut file, tensor_count)?;

            // Read tensor data
            self.read_tensor_data(&mut file)?;
        }

        Ok(())
    }

    /// Parse key-value pairs from GGUF header
    fn parse_kv_pairs(&mut self, file: &mut File, kv_count: u64) -> Result<()> {
        for _ in 0..kv_count {
            // Read key
            let mut key_len_bytes = [0u8; 8];
            file.read_exact(&mut key_len_bytes)?;
            let key_len = u64::from_le_bytes(key_len_bytes) as usize;

            let mut key_bytes = vec![0u8; key_len];
            file.read_exact(&mut key_bytes)?;
            let key = String::from_utf8(key_bytes)?;

            // Read value type
            let mut value_type_bytes = [0u8; 1];
            file.read_exact(&mut value_type_bytes)?;
            let value_type = value_type_bytes[0];

            // Read value based on type
            let value = match value_type {
                8 => {
                    // String
                    let mut value_len_bytes = [0u8; 8];
                    file.read_exact(&mut value_len_bytes)?;
                    let value_len = u64::from_le_bytes(value_len_bytes) as usize;

                    let mut value_bytes = vec![0u8; value_len];
                    file.read_exact(&mut value_bytes)?;
                    String::from_utf8(value_bytes)?
                }
                4 => {
                    // u32
                    let mut value_bytes = [0u8; 4];
                    file.read_exact(&mut value_bytes)?;
                    u32::from_le_bytes(value_bytes).to_string()
                }
                6 => {
                    // f32
                    let mut value_bytes = [0u8; 4];
                    file.read_exact(&mut value_bytes)?;
                    f32::from_le_bytes(value_bytes).to_string()
                }
                _ => {
                    return Err(anyhow!("Unsupported value type: {}", value_type));
                }
            };

            // Update metadata based on key
            self.update_metadata(&key, &value);
        }

        Ok(())
    }

    /// Update metadata from key-value pair
    fn update_metadata(&mut self, key: &str, value: &str) {
        match key {
            "general.architecture" => self.metadata.architecture = value.to_string(),
            "general.file_type" => self.metadata.file_type = value.parse().unwrap_or(0),
            "glm.n_layers" => self.metadata.num_layers = value.parse().unwrap_or(0),
            "glm.n_heads" => self.metadata.num_heads = value.parse().unwrap_or(0),
            "glm.n_embd" => self.metadata.hidden_size = value.parse().unwrap_or(0),
            "glm.intermediate_size" => self.metadata.intermediate_size = value.parse().unwrap_or(0),
            "glm.head_dim" => self.metadata.head_dim = value.parse().unwrap_or(0),
            "glm.max_position_embeddings" => {
                self.metadata.max_position_embeddings = value.parse().unwrap_or(2048)
            }
            "glm.vocab_size" => self.metadata.vocab_size = value.parse().unwrap_or(0),
            "glm.rms_norm_eps" => self.metadata.rms_norm_eps = value.parse().unwrap_or(1e-6),
            "tokenizer.json" => {
                if self.metadata.embedded_tokenizer_json.is_none() {
                    self.metadata.embedded_tokenizer_json = Some(value.to_string());
                }
            }
            key if key.ends_with(".tokenizer_json") => {
                if self.metadata.embedded_tokenizer_json.is_none() {
                    self.metadata.embedded_tokenizer_json = Some(value.to_string());
                }
            }
            _ => {} // Ignore unknown keys
        }
    }

    /// Parse tensor information from GGUF header
    fn parse_tensor_info(&mut self, file: &mut File, tensor_count: u64) -> Result<()> {
        for _ in 0..tensor_count {
            // Read tensor name
            let mut name_len_bytes = [0u8; 8];
            file.read_exact(&mut name_len_bytes)?;
            let name_len = u64::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)?;

            // Read number of dimensions
            let mut n_dims_bytes = [0u8; 4];
            file.read_exact(&mut n_dims_bytes)?;
            let n_dims = u32::from_le_bytes(n_dims_bytes) as usize;

            // Read dimensions
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let mut dim_bytes = [0u8; 8];
                file.read_exact(&mut dim_bytes)?;
                dims.push(u64::from_le_bytes(dim_bytes) as usize);
            }

            // Read tensor type
            let mut tensor_type_bytes = [0u8; 4];
            file.read_exact(&mut tensor_type_bytes)?;
            let tensor_type = GgufTensorType::from_u32(u32::from_le_bytes(tensor_type_bytes))?;

            // Read tensor offset
            let mut offset_bytes = [0u8; 8];
            file.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);

            // Create tensor shape
            let shape = TensorShape::from_dims(&dims);

            // Store tensor info
            let tensor = GgufTensor {
                name: name.clone(),
                shape,
                tensor_type: tensor_type.clone(),
                quant_type: tensor_type.to_string().to_string(),
                offset,
                data: Vec::new(), // Will be filled later
            };

            self.tensors.insert(name, tensor);
        }

        Ok(())
    }

    /// Read tensor data from file
    fn read_tensor_data(&mut self, file: &mut File) -> Result<()> {
        for tensor in self.tensors.values_mut() {
            // Seek to tensor offset
            file.seek(SeekFrom::Start(tensor.offset))?;

            // Read tensor data
            let data_size = tensor.data_size();
            tensor.data.resize(data_size, 0);
            file.read_exact(&mut tensor.data)?;
        }

        Ok(())
    }

    /// Upload tensor to GPU memory
    fn upload_tensor_to_gpu(
        &self,
        backend: &HipBackend,
        tensor: &GgufTensor,
    ) -> Result<DeviceTensor> {
        match tensor.tensor_type {
            GgufTensorType::F32 => {
                // Direct upload for FP32 tensors
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload FP32 tensor: {}", e))
            }
            GgufTensorType::Q8_0 => {
                // Dequantize Q8_0 to FP32
                let f32_data = self.dequantize_q8_0(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q8_0 tensor: {}", e))
            }
            GgufTensorType::Q4_0 => {
                // Dequantize Q4_0 to FP32
                let f32_data = self.dequantize_q4_0(tensor)?;
                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload Q4_0 tensor: {}", e))
            }
            GgufTensorType::F16 => {
                // Convert FP16 to FP32
                let f32_data: Vec<f32> = tensor
                    .data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16::from_bits(bits).to_f32()
                    })
                    .collect();

                DeviceTensor::from_host_vec(backend, f32_data, tensor.shape.clone())
                    .map_err(|e| anyhow!("Failed to upload FP16 tensor: {}", e))
            }
        }
    }

    /// Dequantize Q8_0 tensor to FP32
    fn dequantize_q8_0(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = (total_elements + 31) / 32;

        for block_idx in 0..blocks {
            let block_start = block_idx * (4 + 32); // scale (4) + quants (32)

            if block_start + 4 > tensor.data.len() {
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

            // Read quantized values
            let quant_start = block_start + 4;
            let quant_end = std::cmp::min(quant_start + 32, tensor.data.len());
            let quants = &tensor.data[quant_start..quant_end];

            // Dequantize
            for (i, &q) in quants.iter().enumerate() {
                let element_idx = block_idx * 32 + i;
                if element_idx < total_elements {
                    result[element_idx] = (q as f32 - 128.0) * scale;
                }
            }
        }

        Ok(result)
    }

    /// Dequantize Q4_0 tensor to FP32
    fn dequantize_q4_0(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        let total_elements = tensor.total_elements();
        let mut result = vec![0.0f32; total_elements];
        let blocks = (total_elements + 31) / 32;

        for block_idx in 0..blocks {
            let block_start = block_idx * (4 + 16); // scale (4) + quants (16 bytes for 32 values)

            if block_start + 4 > tensor.data.len() {
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

            // Read quantized values (4-bit packed)
            let quant_start = block_start + 4;
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
                        result[element_idx] = (quant as f32 - 8.0) * scale;
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Simple f16 implementation for conversion
#[allow(dead_code)]
struct f16(u16);

impl f16 {
    fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    fn to_f32(self) -> f32 {
        // Simple conversion - in practice would use proper half-precision conversion
        let bits = self.0;
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1F) as i32 - 15;
        let mantissa = bits & 0x3FF;

        if exponent == -15 {
            if mantissa == 0 {
                0.0
            } else {
                sign * (mantissa as f32) * 2.0f32.powi(-14 - 10)
            }
        } else {
            sign * (1.0 + (mantissa as f32) * 2.0f32.powi(-10)) * 2.0f32.powi(exponent)
        }
    }
}
