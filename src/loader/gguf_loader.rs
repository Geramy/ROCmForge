//! GGUF model loader for CPU-side model loading

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("Invalid GGUF magic number")]
    InvalidMagic,
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Invalid tensor data type: {0}")]
    InvalidDataType(u32),
}

pub type GgufResult<T> = Result<T, GgufError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GgufDataType {
    F32,
    F16,
    Q8_0,
    Q4_0,
    I32,
    I16,
    I8,
}

impl GgufDataType {
    pub fn from_u32(value: u32) -> GgufResult<Self> {
        match value {
            0 => Ok(GgufDataType::F32),
            1 => Ok(GgufDataType::F16),
            2 => Ok(GgufDataType::Q8_0),
            3 => Ok(GgufDataType::Q4_0),
            4 => Ok(GgufDataType::I32),
            5 => Ok(GgufDataType::I16),
            6 => Ok(GgufDataType::I8),
            _ => Err(GgufError::InvalidDataType(value)),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            GgufDataType::F32 => 4,
            GgufDataType::F16 => 2,
            GgufDataType::Q8_0 => 1,
            GgufDataType::Q4_0 => 1,
            GgufDataType::I32 => 4,
            GgufDataType::I16 => 2,
            GgufDataType::I8 => 1,
        }
    }
}

#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: GgufDataType,
    pub offset: u64,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

#[derive(Debug)]
pub struct GgufModel {
    pub header: GgufHeader,
    pub tensors: HashMap<String, GgufTensor>,
    pub metadata: HashMap<String, String>,
}

impl GgufModel {
    pub fn load<P: AsRef<Path>>(path: P) -> GgufResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let header = Self::read_header(&mut reader)?;

        // Read metadata
        let metadata = Self::read_metadata(&mut reader, header.kv_count)?;

        // Read tensor info
        let tensors = Self::read_tensor_info(&mut reader, header.tensor_count)?;

        // Read tensor data
        let tensors = Self::read_tensor_data(&mut reader, tensors)?;

        Ok(GgufModel {
            header,
            tensors,
            metadata,
        })
    }

    fn read_header(reader: &mut BufReader<File>) -> GgufResult<GgufHeader> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != 0x46554747 {
            // "GGUF" in little endian
            return Err(GgufError::InvalidMagic);
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let kv_count = reader.read_u64::<LittleEndian>()?;

        Ok(GgufHeader {
            magic,
            version,
            tensor_count,
            kv_count,
        })
    }

    fn read_metadata(
        reader: &mut BufReader<File>,
        kv_count: u64,
    ) -> GgufResult<HashMap<String, String>> {
        let mut metadata = HashMap::new();

        for _ in 0..kv_count {
            let key_len = reader.read_u64::<LittleEndian>()?;
            let mut key = vec![0u8; key_len as usize];
            reader.read_exact(&mut key)?;
            let key = String::from_utf8_lossy(&key).to_string();

            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_len = reader.read_u64::<LittleEndian>()?;
            let mut value = vec![0u8; value_len as usize];
            reader.read_exact(&mut value)?;

            // For simplicity, convert all values to strings
            let value_str = match value_type {
                1 | 2 | 3 | 4 | 5 | 6 => {
                    // Numeric types
                    format!("{:?}", value)
                }
                8 => {
                    // String
                    String::from_utf8_lossy(&value).to_string()
                }
                _ => {
                    format!("binary_data_{}", value_type)
                }
            };

            metadata.insert(key, value_str);
        }

        Ok(metadata)
    }

    fn read_tensor_info(
        reader: &mut BufReader<File>,
        tensor_count: u64,
    ) -> GgufResult<HashMap<String, GgufTensor>> {
        let mut tensors = HashMap::new();

        for _ in 0..tensor_count {
            let name_len = reader.read_u64::<LittleEndian>()?;
            let mut name = vec![0u8; name_len as usize];
            reader.read_exact(&mut name)?;
            let name = String::from_utf8_lossy(&name).to_string();

            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut shape = Vec::new();
            for _ in 0..n_dims {
                shape.push(reader.read_u64::<LittleEndian>()? as usize);
            }

            let data_type = GgufDataType::from_u32(reader.read_u32::<LittleEndian>()?)?;
            let offset = reader.read_u64::<LittleEndian>()?;

            let tensor = GgufTensor {
                name: name.clone(),
                shape,
                data_type,
                offset,
                data: Vec::new(),
            };

            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }

    fn read_tensor_data(
        reader: &mut BufReader<File>,
        mut tensors: HashMap<String, GgufTensor>,
    ) -> GgufResult<HashMap<String, GgufTensor>> {
        for (_, tensor) in tensors.iter_mut() {
            let total_elements = tensor.shape.iter().product::<usize>();
            let data_size = total_elements * tensor.data_type.size();

            reader.seek(SeekFrom::Start(tensor.offset))?;
            tensor.data.resize(data_size, 0);
            reader.read_exact(&mut tensor.data)?;
        }

        Ok(tensors)
    }

    pub fn get_tensor(&self, name: &str) -> GgufResult<&GgufTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))
    }

    pub fn get_tensor_f32(&self, name: &str) -> GgufResult<Vec<f32>> {
        let tensor = self.get_tensor(name)?;

        match tensor.data_type {
            GgufDataType::F32 => {
                let mut result = Vec::with_capacity(tensor.data.len() / 4);
                for chunk in tensor.data.chunks_exact(4) {
                    result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(result)
            }
            GgufDataType::F16 => {
                let mut result = Vec::with_capacity(tensor.data.len() / 2);
                for chunk in tensor.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // Convert f16 to f32 manually
                    let sign = (bits >> 15) != 0;
                    let exponent = ((bits >> 10) & 0x1f) as i32;
                    let mantissa = bits & 0x3ff;

                    let f32_value = if exponent == 0 {
                        if mantissa == 0 {
                            0.0
                        } else {
                            // Subnormal number
                            let normalized_mantissa = mantissa as f32 / (1 << 10) as f32;
                            let result = if sign {
                                -normalized_mantissa
                            } else {
                                normalized_mantissa
                            };
                            result * (1.0 / (1 << 14) as f32)
                        }
                    } else if exponent == 31 {
                        if mantissa == 0 {
                            if sign {
                                f32::NEG_INFINITY
                            } else {
                                f32::INFINITY
                            }
                        } else {
                            f32::NAN
                        }
                    } else {
                        let bias = 15;
                        let unbiased_exponent = exponent - bias;
                        let normalized_mantissa = (mantissa as f32 / (1 << 10) as f32) + 1.0;
                        let scale = if unbiased_exponent >= 0 {
                            (1 << unbiased_exponent) as f32
                        } else {
                            1.0 / ((1 << (-unbiased_exponent)) as f32)
                        };
                        if sign {
                            -normalized_mantissa * scale
                        } else {
                            normalized_mantissa * scale
                        }
                    };

                    result.push(f32_value);
                }
                Ok(result)
            }
            _ => Err(GgufError::InvalidDataType(0)), // Placeholder error
        }
    }
}

pub struct GgufLoader {
    model: Option<GgufModel>,
}

impl GgufLoader {
    pub fn new() -> Self {
        GgufLoader { model: None }
    }

    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> GgufResult<()> {
        let model = GgufModel::load(path)?;
        self.model = Some(model);
        Ok(())
    }

    pub fn get_model(&self) -> Option<&GgufModel> {
        self.model.as_ref()
    }

    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_dummy_gguf() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();

        // Write GGUF header
        file.write_all(&0x46554747u32.to_le_bytes()).unwrap(); // magic
        file.write_all(&3u32.to_le_bytes()).unwrap(); // version
        file.write_all(&1u64.to_le_bytes()).unwrap(); // tensor_count
        file.write_all(&0u64.to_le_bytes()).unwrap(); // kv_count

        // Write tensor info
        let tensor_name = "test.weight";
        file.write_all(&(tensor_name.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(tensor_name.as_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
        file.write_all(&4u64.to_le_bytes()).unwrap(); // dim1
        file.write_all(&4u64.to_le_bytes()).unwrap(); // dim2
        file.write_all(&0u32.to_le_bytes()).unwrap(); // data_type (F32)
        file.write_all(&100u64.to_le_bytes()).unwrap(); // offset

        // Write tensor data
        file.seek(SeekFrom::Start(100)).unwrap();
        for i in 0..16 {
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
        }

        file
    }

    #[test]
    fn test_gguf_header_parsing() {
        let file = create_dummy_gguf();
        let model = GgufModel::load(file.path()).unwrap();

        assert_eq!(model.header.magic, 0x46554747);
        assert_eq!(model.header.version, 3);
        assert_eq!(model.header.tensor_count, 1);
        assert_eq!(model.header.kv_count, 0);
    }

    #[test]
    fn test_tensor_loading() {
        let file = create_dummy_gguf();
        let model = GgufModel::load(file.path()).unwrap();

        let tensor = model.get_tensor("test.weight").unwrap();
        assert_eq!(tensor.name, "test.weight");
        assert_eq!(tensor.shape, vec![4, 4]);
        assert_eq!(tensor.data_type, GgufDataType::F32);
        assert_eq!(tensor.data.len(), 16 * 4); // 16 elements * 4 bytes each
    }

    #[test]
    fn test_tensor_f32_conversion() {
        let file = create_dummy_gguf();
        let model = GgufModel::load(file.path()).unwrap();

        let data = model.get_tensor_f32("test.weight").unwrap();
        assert_eq!(data.len(), 16);
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(val, i as f32);
        }
    }

    #[test]
    fn test_invalid_magic() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&0x12345678u32.to_le_bytes()).unwrap(); // Invalid magic

        let result = GgufModel::load(file.path());
        assert!(matches!(result, Err(GgufError::InvalidMagic)));
    }
}
