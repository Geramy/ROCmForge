//! Comprehensive TDD tests for loader modules

use rocmforge::loader::{
    GgufDataType, GgufModel, OnnxDataType, OnnxLoader, OnnxSession, OnnxTensor,
};
use std::io::{Seek, Write};
use tempfile::NamedTempFile;

#[test]
fn test_gguf_data_type_conversion() {
    // Test valid conversions
    assert!(matches!(GgufDataType::from_u32(0), Ok(GgufDataType::F32)));
    assert!(matches!(GgufDataType::from_u32(1), Ok(GgufDataType::F16)));
    assert!(matches!(GgufDataType::from_u32(2), Ok(GgufDataType::Q8_0)));
    assert!(matches!(GgufDataType::from_u32(3), Ok(GgufDataType::Q4_0)));
    assert!(matches!(GgufDataType::from_u32(4), Ok(GgufDataType::I32)));
    assert!(matches!(GgufDataType::from_u32(5), Ok(GgufDataType::I16)));
    assert!(matches!(GgufDataType::from_u32(6), Ok(GgufDataType::I8)));

    // Test invalid conversion
    let result = GgufDataType::from_u32(999);
    assert!(result.is_err());
}

#[test]
fn test_gguf_data_type_size() {
    assert_eq!(GgufDataType::F32.size(), 4);
    assert_eq!(GgufDataType::F16.size(), 2);
    assert_eq!(GgufDataType::Q8_0.size(), 1);
    assert_eq!(GgufDataType::Q4_0.size(), 1);
    assert_eq!(GgufDataType::I32.size(), 4);
    assert_eq!(GgufDataType::I16.size(), 2);
    assert_eq!(GgufDataType::I8.size(), 1);
}

fn create_dummy_gguf() -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();

    // Write GGUF header
    file.write_all(&0x46554747u32.to_le_bytes()).unwrap(); // magic
    file.write_all(&3u32.to_le_bytes()).unwrap(); // version
    file.write_all(&2u64.to_le_bytes()).unwrap(); // tensor_count
    file.write_all(&1u64.to_le_bytes()).unwrap(); // kv_count

    // Write metadata
    let key = "test.metadata";
    file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    file.write_all(key.as_bytes()).unwrap();
    file.write_all(&8u32.to_le_bytes()).unwrap(); // string type
    file.write_all(&(12u64).to_le_bytes()).unwrap(); // value length
    file.write_all(b"test_value").unwrap();

    // Write first tensor info
    let tensor_name1 = "weight1";
    file.write_all(&(tensor_name1.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(tensor_name1.as_bytes()).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
    file.write_all(&4u64.to_le_bytes()).unwrap(); // dim1
    file.write_all(&4u64.to_le_bytes()).unwrap(); // dim2
    file.write_all(&0u32.to_le_bytes()).unwrap(); // data_type (F32)
    file.write_all(&200u64.to_le_bytes()).unwrap(); // offset

    // Write second tensor info
    let tensor_name2 = "bias1";
    file.write_all(&(tensor_name2.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(tensor_name2.as_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap(); // n_dims
    file.write_all(&4u64.to_le_bytes()).unwrap(); // dim1
    file.write_all(&1u32.to_le_bytes()).unwrap(); // data_type (F16)
    file.write_all(&300u64.to_le_bytes()).unwrap(); // offset

    // Write first tensor data (F32)
    file.seek(std::io::SeekFrom::Start(200)).unwrap();
    for i in 0..16 {
        file.write_all(&(i as f32).to_le_bytes()).unwrap();
    }

    // Write second tensor data (F16)
    file.seek(std::io::SeekFrom::Start(300)).unwrap();
    for i in 0..4 {
        let f16 = half::f16::from_f32(i as f32);
        file.write_all(&f16.to_bits().to_le_bytes()).unwrap();
    }

    file
}

#[test]
fn test_gguf_model_loading() {
    let file = create_dummy_gguf();
    let model = GgufModel::load(file.path());

    assert!(model.is_ok());

    let model = model.unwrap();
    assert_eq!(model.header.magic, 0x46554747);
    assert_eq!(model.header.version, 3);
    assert_eq!(model.header.tensor_count, 2);
    assert_eq!(model.header.kv_count, 1);
    assert_eq!(model.tensors.len(), 2);
    assert_eq!(model.metadata.len(), 1);

    // Check metadata
    assert!(model.metadata.contains_key("test.metadata"));
    assert_eq!(model.metadata["test.metadata"], "test_value");
}

#[test]
fn test_gguf_tensor_access() {
    let file = create_dummy_gguf();
    let model = GgufModel::load(file.path()).unwrap();

    // Test existing tensors
    let weight1 = model.get_tensor("weight1");
    assert!(weight1.is_ok());
    let weight1 = weight1.unwrap();
    assert_eq!(weight1.name, "weight1");
    assert_eq!(weight1.shape, vec![4, 4]);
    assert_eq!(weight1.data_type, GgufDataType::F32);
    assert_eq!(weight1.data.len(), 16 * 4); // 16 elements * 4 bytes each

    let bias1 = model.get_tensor("bias1");
    assert!(bias1.is_ok());
    let bias1 = bias1.unwrap();
    assert_eq!(bias1.name, "bias1");
    assert_eq!(bias1.shape, vec![4]);
    assert_eq!(bias1.data_type, GgufDataType::F16);
    assert_eq!(bias1.data.len(), 4 * 2); // 4 elements * 2 bytes each

    // Test non-existing tensor
    let missing = model.get_tensor("nonexistent");
    assert!(missing.is_err());
}

#[test]
fn test_gguf_f32_conversion() {
    let file = create_dummy_gguf();
    let model = GgufModel::load(file.path()).unwrap();

    let weight1_data = model.get_tensor_f32("weight1");
    assert!(weight1_data.is_ok());

    let weight1_data = weight1_data.unwrap();
    assert_eq!(weight1_data.len(), 16);
    for (i, &val) in weight1_data.iter().enumerate() {
        assert_eq!(val, i as f32);
    }

    // F16 tensor should also convert to F32
    let bias1_data = model.get_tensor_f32("bias1");
    assert!(bias1_data.is_ok());

    let bias1_data = bias1_data.unwrap();
    assert_eq!(bias1_data.len(), 4);
    for (i, &val) in bias1_data.iter().enumerate() {
        assert_eq!(val, i as f32);
    }
}

#[test]
fn test_gguf_invalid_magic() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&0x12345678u32.to_le_bytes()).unwrap(); // Invalid magic
    file.write_all(&3u32.to_le_bytes()).unwrap(); // version
    file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor_count
    file.write_all(&0u64.to_le_bytes()).unwrap(); // kv_count

    let result = GgufModel::load(file.path());
    assert!(result.is_err());
}

#[test]
fn test_gguf_unsupported_version() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&0x46554747u32.to_le_bytes()).unwrap(); // magic
    file.write_all(&999u32.to_le_bytes()).unwrap(); // unsupported version
    file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor_count
    file.write_all(&0u64.to_le_bytes()).unwrap(); // kv_count

    let result = GgufModel::load(file.path());
    assert!(result.is_err());
}

#[test]
fn test_onnx_tensor_creation() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = OnnxTensor::new("test".to_string(), vec![2, 2], &data);

    assert_eq!(tensor.name, "test");
    assert_eq!(tensor.shape, vec![2, 2]);
    assert!(matches!(tensor.data_type, OnnxDataType::F32));
    assert_eq!(tensor.data.len(), 16); // 4 elements * 4 bytes
}

#[test]
fn test_onnx_tensor_f32_conversion() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = OnnxTensor::new("test".to_string(), vec![2, 2], &data);

    let result = tensor.get_data_f32();
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_onnx_tensor_invalid_conversion() {
    let data = vec![1i32, 2, 3, 4];
    let tensor = OnnxTensor::new("test".to_string(), vec![2, 2], &data);

    // This tensor is I32, not F32
    let result = tensor.get_data_f32();
    assert!(result.is_err());
}

#[test]
fn test_onnx_loader_creation() {
    let loader = OnnxLoader::new();
    assert!(!loader.is_model_loaded());
}

#[test]
fn test_onnx_model_loading() {
    let mut loader = OnnxLoader::new();

    // Create a dummy ONNX model file
    let temp_file = NamedTempFile::new().unwrap();
    let result = loader.load_model(temp_file.path());

    // In our mock implementation, this should succeed
    assert!(result.is_ok());
    assert!(loader.is_model_loaded());
}

#[test]
fn test_onnx_inference() {
    let mut loader = OnnxLoader::new();

    // Create a dummy ONNX model file
    let temp_file = NamedTempFile::new().unwrap();
    loader.load_model(temp_file.path()).unwrap();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor = OnnxTensor::new("input".to_string(), vec![2, 2], &input_data);

    let outputs = loader.run_inference(&[input_tensor]);
    assert!(outputs.is_ok());

    let outputs = outputs.unwrap();
    assert_eq!(outputs.len(), 1);

    // In our mock implementation, this should be an identity operation
    let output_data = outputs[0].get_data_f32().unwrap();
    assert_eq!(output_data, input_data);
}

#[test]
fn test_onnx_inference_without_model() {
    let loader = OnnxLoader::new();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor = OnnxTensor::new("input".to_string(), vec![2, 2], &input_data);

    let result = loader.run_inference(&[input_tensor]);
    assert!(result.is_err());
}

#[test]
fn test_onnx_inference_empty_inputs() {
    let mut loader = OnnxLoader::new();

    let temp_file = NamedTempFile::new().unwrap();
    loader.load_model(temp_file.path()).unwrap();

    let result = loader.run_inference(&[]);
    assert!(result.is_err());
}

#[test]
fn test_onnx_session_input_output_names() {
    let temp_file = NamedTempFile::new().unwrap();
    let session = OnnxSession::new(temp_file.path());

    assert!(session.is_ok());

    let session = session.unwrap();
    assert_eq!(session.input_names(), &["input"]);
    assert_eq!(session.output_names(), &["output"]);
}

#[test]
fn test_onnx_session_run() {
    let temp_file = NamedTempFile::new().unwrap();
    let session = OnnxSession::new(temp_file.path()).unwrap();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor = OnnxTensor::new("input".to_string(), vec![2, 2], &input_data);

    let outputs = session.run(&[input_tensor]);
    assert!(outputs.is_ok());

    let outputs = outputs.unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name, "output");
    assert_eq!(outputs[0].shape, vec![2, 2]);
}

// Property-based tests
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_gguf_tensor_data_properties(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..100)
    ) {
        let file = create_dummy_gguf();
        let mut model = GgufModel::load(file.path()).unwrap();

        // Note: Since get_tensor_mut is not available, we'll test with the existing data
        // In a real implementation, you would need to add a method to modify tensor data

        // Test conversion back to f32
        let converted = model.get_tensor_f32("weight1").unwrap();
        prop_assert_eq!(converted.len(), data.len());

        for (original, &converted) in data.iter().zip(converted.iter()) {
            prop_assert!((original - converted).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_onnx_tensor_properties(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 1..50),
        rows in 1usize..10,
        cols in 1usize..10
    ) {
        let total_elements = rows * cols;
        let truncated_data = data.into_iter().take(total_elements).collect::<Vec<_>>();

        let tensor = OnnxTensor::new(
            "test".to_string(),
            vec![rows, cols],
            &truncated_data,
        );

        prop_assert_eq!(tensor.name.as_str(), "test");
        let expected_shape = vec![rows, cols];
        let tensor_shape = tensor.shape.clone();
        prop_assert_eq!(tensor_shape, expected_shape);
        prop_assert!(matches!(tensor.data_type, OnnxDataType::F32));
        prop_assert_eq!(tensor.data.len(), total_elements * 4);

        let converted = tensor.get_data_f32().unwrap();
        prop_assert_eq!(converted.len(), total_elements);
        prop_assert_eq!(converted, truncated_data);
    }

    #[test]
    fn test_gguf_data_type_properties(
        type_id in 0u32..7u32
    ) {
        let result = GgufDataType::from_u32(type_id);
        prop_assert!(result.is_ok());

        let data_type = result.unwrap();
        let size = data_type.size();
        prop_assert!(size > 0);
        prop_assert!(size <= 4); // Max size for our types
    }
}
