// Debug test to understand the buffer size issue
#[cfg(feature = "rocm")]
#[test]
fn test_debug_device_tensor_sizes() {
    use rocmforge::backend::{HipBackend, DeviceTensor};
    use rocmforge::tensor::TensorShape;
    
    let backend = HipBackend::new().unwrap();
    
    // Test data
    let data = vec![1.0, 2.0, 3.0, 4.0]; // 4 elements
    let shape = TensorShape::from_dims(&[1, 2, 2]); // Should be 4 elements
    
    println!("Data length: {} elements", data.len());
    println!("Shape total elements: {}", shape.total_elements());
    println!("Data bytes: {}", data.len() * 4);
    println!("Shape bytes: {}", shape.total_elements() * 4);
    
    let device_tensor = DeviceTensor::from_host_vec(&backend, data.clone(), shape).unwrap();
    
    println!("Device tensor len(): {} elements", device_tensor.len());
    println!("Device tensor size(): {} bytes", device_tensor.size());
    
    // Try to copy back
    let result = device_tensor.to_host_vec();
    match result {
        Ok(host_data) => println!("Success: got {} elements", host_data.len()),
        Err(e) => println!("Error: {:?}", e),
    }
}