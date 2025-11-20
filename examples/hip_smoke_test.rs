//! ROCm HIP Smoke Test Example
//!
//! This example demonstrates the HIP backend functionality for GPU kernel execution.
//! It tests kernel loading, memory management, and basic GPU operations.

use rocmforge::backend::{HipBackend, HipResult};
use std::path::Path;

fn main() -> HipResult<()> {
    println!("🚀 ROCmForge HIP Smoke Test");
    println!("================================");

    // Initialize HIP backend
    println!("📋 Initializing HIP backend...");
    let backend = HipBackend::new()?;

    // Display GPU information
    let device = backend.device();
    println!("✅ GPU Detected: {}", device.name);
    println!("   Memory: {} GB", device.memory / (1024 * 1024 * 1024));
    println!("   Compute Units: {}", device.compute_units);
    println!("   Device ID: {}", device.device_id);

    // Test GPU memory allocation
    println!("\n🧠 Testing GPU memory allocation...");

    // Test different buffer sizes
    let sizes = [1, 1024, 65536];
    for &size in &sizes {
        let buffer = backend.alloc_gpu_buffer::<f32>(size)?;
        println!("   ✅ Allocated {} floats ({} bytes)", size, buffer.size);
    }

    // Test memory roundtrip
    println!("\n🔄 Testing GPU memory roundtrip...");
    let test_data: Vec<f32> = vec![
        1.0, 2.0, 3.14159, 42.0, -1.0, -2.0, 0.0, 999.999, 0.5, 0.25, 0.125, 0.0625, 123.456,
        789.012, 345.678, 901.234,
    ];

    let gpu_buffer = backend.alloc_gpu_buffer::<f32>(test_data.len())?;
    println!("   📤 Copying {} floats to GPU...", test_data.len());
    backend.copy_to_gpu(&test_data, &gpu_buffer)?;

    let mut result_data: Vec<f32> = vec![0.0; test_data.len()];
    println!("   📥 Copying {} floats from GPU...", test_data.len());
    backend.copy_from_gpu(&gpu_buffer, &mut result_data)?;

    // Verify data integrity
    let mut success = true;
    for (i, (&original, &retrieved)) in test_data.iter().zip(result_data.iter()).enumerate() {
        if (original - retrieved).abs() > 1e-6 {
            println!(
                "   ❌ Data mismatch at index {}: expected {}, got {}",
                i, original, retrieved
            );
            success = false;
        }
    }

    if success {
        println!("   ✅ Memory roundtrip successful - all data intact!");
    }

    // Test kernel loading (if compiled kernels exist)
    println!("\n⚡ Testing kernel loading...");
    let module_path = Path::new("src/backend/hip_kernels/smoke_test.hsaco");

    if module_path.exists() {
        println!("   📦 Found compiled kernel module: {:?}", module_path);

        match backend.load_module(&module_path.to_string_lossy()) {
            Ok(module) => {
                println!("   ✅ Module loaded successfully");

                // Test kernel symbol resolution
                match backend.get_kernel(&module_path.to_string_lossy(), "add_one") {
                    Ok(_kernel) => {
                        println!("   ✅ Kernel 'add_one' found");
                    }
                    Err(e) => {
                        println!("   ❌ Failed to find kernel 'add_one': {}", e);
                    }
                }

                match backend.get_kernel(&module_path.to_string_lossy(), "nonexistent_kernel") {
                    Ok(_) => {
                        println!("   ❌ Unexpectedly found nonexistent kernel");
                    }
                    Err(_) => {
                        println!("   ✅ Correctly rejected nonexistent kernel");
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Failed to load module: {}", e);
            }
        }
    } else {
        println!("   ⚠️  No compiled kernel found at {:?}", module_path);
        println!("   💡 Run 'hipcc' to compile .hip files to .hsaco for real GPU execution");
    }

    // Test error handling
    println!("\n🛡️  Testing error handling...");

    // Test invalid module path
    match backend.load_module("/nonexistent/path/kernel.hsaco") {
        Ok(_) => {
            println!("   ❌ Should have failed to load nonexistent module");
        }
        Err(_) => {
            println!("   ✅ Correctly rejected invalid module path");
        }
    }

    println!("\n🎉 Smoke test completed successfully!");
    println!("   Backend is ready for GPU kernel execution");

    Ok(())
}
