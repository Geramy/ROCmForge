//! Direct test of CPU backend to see debug output

mod rocmforge {
    pub mod attention {
        pub use super::super::rocmforge::attention::*;
    pub mod cpu {
            pub use super::super::super::rocmforge::attention::cpu::*;
        }
    }
}

fn main() {
    println!("DEBUG: Starting direct CPU backend test");
    
    let dim = 4;
    let total_elements = dim * dim; // Simple case
    let q_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let v_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1 + 1.0).collect();
    
    println!("DEBUG: Created test data - q: {}, k: {}, v: {}", q_data.len(), k_data.len(), v_data.len());
    
    // Call CPU backend directly
    let result = rocmforge::attention::cpu::CpuBackend::forward(dim, &q_data, &k_data, &v_data, None, Some(0.1));
    
    match result {
        Ok(output_data) => {
            println!("DEBUG: CPU backend forward succeeded, output length: {}", output_data.len());
            println!("DEBUG: First few output values: {:?}", &output_data[..std::cmp::min(5, output_data.len())]);
        }
        Err(e) => {
            println!("DEBUG: CPU backend forward failed: {}", e);
        }
    }
    
    println!("DEBUG: Test completed");
}