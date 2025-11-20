//! Test our attention module with debug output

use rocmforge::attention::{Attention, AttentionBackend};

fn main() {
    println!("DEBUG: Starting attention module test");
    
    let dim = 4; // Small dimension for simple test
    let attention = Attention::with_backend(dim, AttentionBackend::Cpu);
    
    // Create simple test data - batch_size=1, seq_len=dim, head_dim=dim
    let total_elements = 1 * dim * dim; // batch_size * seq_len * head_dim
    let q_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let v_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1 + 1.0).collect();
    
    println!("DEBUG: Created test data - q: {}, k: {}, v: {}", q_data.len(), k_data.len(), v_data.len());
    
    // This should trigger our debug output in the attention module
    let result = attention.forward(&q_data, &k_data, &v_data, None, Some(0.1));
    
    match result {
        Ok(output_data) => {
            println!("DEBUG: Attention forward succeeded, output length: {}", output_data.len());
            println!("DEBUG: First few output values: {:?}", &output_data[..std::cmp::min(5, output_data.len())]);
        }
        Err(e) => {
            println!("DEBUG: Attention forward failed: {}", e);
        }
    }
    
    println!("DEBUG: Test completed");
}