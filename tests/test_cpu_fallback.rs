//! Test CPU fallback functionality in attention mechanism

use rocmforge::attention::{Attention, AttentionBackend};

fn main() {
    // Force CPU fallback by using CPU backend (since GPU is only available with rocm feature)
    let dim = 64;
    let seq_len = 4;
    
    // Create attention with CPU backend
    let attention = Attention::with_backend(dim, AttentionBackend::Cpu);
    
    // Create simple test data
    let batch_size = 1;
    let num_heads = 1;
    let head_dim = dim / num_heads;
    let total_elements = batch_size * num_heads * seq_len * head_dim;
    
    // Simple test data
    let q_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let v_data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 0.1 + 1.0).collect();
    
    println!("DEBUG: Starting CPU test");
    println!("DEBUG: Q data length: {}", q_data.len());
    println!("DEBUG: K data length: {}", k_data.len());
    println!("DEBUG: V data length: {}", v_data.len());
    
    // This should trigger our debug output
    let result = attention.forward(&q_data, &k_data, &v_data, None, Some(0.1));
    
    match result {
        Ok(output_data) => {
            println!("DEBUG: Attention forward succeeded, output length: {}", output_data.len());
        }
        Err(e) => {
            println!("DEBUG: Attention forward failed: {}", e);
        }
    }
}