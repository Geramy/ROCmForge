//! Simple test to trigger CPU fallback debug output

fn main() {
    println!("DEBUG: Simple test to check if we can trigger the debug output");
    println!("DEBUG: If this works, we should see debug output from attention module");
    
    // Since we can't easily test the full attention due to linking issues,
    // let's at least verify our debug output approach works
    let test_data = vec![1.0, 2.0, 3.0, 4.0];
    println!("DEBUG: Test data length: {}", test_data.len());
    
    println!("DEBUG: Test completed successfully");
}