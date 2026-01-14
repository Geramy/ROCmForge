//! Supported ggml operations.

#[derive(Debug, Clone)]
pub enum Op {
    GetRows,
    MatMul,
    // PHASE 3: Quantized matmul operations
    // These ops keep weights in quantized format on GPU and dequantize during compute
    // TODO: Implement HIP kernels for dequantize + matmul
    MatMulQ4_0,  // Q4_0: block_size=32, 4-bit values + f32 scale per block
    MatMulQ8_0,  // Q8_0: block_size=32, 8-bit values + f32 scale per block
    Add,
    Mask,
    Scale { factor: f32 },
    LayerNorm { eps: f32 },
    RmsNorm { eps: f32 },
    Rope,
    Softmax,
    Attention,
    SwiGlu,
    MlpSwiglu,
    SplitQkv,
    Reshape,
    View,
    Copy,
    // PHASE 5: Accumulate op for efficient KV cache writes
    // Adds source tensor to destination tensor at given offset
    // Used for in-place KV cache updates without Copy overhead
    Accumulate { offset: usize },
}
