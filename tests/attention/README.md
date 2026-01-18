# Attention Benchmarks

This directory contains documentation and reference materials for attention mechanism benchmarks in ROCmForge.

## Running Benchmarks

### Basic Command

```bash
cargo bench --bench attention_bench
```

### With ROCm GPU Support

To benchmark GPU (Flash Attention) backends:

```bash
cargo bench --bench attention_bench --features rocm
```

## Hardware Requirements

### For CPU Benchmarks (Always Available)
- Any x86_64 or ARM64 CPU
- Minimum 4GB RAM

### For GPU Benchmarks (Requires ROCm)
- AMD GPU with RDNA2 or newer architecture
- ROCm 5.4+ installed
- Minimum 8GB VRAM recommended for larger sequence lengths

## What Each Benchmark Measures

### CPU Attention Benchmarks
- **Standard attention computation** on CPU
- Varying sequence lengths (128, 256, 512, 1024)
- Varying head dimensions (32, 64)
- Varying numbers of attention heads (4, 8, 16)

### Flash Attention GPU Benchmarks
- **Fused flash attention kernels** on AMD GPU
- Same configurations as CPU for direct comparison
- Measures GPU kernel launch overhead vs computation time

### CPU vs Flash Attention Comparison
- Direct comparison on same input data
- Calculates speedup/slowdown factor
- Identifies when GPU overhead outweighs benefits

## Expected Performance Characteristics

### CPU Backend
- Linear scaling with sequence length (O(n^2) attention computation)
- Consistent performance across different head dimensions
- Baseline for comparison

### Flash Attention Backend (GPU)
- Expected 2-4x speedup for typical inference workloads
- Larger speedup for longer sequences (amortized kernel launch overhead)
- May show slowdown for very small sequences due to GPU kernel launch overhead

### Performance Factors

| Factor | Impact |
|--------|--------|
| Sequence length | Longer sequences benefit more from GPU |
| Head dimension | Larger dimensions may saturate GPU memory bandwidth |
| Batch size | Currently fixed at 1 (single-token generation) |
| GPU generation | RDNA3 (gfx1100) has optimized kernels |

## Interpreting Results

### Key Metrics

1. **Average time**: Mean execution time across iterations
2. **Tokens/second**: Throughput metric for generation workloads
3. **P50/P95/P99**: Latency percentiles for tail latency analysis

### What to Look For

- **Speedup > 1.5x**: Flash attention is providing significant benefit
- **Speedup ~ 1.0x**: GPU overhead is comparable to benefit (consider CPU)
- **Speedup < 1.0x**: GPU overhead dominates for this config (use CPU)

### Known Limitations

1. **Layout mismatch**: Current implementation has known layout conversion issues
   - GPU kernels expect `[batch, heads, seq, dim]`
   - BackendImplementation provides `[batch, seq, heads*dim]`
   - Tests use constant values, so output correctness not validated

2. **Single batch**: Benchmarks use batch_size=1 (typical for inference)
   - Multi-batch workloads may show different characteristics

3. **No warm GPU state**: First kernel launch includes driver initialization overhead
   - Subsequent launches are representative of steady-state performance

## Benchmark Configuration

### Iteration Counts
- CPU benchmarks: 100 iterations
- GPU benchmarks: 50 iterations (longer per-iteration time)
- Warmup: 10 iterations (min of 10 or benchmark count)

### Test Data
- Generated using sin/cos/tan functions for variation
- Not all zeros (would trivialize softmax)
- Deterministic across runs (same seed pattern)

## Troubleshooting

### "FlashAttention backend not available"
- Ensure `rocm` feature is enabled
- Check ROCm driver installation: `rocminfo`
- Verify GPU is visible: `rocm-smi`

### GPU benchmarks fail with "operation failed"
- Check GPU memory: `rocm-smi`
- Try smaller sequence lengths (reduce memory usage)
- Verify kernel compilation in build.rs

### "CPU forward should succeed"
- Check available system memory
- Reduce sequence length for testing

## Future Enhancements

1. **Layout conversion**: Fix layout mismatch for correctness validation
2. **Batch size sweeps**: Test multi-batch scenarios
3. **Custom masks**: Benchmark custom attention masks
4. **Memory profiling**: Add GPU memory bandwidth measurements
5. **Kernel profiling**: Use ROCm profiler to identify bottlenecks

## Related Files

- `benches/attention_bench.rs` - Main benchmark harness
- `src/attention/flash_attention.rs` - FlashAttention backend implementation
- `src/attention/backend_registry.rs` - Backend selection logic
- `kernels/flash_attention*.hip` - GPU kernel implementations
