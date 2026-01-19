# ROCmForge

**AMD GPU Inference Engine for Large Language Models**

An LLM inference engine for AMD GPUs using ROCm and HIP. Loads GGUF-format models and provides an OpenAI-compatible HTTP API.

## Status

**Version:** 0.1.0 (v1.0 milestone complete)

| Component | Status | Notes |
|-----------|--------|-------|
| GGUF Model Loading | Working | 15 quantization formats supported |
| CPU Backend | Working | SIMD (AVX2/NEON), attention ops |
| GPU Backend | Working | HIP kernels for matmul, dequantization, attention |
| HTTP Server | Working | OpenAI-compatible API |
| CLI | Working | serve, generate, context commands |
| Tests | Passing | 564+ lib tests passing |

### What Works

Based on actual test results:

- **GGUF Loading**: 15 quantization formats (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4, MXFP6)
- **CPU SIMD Backend**: AVX2 (f32x8), NEON (f32x4), runtime detection, matmul + attention + layer ops
- **GPU Kernels**: HIP implementations for dequantization, matmul, attention
- **Hybrid Scheduler**: Automatic CPU/GPU backend selection
- **HTTP API**: `/v1/completions`, `/health`, `/ready`, `/metrics`
- **CLI Tools**: `rocmforge_cli serve`, `generate`, `context` (add/search/list/clear)

### Known Limitations

- GPU sampler kernels use CPU fallback (optimization deferred)
- MQA GPU optimization uses partial CPU fallback
- SIMD feature requires nightly Rust
- AVX-512 is opt-in (feature flag to avoid CPU throttling)
- ~82 compiler warnings (cosmetic: unused imports, variables)

## Requirements

- **Rust**: 1.82+ (for SIMD support)
- **ROCm**: 5.0+ (Linux only)
- **GPU**: AMD GPU with ROCm support (RDNA2/3 or CDNA2/3)
- **Memory**: 8GB+ VRAM recommended

## Build

```bash
# Build release binary
cargo build --release

# Run tests
cargo test --lib

# Run with ROCm feature
cargo build --release --features rocm

# Run with SIMD (requires nightly)
cargo build --release --features simd
```

## Usage

### HTTP Server

```bash
# Start server
./target/release/rocmforge_cli serve --addr 127.0.0.1:8080 --gguf model.gguf

# Health check
curl http://localhost:8080/health

# Generate completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "model", "prompt": "Hello", "max_tokens": 50}'
```

### CLI

```bash
# Generate text
./target/release/rocmforge_cli generate --gguf model.gguf --prompt "Hello world"

# Context management (requires --features context)
./target/release/rocmforge_cli context add "Your text here"
./target/release/rocmforge_cli context search "query terms"
```

## Documentation

- [User Guide](docs/USER_GUIDE.md) - Installation and usage
- [CLI Reference](docs/CLI_REFERENCE.md) - Command-line interface
- [API Documentation](docs/API_DOCUMENTATION.md) - HTTP API endpoints
- [Deployment Guide](docs/DEPLOYMENT.md) - Deployment instructions

## Architecture

```
src/
├── attention/       # Multi-head attention (CPU/GPU backends)
├── backend/         # CPU and HIP/ROCm backends
├── context/         # SQLiteGraph context engine (feature-gated)
├── engine.rs        # Main inference engine
├── error.rs         # Unified error types
├── http/            # HTTP API server
├── kv_cache/        # Paged key-value cache
├── loader/          # GGUF model loader
├── logging.rs       # Logging utilities
├── model/           # Model configuration and execution
├── otel_traces.rs   # OpenTelemetry tracing
├── profiling/       # Performance profiling
├── sampler/         # Token sampling
├── scheduler/       # Request batching
├── tensor/          # Tensor data structures
└── tokenizer.rs     # HuggingFace tokenizer integration
```

## Features

- **Hybrid Execution**: Automatic CPU/GPU backend selection via `CapabilityProvider` trait
- **GGUF Compatibility**: Supports LLaMA, Qwen, Mistral, Yi, Mixtral architectures
- **Quantized Inference**: Q4_0, Q8_0, Q4_K, Q6_K with fused dequant+matmul kernels
- **CPU SIMD**: AVX-512/AVX2/NEON with runtime detection
- **Context Engine**: SQLiteGraph-based semantic context (feature-gated)

## Development

```bash
# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings

# Run tests
cargo test --lib

# Run with specific features
cargo test --features rocm,simd
```

## Contributing

This is a development-focused project. Patches are welcome for:
- GPU kernel optimizations
- Additional quantization format support
- CPU SIMD improvements
- Bug fixes

## License

GPL-3.0

## Acknowledgments

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format and quantization
- [vLLM](https://github.com/vllm-project/vllm) - Paged attention and batching
- [candle](https://github.com/huggingface/candle) - Rust ML design patterns
