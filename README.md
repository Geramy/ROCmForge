# ROCmForge

**AMD GPU Inference Engine for Large Language Models**

A high-performance inference engine specifically designed for AMD GPUs using ROCm and HIP. ROCmForge aims to provide efficient LLM inference capabilities on AMD hardware.

## Project Status

**⚠️ Experimental - Active Development**

This project is in early development stages. Core components are being built and tested. Not recommended for production use yet.

## What Works

- ✅ **Project Structure**: Well-organized codebase (~15,260 LOC across 48 modules)
- ✅ **Core Modules**: Attention mechanisms, backend abstractions, tensor operations
- ✅ **Model Loading**: GGUF and ONNX loader implementations
- ✅ **HTTP Server**: Axum-based REST API for inference requests
- ✅ **KV Cache**: Key-value cache implementation for efficient inference
- ✅ **Scheduler & Sampler**: Request scheduling and token sampling logic

## Architecture

```
src/
├── attention/      # Multi-head attention with GPU/CPU backends
├── backend/        # HIP/ROCm backend abstraction layer
├── engine.rs       # Main inference engine
├── http/           # HTTP API server
├── kv_cache/       # Key-value cache for transformer models
├── loader/         # GGUF and ONNX model loaders
├── model/          # Model configuration and execution
├── ops/            # GPU operations (matmul, attention)
├── sampler/        # Token sampling strategies
├── scheduler/      # Request batching and scheduling
├── tensor/         # Tensor data structures and operations
└── tokenizer.rs    # Tokenization utilities
```

## What's Experimental

- ⚠️ **HIP Backend**: AMD GPU acceleration via HIP/ROCm (placeholder implementations)
- ⚠️ **GPU Kernels**: Custom CUDA-style kernels for attention and matmul operations
- ⚠️ **Performance**: Not yet optimized for production workloads
- ⚠️ **Model Support**: Limited to specific architectures, testing ongoing

## What's Missing

- ❌ **Production ROCm Integration**: Real HIP bindings are commented out (using placeholders)
- ❌ **Quantization**: 4-bit/8-bit quantization support not fully implemented
- ❌ **Multi-GPU**: Single GPU only, no tensor parallelism yet
- ❌ **Benchmarks**: Performance comparisons against llama.cpp, vLLM not available
- ❌ **Documentation**: API docs and user guides need expansion

## Requirements

- Rust 1.70+ (2021 edition)
- AMD GPU with ROCm 5.x+ (for GPU acceleration)
- Linux (ROCm support)

## Build

```bash
# Clone repository
git clone <your-repo-url>
cd ROCmForge

# Build release binary
cargo build --release

# Run tests
cargo test

# Run with ROCm feature (when available)
cargo build --release --features rocm
```

## Usage

```bash
# Start HTTP inference server
cargo run --bin rocmforge_cli -- --port 8080

# Run simple model inference
cargo run --bin run_simple_model
```

## API Example

```bash
# Health check
curl http://localhost:8080/health

# Inference request (OpenAI-compatible)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'
```

## Dependencies

Key libraries used:
- **axum**: HTTP server framework
- **tokio**: Async runtime
- **tokenizers**: HuggingFace tokenizers
- **half**: FP16 support
- **memmap2**: Memory-mapped file I/O for large models
- **serde/serde_json**: Serialization

## Roadmap

**Phase 1 (Current)**: Core Infrastructure
- [x] Project structure and module organization
- [x] GGUF loader implementation
- [x] Basic HTTP API
- [ ] Real HIP backend integration
- [ ] Attention kernel optimization

**Phase 2 (Planned)**: Performance & Features
- [ ] Quantization support (4-bit, 8-bit)
- [ ] Multi-GPU tensor parallelism
- [ ] FlashAttention-2 for AMD
- [ ] Continuous batching
- [ ] Model support: Llama, Mistral, Gemma

**Phase 3 (Future)**: Production Readiness
- [ ] Comprehensive benchmarks
- [ ] Production deployment guide
- [ ] Docker containers with ROCm
- [ ] Monitoring and observability

## Development

```bash
# Run specific test
cargo test test_name

# Run benchmarks
cargo bench

# Format code
cargo fmt

# Check for issues
cargo clippy
```

## Project Structure

```
.
├── src/            # Source code (48 Rust files)
├── tests/          # Integration and unit tests
├── benches/        # Performance benchmarks
├── examples/       # Usage examples
├── build.rs        # Build script for C/C++ integration
└── Cargo.toml      # Dependencies and project config
```

## Contributing

This is a private repository in active development. Contributions and feedback are welcome once the project reaches a more stable state.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU inference optimization techniques
- [vLLM](https://github.com/vllm-project/vllm) - Efficient batching and scheduling
- [candle](https://github.com/huggingface/candle) - Rust ML framework design patterns

## Disclaimer

This project is experimental and under active development. APIs may change without notice. Performance characteristics on AMD GPUs are still being evaluated and optimized.

---

**Status**: 🚧 Active Development | **Version**: 0.1.0 | **Last Updated**: November 2024
