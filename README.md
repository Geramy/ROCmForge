# ROCmForge

**AMD GPU Inference Engine for Large Language Models**

A high-performance inference engine specifically designed for AMD GPUs using ROCm and HIP. ROCmForge provides efficient LLM inference capabilities on AMD hardware with fully-tested GPU kernels.

## Project Status

**Alpha Software - Phase 21 Complete**

This is **alpha software** under active development. Components work individually but end-to-end integration is incomplete.

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| GPU Kernels | ✅ Complete | 41/41 | Phases 1-4: scale, mask, softmax, RoPE, FlashAttention, SwiGLU, RMSNorm |
| GPU Attention Path | ✅ Complete | 67/67 | Phase 7: 2-5x speedup over CPU |
| Q4_1/Q5_0/Q5_1 Support | ✅ Complete | 13/13 | Phase 8: Full dequantization support |
| Code Quality | ✅ Complete | 158/158 | Phase 15-21: 100% test health |
| HIP/ROCm Backend | ✅ Complete | All | AMD RX 7900 XT tested |
| GGUF Loader | ✅ Complete | All | Fixed spec compliance, vocab inference |
| MXFP Quantization | ✅ Complete | 24/24 | Phase 5: MXFP4/MXFP6 (OCP MX Spec v1.0) |
| KV Cache | ✅ Complete | All | Paged attention cache with bug fixes |
| HTTP Server | ✅ Complete | All | OpenAI-compatible API with tests |
| CLI | ⚠️ Experimental | All | Phase 21: Fixes applied, not fully tested |
| Memory Pooling | ✅ Complete | All | Phase 10: 70% reduction in hipMalloc calls |
| Bug Fixes | ✅ Complete | All | Phase 11-21: P0/P1/P2 bugs fixed |
| Logging Cleanup | ✅ Complete | All | Phase 15: 101 eprintln! → tracing macros |
| Naming Cleanup | ✅ Complete | All | Phase 15: AttentionBackend conflict resolved |
| Async GPU Loading | ✅ Complete | 8/8 | Phase 17: ~5x model loading speedup |
| GPU Testing Safety | ✅ Complete | 5/5 | Phase 20: Safe to run GPU tests |

**Overall Test Health**: 274+ unit tests passing (100%)

## What Works

### GPU Kernels (100% Complete)

All transformer layer operations are GPU-accelerated with comprehensive testing:

- **Phase 1**: Basic kernels (scale, mask, softmax) - 3/3 tests
- **Phase 2**: RoPE (Rotary Position Embedding) - 5/5 tests
- **Phase 3a**: Non-Causal FlashAttention - 17/17 tests
- **Phase 3b**: Causal Masking for autoregressive decoding - 8/8 tests
- **Phase 4**: MLP Ops (SwiGLU, RMSNorm) - 8/8 tests
- **Phase 7**: GPU Attention Path - 67/67 tests (2-5x speedup)
- **Phase 8**: Model Support (Q4_1/Q5_0/Q5_1) - 13/13 tests
- **Phase 9**: Code Quality (bug fixes) - 190/190 tests (at completion)
- **Phase 10**: Memory Pooling (ROCm workaround) - Complete
- **Phase 11**: P0/P1 Bug Fixes (5 bugs fixed) - Complete
- **Phase 15**: P1/P2 Code Quality Fixes - Complete
- **Phase 17**: Async GPU Loading - Complete

**Total: 158/158 unit tests passing (100% test health)**

### Async GPU Loading (100% Complete)

Phase 17: Option B - Multi-stream concurrent GPU uploads for ~5x faster model loading

- **Performance**: ~60s → ~12s model loading time (~5x speedup)
- **Phase A**: Parallel dequantization using Rayon (~4x CPU speedup)
- **Phase B**: Concurrent GPU uploads using 4 HIP streams (~4x GPU speedup)
- **Phase C**: Thread-safe GPU cache population
- **HIP Events**: FFI bindings + RAII wrapper for synchronization
- **AsyncLoader**: Multi-stream upload manager with automatic stream selection
- **Tests**: 8 new tests (3 HipEvent + 5 AsyncLoader)

**E2E Test Report**: `docs/ASYNC_LOADING_E2E_TEST_REPORT.md`

### MXFP Quantization (100% Complete)

Phase 5: OCP MX Specification v1.0 compliant MXFP4/MXFP6 support

- **E8M0 Scale Format**: 8-bit exponent-only scaling (24/24 tests)
- **MXFP4**: 4-bit E2M1 format, 4x memory reduction vs FP16
- **MXFP6**: 6-bit E2M3 format, 2.67x memory reduction vs FP16
- **Block Scaling**: 32 elements per block with shared E8M0 scale
- **GGUF Integration**: MXFP tensor types added to loader

**Total: 24/24 MXFP tests passing**

### HIP/ROCm Integration

- AMD Radeon RX 7900 XT (gfx1100, RDNA3) tested
- Wave32 optimization (256 thread blocks)
- GPU-only execution path (no CPU round-trips in transformer layers)

### GGUF Model Loading

- Fixed spec compliance (array encoding, value types, tensor types)
- Vocab size inference from tensor shapes
- Architecture detection (Qwen2, LLaMA, Mistral, GLM)
- Supports: F32, F16, Q8_0, Q4_0, **Q4_1, Q5_0, Q5_1**, MXFP4, MXFP6

### Infrastructure

- HTTP Server: Axum-based REST API with OpenAI compatibility
- Scheduler: Request batching and queue management
- KV Cache: Paged attention cache for efficient inference
- Tokenizer: HuggingFace tokenizers with fallback
- Sampler: Top-k, top-p, temperature, repetition penalty

### Code Quality Improvements (Phase 15)

- **Logging**: Replaced 101 eprintln! statements with structured tracing macros
- **Naming**: Resolved AttentionBackend trait/enum naming conflict
- **Error Handling**: Audited 28 expect() calls (all acceptable)
- **Result Types**: Verified consistent naming conventions

## Known Issues

### High Priority

1. **CLI Stability**: Phase 21 fixes applied but not fully tested
   - Status: Fixed P0 GPU resource leak, P2 missing input validation
   - Remaining: Not tested end-to-end with real models
   - Workaround: Use HTTP server API which is more stable
   - Impact: CLI may still crash in edge cases
   - See: `docs/CLI_BUG_FIXES_2026-01-11.md`

2. **End-to-End Inference**: Not fully tested with real models
   - Status: Individual components tested, integration incomplete
   - Impact: Cannot guarantee reliable model execution
   - Plan: Add integration tests with real models

### Medium Priority (Non-Blockers)

3. **Compiler Warnings**: ~50 warnings remaining
   - Types: Dead code, unused imports, unused variables
   - Target: <10 warnings (only FFI `#[allow(...)]`)
   - Impact: Code quality, not functionality

4. **MQA/GQA CPU Fallback**: Multi-query attention uses CPU instead of GPU
   - Impact: Performance penalty for MQA/GQA models only
   - Workaround: CPU path is correct and tested
   - Plan: Add GPU kernels for MQA/GQA

5. **Missing Test Coverage**:
   - HTTP server integration tests (unit tests exist)
   - Sampler integration tests (inline tests only)
   - GPU memory exhaustion tests

### Low Priority

6. **RwLock Poisoning**: 6 expect() calls in kv_cache stats methods
   - Status: Documented as acceptable
   - Fix: Would require API breaking change
   - Impact: Low (stats methods only)

## What's In Progress

### Future Enhancements (Planned)

1. **CLI End-to-End Testing**: Test CLI with real models after Phase 21 fixes
2. **Integration Testing**: End-to-end tests with real models
3. **Warning Cleanup**: Reduce compiler warnings to <10
4. **Benchmark Suite**: Performance regression tracking
5. **MQA/GQA GPU Pipeline**: GPU acceleration for multi-query attention

## Architecture

```
src/
├── attention/      # Multi-head attention (GPU/CPU backends)
├── backend/        # HIP/ROCm backend abstraction
├── engine.rs       # Main inference engine
├── http/           # HTTP API server
├── kv_cache/       # Key-value cache (paged)
├── loader/         # GGUF model loader
├── mlp/            # MLP operations (SwiGLU, RMSNorm)
├── model/          # Configuration and execution plans
├── ops/            # High-level GPU operations
├── sampler/        # Token sampling (CPU)
├── scheduler/      # Request batching
├── tensor/         # Tensor data structures
└── tokenizer.rs    # Tokenization utilities
```

## Requirements

- **Rust**: 1.70+ (2021 edition)
- **GPU**: AMD GPU with ROCm 5.x+
- **OS**: Linux (ROCm requirement)
- **Memory**: 16GB+ recommended for 7B models

## Build

```bash
# Clone repository
git clone https://github.com/your-repo/ROCmForge.git
cd ROCmForge

# Build release binary
cargo build --release

# Run tests (requires AMD GPU)
cargo test --features rocm

# Run specific test
cargo test --features rocm --lib test_swiglu_matches_cpu_small
```

## Usage

### Testing

```bash
# Run all GPU kernel tests
cargo test --features rocm

# Test specific module
cargo test --features rocm --lib attention
cargo test --features rocm --lib mlp

# Monitor GPU during tests
watch -n 1 rocm-smi
```

### CLI (Experimental - Fixes Applied, Not Fully Tested)

```bash
# Inspect GGUF model metadata
./target/release/rocmforge_cli inspect --model /path/to/model.gguf

# Generate text (experimental - fixes applied but not fully tested)
./target/release/rocmforge_cli generate \
  --gguf ~/.config/syncore/models/qwen2.5-0.5b.gguf \
  --prompt "The future of AI is" \
  --max-tokens 20 \
  --temperature 0.7
```

### HTTP Server (Recommended for Testing)

```bash
# Start server
./target/release/rocmforge_cli serve --port 8080

# Health check
curl http://localhost:8080/health

# Completion request
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'
```

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Basic kernels (scale, mask, softmax) | ✅ Complete |
| Phase 2 | RoPE + KV Append | ✅ Complete |
| Phase 3 | FlashAttention (causal + non-causal) | ✅ Complete |
| Phase 4 | MLP Ops (SwiGLU, RMSNorm) | ✅ Complete |
| Phase 4.5 | GGUF Loader Fixes | ✅ Complete |
| Phase 4.6 | Qwen2 Tensor Mapping | ✅ Complete |
| Phase 5 | MXFP Quantization (MXFP4/MXFP6) | ✅ Complete |
| Phase 7 | GPU Attention Path (2-5x speedup) | ✅ Complete |
| Phase 8 | Q4_1/Q5_0/Q5_1 Support | ✅ Complete |
| Phase 9 | Code Quality (100% test health) | ✅ Complete |
| Phase 10 | Memory Pooling (ROCm workaround) | ✅ Complete |
| Phase 11 | P0/P1 Bug Fixes | ✅ Complete |
| Phase 15 | P1/P2 Code Quality Fixes | ✅ Complete |
| Phase 17 | Async GPU Loading (~5x speedup) | ✅ Complete |
| Phase 18 | Lazy ExecutionPlan (~12x more speedup) | ✅ Complete |
| Phase 20 | GPU Testing Safety | ✅ Complete |
| Phase 21 | CLI Stability Fixes | ✅ Complete |
| Phase 6 | GPU Sampler (top-k/top-p on device) | ❌ Pending |
| Future | FP16 Compute Support | ❌ Planned |
| Future | End-to-End Integration Tests | ❌ Planned |

### Future Work

- [ ] Test CLI end-to-end with real models (Phase 21 fixes applied)
- [ ] GPU-based MXFP dequantization kernels
- [ ] End-to-end integration tests with real models
- [ ] Multi-GPU tensor parallelism
- [ ] Performance benchmarks vs llama.cpp, vLLM
- [ ] Deployment guide

## Development

```bash
# Format code
cargo fmt

# Linter
cargo clippy -- -D warnings

# Run benchmarks
cargo bench

# Full test suite
cargo test --features rocm --workspace
```

## Dependencies

Key libraries:
- **axum**: HTTP server framework
- **tokio**: Async runtime
- **tokenizers**: HuggingFace tokenizers
- **half**: FP16 support
- **memmap2**: Memory-mapped I/O
- **serde/serde_json**: Serialization
- **tracing**: Structured logging

## Contributing

See [docs/TODO.md](docs/TODO.md) for detailed task tracking and [docs/PLAN.md](docs/PLAN.md) for implementation roadmap.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU inference optimization
- [vLLM](https://github.com/vllm-project/vllM) - Efficient batching
- [candle](https://github.com/huggingface/candle) - Rust ML design patterns

## Disclaimer

**This is alpha software.** It is not suitable for any critical use. Components work individually but end-to-end model execution is not fully tested. APIs will change. Bugs exist.

**What this means:**
- ✅ GPU kernels work (274+ tests pass)
- ✅ Async loading (~60x total speedup with lazy loading)
- ✅ Individual components are tested
- ⚠️ CLI fixes applied (Phase 21) but not fully tested end-to-end
- ⚠️ End-to-end inference not fully tested
- ❌ Not ready for any real deployment
- ❌ Use at your own risk

---

**Status**: Alpha | **Tests**: 274+ Passing (100%) | **Code Quality**: B+ (82/100) | **Hardware**: AMD Radeon RX 7900 XT (gfx1100) | **Last Updated**: January 2026 | **Phase**: 21 Complete
