# ROCmForge Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-19

### Added
- GGUF model loading with 15 quantization formats (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4, MXFP6)
- CPU SIMD backend with AVX2 (f32x8), NEON (f32x4), AVX-512 (f32x16, opt-in)
- Runtime CPU feature detection using `raw-cpuid`
- SIMD operations: matmul, attention (softmax, QK^T, weighted value), RMSNorm, RoPE, SiLU, SwiGLU, GELU
- GPU kernels: dequantization (Q4_0, Q8_0, Q4_K, Q6_K), fused dequant+matmul
- Flash attention backend with GPU implementation
- Hybrid execution scheduler with automatic CPU/GPU backend selection
- HTTP server with OpenAI-compatible API (`/v1/completions`, `/health`, `/ready`, `/metrics`)
- CLI tools: `serve`, `generate`, `context` (add/search/list/clear)
- Paged KV cache for efficient inference
- Token sampling: top-k, top-p, temperature, repetition penalty
- Error handling with unified error types
- Tracing with OpenTelemetry integration
- Performance profiling infrastructure
- Documentation: User Guide, CLI Reference, API Documentation, Deployment Guide
- `.env.example` with environment variable reference

### Fixed
- GPU stream synchronization bug (hipBLAS vs hipMemcpy mismatch)
- Race condition in inference loop spawn
- Engine cleanup issues in CLI
- 98+ test compilation errors

### Test Coverage
- 564+ lib tests passing
- Unit + integration + E2E coverage

### Known Limitations
- GPU sampler kernels use CPU fallback (optimization deferred)
- MQA GPU optimization uses partial CPU fallback
- SIMD feature requires nightly Rust
- ~82 compiler warnings (cosmetic: unused imports, variables)

## [0.0.1] - Earlier Development

### Added
- Initial project structure
- Basic HIP kernel stubs
- GGUF loader (initial implementation)
- HTTP server skeleton
- Test infrastructure

---

**Hardware Tested**: AMD Radeon RX 7900 XT (gfx1100, RDNA3)
**ROCm Version**: 5.0+
**Rust Version**: 1.82+
