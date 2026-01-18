# ROCmForge

## What This Is

A high-performance LLM inference engine for AMD GPUs, written in Rust. ROCmForge loads GGUF-format models and runs them on ROCm/HIP with optimized CPU fallback using SIMD. It provides an OpenAI-compatible HTTP API for seamless integration with existing LLM applications.

## Core Value

**Reliable, fast inference on AMD GPUs with transparent CPU fallback.**

If ROCm is available, use it. If not, fall back to optimized CPU execution seamlessly. Any GGUF model should just work.

## Requirements

### Validated

*Capabilities that exist and work (inferred from existing codebase):*

- ✓ GGUF model loading — `src/loader/gguf.rs` (2832 lines, functional)
- ✓ HuggingFace tokenizer integration — `src/tokenizer.rs`
- ✓ OpenAI-compatible HTTP API with SSE streaming — `src/http/server.rs`
- ✓ Token sampling (top-k, top-p, temperature) — `src/sampler/`
- ✓ Multi-head attention (CPU backend) — `src/attention/`
- ✓ Paged KV cache — `src/kv_cache/`
- ✓ Basic tensor operations — `src/tensor/`
- ✓ GPU kernels: matmul, softmax, rope, swiglu — `src/ggml/hip_backend/ops/`
- ✓ CLI tools: serve, generate, inspect — `src/bin/rocmforge_cli.rs`
- ✓ Model configs: LLaMA, Qwen — `src/model/config.rs`

### Active

*What we're building toward:*

- [ ] Fix inference hangs (GPU stream synchronization bug)
- [ ] Complete quantized matmul with native HIP dequantization kernel
- [ ] Implement flash attention detection and GPU kernels
- [ ] Add CPU SIMD backend for all tensor operations
- [ ] Hybrid execution scheduler (automatic CPU/GPU op selection)
- [ ] Universal GGUF compatibility (all architectures, quantizations)
- [ ] Performance optimization (balanced: throughput, latency, memory)
- [ ] Production-ready reliability and error handling

### Out of Scope

- **Training features** (LoRA adapters, fine-tuning, training modes) — Focus is inference-only
- **Non-text modalities** (vision, audio, multimodal models) — Text-only for v1
- **Multi-GPU/distributed execution** — Single GPU focus for v1
- **Non-AMD GPU support** — ROCm/HIP only (CPU fallback covers non-GPU systems)

## Context

**Existing Codebase State:**
- Monolithic Rust application with layered architecture (API → Service → Engine → Data → Kernel)
- 96 test files with good coverage, but 20+ tests commented out pending API rewrite
- 3 files exceed 3000 LOC (execution_plan.rs, hip_backend.rs, gguf.rs) — modularization needed
- Known GPU synchronization bugs causing inference hangs (hipBLAS vs hipMemcpy stream mismatch)
- Transition in progress: eprintln! → tracing for logging

**Technical Environment:**
- Rust 2021 edition with Tokio async runtime
- ROCm/HIP for AMD GPU (targeting gfx1100 / RX 7900 XT, broader support goal)
- Axum web framework for HTTP API
- GGUF format for model weights (llama.cpp compatibility)

**Known Issues (from CONCERNS.md):**
- GPU stream synchronization bug in `src/ggml/hip_backend/ops/matmul.rs`
- Race condition in inference loop spawn
- Engine cleanup issues in CLI
- Missing .env.example for environment variables

## Constraints

- **Platform**: ROCm on Linux only — AMD GPU driver limitation
- **Model Format**: GGUF only — Leverage llama.cpp ecosystem
- **Hardware**: AMD GPU or CPU SIMD — No NVIDIA, no other accelerators
- **Architecture**: Single binary, self-contained — No external runtime dependencies beyond ROCm

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rust implementation | Performance, safety, GPU FFI control | — Pending |
| GGUF format only | llama.cpp ecosystem compatibility | — Pending |
| Hybrid CPU/GPU execution | Maximum compatibility, graceful degradation | — Pending |
| OpenAI-compatible API | Drop-in replacement for existing apps | — Pending |
| Modular architecture with trait backends | Easy CPU/GPU switching, testability | — Pending |

---

*Last updated: 2026-01-18 after initialization*
