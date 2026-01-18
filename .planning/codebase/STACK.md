# Technology Stack

**Analysis Date:** 2026-01-18

## Languages

**Primary:**
- Rust 2021 edition - All application code (100+ .rs files)

**Secondary:**
- Not detected (no Python, JavaScript, TypeScript, or other languages)

## Runtime

**Environment:**
- Rust native binary - No interpreted runtime required
- Tokio async runtime v1.0 - For async HTTP and concurrent operations
- ROCm/HIP - AMD GPU computing platform (placeholder dependencies)

**Package Manager:**
- Cargo - Rust's package manager
- Lockfile: Cargo.lock present

## Frameworks

**Core:**
- Axum 0.7 - HTTP server framework with JSON support
- Tower 0.4 - Service composition and middleware
- Tokio 1.0 - Async runtime (features: full)

**Testing:**
- Built-in Rust test harness - Unit and integration tests
- Criterion 0.5 - Benchmarking framework
- Mockall 0.12 - Mocking framework
- Proptest 1.4 - Property-based testing
- Serial Test 3.0 - Thread-safe test execution

**Build/Dev:**
- Cargo - Build system and package manager
- Clippy - Linting (with extensive GPU-specific allowances)
- Rustfmt - Code formatting (no explicit config found)

## Key Dependencies

**Critical:**
- tokenizers 0.15 - HuggingFace tokenizer support
- half 2.4 - FP16 numerical format for GPU tensors
- memmap2 0.9 - Memory-mapped file I/O for model loading
- reqwest 0.11 - HTTP client with JSON and streaming
- reqwest-eventsource 0.4 - Server-Sent Events support

**Infrastructure:**
- serde/serde_json 1.0 - Serialization
- tracing 0.1 - Structured logging
- rayon 1.10 - Parallel processing
- anyhow/thiserror 1.0 - Error handling

## Configuration

**Environment:**
- No .env files detected
- Environment variables: ROCMFORGE_GGUF, ROCMFORGE_TOKENIZER, ROCMFORGE_MODELS
- Configuration through CLI arguments - `src/bin/rocmforge_cli.rs`

**Build:**
- `Cargo.toml` - Rust package configuration
- `Makefile` - Common build tasks

## Platform Requirements

**Development:**
- Linux required (ROCm is Linux-only)
- AMD GPU with ROCm support
- Rust toolchain (1.70+ edition 2021)

**Production:**
- Native binary distribution
- AMD GPU hardware required
- No containerization (no Dockerfile detected)

---

*Stack analysis: 2026-01-18*
*Update after major dependency changes*
