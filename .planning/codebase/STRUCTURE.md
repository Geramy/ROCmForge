# Codebase Structure

**Analysis Date:** 2026-01-18

## Directory Layout

```
rocmtree/
├── src/               # All source code
├── tests/             # Integration tests
├── benches/           # Criterion benchmarks
├── docs/              # Documentation files
├── examples/          # Example code (if present)
├── Cargo.toml         # Package manifest
└── Makefile           # Build tasks
```

## Directory Purposes

**src/**
- Purpose: All Rust source code
- Contains: 100+ .rs files organized by domain
- Key files: `lib.rs` (entry), `engine.rs` (orchestrator)
- Subdirectories: See detailed breakdown below

**src/bin/**
- Purpose: CLI and tool binaries
- Contains: `rocmforge_cli.rs`, `inspect_gguf.rs`, `test_gguf_load.rs`
- Key files: `rocmforge_cli.rs` - main CLI interface

**src/backend/**
- Purpose: GPU abstraction layer
- Contains: `hip_backend.rs`, `hip_blas.rs`, `gpu_executor.rs`, `scratch.rs`
- Key files: `hip_backend.rs` - main HIP/ROCm backend

**src/ggml/**
- Purpose: GGML-style IR implementation
- Contains: `backend.rs`, `graph.rs`, `tensor.rs`
- Subdirectories: `hip_backend/ops/` - GPU kernel implementations

**src/attention/**
- Purpose: Attention mechanisms
- Contains: `backend_registry.rs`, `gpu.rs`, `cpu.rs`
- Key files: `backend_registry.rs` - pluggable backend system

**src/kv_cache/**
- Purpose: Key-Value cache for inference
- Contains: `kv_cache.rs` (1,439 lines - needs splitting)
- Key files: `kv_cache.rs` - paged attention implementation

**src/scheduler/**
- Purpose: Request scheduling and batching
- Contains: `scheduler.rs`, `mod.rs`
- Key files: `scheduler.rs` - continuous batching logic

**src/loader/**
- Purpose: Model file loading
- Contains: `gguf.rs`, `mmap_loader.rs`, `mod.rs`
- Key files: `gguf.rs` - GGUF format parser

**src/model/**
- Purpose: Model implementations and configs
- Contains: `config.rs`, `execution_plan/`
- Key files: `config.rs` - model configurations

**src/sampler/**
- Purpose: Token sampling strategies
- Contains: `gpu.rs`, `mod.rs`
- Key files: GPU-based sampling implementations

**src/http/**
- Purpose: HTTP API server
- Contains: `server.rs`, `mod.rs`
- Key files: `server.rs` - Axum server with SSE

**src/tensor/**
- Purpose: Tensor operations
- Contains: `matmul.rs`, other tensor ops
- Key files: `matmul.rs` - matrix multiplication

**tests/**
- Purpose: Integration tests
- Contains: 12 test files (e.g., `hip_blas_matmul_tests.rs`, `attention_tests.rs`)
- Naming: `{feature}_tests.rs` pattern

**benches/**
- Purpose: Criterion benchmarks
- Contains: `phase12_benchmark.rs`
- Key files: Performance benchmarks for attention and batching

**docs/**
- Purpose: Project documentation
- Contains: `.md` files including ROADMAP.md, DATABASE_SCHEMA.md
- Key files: Project planning and architecture docs

## Key File Locations

**Entry Points:**
- `src/lib.rs` - Library entry point, public API exports
- `src/bin/rocmforge_cli.rs` - Main CLI interface
- `src/engine.rs` - Core inference orchestrator

**Configuration:**
- `Cargo.toml` - Package manifest and dependencies
- `Makefile` - Build tasks

**Core Logic:**
- `src/engine.rs` - Inference engine
- `src/scheduler/scheduler.rs` - Continuous batching
- `src/backend/hip_backend.rs` - GPU backend
- `src/kv_cache/kv_cache.rs` - KV cache

**Testing:**
- `tests/` - Integration tests
- `benches/` - Performance benchmarks

**Documentation:**
- `README.md` - User documentation
- `docs/` - Developer documentation

## Naming Conventions

**Files:**
- snake_case.rs for modules (e.g., `hip_backend.rs`, `matmul.rs`)
- {name}_tests.rs for test files (e.g., `hip_blas_matmul_tests.rs`)
- mod.rs for module exports

**Directories:**
- snake_case for all directories
- Singular names for modules (e.g., `backend/`, not `backends/`)
- Plural for collections (e.g., `benches/`, `tests/`)

**Special Patterns:**
- bin/ for CLI binaries
- ops/ for kernel operations
- execution_plan/ for sub-modules

## Where to Add New Code

**New Feature (inference):**
- Primary code: `src/` (create new directory if major feature)
- Tests: `tests/{feature}_tests.rs`
- Benchmarks: `benches/{feature}_benchmark.rs`

**New GPU Operation:**
- Implementation: `src/ggml/hip_backend/ops/{operation}.rs`
- Tests: `tests/{operation}_tests.rs`
- Export from: `src/ggml/hip_backend/ops/mod.rs`

**New Attention Implementation:**
- Implementation: `src/attention/{name}.rs`
- Register in: `src/attention/backend_registry.rs`

**New Model Format Support:**
- Implementation: `src/loader/{format}.rs`
- Export from: `src/loader/mod.rs`

**Utilities:**
- Shared helpers: `src/util/` (create if needed)
- GPU utilities: `src/backend/`

## Special Directories

**src/bin/**
- Purpose: Executable binaries
- Source: Compiled by Cargo as separate binaries
- Committed: Yes

**tests/**
- Purpose: Integration tests
- Source: Compiled and run by `cargo test`
- Committed: Yes

**benches/**
- Purpose: Criterion benchmarks
- Source: Run by `cargo bench`
- Committed: Yes

**.planning/**
- Purpose: Project planning documents (GSD workflow)
- Source: Created by `/gsd:*` commands
- Committed: Yes

---

*Structure analysis: 2026-01-18*
*Update when directory structure changes*
