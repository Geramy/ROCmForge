# Architecture

**Analysis Date:** 2026-01-18

## Pattern Overview

**Overall:** Layered Monolith with Microkernel Design

**Key Characteristics:**
- Single-purpose inference engine (not general-purpose framework)
- Layered architecture with clear separation of concerns
- GPU-backend abstraction for portability
- GGML-style intermediate representation for tensor operations
- Continuous batching scheduler for throughput optimization
- Pluggable attention implementations

## Layers

**API Layer:**
- Purpose: External interface for inference requests
- Contains: HTTP server with SSE streaming, CLI tools
- Location: `src/http/server.rs`, `src/bin/`
- Depends on: Service layer
- Used by: External clients

**Service Layer:**
- Purpose: Orchestration and business logic
- Contains: InferenceEngine, Scheduler, Sampler, Tokenizer
- Location: `src/engine.rs`, `src/scheduler/`, `src/sampler/`
- Depends on: Computation layer, Data layer
- Used by: API layer

**Computation Layer:**
- Purpose: GPU kernel execution and tensor operations
- Contains: HipBackend, GgmlBackend, attention implementations
- Location: `src/backend/`, `src/ggml/`, `src/attention/`
- Depends on: Hardware (ROCm/HIP)
- Used by: Service layer

**Data Layer:**
- Purpose: Model loading, caching, tensor storage
- Contains: GGUF loader, KV cache, tensor storage
- Location: `src/loader/`, `src/kv_cache/`, `src/tensor/`
- Depends on: File system, GPU memory
- Used by: Service layer, Computation layer

## Data Flow

**HTTP Inference Request:**

1. Client sends POST request to `/v1/chat/completions`
2. Axum router receives request at `src/http/server.rs`
3. Scheduler queues request and manages batching (`src/scheduler/`)
4. InferenceEngine coordinates execution (`src/engine.rs`)
5. Backend executes GPU kernels via HIP (`src/backend/hip_backend.rs`)
6. GGML IR optimizes tensor operations (`src/ggml/`)
7. Response streamed via Server-Sent Events

**State Management:**
- KV cache stored in GPU memory (`src/kv_cache/`)
- Model weights loaded once and kept resident
- No persistent database (in-memory only)

## Key Abstractions

**InferenceEngine:**
- Purpose: Central orchestrator for inference execution
- Examples: `src/engine.rs`
- Pattern: Singleton-like service coordinating model, cache, backend

**Backend:**
- Purpose: GPU abstraction layer for computation
- Examples: `src/backend/hip_backend.rs`, `src/ggml/backend.rs`
- Pattern: Trait-based abstraction for portability

**AttentionBackend:**
- Purpose: Pluggable attention mechanism implementations
- Examples: `src/attention/backend_registry.rs`, `src/attention/gpu.rs`
- Pattern: Registry pattern for dynamic backend selection

**DeviceTensor:**
- Purpose: GPU memory management wrapper
- Examples: `src/backend/hip_backend.rs`
- Pattern: RAII for GPU resource cleanup

## Entry Points

**Library Entry:**
- Location: `src/lib.rs`
- Triggers: Used as dependency by other crates
- Responsibilities: Public API exports, module declarations

**HTTP Server:**
- Location: `src/bin/rocmforge_cli.rs` (serve command)
- Triggers: User runs `rocmforge serve`
- Responsibilities: Start Axum server, bind to address

**CLI Tools:**
- Location: `src/bin/inspect_gguf.rs`, `src/bin/test_gguf_load.rs`
- Triggers: User runs CLI commands
- Responsibilities: Model inspection, testing utilities

## Error Handling

**Strategy:** Result types with thiserror for custom errors

**Patterns:**
- Custom error types derive `Debug` and `Error` via thiserror
- Backend-specific error variants (`HipError`, `GgufError`)
- Many `unwrap()` and `expect()` calls (technical debt)

## Cross-Cutting Concerns

**Logging:**
- Structured logging via tracing crate
- Debug-level tracing for GPU operations
- Some temporary debug prints remain (e.g., in matmul)

**Validation:**
- Dimension validation for tensor operations (`src/tensor/matmul.rs`)
- File format validation for GGUF loading

**Memory Management:**
- Scratch allocator for GPU temporary memory (`src/backend/scratch.rs`)
- RAII patterns for GPU buffers
- Paged KV cache for memory efficiency

---

*Architecture analysis: 2026-01-18*
*Update when major patterns change*
