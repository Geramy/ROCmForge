# ROCmForge Codebase Audit

> Generated: 2026-01-03
> Status: Infrastructure Complete, Kernels Missing
>
> **See also:** `implementation_roadmap.md` for exact kernel contracts and test harness patterns.

---

## Executive Summary

ROCmForge has solid architectural foundations with **complete infrastructure** but **missing critical GPU kernels**. The project has ~1,323 symbols across 93 files with well-organized modules. The bottleneck is not architecture—it's the absence of real HIP kernels for the performance-critical path.

**Bottom line:** You have a well-structured shell that still depends on CPU fallback for the heavy lifting.

---

## A. What We Have (Implementation Status)

### ✅ Complete Infrastructure

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| **Project Layout** | `lib.rs` | ✅ Complete | Clean module organization |
| **Backend Abstraction** | `backend/` | ✅ Complete | HipBackend, DeviceTensor, HipBuffer |
| **HIP FFI Bindings** | `hip_backend.rs` | ✅ Complete | Full FFI to amdhip64 |
| **hipBLAS Wrapper** | `hip_blas.rs` | ✅ Complete | sgemm, saxpy, sscal |
| **Memory Management** | `kv_cache/` | ✅ Complete | Paged KV cache with GPU buffers |
| **Scheduler** | `scheduler/` | ✅ Complete | Request batching, queue management |
| **Sampler** | `sampler/` | ✅ Complete | CPU-based top-k/top-p (well-tested) |
| **Engine** | `engine.rs` | ✅ Complete | Async inference loop |
| **HTTP Server** | `http/` | ✅ Complete | Axum-based REST API |
| **Tokenizer** | `tokenizer.rs` | ✅ Complete | tokenizers crate integration |
| **Model Loaders** | `loader/` | ✅ Complete | GGUF, ONNX, mmap loaders |
| **CPU Attention** | `attention/cpu.rs` | ✅ Complete | Reference implementation |
| **RoPE (CPU)** | `attention/rope.rs` | ✅ Complete | With precomputed cos/sin |
| **Tensor Ops** | `tensor/` | ✅ Complete | CPU matmul, shape handling |
| **QKV Fused Op** | `ops/qkv.rs` | ✅ Complete | Uses hipBLAS GEMM |

---

## B. What Is Mocked / Stub Implementation

### ⚠️ GPU Kernels (Shell Only - No Real HIP Code)

| Kernel | File | What It Does | Reality |
|--------|------|--------------|---------|
| `scale_gpu_kernel` | `attention/kernels.rs:17` | Returns 0 (success) | **No operation** |
| `mask_gpu_kernel` | `attention/kernels.rs:32` | Returns 0 (success) | **No operation** |
| `softmax_gpu_kernel` | `attention/kernels.rs:52` | Returns 0 (success) | **No operation** |

**Impact:** GPU attention path in `attention/gpu.rs` calls these but they do nothing. Data gets copied to GPU, "processed" by no-ops, then copied back to CPU for real computation.

### ⚠️ GPU Attention (Hybrid CPU/GPU)

| Function | File | Behavior |
|----------|------|----------|
| `GpuBackend::forward` | `attention/gpu.rs:30` | Copies to GPU, calls no-op kernels, copies back to CPU for actual work |
| `GpuBackend::forward_device` | `attention/gpu.rs:338` | Falls back to CPU via `to_host_vec()` |

### ⚠️ RoPE GPU (CPU Fallback)

| Function | File | Behavior |
|----------|------|----------|
| `Rope::apply_rope_device` | `attention/rope.rs:226` | Copy to host → CPU RoPE → copy back to device |

**Comment in code (line 233):** `// TODO: Implement GPU kernel for RoPE`

### ⚠️ MLP / SwiGLU (CPU Fallback)

| Function | File | Behavior |
|----------|------|----------|
| `HipBackend::mlp_swiglu` | `backend/hip_backend.rs:1183` | Uses hipBLAS for GEMM but CPU for SwiGLU activation (lines 1284-1300) |
| `HipBackend::layernorm` | `backend/hip_backend.rs:1328` | CPU-only implementation (lines 1379-1421) |

### ⚠️ No Actual HIP Kernel Files

```bash
$ ls kernels/**/*.hip
# No files found
```

The code references `kernels/softmax.hip`, `kernels/mask.hip`, `kernels/scale.hip` in `hip_backend.rs:976-992` but **these files don't exist**.

---

## C. What Is Missing (Critical Path)

### 1. **Fused Attention Kernel** (Highest Impact)

**What:** Single HIP kernel doing QK^T → scale → mask → softmax → (softmax × V)

**NVIDIA equivalent:** FlashAttention-2

**AMD equivalent needed:** Wave64-aligned persistent kernel with LDS tiling

**Why it matters:** This is the single biggest performance unlock. Current implementation does ~5 memory round trips (GPU→CPU→GPU→CPU→GPU).

**Files to create:**
- `kernels/flash_attention.hip` - Real fused attention

---

### 2. **GPU RoPE + KV Append Kernel**

**What:** Apply rotary embeddings AND append to KV cache in one kernel

**Current state:** Separate CPU RoPE, then copy

**Why it matters:** Eliminates host round-trip per token during decode

---

### 3. **GPU Sampler (top-k/top-p)**

**What:** GPU-side sampling kernel

**Current state:** CPU-only in `sampler/sampler.rs`

**Why it matters:** Overlaps sampling with next-token compute; decode latency drops

---

### 4. **SwiGLU Activation Kernel**

**What:** Element-wise SwiGLU: `gate(x) * swish(up(x))`

**Current state:** CPU fallback in `hip_backend.rs:1284-1300`

**Why it matters:** MLP is ~1/3 of compute; can't have CPU round-trip here

---

### 5. **LayerNorm Kernel**

**What:** GPU-side LayerNorm

**Current state:** CPU fallback in `hip_backend.rs:1328`

**Why it matters:** Runs before every attention and MLP

---

### 6. **Wave64-Optimized GEMM Layout**

**What:** Input tensors pre-transposed for wave64 MMA

**Current state:** Uses hipBLAS (good) but no layout optimization

**Why it matters:** hipBLAS is decent, but custom layout = 2-3x win for attention shapes

---

## D. Build System Status

### `Cargo.toml` Analysis

```toml
[features]
default = []      # No default features!
rocm = []         # Empty - just a flag, no actual ROCm deps
```

**Issues:**
1. No `hip-sys` or ROCm dependency crates (commented out lines 10-12)
2. `rocm` feature does nothing - it's just a cfg flag
3. No build script to compile `.hip` files
4. No link to `amdhip64` (the FFI in `hip_backend.rs` won't find amdhip64 at runtime without proper linking)

**What's needed:**
- `build.rs` that calls `hipcc` to compile `.hip` to `.hsaco`
- Proper linkage to `amdhip64`
- Optional: `half` crate for FP16 support

---

## E. Dependency Graph (Critical Path)

```
InferenceEngine::run_forward_pass()
    └─> ModelRuntime::decode_step()
        └─> ExecutionPlan::forward_layer()
            ├─> HipBackend::layernorm()           [CPU MOCK]
            ├─> Attention::forward_device()
            │   └─> GpuBackend::forward_device()  [CPU FALLBACK]
            │       └─> scale_gpu_kernel()        [NO-OP]
            │       └─> mask_gpu_kernel()         [NO-OP]
            │       └─> softmax_gpu_kernel()      [NO-OP]
            └─> HipBackend::mlp_swiglu()          [CPU ACTIVATION]
                └─> SwiGLU activation             [CPU MOCK]

Sampler::sample()                                  [CPU ONLY]
```

**Every `decode_step` does ~4-5 host round-trips.**

---

## F. Mapping to "NVIDIA Stack Decomposed"

| NVIDIA Component | ROCmForge Status |
|-----------------|------------------|
| **A. Linear Algebra** | ✅ hipBLAS via `HipBlasHandle` |
| **B. Attention Kernels** | ❌ Kernels are no-ops; CPU fallback |
| **C. Memory Model** | ✅ Paged KV cache structure exists |
| **D. Execution Model** | ✅ Streams, async engine exist |
| **E. Sampler/Decode** | ⚠️ CPU-only; no GPU kernels |
| **F. Glue Layer** | ✅ HTTP API, tokenizer, loaders all present |

---

## G. The Three Missing Wins

Per the "practical order of attack" document:

| Phase | What | ROCmForge Status |
|-------|------|------------------|
| 1 | CPU SIMD matmul + RMSNorm + RoPE | ✅ Complete |
| 2 | GPU QKV GEMM via rocBLAS | ✅ Complete (via hipBLAS) |
| 3 | GPU KV append + RoPE kernel | ❌ Missing |
| 4 | Fused attention kernel | ❌ Missing (kernels are no-ops) |
| 5 | GPU sampler (top-k/top-p) | ❌ CPU-only |

---

## H. Concrete Next Steps

### Immediate (High Impact, Low Effort)

1. **Create `build.rs`** - Compile `.hip` files with `hipcc`
2. **Implement `softmax.hip`** - Row-wise softmax with LDS reduction
3. **Implement `scale.hip`** - Simple element-wise multiply
4. **Implement `mask.hip`** - Element-wise conditional replace

### Medium (Write Real Kernels)

5. **Fused attention kernel** - `kernels/flash_attention.hip`
6. **RoPE + KV append kernel** - Single fused op
7. **SwiGLU kernel** - Element-wise activation

### Long (Optimization)

8. **GPU sampler** - top-k/top-p on device
9. **FP16 support** - Add `half` crate usage
10. **Wave64 tuning** - Optimize block sizes for AMD architecture

---

## I. File Summary

```
src/
├── lib.rs                           ✅ Public API exports
├── engine.rs                        ✅ Async inference loop
├── backend/
│   ├── mod.rs                       ✅ Exports
│   ├── hip_backend.rs       ✅ FFI + DeviceTensor (1934 lines)
│   ├── hip_blas.rs          ✅ hipBLAS wrapper (246 lines)
│   ├── gpu_executor.rs              ✅ GPU execution planning
│   └── scratch.rs                   ✅ Scratch buffer manager
├── attention/
│   ├── mod.rs                       ✅ Attention struct
│   ├── backend.rs                   ✅ Backend enum
│   ├── cpu.rs                       ✅ CPU reference impl
│   ├── gpu.rs                       ⚠️ Calls no-op kernels
│   ├── kernels.rs                   ❌ No-op kernel stubs
│   ├── rope.rs                      ⚠️ CPU only (GPU fallback)
│   ├── softmax.rs                   ✅ CPU softmax
│   ├── mask.rs                      ✅ CPU masking
│   └── compute.rs                   ✅ CPU matmul
├── kv_cache/
│   ├── mod.rs                       ✅ Exports
│   └── kv_cache.rs          ✅ Paged cache (446 lines)
├── sampler/
│   ├── mod.rs                       ✅ Exports
│   └── sampler.rs           ✅ CPU sampling (475 lines)
├── scheduler/
│   └── scheduler.rs                 ✅ Request batching
├── ops/
│   ├── mod.rs                       ✅ Exports
│   └── qkv.rs                       ✅ Fused QKV (273 lines)
├── tensor/
│   └── mod.rs                       ✅ Tensor abstraction
├── loader/
│   ├── mod.rs                       ✅ Exports
│   ├── gguf.rs                      ✅ GGUF parser
│   ├── gguf_loader.rs               ✅ GGUF loader
│   └── mmap_loader.rs               ✅ mmap support
├── model/
│   └── ...                          ✅ Config, execution plan
├── http/                            ✅ REST API
└── bin/
    └── rocmforge_cli.rs             ✅ CLI entry point

kernels/                             ❌ EMPTY (no .hip files)
```

---

## J. Statistics

| Metric | Value |
|--------|-------|
| Total Rust files | ~50 |
| Total lines (Rust) | ~15,000+ |
| HIP kernel files | **0** |
| CPU fallback points | 5+ |
| `// TODO: Implement GPU kernel` | 3+ confirmed |

---

## K. The "Why It Still Works" Explanation

The codebase doesn't crash because:

1. **CPU fallbacks are everywhere** - `to_host_vec()` is liberally used
2. **No-op kernels return success** - The GPU functions return 0 without doing anything
3. **Data copies hide the issue** - GPU→CPU→GPU happens transparently

This means **correctness is fine** but **performance is CPU-bound**.

---

## Conclusion

> "The hard part is already done." - Practical Order of Attack

| What's Done | What's Missing |
|-------------|---------------|
| Project layout ✅ | Real HIP kernels ❌ |
| Backend abstraction ✅ | build.rs for .hip compilation ❌ |
| HTTP integration ✅ | Wave64 optimizations ❌ |
| Scheduler ✅ | GPU sampler ❌ |
| KV cache structure ✅ | Fused attention ❌ |

**Assessment:** This is exactly the situation described in "practical_order_of_attack.md"—you built the tools first. Now the kernels are the clear, narrow next step.
