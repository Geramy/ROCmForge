# CPU-First Architecture Implementation TODO

**Created:** 2026-01-11
**Status:** Planning Phase
**Reference:** `CPU_FIRST_ARCHITECTURE_PLAN_2026-01-11.md`

## Overview

This TODO tracks the implementation of CPU-first architecture for ROCmForge.
The goal is to make CPU a first-class citizen, using it for 80-90% of operations
while reserving GPU for the remaining 10-20% that truly require massive parallelism.

**CRITICAL:** Phase 0 must be completed FIRST. The application currently hard-fails
if no AMD dGPU is present. We must fix this before any other CPU-first work.

---

## Design Rules (INVARIANTS - ENFORCE DURING IMPLEMENTATION)

**These rules are NON-NEGOTIABLE. They prevent bugs we've already discovered.**

### Rule #1: Sampling NEVER on GPU
- [ ] **ENFORCEMENT:** All sampling code paths route to CPU
- [ ] **REASON:** Phase 6 proved GPU sampling triggers watchdog timeout (120s+)
- [ ] **EVIDENCE:** CPU sampling: 0.10s | GPU sampling: TIMEOUT
- [ ] **TEST:** Add test that GPU sampling code path does not exist

### Rule #2: Detection NEVER Fails
- [ ] **ENFORCEMENT:** `detect_hardware()` returns `ComputeTier`, never `Result`
- [ ] **REASON:** Application must start on any hardware (CPU-only is valid)
- [ ] **CURRENT BUG:** `src/backend/hip_backend.rs:746` returns `Err` when no GPU
- [ ] **TEST:** Verify app starts on machine with no GPU

### Rule #3: CPU Default, GPU Exception
- [ ] **ENFORCEMENT:** New operations default to CPU implementation
- [ ] **REASON:** Portability, debuggability, simplicity
- [ ] **TEST:** Code review verifies CPU-first mindset

### Rule #4: Graceful Degradation
- [ ] **ENFORCEMENT:** Each compute tier is optional (except CPU)
- [ ] **REASON:** User hardware varies widely
- [ ] **TEST:** Test matrix: CPU-only, CPU+iGPU, CPU+dGPU, all three

### Rule #5: Zero Config Required
- [ ] **ENFORCEMENT:** Auto-detect hardware at startup
- [ ] **REASON:** "Just works" user experience
- [ ] **TEST:** Run with no flags, verify optimal config selected

### Rule #6: CPU NEVER IDLES (THE ORCHESTRATOR RULE)
- [ ] **ENFORCEMENT:** CPU always working ahead/behind/parallel to GPU
- [ ] **REASON:** GPU only fast if fed fast. Idle CPU = Starving GPU.
- [ ] **METRIC:** Target 60-80% CPU utilization (not 5-10%)
- [ ] **TEST:** Profile inference - verify no CPU idle gaps

---

## Phase 0: Hardware Detection Layer (1 week) - CRITICAL PRE-REQUISITE

**Goal:** Application starts successfully on ANY hardware, with graceful fallback.

### 0.1 Hardware Detection Module
- [ ] Create `src/hardware/detection.rs` module
- [ ] Create `ComputeTier` enum (CpuOnly, CpuWithIgpu, CpuWithIgpuAndDGpu)
- [ ] Create `IGpuType` enum (IntelIrisXe, IntelArc, AmdRadeonApu, AppleM1/M2/M3)
- [ ] Create `DGpuType` enum (AmdRdna2/3, AmdCdna2/3)
- [ ] Implement `detect_hardware() -> ComputeTier` (NEVER fails)
- [ ] Tests: Verify detection on various hardware configurations

### 0.2 CPU Detection (ALWAYS AVAILABLE)
- [ ] Detect CPU core count (num_cpus)
- [ ] Detect SIMD capabilities (SSE4.2, AVX, AVX2, AVX-512, NEON, SVE)
- [ ] Detect available system memory
- [ ] Tests: Verify CPU detection works on x86_64 and ARM64

### 0.3 iGPU Detection (OPTIONAL)
- [ ] **macOS**: Apple Silicon detection via sysctl
  - [ ] Detect M1/M2/M3 variants
  - [ ] Query GPU memory via ioreg/system_profiler
- [ ] **Linux x86_64**: Intel/AMD iGPU detection
  - [ ] Check `/sys/class/drm/card*/device/vendor` (0x8086=Intel, 0x1002=AMD)
  - [ ] Query shared memory size
- [ ] **Linux ARM**: Mobile SoC GPU detection
  - [ ] Check `/sys/class/devfreq/` for Mali/Adreno GPUs
- [ ] **Windows**: DXGI/Direct3D device enumeration
- [ ] Tests: Verify iGPU detection on supported platforms

### 0.4 dGPU Detection (OPTIONAL)
- [ ] Fix `src/backend/hip_backend.rs::detect_amd_gpu()` to return `Option<HipDevice>` instead of `HipResult<HipDevice>`
- [ ] Add >4GB filter to distinguish dGPU from iGPU
- [ ] Return `None` instead of `Err(HipError::DeviceNotFound)` when no GPU found
- [ ] Tests: Verify graceful handling when no dGPU present

### 0.5 Singleton Backend Refactor
- [ ] Change `GLOBAL_BACKEND: Mutex<Option<Arc<HipBackend>>>` to support CPU-only mode
- [ ] Create `CpuBackend` struct for CPU-only operations
- [ ] Refactor `HipBackend::new()` to call `detect_hardware()` first
- [ ] If no dGPU found, use `CpuBackend` instead of failing
- [ ] Tests: Verify application starts on CPU-only system

### 0.6 CLI Integration
- [ ] Update CLI to print detected hardware at startup
- [ ] Example output: "Detected: CPU (16 cores, AVX2) + AMD Radeon iGPU + No dGPU"
- [ ] Add `--hardware` flag to show detected configuration
- [ ] Tests: Verify CLI starts on various hardware configurations

**Deliverables:**
- Hardware detection module
- Application starts on CPU-only systems (no hard failure)
- iGPU detection on macOS/Linux/Windows
- Graceful fallback: dGPU → iGPU+CPU → CPU-only

**Success Criteria:**
- ✅ Application starts on a laptop with iGPU only
- ✅ Application starts on a desktop with dGPU only
- ✅ Application starts on a system with no GPU at all
- ✅ Hardware detection prints clear, informative message

---

## Phase 1: CPU Optimization Foundation (1-2 weeks)

### 1.1 SIMD Detection & Dispatch
- [ ] Create `src/cpu/simd.rs` module
- [ ] Implement CPU capability detection (SSE4.2, AVX, AVX2, AVX-512, NEON, SVE)
- [ ] Create `SimdLevel` enum
- [ ] Add runtime detection function
- [ ] Create compile-time feature flags for each SIMD level
- [ ] Tests: Verify detection on x86_64 and ARM64

### 1.2 SIMD-Optimized MatMul
- [ ] Create `src/cpu/matmul.rs` module
- [ ] Implement AVX2-optimized matmul kernel
- [ ] Implement AVX-512-optimized matmul kernel (fallback to AVX2)
- [ ] Implement NEON-optimized matmul for ARM64
- [ ] Add scalar fallback for unsupported CPUs
- [ ] Tests: Verify correctness vs naive matmul
- [ ] Benchmarks: Measure speedup over current implementation

### 1.3 Multi-Threading with Rayon
- [ ] Add rayon dependency to Cargo.toml
- [ ] Create `src/cpu/thread_pool.rs` module
- [ ] Implement thread pool for CPU operations
- [ ] Add parallel iterators for batch operations
- [ ] Configurable thread count (default: num_cpus - 2)
- [ ] Tests: Verify thread safety and correctness

### 1.4 Baseline Benchmarks
- [ ] Create `benches/cpu_baseline.rs` benchmark suite
- [ ] Benchmark current CPU matmul
- [ ] Benchmark GPU matmul for different sizes
- [ ] Identify crossover point where GPU becomes faster
- [ ] Document findings in `docs/CPU_BASELINE_BENCHMARKS_2026-01-11.md`

**Deliverables:**
- SIMD detection module
- SIMD-optimized matmul kernel
- Multi-threading infrastructure
- Baseline performance benchmarks

---

## Phase 2: CPU-First Operations (2-3 weeks)

### 2.1 Sampling (✅ COMPLETE in Phase 6)
- [x] Top-p sampling on CPU
- [x] Top-k sampling on CPU
- [x] Fused sampler on CPU
- [x] Tests passing
- [ ] Move to new `src/cpu/sampling.rs` module (refactor)

### 2.2 Element-Wise Operations (SIMD-Optimized)
- [ ] Create `src/cpu/elementwise.rs` module
- [ ] SIMD-optimized RMSNorm
- [ ] SIMD-optimized LayerNorm
- [ ] SIMD-optimized RoPE (rotary embeddings)
- [ ] SIMD-optimized SwiGLU activation
- [ ] SIMD-optimized SiLU activation
- [ ] SIMD-optimized GeLU activation
- [ ] Tests: Verify correctness vs GPU implementations
- [ ] Benchmarks: Measure speedup

### 2.3 Small Matrix Operations
- [ ] Implement size-based dispatch for matmul
- [ ] CPU path for matmul with all dimensions < 1024
- [ ] GPU path for matmul with any dimension > 4096
- [ ] Gray zone (1024-4096): Measure and adapt
- [ ] Tests: Verify correctness at boundary conditions

### 2.4 Memory Operations
- [ ] CPU token embedding lookup (cache-friendly)
- [ ] CPU KV cache read/write
- [ ] CPU quantization/dequantization
- [ ] Memory pooling for CPU allocations
- [ ] Tests: Verify correctness vs GPU

**Deliverables:**
- CPU implementations for all element-wise operations
- Size-based dispatch for matmul
- Memory-optimized operations
- Integration tests

---

## Phase 3: Adaptive Dispatcher (1 week)

### 3.1 Dispatcher Module
- [ ] Create `src/cpu/dispatcher.rs` module
- [ ] Define `ComputeTarget` enum (CPU, GPU, Hybrid)
- [ ] Implement size-based heuristics
- [ ] Add operation cost estimation
- [ ] Tests: Verify correct routing

### 3.2 Performance Monitoring
- [ ] Add performance counters for CPU/GPU operations
- [ ] Track latency for each operation type
- [ ] Track memory usage
- [ ] Adaptive routing based on measurements
- [ ] Logging/tracing for debugging

### 3.3 Fallback Mechanisms
- [ ] GPU → CPU fallback on OOM
- [ ] GPU → CPU fallback on kernel failure
- [ ] CPU → GPU fallback on timeout
- [ ] Graceful degradation
- [ ] Tests: Verify fallback behavior

**Deliverables:**
- Intelligent dispatcher
- Performance monitoring
- Robust fallback mechanisms

---

## Phase 4: BLAS Integration (1-2 weeks)

### 4.1 BLAS Abstraction Layer
- [ ] Create `src/cpu/blas.rs` module
- [ ] Define BLAS trait
- [ ] Implement for BLIS library
- [ ] Implement for oneDNN (Intel/AMD)
- [ ] Implement for Accelerate (macOS)
- [ ] Scalar fallback

### 4.2 Library Integration
- [ ] Add BLIS as optional dependency
- [ ] Add build script for BLIS detection
- [ ] Add oneDNN as optional dependency
- [ ] Add Accelerate framework detection (macOS)
- [ ] Feature flags: `blas`, `onednn`, `accelerate`

### 4.3 Benchmark: Hand-Written vs BLAS
- [ ] Benchmark SIMD matmul vs BLAS matmul
- [ ] Choose fastest implementation
- [ ] Document results

**Deliverables:**
- BLAS abstraction layer
- Library integrations
- Performance comparison

---

## Phase 5: End-to-End Optimization (1-2 weeks)

### 5.1 Memory Layout Optimization
- [ ] Cache-friendly data structures
- [ ] Reduced allocations (memory pool)
- [ ] Memory reuse for intermediate results
- [ ] Contiguous memory layouts

### 5.2 Pipeline Parallelism
- [ ] CPU and GPU working simultaneously
- [ ] Overlapping compute and data transfer
- [ ] Async operation scheduling

### 5.3 Batch Size Optimization
- [ ] Dynamic batch sizing based on load
- [ ] Micro-batching for low latency
- [ ] Continuous batching support

### 5.4 Final Benchmarks
- [ ] End-to-end latency comparison
- [ ] Throughput comparison
- [ ] Memory usage comparison
- [ ] Document in `docs/CPU_FIRST_RESULTS_2026-01-11.md`

**Deliverables:**
- Optimized inference pipeline
- Comprehensive benchmarks
- Final documentation

---

## Phase 6: Tri-Tier Execution (2-3 weeks) - CPU + iGPU + dGPU

**Goal:** Optimize execution when all three compute tiers are available.

### 6.1 iGPU Backend Implementation
- [ ] Create `src/igpu/mod.rs` module
- [ ] Create `src/igpu/opencl_backend.rs` for Intel/AMD iGPU (Linux/Windows)
- [ ] Create `src/igpu/metal_backend.rs` for Apple Silicon (macOS)
- [ ] Create `src/igpu/vulkan_backend.rs` as cross-platform alternative
- [ ] Implement unified memory-aware buffer management
- [ ] Tests: Verify iGPU kernel execution on supported platforms

### 6.2 Layer Split Strategy
- [ ] Implement `split_model_by_memory()` function (llama.cpp strategy)
- [ ] Add `LayerRange` struct for device layer assignment
- [ ] Implement `split-mode layer` (layer-wise distribution)
- [ ] Implement `split-mode row` (tensor parallelism) - FUTURE
- [ ] Add `--tensor-split` CLI flag for custom distribution
- [ ] Tests: Verify correct layer assignment across devices

### 6.3 Tri-Tier Executor
- [ ] Create `src/executor/tri_tier.rs` module
- [ ] Implement `TriTierExecutor` struct with CPU/iGPU/dGPU engines
- [ ] Implement `generate_token()` with pipelined execution
- [ ] Implement KV cache distribution across devices
- [ ] Add stream synchronization for concurrent execution
- [ ] Tests: Verify correctness vs single-device execution

### 6.4 Pipeline Parallelism
- [ ] Implement device prefetching (overlap compute + data transfer)
- [ ] Pipeline stages: CPU → iGPU → dGPU → CPU (sampling)
- [ ] Implement double-buffering for hidden states
- [ ] Add async execution with proper synchronization
- [ ] Tests: Measure pipeline utilization

### 6.5 Cross-Device Communication
- [ ] CPU → iGPU: Zero-copy shared memory paths
- [ ] iGPU → dGPU: PCIe async transfer optimization
- [ ] dGPU → CPU: PCIe transfer for logits only
- [ ] Implement pinned memory for faster PCIe transfers
- [ ] Tests: Measure transfer overhead vs compute gain

### 6.6 Adaptive Workload Balancing
- [ ] Monitor per-device utilization during inference
- [ ] Dynamic layer reassignment based on performance
- [ ] Hot-swapping: Move slow layers between devices
- [ ] Implement performance feedback loop
- [ ] Tests: Verify adaptive behavior under load

### 6.7 CLI Integration
- [ ] Add `--split-mode` flag (layer/row/none)
- [ ] Add `--tensor-split` flag (cpu:N,igpu:N,dgpu:N)
- [ ] Add `--device-override` for manual device selection
- [ ] Print tri-tier configuration at startup
- [ ] Example: "Tri-Tier: CPU(4L) + iGPU(12L) + dGPU(16L)"
- [ ] Tests: Verify all flag combinations

**Deliverables:**
- iGPU backend (OpenCL/Metal/Vulkan)
- Tri-tier executor with pipelining
- Layer split strategy (llama.cpp-compatible)
- CLI flags for device distribution

**Success Criteria:**
- ✅ Application uses CPU + iGPU + dGPU simultaneously
- ✅ All three devices show utilization during inference
- ✅ Throughput > single GPU (despite transfer overhead)
- ✅ Fallback to 2-tier or 1-tier works automatically

---

## Phase 7: CPU Orchestrator (3-4 weeks) - THE NOVEL FEATURES

**Goal:** CPU as active orchestrator - always working ahead of, behind, or parallel to GPU.

**This is what makes ROCmForge unique.** Nobody else is building true hybrid CPU/GPU orchestration.

### 7.1 KV Cache Orchestrator (10-25% improvement)
- [ ] Create `src/orchestrator/kv_orchestrator.rs` module
- [ ] Implement `predict_next_blocks()` - predict which KV blocks will be needed
- [ ] Implement `prefetch_blocks()` - async prefetch from disk/RAM
- [ ] Implement `compress_kv_block()` - Q4, Q6, MXFP4, MXFP6 compression
- [ ] Implement `stage_to_gpu()` - async decompress + stage to GPU
- [ ] Implement `reorder_for_gpu locality()` - optimize access patterns
- [ ] Tests: Measure improvement in KV cache hit rate

### 7.2 Attention Metadata Precomputation (5-15% improvement)
- [ ] Create `src/orchestrator/metadata_builder.rs` module
- [ ] Implement `sort_tokens()` - sort by position/pattern
- [ ] Implement `compute_rope_positions()` - sin/cos tables on CPU
- [ ] Implement `build_attention_mask()` - causal, block-diagonal masks
- [ ] Implement `build_block_table()` - PagedAttention block tables
- [ ] Implement `build_sequence_map()` - ragged batch sequence maps
- [ ] Move existing `src/attention/rope.rs` to CPU (already CPU, verify)
- [ ] Move existing `src/attention/multi_query.rs` metadata to CPU
- [ ] Tests: Measure GPU sync stall reduction

### 7.3 Fact Cache + Local Reasoning (30-60% on repetitive patterns)
- [ ] Create `src/orchestrator/fact_cache.rs` module
- [ ] Implement `detect_repetition()` - hash-based query detection
- [ ] Implement `can_skip_layer()` - predict if layer computation needed
- [ ] Implement `reuse_activations()` - speculative execution cache
- [ ] Add LRU eviction for cache management
- [ ] Tests: Measure speedup on repetitive queries (paper-worthy result)

### 7.4 Layer Skip Predictor (10-30% improvement)
- [ ] Create `src/orchestrator/layer_skip.rs` module
- [ ] Implement `predict_token_spike()` - detect predictable distributions
- [ ] Implement `skip_matmul_if_predictable()` - skip unnecessary compute
- [ ] Implement `reuse_activation()` - hardware branch prediction analogy
- [ ] Add confidence scoring for predictions
- [ ] Tests: Measure accuracy vs speedup tradeoff

### 7.5 Batch Optimizer (15-25% improvement)
- [ ] Create `src/orchestrator/batch_optimizer.rs` module
- [ ] Implement `reorder_sequences()` - maximize GPU utilization
- [ ] Implement `pack_tokens()` - dense packing, no holes
- [ ] Implement `build_batch_groups()` - group by length/complexity
- [ ] Implement `fill_holes()` - fill unused batch slots
- [ ] Implement `predict_next_batch_shape()` - pre-allocate for next batch
- [ ] Tests: Measure GPU utilization improvement

### 7.6 Hybrid KV Cache (enables larger contexts)
- [ ] Create `src/orchestrator/hybrid_kv.rs` module
- [ ] Implement `CompressedKVStore` - CPU RAM cache (64-128 GB)
- [ ] Implement compression formats: Q4, Q6, MXFP4, MXFP6
- [ ] Implement async decompression + GPU staging
- [ ] Add LRU eviction for CPU-side cache
- [ ] Tests: Measure effective context size increase

### 7.7 CPU Utilization Monitoring
- [ ] Add CPU utilization metrics to inference loop
- [ ] Detect idle gaps (CPU waiting on GPU)
- [ ] Profile each orchestrator task
- [ ] Target: 60-80% CPU utilization (not 5-10%)
- [ ] Tests: Continuous profiling during inference

**Deliverables:**
- 6 orchestrator modules (KV, metadata, fact cache, skip, batch, hybrid KV)
- CPU utilization monitoring and dashboards
- Performance measurements for each optimization

**Success Criteria:**
- ✅ CPU utilization 60-80% during inference (not 5-10%)
- ✅ No CPU idle gaps in profiling
- ✅ 20-40% overall throughput improvement
- ✅ Fact cache shows measurable speedup on repetitive workloads
- ✅ Larger effective context sizes via hybrid KV

**Potential Publication:** The fact cache + layer skipping combination could be a major paper if results are strong.

---

## Code Structure

```
src/
├── hardware/
│   ├── mod.rs                 # Hardware detection module
│   ├── detection.rs           # CPU/iGPU/dGPU detection
│   └── tiers.rs               # ComputeTier enums
├── cpu/
│   ├── mod.rs                 # CPU module exports
│   ├── simd.rs                # SIMD detection and utilities
│   ├── matmul.rs              # SIMD-optimized matrix multiplication
│   ├── elementwise.rs         # SIMD-optimized element-wise operations
│   ├── thread_pool.rs         # Multi-threading infrastructure
│   ├── dispatcher.rs          # CPU/GPU dispatcher
│   ├── blas.rs                # BLAS abstraction layer
│   ├── memory.rs              # Memory pooling
│   └── sampling.rs            # CPU sampling (from Phase 6)
├── igpu/                      # NEW: Integrated GPU backends
│   ├── mod.rs                 # iGPU module exports
│   ├── opencl_backend.rs      # Intel/AMD iGPU (Linux/Windows)
│   ├── metal_backend.rs       # Apple Silicon (macOS)
│   ├── vulkan_backend.rs      # Cross-platform alternative
│   └── memory.rs              # Unified memory management
├── gpu/
│   └── ...                    # Existing HIP/ROCm code
├── orchestrator/              # NEW: CPU orchestrator (Phase 7)
│   ├── mod.rs                 # Orchestrator module exports
│   ├── kv_orchestrator.rs     # KV cache prefetch + compression
│   ├── metadata_builder.rs    # Attention metadata precompute
│   ├── fact_cache.rs          # Fact memory + local reasoning
│   ├── layer_skip.rs          # Speculative execution predictor
│   ├── batch_optimizer.rs     # Dynamic batching + packing
│   └── hybrid_kv.rs           # CPU RAM + GPU VRAM hybrid cache
├── executor/                  # NEW: Multi-device orchestration
│   ├── mod.rs                 # Executor module exports
│   ├── tri_tier.rs            # CPU+iGPU+dGPU execution
│   ├── layer_split.rs         # Layer distribution strategy
│   └── pipeline.rs            # Pipeline parallelism
└── ops/
    └── dispatch.rs            # Unified operation dispatcher

benches/
├── cpu_baseline.rs            # CPU performance benchmarks
├── igpu_baseline.rs           # iGPU performance benchmarks
├── tri_tier.rs                # CPU+iGPU+dGPU benchmarks
├── orchestrator.rs            # Orchestrator feature benchmarks
└── end_to_end.rs              # Full pipeline benchmarks
```

---

## Dependencies to Add

```toml
[dependencies]
# Existing
rayon = "1.10"                  # Multi-threading

# New for CPU-first
num_cpus = "1.16"                # CPU count detection
paste = "1.0"                    # SIMD trait
wide = "0.7"                     # SIMD wrappers

# Optional BLAS libraries
blis = { version = "0.2", optional = true }
onednn = { version = "3.3", optional = true }

# iGPU backends (all optional)
ocl = { version = "0.19", optional = true }           # OpenCL for Intel/AMD iGPU
metal = { version = "0.27", optional = true }         # Metal for Apple Silicon
vulkan = { version = "0.10", optional = true }        # Vulkan cross-platform

[features]
default = []
cpu-simd = []                    # Enable SIMD optimizations
cpu-blas-blis = ["blis"]         # Use BLIS for matmul
cpu-blas-onednn = ["onednn"]     # Use oneDNN
cpu-blas-accelerate = []         # Use macOS Accelerate

# iGPU support
igpu-opencl = ["ocl"]            # Intel/AMD iGPU via OpenCL
igpu-metal = ["metal"]           # Apple Silicon via Metal
igpu-vulkan = ["vulkan"]         # Cross-platform via Vulkan
```

---

## Testing Strategy

### Unit Tests
- Each CPU implementation must match GPU correctness
- Test with various input sizes
- Test edge cases (empty, single element, very large)

### Integration Tests
- Full transformer forward pass with CPU-first
- Verify output matches GPU-only path
- Test fallback mechanisms

### Benchmark Tests
- Measure performance for different operation sizes
- Compare CPU vs GPU
- Track improvements over time

---

## Success Criteria

- [ ] All CPU implementations match GPU correctness
- [ ] Overall inference latency ≤ current GPU-only
- [ ] Memory usage ≤ current (no regression)
- [ ] Code maintainability improved (simpler CPU paths)
- [ ] 40-60% of GPU performance on CPU for small models
- [ ] No GPU required for models < 1B parameters

---

## Timeline Estimate

| Phase | Duration | Dependencies | Priority |
|-------|----------|--------------|----------|
| **Phase 0: Hardware Detection** | **1 week** | **None** | **P0 - BLOCKS EVERYTHING** |
| Phase 1: Foundation | 1-2 weeks | Phase 0 | P1 |
| Phase 2: CPU Operations | 2-3 weeks | Phase 1 | P1 |
| Phase 3: Dispatcher | 1 week | Phase 2 | P1 |
| Phase 4: BLAS Integration | 1-2 weeks | Phase 1 | P2 |
| Phase 5: Optimization | 1-2 weeks | Phase 2, 3, 4 | P2 |
| **Phase 6: Tri-Tier Execution** | **2-3 weeks** | **Phase 0, 2, 5** | **P1 - HIGH VALUE** |
| **Phase 7: CPU Orchestrator** | **3-4 weeks** | **Phase 0, 2** | **P1 - NOVEL, PUBLICATION-WORTHY** |
| **Total** | **12-18 weeks** | | |

**CRITICAL:** Phase 0 is a **BLOCKER** for all other work. The current application
hard-fails if no AMD dGPU is present. This must be fixed before any CPU-first
optimization can be tested on systems without discrete GPUs.

**Phase 6 Note:** Tri-tier execution is the "killer feature" - using CPU + iGPU + dGPU
simultaneously provides maximum performance on high-end systems.

**Phase 7 Note:** The CPU Orchestrator is what makes ROCmForge **unique**. Nobody else is building true hybrid CPU/GPU orchestration for LLM inference. The fact cache + layer skipping features could be **paper-worthy** if results are strong.

**Parallelization:** Phases 4, 6, and 7 can partially overlap:
- Phase 4 (BLAS) can happen during Phase 6/7
- Phase 6 (Tri-tier) and Phase 7 (Orchestrator) share dependencies but can be developed incrementally

---

## References

- Architecture Plan: `docs/CPU_FIRST_ARCHITECTURE_PLAN_2026-01-11.md`
- Phase 6 Report: `docs/PHASE_6_GPU_SAMPLER_FINAL_REPORT_2026-01-11.md`
- llama.cpp: https://github.com/ggml-org/llama.cpp
- oneDNN: https://github.com/uxlfoundation/onednn
- BLIS: https://github.com/flame/blis
