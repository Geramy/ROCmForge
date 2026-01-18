# Phase 09: Performance Optimization - Execution Plan

**Phase:** 09
**Mode:** revised
**Created:** 2026-01-18
**Status:** Ready for execution

---

## Phase Goal

Balanced optimization of throughput, latency, and memory efficiency for LLM inference on AMD GPUs. Profile current performance, identify bottlenecks, add benchmarks, and optimize until targets are met.

**From ROADMAP.md:**
- Profile and optimize throughput (tokens/second)
- Profile and optimize latency (first token time)
- Profile and optimize memory efficiency (KV cache, allocations)
- Add performance benchmarks and regression tests

---

## Must Haves (Verification Criteria)

After phase completion, the following must be true:

1. **Profiling Infrastructure**
   - ROCm profiling tools integrated (rocprof, rocperf, omniperf)
   - Baseline measurements established for target hardware
   - Kernel timing wrappers in place
   - Performance can be measured and compared

2. **Benchmark Suite**
   - MatMul benchmarks (dense and quantized) covering all major formats
   - Attention benchmarks (CPU vs GPU)
   - Dequantization benchmarks (format comparison)
   - End-to-end inference benchmarks
   - Baseline storage and regression detection

3. **Throughput Optimization**
   - Memory bandwidth bottlenecks identified and addressed
   - Kernel tuning for RDNA2/RDNA3 (block sizes, occupancy)
   - Operator fusion where beneficial
   - Target: >40 tokens/sec for 7B Q4_K_M on RDNA3

4. **Latency Optimization**
   - Time to first token (TTFT) measured and optimized
   - Prompt processing path optimized
   - Kernel launch overhead reduced
   - Target: TTFT <200ms for 512 token prompts

5. **Memory Optimization**
   - KV cache memory usage profiled
   - Allocation patterns optimized
   - Memory efficiency documented

---

## Dependencies

- **Phase 6 (Attention Optimization):** Complete - Flash attention kernels available
- **Phase 7 (Hybrid Execution):** Complete - Telemetry infrastructure exists
- **Hardware:** RDNA2 or RDNA3 GPU required for GPU-specific profiling

---

## Files Modified (Expected)

### New Files
```
benches/matmul_bench.rs
benches/dequant_bench.rs
benches/inference_bench.rs
benches/memory_bench.rs
src/profiling/mod.rs
src/profiling/kernel_timer.rs
src/profiling/rocprof_integration.rs
src/profiling/baseline.rs
benchmarks/baselines/rdna3-baseline.json
docs/PROFILING_GUIDE.md
docs/PERFORMANCE.md
```

### Modified Files
```
Cargo.toml            # Add new benchmarks
benches/attention_bench.rs  # Enhance with GPU comparison
src/ggml/hip_backend/mod.rs # Profiling hooks
src/ggml/hybrid_scheduler.rs  # Enhanced telemetry
kernels/q4_0_matmul.hip      # Tuning adjustments
kernels/flash_attention.hip  # Tuning adjustments
```

---

## Waves

### Wave 1: Profiling Infrastructure (Parallel)
**Goal:** Establish baseline measurements and tooling

### Wave 2: Benchmark Suite (Sequential)
**Goal:** Create comprehensive benchmarks for all hot paths

### Wave 3: Throughput Optimization (Sequential)
**Goal:** Optimize memory bandwidth and kernel efficiency

### Wave 4: Latency Optimization (Sequential)
**Goal:** Reduce time to first token

### Wave 5: Memory Optimization (Sequential)
**Goal:** Optimize KV cache and allocation patterns

---

## Tasks

### Wave 1: Profiling Infrastructure

#### 09-01: Create Kernel Timing Infrastructure

```yaml
wave: 1
depends_on: []
autonomous: true
files_modified:
  - src/profiling/mod.rs
  - src/profiling/kernel_timer.rs
```

**Description:** Add HIP event-based timing wrappers for accurate GPU kernel duration measurement. This provides the foundation for all profiling work.

**Files:**
- `src/profiling/mod.rs` (new)
- `src/profiling/kernel_timer.rs` (new)

**Actions:**
1. Create profiling module with kernel timing wrapper
2. Implement HIP event-based timing (hipEventCreate, hipEventRecord, hipEventSynchronize, hipEventElapsedTime)
3. Add scoped timer macro for automatic measurement
4. Add CPU-side timing for comparison
5. Document usage in module docs

**Acceptance Criteria:**
- `KernelTimer` type with `for_kernel()`, `start()`, `stop()`, `elapsed()` methods
- Scoped `time_kernel!` macro for automatic timing
- HIP events properly managed (create/record/destroy)
- CPU and GPU timing available
- Unit tests for timer functionality
- Module documentation complete

**Dependencies:** None
**Estimated Time:** 2 hours
**Autonomous:** true

---

#### 09-02: Integrate ROCm Profiling Tools

```yaml
wave: 1
depends_on: []
autonomous: true
files_modified:
  - src/profiling/rocprof_integration.rs
  - docs/PROFILING_GUIDE.md
```

**Description:** Create integration layer for rocprof, rocperf, and omniperf to enable comprehensive GPU profiling.

**Files:**
- `src/profiling/rocprof_integration.rs` (new)
- `docs/PROFILING_GUIDE.md` (new)

**Actions:**
1. Create rocprof wrapper for HSA trace collection
2. Add rocperf counter collection helpers
3. Create command-line helpers for common profiling scenarios
4. Document profiling workflow in PROFILING_GUIDE.md
5. Add examples for bottleneck detection

**Acceptance Criteria:**
- Helper functions for launching application under rocprof
- Counter collection helpers for memory bandwidth, occupancy
- PROFILING_GUIDE.md with step-by-step profiling instructions
- Examples: kernel profiling, memory analysis, bottleneck detection
- No external process spawning (documentation only)

**Dependencies:** None
**Estimated Time:** 2 hours
**Autonomous:** true

---

#### 09-03: Establish Performance Baselines

```yaml
wave: 1
depends_on: [09-01]
autonomous: true
files_modified:
  - benchmarks/baselines/rdna3-baseline.json
  - .planning/phases/09-performance-optimization/09-03-BASELINE.md
```

**Description:** Run initial benchmarks to establish baseline performance before optimization. This provides the "before" measurement.

**Files:**
- `benchmarks/baselines/rdna3-baseline.json` (new)
- `.planning/phases/09-performance-optimization/09-03-BASELINE.md` (new)

**Actions:**
1. Run existing attention_bench for CPU baseline
2. If GPU available, run GPU benchmarks
3. Document baseline metrics: throughput, latency, memory
4. Store baseline in JSON format for regression detection
5. Identify hardware specifics (GPU model, ROCm version)

**Acceptance Criteria:**
- Baseline JSON file with hardware metadata
- Documented baseline metrics for CPU path
- Documented baseline metrics for GPU path (if available)
- Baseline summary in 09-03-BASELINE.md
- Benchmark methodology documented

**Dependencies:** 09-01
**Estimated Time:** 1 hour
**Autonomous:** true

---

### Wave 2: Benchmark Suite

#### 09-04: Create MatMul Benchmark Suite

```yaml
wave: 2
depends_on: [09-01]
autonomous: true
files_modified:
  - benches/matmul_bench.rs
  - Cargo.toml
```

**Description:** Comprehensive benchmark for dense and quantized matrix multiplication operations. MatMul is the core operation for transformer models.

**Files:**
- `benches/matmul_bench.rs` (new)
- `Cargo.toml` (modify)

**Actions:**
1. Create matmul benchmark harness using existing Benchmark pattern
2. Test dense matmul at sizes: 512x512, 1024x1024, 2048x4096, 4096x4096
3. Test quantized formats: Q4_0, Q4_K, Q6_K, Q8_0
4. Measure: GFLOPS, memory bandwidth, tokens/sec
5. Report: avg, min, max, p50, p95, p99
6. Register in Cargo.toml

**Acceptance Criteria:**
- `benches/matmul_bench.rs` runs with `cargo bench --bench matmul_bench`
- Dense matmul tested at 4+ sizes
- All 4 quantization formats tested
- Output includes: time, GFLOPS, throughput
- Percentiles reported (p50, p95, p99)
- Works with and without rocm feature

**Dependencies:** 09-01
**Estimated Time:** 3 hours
**Autonomous:** true

---

#### 09-05: Create Dequantization Benchmark

```yaml
wave: 2
depends_on: [09-01]
autonomous: true
files_modified:
  - benches/dequant_bench.rs
  - Cargo.toml
```

**Description:** Benchmark dequantization kernels for all 15 GGUF quantization formats to identify format-specific bottlenecks.

**Files:**
- `benches/dequant_bench.rs` (new)
- `Cargo.toml` (modify)

**Actions:**
1. Create dequant benchmark using Benchmark pattern
2. Test all quant formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
3. Measure: dequantization time per element, memory bandwidth utilization
4. Compare CPU vs GPU dequantization
5. Format comparison table in output

**Acceptance Criteria:**
- `benches/dequant_bench.rs` runs successfully
- All 10 quantization formats benchmarked
- CPU vs GPU comparison (when rocm available)
- Output includes: time/element, bandwidth, format comparison
- Identifies slowest format for optimization priority

**Dependencies:** 09-01
**Estimated Time:** 2 hours
**Autonomous:** true

---

#### 09-06: Create End-to-End Inference Benchmark

```yaml
wave: 2
depends_on: [09-01, 09-03]
autonomous: true
files_modified:
  - benches/inference_bench.rs
  - Cargo.toml
```

**Description:** Benchmark complete inference pipeline including tokenization, prompt processing, generation, and sampling.

**Files:**
- `benches/inference_bench.rs` (new)
- `Cargo.toml` (modify)

**Actions:**
1. Create inference benchmark harness
2. Measure: time to first token (TTFT), tokens/sec generation
3. Test scenarios: short prompt (32 tokens), medium (256), long (512)
4. Measure memory usage during inference
5. Report prompt processing time vs generation time
6. Handle model path via environment variable

**Acceptance Criteria:**
- `benches/inference_bench.rs` runs with `cargo bench --bench inference_bench`
- Measures TTFT for 3 prompt lengths
- Reports tokens/sec for generation phase
- Memory usage tracking (peak, KV cache growth)
- Uses ROCFORGE_TEST_MODEL for model path
- Graceful skip when no model available

**Dependencies:** 09-01, 09-03
**Estimated Time:** 3 hours
**Autonomous:** true

---

#### 09-07: Create Memory Benchmark

```yaml
wave: 2
depends_on: [09-01]
autonomous: true
files_modified:
  - benches/memory_bench.rs
  - Cargo.toml
```

**Description:** Benchmark KV cache memory usage, allocation patterns, and identify memory-related bottlenecks.

**Files:**
- `benches/memory_bench.rs` (new)
- `Cargo.toml` (modify)

**Actions:**
1. Create memory benchmark harness
2. Measure KV cache growth per token
3. Test allocation patterns: single allocation vs incremental
4. Profile paged attention memory overhead
5. Report: memory per token, total memory for 1k/2k/4k contexts
6. Identify memory fragmentation if any

**Acceptance Criteria:**
- `benches/memory_bench.rs` runs successfully
- KV cache memory growth measured per token
- Allocation pattern comparison complete
- Memory overhead of paged attention quantified
- Report includes: bytes/token, peak memory, fragmentation analysis

**Dependencies:** 09-01
**Estimated Time:** 2 hours
**Autonomous:** true

---

#### 09-08: Implement Baseline Storage and Regression Detection

```yaml
wave: 2
depends_on: [09-04, 09-05, 09-06]
autonomous: true
files_modified:
  - src/profiling/baseline.rs
  - benchmarks/baselines/rdna3-baseline.json
```

**Description:** Create system for storing benchmark baselines and detecting performance regressions.

**Files:**
- `src/profiling/baseline.rs` (new)
- `benchmarks/baselines/rdna3-baseline.json` (modify)

**Actions:**
1. Create baseline storage format (JSON)
2. Implement baseline loading/saving
3. Add regression detection (10% threshold)
4. Create `--save-baseline` option for benchmarks
5. Create `--compare-baseline` option for regression checking
6. Include hardware metadata in baseline

**Acceptance Criteria:**
- Baseline JSON format with hardware metadata
- Baseline save/load functions working
- Regression detection at 10% threshold
- Command-line options for save/compare
- Baseline includes: hardware info, metrics, timestamp

**Dependencies:** 09-04, 09-05, 09-06
**Estimated Time:** 3 hours
**Autonomous:** true

---

### Wave 3: Throughput Optimization

#### 09-09: Profile Memory Bandwidth Bottlenecks

```yaml
wave: 3
depends_on: [09-01, 09-02, 09-04]
autonomous: false
files_modified:
  - .planning/phases/09-performance-optimization/09-09-PROFILE.md
  - kernels/q4_0_matmul.hip
```

**Description:** Use rocprof to identify memory bandwidth bottlenecks in hot path operations.

**Files:**
- `.planning/phases/09-performance-optimization/09-09-PROFILE.md` (new)
- `kernels/q4_0_matmul.hip` (potential modify)

**Actions:**
1. Profile matmul operations with rocprof memory counters
2. Profile attention operations
3. Profile dequantization kernels
4. Identify: cache miss rates, memory stall cycles, bandwidth utilization
5. Document findings in profile report
6. Identify optimization candidates

**Acceptance Criteria:**
- Profile report with memory metrics for all hot operations
- Top 3 bottlenecks identified
- Quantified: cache hit rate, memory bandwidth, stall %
- Optimization candidates listed with priority

**Dependencies:** 09-01, 09-02, 09-04
**Estimated Time:** 3 hours
**Autonomous:** false (requires GPU and manual analysis)

---

#### 09-10: Optimize MatMul Kernel Tuning

```yaml
wave: 3
depends_on: [09-09]
autonomous: false
files_modified:
  - kernels/q4_0_matmul.hip
  - kernels/flash_attention.hip
  - .planning/phases/09-performance-optimization/09-10-TUNING.md
```

**Description:** Adjust block sizes, tiling, and memory access patterns for RDNA3 Wave32 execution.

**Files:**
- `kernels/q4_0_matmul.hip` (modify)
- `kernels/flash_attention.hip` (modify)
- `.planning/phases/09-performance-optimization/09-10-TUNING.md` (new)

**Actions:**
1. Review existing kernel tuning constants (BLOCK_SIZE, WARP_SIZE)
2. Test alternative block sizes: 128, 256, 384, 512
3. Optimize memory coalescing patterns
4. Tune for RDNA3 Wave32 (ensure wave32 reductions)
5. Benchmark each configuration
6. Document optimal settings

**Acceptance Criteria:**
- Block size tuning tested for 3+ kernels
- Performance measured for each configuration
- Optimal settings documented
- At least 10% improvement in worst-case scenario
- Tuning report with before/after numbers

**Dependencies:** 09-09
**Estimated Time:** 4 hours
**Autonomous:** false (requires GPU testing)

---

#### 09-11: Implement Operator Fusion

```yaml
wave: 3
depends_on: [09-10]
autonomous: false
files_modified:
  - kernels/fused_dequant_rmsnorm.hip
  - kernels/fused_rope_kvappend.hip
  - src/ggml/hip_backend/ops/fused_ops.rs
  - build.rs
  - src/ggml/hip_backend/ops/mod.rs
```

**Description:** Fuse commonly co-occurring operations to reduce memory traffic and kernel launches.

**Files:**
- `kernels/fused_dequant_rmsnorm.hip` (new)
- `kernels/fused_rope_kvappend.hip` (new)
- `src/ggml/hip_backend/ops/fused_ops.rs` (new)
- `build.rs` (modify)
- `src/ggml/hip_backend/ops/mod.rs` (modify)

**Actions:**
1. Analyze operation sequences in execution plan for fusion candidates
2. Implement fused dequantization + RMSNorm kernel
3. Implement fused RoPE + KV cache append
4. Add fusion detection to execution plan optimizer
5. Benchmark fusion benefits
6. Only fuse when beneficial (measure first)

**Acceptance Criteria:**
- At least 2 fused kernels implemented
- Fusion detection in optimizer
- Benchmark shows benefit for fused ops
- No performance regression for unfused path
- Documentation of fusion strategy

**Dependencies:** 09-10
**Estimated Time:** 5 hours
**Autonomous:** false (requires GPU testing)

---

#### 09-12: Optimize Quantized MatMul Throughput

```yaml
wave: 3
depends_on: [09-11]
autonomous: false
files_modified:
  - kernels/q4_0_matmul.hip
  - src/ggml/hip_backend/ops/quantized_matmul.rs
  - .planning/phases/09-performance-optimization/09-12-QUANTOPT.md
```

**Description:** Focus optimization on quantized matmul variants (Q4_0, Q4_K, Q6_K) which are the bottleneck for inference throughput.

**Files:**
- `kernels/q4_0_matmul.hip` (modify)
- `src/ggml/hip_backend/ops/quantized_matmul.rs` (modify)
- `.planning/phases/09-performance-optimization/09-12-QUANTOPT.md` (new)

**Actions:**
1. Profile each quantization format individually
2. Optimize block layout for each format
3. Tune for Q4_K_M (most common format)
4. Optimize shared memory usage
5. Reduce register pressure if needed
6. Benchmark improvement

**Acceptance Criteria:**
- Each quant format profiled
- Q4_K_M optimized (primary target)
- At least 15% throughput improvement for Q4_K_M
- No regression for other formats
- Optimization report with before/after

**Dependencies:** 09-11
**Estimated Time:** 4 hours
**Autonomous:** false (requires GPU testing)

---

### Wave 4: Latency Optimization

#### 09-13: Profile Time to First Token (TTFT)

```yaml
wave: 4
depends_on: [09-06]
autonomous: false
files_modified:
  - .planning/phases/09-performance-optimization/09-13-TTFT.md
  - benches/inference_bench.rs
```

**Description:** Measure and analyze TTFT to identify latency bottlenecks in prompt processing.

**Files:**
- `.planning/phases/09-performance-optimization/09-13-TTFT.md` (new)
- `benches/inference_bench.rs` (modify)

**Actions:**
1. Add detailed TTFT breakdown to inference benchmark
2. Measure: tokenization, model loading, prompt processing, first token generation
3. Identify which phase dominates latency
4. Profile GPU kernels during prompt processing
5. Identify optimization targets

**Acceptance Criteria:**
- TTFT broken down by phase
- Dominant latency source identified
- Prompt processing kernels profiled
- Optimization targets prioritized
- Target: TTFT <200ms for 512 tokens

**Dependencies:** 09-06
**Estimated Time:** 2 hours
**Autonomous:** false (requires GPU and model)

---

#### 09-14: Reduce Kernel Launch Overhead

```yaml
wave: 4
depends_on: [09-13]
autonomous: false
files_modified:
  - src/ggml/hip_backend/mod.rs
  - src/ggml/executor.rs
```

**Description:** Minimize CPU-GPU synchronization and kernel launch overhead for low-latency inference.

**Files:**
- `src/ggml/hip_backend/mod.rs` (modify)
- `src/ggml/executor.rs` (modify)

**Actions:**
1. Profile kernel launch frequency and duration
2. Batch small operations where possible
3. Reduce CPU-GPU synchronization points
4. Use streams for concurrent operations
5. Implement async kernel launching
6. Measure latency improvement

**Acceptance Criteria:**
- Kernel launch profile complete
- Small operations identified
- At least 3 operations batched
- Synchronization points reduced
- Measured latency improvement

**Dependencies:** 09-13
**Estimated Time:** 4 hours
**Autonomous:** false (requires GPU testing)

---

#### 09-15: Optimize Prompt Processing Path

```yaml
wave: 4
depends_on: [09-14]
autonomous: false
files_modified:
  - kernels/flash_attention.hip
  - src/attention/flash_attention.rs
  - .planning/phases/09-performance-optimization/09-15-PROMPT.md
```

**Description:** Specialized optimizations for the prompt processing phase (parallel attention computation).

**Files:**
- `kernels/flash_attention.hip` (modify)
- `src/attention/flash_attention.rs` (modify)
- `.planning/phases/09-performance-optimization/09-15-PROMPT.md` (new)

**Actions:**
1. Profile attention computation during prompt processing
2. Optimize for batch processing (not single-token)
3. Tune flash attention for prompt phase
4. Reduce KV cache write overhead
5. Optimize RoPE computation for batch
6. Benchmark improvement

**Acceptance Criteria:**
- Prompt processing profiled
- Flash attention tuned for batch
- RoPE optimized for batch
- Measured TTFT improvement
- Target: <200ms for 512 token prompts

**Dependencies:** 09-14
**Estimated Time:** 3 hours
**Autonomous:** false (requires GPU testing)

---

### Wave 5: Memory Optimization

#### 09-16: Profile KV Cache Memory Usage

```yaml
wave: 5
depends_on: [09-07]
autonomous: false
files_modified:
  - src/kv_cache/kv_cache.rs
  - .planning/phases/09-performance-optimization/09-16-KVMEM.md
  - benches/memory_bench.rs
```

**Description:** Detailed analysis of KV cache memory patterns and growth during generation.

**Files:**
- `src/kv_cache/kv_cache.rs` (modify)
- `.planning/phases/09-performance-optimization/09-16-KVMEM.md` (new)
- `benches/memory_bench.rs` (modify)

**Actions:**
1. Add memory tracking to KV cache
2. Profile memory per token, per layer
3. Analyze allocation patterns
4. Identify fragmentation sources
5. Profile paged attention overhead
6. Document findings

**Acceptance Criteria:**
- Per-token memory growth measured
- Allocation pattern analyzed
- Fragmentation quantified (if any)
- Paged attention overhead measured
- Memory profile report complete

**Dependencies:** 09-07
**Estimated Time:** 3 hours
**Autonomous:** false (requires GPU and model)

---

#### 09-17: Optimize KV Cache Allocation Patterns

```yaml
wave: 5
depends_on: [09-16]
autonomous: false
files_modified:
  - src/kv_cache/block_allocator.rs
  - src/kv_cache/page_table.rs
  - .planning/phases/09-performance-optimization/09-17-ALLOC.md
```

**Description:** Improve memory allocation patterns for KV cache to reduce overhead and fragmentation.

**Files:**
- `src/kv_cache/block_allocator.rs` (modify)
- `src/kv_cache/page_table.rs` (modify)
- `.planning/phases/09-performance-optimization/09-17-ALLOC.md` (new)

**Actions:**
1. Implement pre-allocation for predictable growth
2. Optimize block size selection
3. Reduce allocation calls during generation
4. Add memory pool if beneficial
5. Benchmark improvement
6. Document strategy

**Acceptance Criteria:**
- Allocation pattern optimized
- Pre-allocation for KV cache
- Reduced allocation calls during generation
- Measured memory improvement
- Documentation of allocation strategy

**Dependencies:** 09-16
**Estimated Time:** 3 hours
**Autonomous:** false (requires GPU testing)

---

#### 09-18: Create Performance Summary Report

```yaml
wave: 5
depends_on: [09-15, 09-17]
autonomous: true
files_modified:
  - .planning/phases/09-performance-optimization/09-18-SUMMARY.md
  - benchmarks/baselines/rdna3-final.json
  - docs/PERFORMANCE.md
```

**Description:** Compile all profiling data, benchmark results, and optimizations into comprehensive performance report.

**Files:**
- `.planning/phases/09-performance-optimization/09-18-SUMMARY.md` (new)
- `benchmarks/baselines/rdna3-final.json` (new)
- `docs/PERFORMANCE.md` (new)

**Actions:**
1. Collect all benchmark results
2. Compare initial vs final performance
3. Document all optimizations applied
4. Create performance summary table
5. Document remaining bottlenecks
6. Create user-facing performance guide

**Acceptance Criteria:**
- Performance summary report complete
- Before/after comparison for all optimizations
- Remaining bottlenecks documented
- User-facing performance guide
- Final baseline stored

**Dependencies:** 09-15, 09-17
**Estimated Time:** 2 hours
**Autonomous:** true

---

## Estimated Total Time

| Wave | Tasks | Time |
|------|-------|------|
| Wave 1: Profiling Infrastructure | 3 | 5 hours |
| Wave 2: Benchmark Suite | 5 | 13 hours |
| Wave 3: Throughput Optimization | 4 | 16 hours |
| Wave 4: Latency Optimization | 3 | 9 hours |
| Wave 5: Memory Optimization | 3 | 8 hours |
| **Total** | **18** | **51 hours** |

---

## Phase Completion Checklist

- [ ] Profiling infrastructure operational
- [ ] All benchmarks created and passing
- [ ] Baseline measurements established
- [ ] Throughput optimizations applied and measured
- [ ] Latency optimizations applied and measured
- [ ] Memory optimizations applied and measured
- [ ] Performance summary report complete
- [ ] Targets met (within reasonable margin for hardware):
  - [ ] >40 tokens/sec (7B Q4_K_M, RDNA3)
  - [ ] <200ms TTFT (512 token prompts)
  - [ ] >80% GPU utilization during generation
- [ ] Documentation updated (PROFILING_GUIDE.md, PERFORMANCE.md)

---

*Plan created: 2026-01-18*
*Phase: 09 - Performance Optimization*
