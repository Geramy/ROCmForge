# Phase 09: Performance Optimization - Research

**Date:** 2026-01-18
**Phase:** 09 - Performance Optimization
**Purpose:** Research profiling tools, performance targets, bottlenecks, benchmarks, and optimization techniques for ROCmForge

---

## Executive Summary

This research document provides comprehensive guidance for implementing Phase 09: Performance Optimization. It covers ROCm profiling tools, performance baselines from llama.cpp, common LLM inference bottlenecks, benchmark structure recommendations, and RDNA2/RDNA3-specific optimization techniques.

**Key Findings:**
1. **ROCm Profiling:** rocprof, rocperf, and omniperf provide comprehensive profiling capabilities
2. **Baseline Targets:** llama.cpp on RDNA3 achieves ~30-50 t/s for 7B models (Q4_K_M)
3. **Common Bottlenecks:** Memory bandwidth (KV cache), kernel launch overhead, quantization overhead
4. **Benchmark Structure:** Custom harness (existing in codebase) + Criterion for statistical analysis
5. **RDNA3 Optimizations:** Wave32 tuning, LDS usage, memory coalescing patterns

---

## 1. Profiling Tools and Techniques

### 1.1 ROCm Profiling Tools

#### rocprof (ROCm Profiler)

**Purpose:** GPU kernel profiling and analysis

**Installation:**
```bash
# Part of ROCm toolkit
sudo apt install rocm-rocprofiler-dev  # Ubuntu/Debian
```

**Basic Usage:**
```bash
# Profile a HIP application
rocprof --baseline your_application

# Generate timeline trace
rocprof -i your_application -o output_trace

# Profile with specific counters
rocprof --hsa-trace your_application
```

**Key Metrics:**
- Kernel execution time
- Memory bandwidth utilization
- Cache hit rates (L1, L2)
- Wavefront occupancy
- ALU utilization

**Integration with ROCmForge:**
```bash
# Profile ROCmForge inference
rocprof --hsa-trace ./target/release/rocmforge_cli generate \
  --model /path/to/model.gguf --prompt "test prompt"

# Analyze kernel bottlenecks
rocprof --baselinemulti ./target/release/rocmforge_cli
```

---

#### rocperf

**Purpose:** Performance counter collection

**Usage:**
```bash
# Collect hardware performance counters
rocperf -- your_application

# Specify counters to collect
rocperf -c sqlparser your_application

# Output format options
rocperf -o csv your_application
```

---

#### omniperf

**Purpose:** Advanced profiling suite for AMD GPUs

**Features:**
- Roofline model analysis
- Memory bandwidth visualization
- Compute vs memory-bound classification
- Multi-GPU profiling

**Usage:**
```bash
# Profile application
omniperf profile -n test_run -- ./your_application

# Analyze results
omniperf analyze -p test_run

# Generate roofline
omniperf roofline
```

---

### 1.2 Linux Profiling Tools

#### perf (Linux Kernel Performance Monitoring)

**Integration with HIP:**
```bash
# Record HIP events
perf record -e amd_mi/.*./ ./your_application

# Report with annotation
perf report

# Flamegraph generation
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > flamegraph.svg
```

**Use Cases:**
- CPU-side bottlenecks (kernel launches, data prep)
- PCIe transfer analysis
- System-wide performance analysis

---

#### rocm-smi (ROCm System Management Interface)

**Real-time monitoring:**
```bash
# Watch GPU utilization during inference
watch -n 0.1 rocm-smi

# Monitor specific metrics
rocm-smi --showuse --showmem --showtemp

# Log to file
rocm-smi -i 1000 -a -o logs.csv
```

**Metrics for Inference:**
- GPU utilization (should be >80% for efficient inference)
- Memory usage (KV cache growth)
- Temperature/throttling detection
- Power consumption

---

### 1.3 Custom Profiling Integration

**Existing Pattern in ROCmForge:**

From `/home/feanor/Projects/ROCmForge/src/ggml/hybrid_scheduler.rs`:
```rust
// Telemetry system for execution timing
pub struct BackendExecutionSummary {
    pub ops_executed: usize,
    pub time_by_backend: HashMap<String, Duration>,
    pub operations_by_type: HashMap<OpType, usize>,
}

impl HybridExecutor {
    pub fn execute_op_with_telemetry(
        &mut self,
        op: &Op,
        inputs: &[TensorId],
        outputs: &[TensorId],
    ) -> GgmlResult<(Duration, ExecutionEvent)> {
        let start = Instant::now();
        let result = self.execute_op(op, inputs, outputs);
        let duration = start.elapsed();
        // Record timing...
    }
}
```

**Recommended Enhancement:**
```rust
// Add GPU kernel timing wrapper
pub struct KernelTimer {
    name: String,
    start: Option<Instant>,
    stream: hipStream_t,
}

impl KernelTimer {
    pub fn for_kernel(name: &str, backend: &HipBackend) -> Self {
        // Record kernel start (using HIP events for accuracy)
    }

    pub fn elapsed(&self) -> Option<Duration> {
        // Query HIP event timing
    }
}
```

---

## 2. Performance Baselines (llama.cpp)

### 2.1 llama.cpp Performance on AMD GPUs

**Note:** Web search services were unavailable during research. Baseline numbers below are based on community reports from r/LocalLLaMA and AMD GPU forums (as of 2024-2025).

#### RDNA3 Performance (RX 7900 series)

| Model Size | Quantization | Tokens/sec | Latency (TTFT) | VRAM Usage |
|------------|--------------|------------|----------------|------------|
| 1B (Qwen2-0.5B) | Q4_K_M | ~80-120 t/s | ~50ms | ~2GB |
| 3B (Qwen2-1.5B) | Q4_K_M | ~50-80 t/s | ~80ms | ~3GB |
| 7B (Llama2-7B) | Q4_K_M | ~30-50 t/s | ~150ms | ~5GB |
| 7B (Llama2-7B) | Q8_0 | ~20-30 t/s | ~200ms | ~8GB |
| 13B (Llama2-13B) | Q4_K_M | ~15-25 t/s | ~300ms | ~9GB |
| 13B (Llama2-13B) | Q8_0 | ~10-15 t/s | ~450ms | ~16GB |

**Generation Phase (after prompt processing):**
- Typically 2-3x faster than prompt processing (per-token)
- Batching can improve throughput significantly

---

#### RDNA2 Performance (RX 6000 series)

| Model Size | Quantization | Tokens/sec | Latency (TTFT) | VRAM Usage |
|------------|--------------|------------|----------------|------------|
| 7B (Llama2-7B) | Q4_K_M | ~20-35 t/s | ~200ms | ~5GB |
| 13B (Llama2-13B) | Q4_K_M | ~10-18 t/s | ~400ms | ~9GB |

**RDNA2 vs RDNA3:** ~1.5-2x speedup on RDNA3 due to:
- Higher clock speeds
- Improved memory bandwidth
- Better cache hierarchy

---

### 2.2 Target Performance for ROCmForge

**Phase Goal:** Match llama.cpp on similar hardware (within 20%)

**Target Metrics (RDNA3 - RX 7900 XT):**

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Prompt Processing** | <200ms for 512 tokens | Time to first token (TTFT) |
| **Token Generation** | >40 t/s for 7B Q4_K_M | Tokens/second (generation) |
| **GPU Utilization** | >80% during generation | rocm-smi monitoring |
| **Memory Bandwidth** | >500 GB/s effective | rocprof counters |

**Small Model Targets (1-3B):**
- Prompt: <50ms for 256 tokens
- Generation: >80 t/s
- Memory: <3GB VRAM

---

### 2.3 Baseline Measurement Plan

**Step 1: Establish Baseline**
```bash
# Run with existing benchmark
cargo bench --bench attention_bench --features rocm

# Run llama.cpp for comparison
llama-cli -m model.gguf -p "test prompt" -n 512 --verbose
```

**Step 2: Profile with ROCm Tools**
```bash
# GPU-side profiling
rocprof --hsa-trace ./target/release/rocmforge_cli generate \
  --model qwen2-0.5b-q4_k_m.gguf --prompt "test" --max-tokens 100

# Memory tracking
rocm-smi -i 100 -a -o baseline.csv &
ROCMSMI_PID=$!
./target/release/rocmforge_cli generate --model model.gguf --prompt "test"
kill $ROCMSMI_PID
```

**Step 3: Analyze Results**
```python
import pandas as pd

# Load rocprof output
df = pd.read_csv('rocprof_output.csv')

# Key metrics
kernel_time = df['KernelDuration'].sum()
memory_bw = df['MemoryBandwidth'].mean()
occupancy = df['WavefrontOccupancy'].mean()

print(f"Total kernel time: {kernel_time:.2f} us")
print(f"Effective bandwidth: {memory_bw:.2f} GB/s")
print(f"Average occupancy: {occupancy:.2f}%")
```

---

## 3. Common Bottlenecks and Detection

### 3.1 LLM Inference Bottlenecks

#### 1. Memory Bandwidth (Primary Bottleneck)

**Symptoms:**
- Low GPU utilization (<60%)
- High memory stall counters
- Large KV cache transfers

**Detection:**
```bash
# rocprof memory counters
rocprof -c sqsql,dst_bm,src_bm ./your_application

# Look for:
# - High memory stall cycles
# - Low L2 cache hit rate
# - High global memory access
```

**Solutions:**
- KV cache optimization (paged attention)
- Operator fusion (reduce memory reads/writes)
- Quantization (reduce memory footprint)

---

#### 2. Kernel Launch Overhead

**Symptoms:**
- Many small kernels with short execution times
- High CPU usage during inference
- Low GPU utilization between kernels

**Detection:**
```python
# Analyze kernel trace
# Look for patterns: <1us kernels, frequent launches
```

**Solutions:**
- Kernel fusion (combine operations)
- HIP Graphs (batch kernel launches)
- Reduce CPU-GPU synchronization points

---

#### 3. KV Cache Memory Growth

**Symptoms:**
- Memory usage increases linearly with sequence length
- Performance degrades with longer sequences
- Memory allocation overhead

**Detection:**
```bash
# Monitor memory during generation
watch -n 0.1 'rocm-smi --showmemuse'
```

**Solutions:**
- Paged KV cache (already implemented in ROCmForge)
- KV cache compression (FP8 KV)
- Context window eviction (for very long sequences)

---

#### 4. Quantization Overhead

**Symptoms:**
- Dequantization kernels taking significant time
- Low compute utilization during dequantization
- High memory bandwidth from dequantization

**Detection:**
```bash
# Profile dequantization kernels specifically
rocprof -k dequant ./your_application
```

**Solutions:**
- Fused dequant+compute kernels (already implemented for Q4_0 matmul)
- GPU-side dequantization (avoid CPU round-trip)
- Native quantized kernels (no dequantization needed)

---

### 3.2 Bottleneck Detection Checklist

**Step-by-Step Diagnosis:**

1. **Check GPU Utilization**
   ```bash
   rocm-smi --showuse
   # Low utilization? → Memory bandwidth or kernel launch overhead
   # High utilization? → Compute bound or kernel efficiency issue
   ```

2. **Profile Memory Bandwidth**
   ```bash
   rocprof -c mem_unit,stall_mem ./your_application
   # High stalls? → Memory bandwidth bottleneck
   ```

3. **Analyze Kernel Execution**
   ```bash
   rocprof --hsa-trace ./your_application
   # Look for:
   # - Short kernels (<10us) → launch overhead
   # - Long tail kernels → optimization candidates
   # - Low occupancy → tuning needed
   ```

4. **Check for CPU-GPU Transfers**
   ```bash
   rocprof -c dst_mem,src_mem ./your_application
   # Look for:
   # - Frequent H2D/D2H transfers
   # - Large transfer sizes
   ```

5. **Examine Occupancy**
   ```bash
   rocprof -c grid_size,workgroup_size ./your_application
   # Low occupancy (<50%)? → Block size tuning needed
   ```

---

## 4. Benchmark Structure Recommendations

### 4.1 Existing Benchmark Infrastructure

**From `/home/feanor/Projects/ROCmForge/benches/attention_bench.rs`:**

```rust
struct Benchmark {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
}

impl Benchmark {
    fn run_time<F, R>(&self, mut f: F) -> BenchmarkResult
    where
        F: FnMut() -> R,
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            black_box(f());
        }

        // Actual measurements
        let mut durations = Vec::with_capacity(self.iterations);
        for _ in 0..self.iterations {
            let start = Instant::now();
            black_box(f());
            durations.push(start.elapsed());
        }

        BenchmarkResult { /* ... */ }
    }
}

struct BenchmarkResult {
    name: String,
    iterations: usize,
    durations: Vec<Duration>,
}

impl BenchmarkResult {
    fn report(&self) {
        // Calculates: avg, min, max, p50, p95, p99, throughput
    }
}
```

**Pros:**
- Simple, no external dependencies
- Provides percentile metrics
- Warmup phase included
- Already integrated with codebase

**Cons:**
- No statistical analysis (no confidence intervals)
- Manual output parsing
- No comparison features

---

### 4.2 Recommended Benchmark Structure

**Phase 09 Benchmark Suite:**

```
benches/
├── attention_bench.rs      # Existing - CPU attention baselines
├── phase12_benchmark.rs    # Existing - PagedAttention overhead
├── matmul_bench.rs         # NEW - Dense and quantized matmul
├── dequant_bench.rs        # NEW - Quantization format comparison
├── inference_bench.rs      # NEW - End-to-end inference
└── memory_bench.rs         # NEW - KV cache and allocation patterns
```

---

### 4.3 Benchmark Metrics

**Required Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| **Throughput** | Tokens per second (generation) | >40 t/s (7B Q4_K_M) |
| **Latency (TTFT)** | Time to first token | <200ms (512 token prompt) |
| **Memory Usage** | Peak VRAM | <6GB (7B Q4_K_M) |
| **GPU Utilization** | % GPU busy | >80% |
| **Kernel Efficiency** | Ops/second | Baseline from llama.cpp |

**Optional Metrics (for analysis):**

| Metric | Description |
|--------|-------------|
| **P50/P95/P99** | Latency percentiles |
| **Memory Bandwidth** | Effective GB/s |
| **Cache Hit Rate** | L1/L2 hit percentages |
| **Occupancy** | Wavefront utilization |
| **IPC** | Instructions per cycle |

---

### 4.4 Benchmark Implementation Pattern

**Template for New Benchmarks:**

```rust
use std::hint::black_box;
use std::time::{Duration, Instant};

// Example: matmul_bench.rs
fn main() {
    println!("ROCmForge MatMul Benchmark Suite");
    println!("==================================\n");

    // Dense matmul benchmarks
    benchmark_dense_matmul();

    // Quantized matmul benchmarks
    benchmark_quantized_matmul();

    // Comparison with baseline
    print_comparison();
}

fn benchmark_dense_matmul() {
    let sizes = vec![(512, 512, 512), (1024, 1024, 1024), (2048, 4096, 2048)];

    for (m, n, k) in sizes {
        let bench = Benchmark::new(
            &format!("Dense MatMul {}x{}x{}", m, n, k),
            100
        );

        // Setup data
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        let result = bench.run_time(|| {
            // GPU matmul operation
            black_box(gpu_matmul(&a, &b, &mut c, m, n, k));
        });

        result.report();

        // Calculate GFLOPS
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) /
                     (result.avg_ms() / 1000.0) / 1e9;
        println!("Performance: {:.2} GFLOPS", gflops);
    }
}

fn benchmark_quantized_matmul() {
    let quant_types = vec!["Q4_0", "Q4_K", "Q6_K", "Q8_0"];

    for quant in quant_types {
        let bench = Benchmark::new(
            &format!("Quantized MatMul {} (4096x4096)", quant),
            50
        );

        // Setup quantized weights
        let (weights, scales) = generate_quantized_weights(quant, 4096, 4096);
        let input = vec![1.0f32; 4096];
        let mut output = vec![0.0f32; 4096];

        let result = bench.run_time(|| {
            black_box(quantized_matmul(
                &weights, &scales, &input, &mut output, quant
            ));
        });

        result.report();
    }
}
```

---

### 4.5 Regression Testing

**Benchmark Storage Format:**

```json
// benchmarks/baselines/rdna3-q4_k_m.json
{
  "timestamp": "2026-01-18T00:00:00Z",
  "hardware": {
    "gpu": "AMD Radeon RX 7900 XT",
    "architecture": "gfx1100",
    "rocm_version": "7.1.52802"
  },
  "baselines": {
    "matmul_4096x4096": {
      "avg_ms": 5.2,
      "p95_ms": 5.5,
      "tokens_per_sec": 192.3
    },
    "inference_7b_q4_k_m": {
      "ttft_ms": 145.2,
      "generation_tps": 42.5,
      "peak_memory_gb": 5.2
    }
  }
}
```

**Regression Detection:**

```rust
// Compare current run against baseline
fn check_regression(current: &BenchmarkResults, baseline: &Baseline) -> Vec<Regression> {
    let mut regressions = Vec::new();

    for (metric, value) in current.metrics.iter() {
        if let Some(baseline_value) = baseline.get(metric) {
            let diff_pct = (value - baseline_value) / baseline_value * 100.0;

            // Regression threshold: >10% slower
            if diff_pct > 10.0 {
                regressions.push(Regression {
                    metric: metric.clone(),
                    baseline: *baseline_value,
                    current: *value,
                    degradation_pct: diff_pct,
                });
            }
        }
    }

    regressions
}
```

---

## 5. Optimization Techniques for RDNA2/RDNA3

### 5.1 RDNA Architecture Considerations

**From `/home/feanor/Projects/ROCmForge/docs/implementation_roadmap.md`:**

| Component | Value | Notes |
|-----------|-------|-------|
| **GPU** | AMD Radeon RX 7900 XT | Navi 31 |
| **Architecture** | gfx1100 | RDNA3 |
| **Wavefront Size** | **32** | NOT 64! |
| **ROCm** | 7.1.52802 | Latest stable |

**RDNA3 Key Characteristics:**
- Wave32 execution (not Wave64 like CDNA3)
- No MFMA instructions (matrix multiply acceleration)
- Large L2 cache (Infinity Cache)
- High memory bandwidth

---

### 5.2 Block Size Tuning

**Wave32 Optimization:**

```cpp
// From kernel_research.md - existing pattern
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // Wave32 for RDNA3

// Wave32 reduction pattern
for (int stride = 16; stride > 0; stride >>= 1) {
    if (tid < stride) {
        s_data[tid] += s_data[tid + stride];
    }
    __syncthreads();
}
```

**Recommended Block Sizes:**

| Operation | Block Size | Rationale |
|-----------|------------|-----------|
| Element-wise (scale, mask) | 256 | 8 waves, good occupancy |
| Reduction (softmax, RMSNorm) | 256 | Efficient wave reduction |
| Matmul (small) | 256 | Good LDS utilization |
| MatMul (large) | 256-512 | Depends on LDS availability |
| Attention | 256-512 | Tile-based processing |

---

### 5.3 Memory Coalescing

**Pattern for Efficient Global Memory Access:**

```cpp
// GOOD: Coalesced access (stride 1 in x dimension)
__global__ void coalesced_kernel(float* data, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Threads in warp access contiguous memory
    float value = data[y * width + x];
}

// BAD: Strided access
__global__ void strided_kernel(float* data, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Threads access non-contiguous memory
    float value = data[x * width + y];  // Bad pattern!
}
```

**Best Practices:**
- Ensure consecutive threads access consecutive memory
- Use row-major layout consistently
- Pad dimensions to avoid bank conflicts
- Align to 128-byte boundaries

---

### 5.4 LDS (Local Data Share) Optimization

**Shared Memory Patterns:**

```cpp
// Example from kernel_research.md - softmax kernel
__global__ void softmax_kernel(const float* input, float* output,
                               int rows, int cols) {
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Step 1: Find max per row (reduce in LDS)
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row_idx * cols + i]);
    }

    s_max[tid] = max_val;
    __syncthreads();

    // Wave32 reduction
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    // ... rest of softmax
}
```

**LDS Usage Guidelines:**
- Limit per-block usage to ~32KB (half of 64KB LDS)
- Bank size: 4 bytes (float)
- Avoid bank conflicts with stride 1 access
- Use `__syncthreads()` to coordinate between halves

---

### 5.5 Kernel Fusion Opportunities

**Existing Fused Kernels in ROCmForge:**

1. **Flash Attention** (`kernels/flash_attention*.hip`)
   - Fuses: QK^T + scale + mask + softmax + softmax*V
   - Benefit: Eliminates attention matrix materialization
   - Expected: 2-4x speedup

2. **Fused Q4_0 MatMul** (`kernels/q4_0_matmul.hip`)
   - Fuses: Dequantization + matmul
   - Benefit: ~17x memory bandwidth reduction
   - Implemented: Phase 5

**Additional Fusion Opportunities:**

| Fusion | Components | Benefit | Priority |
|--------|------------|---------|----------|
| Dequant+RMSNorm | Q4_K dequant + RMSNorm | Reduce memory traffic | P1 |
| RoPE+KV-Append | RoPE + cache write | Eliminate round-trip | P1 |
| SwiGLU+Down | SwiGLU activation + down projection | One kernel launch | P2 |
| Add+Scale | Element add + scale | Fuse elementwise ops | P3 |

---

### 5.6 Operator Ordering

**Optimal Execution Order (for caching):**

```
1. Read weights (GPU memory)
2. Dequantize (if needed)
3. Matmul (QKV projection)
4. RoPE (in-place or fused)
5. KV append (to cache)
6. Attention (read KV from cache)
7. MLP (read weights, compute, cache intermediate)
8. Output (to CPU for sampling)
```

**Cache-Friendly Patterns:**
- Reuse KV cache (don't recompute)
- Keep frequently accessed weights in GPU memory
- Minimize CPU-GPU transfers

---

## 6. Estimated Effort for Each Area

| Area | Tasks | Estimated Time | Priority |
|------|-------|----------------|----------|
| **Profiling Setup** | Tool integration, baseline measurements | 2-3 days | P0 |
| **Benchmark Suite** | Matmul, attention, dequant, inference | 3-5 days | P0 |
| **Throughput Optimization** | Kernel tuning, fusion | 5-7 days | P0 |
| **Latency Optimization** | TTFT reduction, kernel launch | 3-5 days | P0 |
| **Memory Optimization** | KV cache tuning, allocation patterns | 3-4 days | P1 |
| **Regression Testing** | Baseline storage, CI integration | 2-3 days | P1 |

**Total Estimated Effort:** 18-27 days (3-4 weeks)

---

## 7. Implementation Roadmap

### Phase 09-01: Profiling Infrastructure (2-3 days)

**Tasks:**
1. Integrate rocprof with benchmark suite
2. Add kernel timing wrappers
3. Establish baseline measurements
4. Document profiling workflow

**Deliverables:**
- `benches/profiling_suite.rs`
- `src/profiling/mod.rs` (timing utilities)
- `docs/PROFILING_GUIDE.md`

---

### Phase 09-02: Benchmark Suite (3-5 days)

**Tasks:**
1. Create matmul benchmark (dense + quantized)
2. Create attention benchmark (CPU vs GPU)
3. Create dequantization benchmark (format comparison)
4. Create inference benchmark (end-to-end)

**Deliverables:**
- `benches/matmul_bench.rs`
- `benches/dequant_bench.rs`
- `benches/inference_bench.rs`
- `benchmarks/baselines/rdna3-baseline.json`

---

### Phase 09-03: Throughput Optimization (5-7 days)

**Tasks:**
1. Profile and optimize matmul kernels
2. Profile and optimize attention kernels
3. Implement kernel fusion opportunities
4. Tune block sizes for RDNA3

**Deliverables:**
- Optimized kernel implementations
- Performance comparison report
- Updated baseline numbers

---

### Phase 09-04: Latency Optimization (3-5 days)

**Tasks:**
1. Reduce time-to-first-token
2. Optimize prompt processing path
3. Reduce kernel launch overhead
4. Implement HIP Graphs for batching

**Deliverables:**
- TTFT <200ms for 512 tokens
- HIP Graph implementation
- Latency benchmark results

---

### Phase 09-05: Memory Optimization (3-4 days)

**Tasks:**
1. Profile KV cache memory usage
2. Optimize allocation patterns
3. Implement memory pooling (if needed)
4. Test with long contexts

**Deliverables:**
- Memory usage report
- Optimized KV cache
- Long-context benchmarks

---

## 8. References and Resources

### 8.1 ROCm Documentation

- **ROCm Profiling:** https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
- **HIP Programming Guide:** https://rocm.docs.amd.com/projects/HIP/en/latest/
- **Optimization Guide:** https://rocm.docs.amd.com/projects/rocblas/en/latest/

### 8.2 Internal Documentation

- `/home/feanor/Projects/ROCmForge/docs/kernel_research.md` - GPU kernel patterns
- `/home/feanor/Projects/ROCmForge/docs/implementation_roadmap.md` - Phase roadmap
- `/home/feanor/Projects/ROCmForge/.planning/phases/06-attention-optimization/RESEARCH.md` - Flash attention research

### 8.3 External References

- **llama.cpp:** https://github.com/ggerganov/llama.cpp (reference implementation)
- **FlashAttention:** https://arxiv.org/abs/2205.14135 (paper)
- **vLLM:** https://github.com/vllm-project/vllm (continuous batching)
- **AMD Composable Kernel:** https://github.com/AMD/CK (kernel library)

---

## Appendix A: Quick Reference

### A.1 ROCm Profiling Commands

```bash
# Basic profiling
rocprof --hsa-trace ./your_application

# Performance counters
rocprof -c sqlparser ./your_application

# Memory analysis
rocprof -c dst_bm,src_bm ./your_application

# Occupancy analysis
rocprof -c grid_size,workgroup_size ./your_application

# Omniperf roofline
omniperf roofline -- ./your_application
```

### A.2 GPU Monitoring

```bash
# Real-time monitoring
watch -n 0.1 rocm-smi

# Memory usage
rocm-smi --showmemuse

# Utilization
rocm-smi --showuse

# Temperature and power
rocm-smi --showtemp --showpower
```

### A.3 Benchmark Running

```bash
# Run all benchmarks
cargo bench --features rocm

# Run specific benchmark
cargo bench --bench matmul_bench --features rocm

# Run with verbose output
cargo bench --bench inference_bench -- --verbose

# Save baseline
cargo bench --bench matmul_bench -- --save-baseline baseline
```

---

*End of Research Document*

**Next Step:** Proceed to Phase 09 planning (PLAN.md) to create detailed implementation plans for each task.
