# ROCmForge Implementation Principles

> "Make it correct → make it measurable → then make it fast."

---

## 1. Start with Contracts, Not Kernels

### What This Means

Before writing a single line of HIP code, establish:

**Tensor Layout Contracts**
```
Freeze these decisions upfront:
- Row-major vs column-major
- Stride patterns (contiguous vs strided dimensions)
- Alignment requirements (4-byte, 16-byte, etc.)
- Padding (if any) between dimensions
```

**Kernel Signature Contracts**
```
Every GPU function must have:
1. Exact Rust FFI signature (no "we'll fix later")
2. Documented input/output layouts
3. Documented invariants (e.g., "scores must be non-negative")
4. Error handling strategy (return codes vs panics)
```

**Correctness Oracle**
```
For every GPU kernel, write:
1. CPU reference function (same algorithm, no shortcuts)
2. Property-based test (proptest for random inputs)
3. Edge case test suite (zeros, infinities, NaNs)
4. Comparison tolerance (1e-5 for f32 is reasonable)
```

### Example Contract Template

```rust
// CONTRACT: scale_kernel
//
// Input:
//   - scores: *mut f32, [batch_size * seq_len * seq_len] in row-major
//   - scale: f32, multiplicative factor
//   - batch_size: u32, number of batches
//   - seq_len: u32, sequence length
//
// Output:
//   - scores is modified in-place: scores[i] *= scale for all i
//
// Invariants:
//   - scores pointer must be valid GPU memory
//   - batch_size * seq_len * seq_len must fit in u32
//   - scale must be finite
//
// Correctness:
//   - CPU reference: |scores[i] - (input[i] * scale)| < 1e-5
```

---

## 2. Fix the Biggest Latency Leaks First

### Identify the Leaks

From `codebase_audit.md`, the decode path does ~4-5 GPU↔CPU round-trips:

```
decode_step() →
  1. QKV projection (GPU: hipBLAS)
  2. RoPE (CPU: to_host → apply → to_device)
  3. KV append (CPU: to_host → write → to_device)
  4. Attention (GPU: copy → no-op kernels → copy to CPU → real work)
  5. MLP (GPU: hipBLAS for GEMM, CPU for SwiGLU)
  6. LayerNorm (CPU)
  7. Sampler (CPU)
```

### Priority Order

| Leak | Impact | First Kernel |
|------|--------|--------------|
| RoPE CPU round-trip | Every decode token | `rope_kv_append_kernel` |
| Attention no-op | Every decode token | softmax, mask, scale (stubs) |
| MLP activation | Every decode token | SwiGLU |
| LayerNorm CPU | Every decode token | RMSNorm |
| Sampler CPU | Every decode token | GPU sampler (last) |

### First Real Win: RoPE + KV-Append Fused Kernel

```cpp
// Single kernel does both:
// 1. Apply rotary embedding to Q, K
// 2. Append K, V to KV cache
//
// This eliminates 2 GPU↔CPU round-trips per decode step
```

**Don't touch fancy FlashAttention until RoPE+KV path is solid.**

---

## 3. Implement the "Stub" GPU Path Before Optimizing

### Current State

`src/attention/kernels.rs` has three no-op stubs:

```rust
pub unsafe fn scale_gpu_kernel(...) -> i32 { 0 }  // No-op!
pub unsafe fn mask_gpu_kernel(...) -> i32 { 0 }  // No-op!
pub unsafe fn softmax_gpu_kernel(...) -> i32 { 0 }  // No-op!
```

### Replace Stubs with Correct GPU Versions

**Order:**
1. `scale_kernel` - simplest (element-wise multiply)
2. `mask_kernel` - simple (conditional replace)
3. `softmax_kernel` - medium (LDS reduction, but well-understood)

**Principles:**
- Use simple, readable HIP first
- Don't optimize shared memory usage yet
- Don't worry about wave64 tuning yet
- **Correct + slow > fast + wrong**

### Example: scale_kernel (First Implementation)

```cpp
// Naive but correct first version
extern "C" __global__ void scale_kernel(
    float* scores,
    const float scale,
    const int batch_size,
    const int seq_len
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * seq_len;

    if (idx < total) {
        scores[idx] *= scale;
    }
}
```

This is **slow** (one thread per element) but **correct**. Optimize later.

---

## 4. Use rocBLAS Before Custom MFMA

### Why rocBLAS First

| rocBLAS | Custom MFMA |
|---------|-------------|
| Tested, debugged, optimized | You debug math + memory + bugs |
| Handles edge cases | You handle edge cases |
| Performance is "good enough" | Performance is "maybe better if you're lucky" |

### Strategy

```
Phase 1: Use rocBLAS for everything
  - QKV projection: hipBLAS sgemm
  - Attention output: hipBLAS sgemm
  - MLP: hipBLAS sgemm

Phase 2: Profile to find bottlenecks
  - Is GEMM actually the bottleneck?
  - Or is it memory transfer?
  - Or is it kernel launch overhead?

Phase 3: Custom MFMA only if needed
  - After attention + KV path is stable
  - After profiling proves GEMM is bottleneck
  - After you understand rocBLAS performance baseline
```

### Don't Debug Math + Memory + Scheduling at Once

```
Too hard: "My MFMA attention kernel crashes"

Easier: "My hipBLAS attention works, now let's profile"
Then: "RoPE kernel is slow, let's optimize"
Then: "KV append is slow, let's optimize"
Then: "Now I understand the full path, can try MFMA"
```

---

## 5. One Kernel = One Responsibility (At First)

### Kernel Catalog

| Kernel | Responsibility | Complexity |
|--------|---------------|------------|
| `scale_kernel` | Element-wise multiply | trivial |
| `mask_kernel` | Conditional replace | trivial |
| `softmax_kernel` | Row-wise reduction | medium |
| `rms_norm_kernel` | Row-wise mean + normalize | medium |
| `swiglu_kernel` | Element-wise activation | medium |
| `rope_kernel` | Rotation in pairs | medium |
| `rope_kv_append_kernel` | RoPE + memory write | high |
| `flash_attention_kernel` | Fused QK→softmax→V | very high |

### Fusion Only After Correctness + Profiling

```
Don't: "Implement fused FlashAttention with RoPE and SwiGLU"

Do:
1. Separate softmax (verify correctness)
2. Separate RoPE (verify correctness)
3. Profile: where's time spent?
4. If softmax is bottleneck, consider fusing with matmul
5. Verify fused version matches unfused
```

### Fusion Checklist

Before fusing, confirm:
- [ ] Individual kernels are correct
- [ ] Individual kernels have tests
- [ ] Profiling identifies fusion opportunity
- [ ] Fused kernel has correctness test
- [ ] Fused kernel is faster (measure it!)

---

## 6. Build System Discipline

### build.rs Requirements

```rust
// build.rs MUST:
// 1. Compile .hip to .hsaco using hipcc
// 2. Fail fast if hipcc not found
// 3. Fail fast if .hsaco not produced
// 4. Set cargo:rustc-env for HSACO paths
// 5. Link against amdhip64

fn main() {
    // Detect ROCm
    let hipcc = find_hipcc().expect("hipcc not found - is ROCm installed?");

    // Compile each kernel
    for kernel in KERNELS {
        let hsaco = compile_kernel(&hipcc, kernel)
            .expect(&format!("Failed to compile {}", kernel));

        // Verify output exists
        assert!(hsaco.exists(), "hipcc produced no output for {}", kernel);

        println!("cargo:rustc-env={}_HSACO={}", kernel.to_uppercase(), hsaco.display());
    }

    // Link
    println!("cargo:rustc-link-lib=amdhip64");
}
```

### Runtime Kernel Loading

```rust
// Fail fast, not silently
impl HipBackend {
    pub fn load_kernel_checked(&self, name: &str) -> HipResult<HipKernel> {
        let hsaco_path = env::var(format!("{}_HSACO", name.to_uppercase()))
            .map_err(|_| HipError::KernelLoadFailed(
                format!("{} HSACO path not set - was build.rs run?", name)
            ))?;

        let path = PathBuf::from(&hsaco_path);
        if !path.exists() {
            return Err(HipError::KernelLoadFailed(
                format!("{} HSACO not found at {}", name, hsaco_path)
            ));
        }

        let module = self.load_module(&hsaco_path)?;
        let kernel = self.get_kernel_function(&module, name)?;

        Ok(kernel)
    }
}
```

---

## 7. Profiling Rules

### Start Small

```
DO:
- Test with batch_size=1, seq_len=32
- Run single decode step
- Measure one token latency

DON'T:
- Test with batch_size=32, seq_len=4096
- Run 1000 token decode loop
- Measure "average throughput" while GPU thermal throttles
```

### Measurement Strategy

```rust
// Measure latency, not throughput
let start = Instant::now();
runtime.decode_step(&input)?;
let latency = start.elapsed();

println!("Token latency: {:?}", latency);

// NOT:
// let total = run_1000_tokens();
// println!("Average: {}", total / 1000);  // Hides outliers
```

### Watch Physical Metrics

```
Before profiling run:
- rocm-smi for clocks (should be at max)
- rocm-smi for temps (should be < 80°C)
- rocm-smi for power (should be stable)

During run:
- Watch for thermal throttling
- Watch for power spikes (memory vs compute)
- Watch for clock drops

If thermal throttling: STOP, fix cooling, re-measure
```

---

## 8. Debugging Mindset

### Treat GPU Kernels Like Kernel-Space Code

```
GPU kernel debugging = kernel debugging, not userspace:
- No printf (mostly)
- Hard to attach debugger
- Driver resets on crashes
- Wrong answers, not panic messages
```

### Debugging Workflow

```
1. Log inputs BEFORE kernel launch
   println!("Launching kernel: shape={:?}, ptr={:?}", shape, ptr);

2. Verify with CPU first
   let cpu_result = cpu_reference(&input);
   // Now try GPU

3. Start with tiny problem
   // batch=1, seq_len=2, head_dim=4
   // Can compute by hand if needed

4. Never "guess-fix"
   // Bad: "maybe if I change this number..."
   // Good: "reproduce crash, add log, confirm fix"

5. Always reproduce before fixing
   // Write test that fails
   // Fix code
   // Confirm test passes
```

### Common Issues

| Symptom | Likely Cause | Debug Step |
|---------|-------------|------------|
| Wrong answers | Out of bounds write | Add bounds check, reduce problem size |
| Random crashes | Race condition | Add __syncthreads(), check shared memory |
| All zeros | Wrong pointer | Log pointer value, check hipMalloc |
| NaN everywhere | Division by zero | Check denominator, add epsilon |
| Driver reset | Kernel too long | Reduce problem size, check for infinite loop |

---

## 9. LLM Usage Strategy

### Use LLMs For

```
✅ Boilerplate HIP code
✅ CUDA → HIP translation (mechanical)
✅ Documentation extraction (summaries)
✅ Test case generation
```

### You Decide

```
❌ Architecture (you own the design)
❌ Invariants (you define correctness)
❌ Merge strategy (you review, not the LLM)
❌ "Is this right?" (you verify with tests)
```

### Ground Truth Enforcement

```
LLM generates → You verify → Compiler checks → Tests pass
    ↑                                                          ↓
    └────────────────── No, you don't skip steps ←────────────┘

Not: LLM generates → Ship it
Not: LLM generates → "Looks right" → Ship it
```

### Example Workflow

```
You: "Write a softmax kernel for row-major [batch, seq, seq] tensor"
LLM: [generates kernel]
You: [Review against contract]
You: [Add bounds check if missing]
You: [Write test]
Compiler: [Compile with warnings]
Test: [Pass or fail]
You: [If pass, commit; if fail, debug]

Not: LLM → "Ship it"
```

---

## 10. Roadmap Order (Don't Reorder)

### Phase 1: Replace Stubs (Week 1)

```
Priority: Fix the no-ops first

□ scale_kernel (element-wise multiply)
□ mask_kernel (conditional replace)
□ softmax_kernel (row-wise reduction)

Verify: CPU vs GPU tests pass for all three
```

### Phase 2: RoPE + KV Append (Week 2)

```
Priority: Biggest latency win

□ rope_kernel (rotation)
□ kv_append_kernel (memory write)
□ rope_kv_append_fused (both together)

Verify: Single decode step stays on GPU
```

### Phase 3: Fused Attention (Week 3-4)

```
Priority: Performance unlock

□ flash_attention_kernel (QK→softmax→V)
□ Profile vs separate kernels
□ Optimize block sizes

Verify: Faster than separate kernels
```

### Phase 4: MLP Ops (Week 5)

```
Priority: Complete GPU path

□ swiglu_kernel (activation)
□ rms_norm_kernel (normalization)

Verify: Full layer stays on GPU
```

### Phase 5: Optional Optimizations (After)

```
□ GPU sampler (top-k/top-p)
□ Custom MFMA GEMM
□ FP16 support
□ Wave64 tuning
```

---

## Rule of Thumb

> **Make it correct → make it measurable → then make it fast.**

| Phase | Question | Answer Means |
|-------|----------|--------------|
| Correct | "Does it match CPU reference?" | Tests pass, known-good baseline |
| Measurable | "How fast is it?" | Profiling data, latency numbers |
| Fast | "Can we optimize?" | Identified bottleneck, targeted fix |

**Never optimize before you measure.**

---

## Summary

```
1. Contracts first, kernels second
2. Fix biggest leaks first (RoPE+KV)
3. Replace stubs before optimizing
4. rocBLAS before MFMA
5. One kernel = one responsibility
6. Build system discipline (fail fast)
7. Profile small, watch physical metrics
8. Debug like kernel-space code
9. LLMs assist, you decide
10. Follow roadmap order
```

**Above all: Correct + slow > fast + wrong.**
