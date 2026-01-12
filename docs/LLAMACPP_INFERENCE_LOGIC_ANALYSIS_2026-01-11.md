# llama.cpp Inference Logic Analysis (Patterns Only)

**Date:** 2026-01-11
**Purpose:** Extract LOGIC and PATTERNS from llama.cpp for Rust implementation
**Rule:** NO C/C++/Python/CUDA code contamination - understand the logic, reimplement in Rust

---

## Executive Summary

llama.cpp achieves fast inference through these **logic patterns**:

| Pattern | What It Does | Why It's Fast |
|---------|--------------|---------------|
| Memory-mapped GGUF | Loads tensor metadata only, data on-demand | No upfront loading delay |
| Lazy tensor access | Reads tensor bytes when first used | Skip unused weights |
| Ring buffer KV cache | Circular buffer for token history | Fixed memory, no re-allocation |
| Computation graph reuse | Build graph once, execute many times | No per-token overhead |
| SIMD batch processing | Process 8/16/32 values simultaneously | CPU vector units |
| Quantized compute | Do math directly on compressed weights | Less memory bandwidth |
| Async buffer staging | 4 parallel 1MB buffers for uploads | Overlap I/O with compute |
| Token batching | Process prompt in parallel | Vectorization |
| Sampler chain | Composable sampling strategies | Flexible, efficient |

---

## 1. GGUF Loading Logic

### The Pattern (Not Implementation)

```
┌─────────────────────────────────────────────────────────────┐
│  GGUF File Structure (Logical View)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Header Section:                                            │
│  - Magic: "GGUF" (4 bytes)                                  │
│  - Version: uint32                                          │
│  - Tensor count: int64                                      │
│  - KV count: int64                                          │
│                                                             │
│  Metadata Section:                                          │
│  - Key-value pairs (architecture info)                      │
│  - Tensor metadata (name, shape, type, offset)             │
│                                                             │
│  Data Section:                                               │
│  - Raw tensor bytes (aligned)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Loading Strategy:
1. Memory-map the entire file (zero-copy)
2. Parse header + metadata into RAM
3. Keep tensor data on disk (in mmap)
4. Load specific tensors on-demand when needed
5. Cache loaded tensors to avoid repeated disk I/O
```

### Key Insight

**Don't load the whole model upfront.**

- llama.cpp: Parse metadata (fast), load tensors lazily
- Our code: Loading ALL tensors immediately (slow!)

### Why Our Loading Is Slow

```
Current ROCmForge logic:
1. Open GGUF file
2. Read EVERY tensor into RAM
3. Upload EVERY tensor to GPU
4. Then start inference

Problem: Steps 2-3 take forever, and we're loading tensors we might not use.

llama.cpp logic:
1. Open GGUF file (memory-mapped)
2. Read metadata only (tiny, fast)
3. Start inference
4. Load each tensor when first accessed
5. Cache loaded tensors for reuse
```

---

## 2. Main Inference Loop Logic

### The Generation Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  Token Generation Loop (Pure Logic)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Initialize:                                                │
│  - Load prompt tokens                                       │
│  - Create KV cache (empty)                                  │
│  - Position = 0                                             │
│                                                             │
│  Loop until max_tokens or EOS:                              │
│                                                             │
│  1. Build batch:                                            │
│     - If first iteration: all prompt tokens                 │
│     - Else: single last token                               │
│                                                             │
│  2. Run transformer layers:                                 │
│     - For each layer:                                       │
│       a. Compute QKV from hidden state                      │
│       b. Compute attention (using cached K+V for past)      │
│       c. Compute MLP (feed-forward)                          │
│       d. Add residual, normalize                             │
│                                                             │
│  3. Sample next token:                                      │
│     - Get logits from final layer                           │
│     - Apply temperature, top-k, top-p                       │
│     - Sample weighted distribution                          │
│                                                             │
│  4. Append to KV cache:                                     │
│     - Store new K,V for this token                          │
│                                                             │
│  5. Decode and output token                                 │
│                                                             │
│  6. Position += 1                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: Prompt vs Generation

```
Prompt Processing (parallel):
- Input: [token1, token2, ..., tokenN]  (N tokens)
- Process: All N tokens simultaneously
- Output: N hidden states, N K,V pairs
- Why: No causal mask needed within prompt

Generation (sequential):
- Input: [last_token]  (1 token)
- Process: Single token
- Output: 1 hidden state, 1 K,V pair
- Why: Each token depends on previous ones
- Repeat: One token at a time

Our code currently doesn't distinguish these cases!
```

---

## 3. Memory Management Logic

### The Ring Buffer Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  KV Cache Ring Buffer (Logical Concept)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Context Window: 2048 tokens (example)                      │
│                                                             │
│  [0][1][2][3] ... [2045][2046][2047]                        │
│   ↑                                                         │
│   Write position                                           │
│                                                             │
│  When context exceeds window:                               │
│  - Oldest entries are overwritten                           │
│  - Physical buffer is fixed size                            │
│  - No new allocations during inference                      │
│                                                             │
│  Benefit:                                                   │
│  - Predictable memory usage                                 │
│  - No malloc/free during hot path                           │
│  - Cache-friendly access pattern                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Layer Offloading Logic

```
┌─────────────────────────────────────────────────────────────┐
│  Partial GPU Offloading Strategy                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model has N transformer layers                             │
│  GPU has memory for M layers (M < N)                        │
│                                                             │
│  Strategy:                                                  │
│  - Layers 0 to M-1: GPU                                     │
│  - Layers M to N-1: CPU                                     │
│                                                             │
│  Data flow:                                                 │
│  Hidden state → GPU layers → CPU layers → output           │
│        (transfer)                                           │
│                                                             │
│  Why:                                                      │
│  - Early layers have largest activations                   │
│  - Benefit most from GPU compute                            │
│  - Later layers are smaller, OK on CPU                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Our Memory Problem

```
Current issue: Too many small allocations

llama.cpp approach:
- Allocate large buffers once at startup
- Sub-allocate from these buffers during inference
- NO per-tensor allocations during hot path

Our code:
- Allocating per tensor (thousands of hipMalloc calls)
- This triggers ROCm driver issues
```

---

## 4. CPU Performance Logic

### The Vectorization Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  SIMD Processing Logic                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scalar operation (slow):                                  │
│  for i in 0..N:                                             │
│    result[i] = a[i] + b[i]                                 │
│                                                             │
│  Vector operation (fast):                                  │
│  Load 8/16/32 values at once                               │
│  Add 8/16/32 pairs simultaneously                           │
│  Store 8/16/32 results                                     │
│  Repeat                                                    │
│                                                             │
│  Hardware:                                                 │
│  - AVX2: 256-bit = 8 float32                               │
│  - AVX512: 512-bit = 16 float32                            │
│  - NEON: 128-bit = 4 float32 (ARM)                         │
│                                                             │
│  Speedup: 8-16x for simple operations                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Precomputation Pattern

```
Expensive functions → Compute once, lookup table

Example: GELU activation
- Exact: Expensive computation per value
- LUT: Precompute 65536 values, array lookup

Trade-off:
- Memory: 256KB for table
- Speed: ~100x faster

llama.cpp uses LUT for:
- GELU activation
- SILU activation
- RoPE sin/cos (position embeddings)
- Quick GELU
```

### The Thread Pool Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  Parallel Execution Logic                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Problem: Process N independent items                      │
│                                                             │
│  Solution:                                                  │
│  1. Create thread pool (fixed number of threads)           │
│  2. Split work into chunks                                  │
│  3. Assign chunks to threads                                │
│  4. Wait for all threads to complete                       │
│  5. Combine results                                        │
│                                                             │
│  Applications in llama.cpp:                                │
│  - Matrix multiplication (tile-based)                       │
│  - Attention computation (parallel heads)                   │
│  - Layer processing (when batching independent sequences)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. KV Cache Logic

### The Multi-Head Cache Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  KV Cache Structure (Logical)                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each layer:                                            │
│  ├── Key cache: [seq_len, num_heads, head_dim]             │
│  └── Value cache: [seq_len, num_heads, head_dim]           │
│                                                             │
│  For each new token:                                        │
│  1. Compute K, V for this token                             │
│  2. Append to cache (at position pos)                       │
│  3. For attention: use all cached K,V + new K,V             │
│  4. Position += 1                                          │
│                                                             │
│  Optimization:                                              │
│  - Cache persists across generation steps                   │
│  - Only compute K,V once per token                          │
│  - Reuse for all future attention computations             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Quantized KV Cache Logic

```
Problem: KV cache takes lots of memory

Solution: Store in compressed format

Trade-offs:
Format    | Memory | Speed | Quality
----------|--------|-------|--------
FP16      | 50%    | Fast  | Best
Q8_0      | 25%    | Fast  | Good
Q4_0      | 12.5%  | Slower| OK
Q2_K      | 6.25%  | Slower| Acceptable

llama.cpp approach:
- Store KV in Q4_0 or Q8_0
- Dequantize on-the-fly during attention
- Saves memory with minimal quality loss

Our code:
- KV in FP32 (wasteful)
- Could implement quantized KV
```

---

## 6. Sampling Logic

### The Sampler Chain Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  Composable Samplers (Pure Logic)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Logits [vocab_size]                                 │
│                                                             │
│  Pipeline (apply in order):                                 │
│  1. Temperature → scale logits                              │
│  2. Top-K → keep only top K tokens                          │
│  3. Top-P → keep tokens until cumulative prob ≥ P          │
│  4. Min-P → remove tokens below min probability            │
│  5. Repetition penalty → penalize repeated tokens           │
│  6. Sample → weighted random choice                         │
│                                                             │
│  Output: Single token ID                                    │
│                                                             │
│  Benefits:                                                 │
│  - Each sampler is independent                              │
│  - Can be reordered or disabled                             │
│  - Easy to add new sampling strategies                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Sampling Optimization

```
Problem: Sampling from 150K tokens is slow

Solution 1: Pre-filter
- Apply top-k BEFORE softmax
- Softmax only on filtered tokens (much smaller)

Solution 2: Early exit
- If top-1 token has probability > 0.9, just take it
- Skip sampling entirely

Solution 3: Batch processing
- Sample multiple sequences simultaneously
- Share softmax computation

Our code already has good CPU sampling!
The issue is elsewhere (model loading, inference execution)
```

---

## 7. Computation Graph Logic

### The Graph Building Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  Computation Graph (Logical Concept)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Idea: Build once, execute many times                      │
│                                                             │
│  Single token execution graph:                              │
│                                                             │
│  input_tokens → embedding                                  │
│               ↓                                            │
│  [N x transformer layers]                                  │
│               ↓                                            │
│  lm_head → logits → sample → output_token                   │
│                                                             │
│  Graph is built AT INITIALIZATION:                          │
│  - Allocate all needed buffers                              │
│  - Create all tensor operations                            │
│  - Optimize operation order                                │
│                                                             │
│  At inference time:                                        │
│  - Just update input data                                   │
│  - Execute graph (no allocations!)                          │
│                                                             │
│  Benefit:                                                   │
│  - Zero allocations during hot path                         │
│  - Predictable memory usage                                 │
│  - Can optimize graph structure                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. What We're Missing (Logic Gap Analysis)

### Comparison: Our Code vs llama.cpp Logic

| Aspect | llama.cpp Logic | Our Current Logic | Gap |
|--------|-----------------|-------------------|-----|
| **Model Loading** | Lazy load, mmap | Load all upfront | **Huge** |
| **Prompt Processing** | Parallel batch | Single token? | **Missing** |
| **KV Cache** | Ring buffer, quantized | FP32, linear? | **Partial** |
| **Memory** | Pre-allocated buffers | Per-tensor alloc | **Critical** |
| **Graph** | Built once, reused | ? | **Missing** |
| **SIMD** | Everywhere | Minimal | **Large** |
| **Threading** | Thread pools | ? | **Missing** |

### Root Cause of Our Slowness

```
Model loading phase:
✓ We parse GGUF correctly
✗ We load ALL tensors immediately (llama.cpp: lazy)
✗ We upload ALL tensors to GPU (llama.cpp: on-demand)
✗ Thousands of small allocations (llama.cpp: few large)

Inference phase:
? Model loads but inference doesn't complete in 120s
? Possible issues:
  - Inference loop not actually running
  - Per-token allocations blocking
  - Not reusing computation graphs
  - KV cache not working correctly
  - Missing prompt vs generation distinction
```

---

## 9. Implementation Plan for Rust

### Phase 1: Fix Model Loading (Immediate)

**Logic Pattern to Implement:**

```rust
// PSEUDOCODE (not C++!)

struct LazyGgufLoader {
    metadata: GgufMetadata,     // Loaded once
    tensor_registry: HashMap<String, Option<Tensor>>,  // None = not loaded
    file_mmap: Mmap,             // Memory-mapped file
}

impl LazyGgufLoader {
    fn get_tensor(&mut self, name: &str) -> &Tensor {
        if !self.tensor_registry.contains_key(name) {
            return Err(TensorNotFound);
        }

        // Load on first access
        if self.tensor_registry[name].is_none() {
            let tensor = self.load_tensor_from_mmap(name)?;
            self.tensor_registry[name] = Some(tensor);
        }

        self.tensor_registry[name].as_ref().unwrap()
    }
}
```

**Benefits:**
- Instant model load (metadata only)
- Only load used tensors
- Can start inference immediately

### Phase 2: Implement Prompt vs Generation

**Logic Pattern:**

```rust
enum InferenceMode {
    PromptProcessing(Vec<Token>),  // Many tokens, parallel
    SingleToken(Token),            // One token, sequential
}

fn run_inference(mode: InferenceMode) -> Result<Vec<HiddenState>> {
    match mode {
        InferenceMode::PromptProcessing(tokens) => {
            // Process all tokens in parallel
            // No causal mask between tokens
            // Vectorize across tokens
        }
        InferenceMode::SingleToken(token) => {
            // Process single token
            // Use cached KV from all previous tokens
            // Causal mask applies
        }
    }
}
```

### Phase 3: Computation Graph

**Logic Pattern:**

```rust
struct ComputationGraph {
    operations: Vec<Operation>,
    buffers: Vec<Buffer>,  // Pre-allocated
}

impl ComputationGraph {
    fn new() -> Self {
        // Build graph once
        // Allocate all buffers
        // Return executable graph
    }

    fn execute(&mut self, input: &Tensor) -> Result<&Tensor> {
        // Just update input buffer
        // Execute operations
        // Return output
        // NO ALLOCATIONS
    }
}
```

### Phase 4: SIMD in Rust

**Use existing crates:**
- `std::simd` (nightly) or `wide` (stable)
- `core::simd` for portable SIMD
- No unsafe C/C++ needed!

### Phase 5: Thread Pool

**Use existing crates:**
- `rayon` for data parallelism
- `tokio` for async I/O
- `threadpool` for custom thread pools

---

## 10. Key Takeaways

### The Logic Secrets (Not Code)

1. **Lazy is fast** - Don't load what you don't need
2. **Batch is fast** - Process independent items together
3. **Pre-allocated is fast** - Avoid malloc during hot path
4. **SIMD is fast** - Use vector units, not scalar loops
5. **Cache is fast** - Never recompute what you can reuse
6. **Graph is fast** - Build once, execute many times
7. **Quantized is fast** - Less memory = more cache hits

### What NOT to Copy from llama.cpp

- C/C++ code (unsafe, manual memory management)
- Python glue code (we don't need it)
- CUDA kernels (we have ROCm/hip)
- Preprocessor macros (Rust has generics)
- Manual SIMD intrinsics (use portable SIMD crates)

### What TO Learn from llama.cpp

- The LOGIC of lazy loading
- The PATTERN of computation graphs
- The STRATEGY of memory pooling
- The APPROACH to quantized storage
- The ARCHITECTURE of sampler chains

---

## Summary

llama.cpp is fast because of **logic patterns**, not because it's C++.

We can implement all these patterns in pure Rust:
- Lazy loading via `memmap2` crate
- Computation graphs via Rust structs
- SIMD via `std::simd` or `wide`
- Thread pools via `rayon`
- Zero-copy via slices and views

The key is understanding WHAT they do, not HOW they do it in C++.

---

**Next Steps:**
1. Implement lazy GGUF loading (immediate priority)
2. Fix inference execution loop (debug why it's slow)
3. Add prompt vs generation distinction
4. Implement computation graph pattern
5. Add SIMD optimizations for CPU paths
