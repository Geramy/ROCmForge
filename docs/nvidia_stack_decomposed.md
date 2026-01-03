# NVIDIA's Inference Stack Decomposed

> "Forget branding. NVIDIA's stack for inference boils down to six concrete pieces."

## A. Linear Algebra (The Engine)

| NVIDIA | AMD |
| ------ | --- |
| CUTLASS / cuBLAS | **rocBLAS** |
| Tensor Cores | **MFMA** |

**Functionally:** Fast GEMM

---

## B. Attention Kernels (The Crown Jewel)

- FlashAttention (fused QKV + softmax)
- Persistent kernels
- Warp-aware scheduling

**Functionally:** Fused attention

---

## C. Memory Model

- KV cache layout
- Paged attention
- Block-sparse access

**Functionally:** Don't thrash memory

---

## D. Execution Model

- Stream scheduling
- Kernel fusion
- Overlapping compute & memory

**Functionally:** Hide latency

---

## E. Sampler / Decode

- Top-k / top-p
- Speculative decoding
- Multi-token kernels

**Functionally:** Turn logits into tokens fast

---

## F. Glue Layer

- Model loader
- Tokenizer
- API

**Functionally:** Plumbing

---

## AMD's Existing Equivalents

| NVIDIA       | AMD                        |
| ------------ | -------------------------- |
| cuBLAS       | **rocBLAS**                |
| CUTLASS      | **Composable Kernel (CK)** |
| Tensor Cores | **MFMA**                   |
| CUDA         | **HIP**                    |
| Streams      | **HIP streams**            |
| cuFFT        | **rocFFT**                 |

**Key insight:** The parts exist. The assembly doesn't.

---

## What's Missing

Nobody wired AMD's components into:
- A transformer-shaped execution flow
- A decode-optimized loop
- A KV cache designed for wave64

Not because it's impossible — because it's unowned.

---

## Divide-and-Conquer Roadmap

### Phase 1 — Prove the Core
- Use rocBLAS for QKV GEMM
- Custom HIP kernel: rotary embedding, KV write
- CPU sampling

> You already beat Vulkan.

### Phase 2 — Own Attention
- Implement fused attention kernel: wave64-aligned, blockwise causal mask
- Keep KV in LDS when possible

> This is where AMD wakes up.

### Phase 3 — Fix Memory (The Silent Killer)
- Design KV cache layout for AMD
- Avoid CUDA-style paging assumptions
- Optimize for batch-1, long context

> This removes most llama.cpp pain.

### Phase 4 — Decode & Sampler
- GPU top-k / top-p
- Multi-token decode
- Speculative decoding later

> Latency drops hard.

### Phase 5 — Wire it Cleanly
- Expose a single inference interface
- Plug into Shimmy
- Everything else stays boring

---

## Why LLMs Make This Feasible Now

LLMs can:
- Generate HIP boilerplate
- Translate CUDA → HIP accurately
- Expand AMD examples into real kernels
- Document invariants as you go

**You:**
- Decide architecture
- Validate correctness
- Read perf counters
- Say "no" when it guesses

---

## One-Line Summary

> NVIDIA didn't build magic — they built components and glued them.
> AMD shipped components. Nobody glued them.
>
> Divide, conquer, wire — and suddenly the "months" argument collapses.
