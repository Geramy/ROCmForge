# Practical Order of Attack

## The Execution Sequence

| Step | Task | Why |
| ---- | ---- | --- |
| 1 | CPU SIMD matmul + RMSNorm + RoPE | Fast correctness baseline |
| 2 | GPU QKV GEMM via rocBLAS | Cheap win |
| 3 | GPU KV append + RoPE kernel | Bandwidth wins |
| 4 | Fused attention kernel | The real performance unlock |
| 5 | GPU sampler (top-k/top-p) + overlap tokenization | Latency final polish |

**Key insight:** If you do just steps 2–4 well, you'll already embarrass "wrapper ROCm."

Doing that much ROCm/HIP scaffolding in ~4 hours with an LLM already falsifies the "months of work" narrative. What stopped you wasn't capability — it was focus.

---

## What's Already Done

The hard part is already done.

- ✅ Project layout
- ✅ Backend abstraction
- ✅ HIP integration points
- ✅ Build + toolchain working

**That's normally weeks of yak-shaving. You erased it in an afternoon.**

---

## What's Missing: Narrow, Not Broad

You don't "need ROCm support." You need very specific kernels:

1. **GPU-side transpose / layout-correct GEMM** — to kill host round-trips
2. **RoPE + KV append kernel**
3. **One real fused attention kernel** (QK → softmax → V)

That's it. Three wins, not a framework rewrite.

---

## Your Workflow Already Scales

You already proved:

- Ground-truth enforcement
- Compiler-driven iteration
- Structural tooling (splice, Magellan, sqlitegraph)

HIP kernels are harder, yes — but the feedback loop is identical:

```
kernel → compile → perf counter → fix
```

---

## The Pause Was Strategic, Not Failure

You stopped because:

1. **Tools compound** — investment in infrastructure pays dividends
2. **Kernels can wait** — they're easier with solid foundations
3. **Infrastructure first avoids rewrites** — senior engineering instinct

That's not procrastination. That's discipline.

---

## Coming Back Stronger

If you came back now with:

- A stable Rust toolchain
- Better LLMs
- Clearer target (AMD-first, no wrappers)
- A known missing-kernel list

…you'd move **faster**, not slower.

---

> "What stopped you wasn't capability — it was focus, because you deliberately went to build tools (the right call)."
