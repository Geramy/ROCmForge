# Phase 6: GPU Kernel Hang Research (2026-01-11)

## Problem
GPU kernel hangs during top-p sampling test execution. Tests timeout after >120 seconds.

## Research Findings

### 1. GPU Watchdog Timer
- Source: [HIP Error Codes Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/error_codes.html)
- The GPU's watchdog timer is a safety mechanism to prevent a hanging kernel from making the system unresponsive
- Typical timeout: ~1-2 seconds for kernel execution
- Common cause: Long-running loops or excessive computation in a single kernel launch

### 2. Thread-0-Only Execution Pattern
- Source: StackOverflow - "CUDA Kernel executing a statement by a single thread only"
- Our kernel uses `if (tid == 0)` pattern where only thread 0 does all the work
- This is extremely inefficient and may trigger watchdog
- The other 255 threads in the block are idle

### 3. FlashInfer Sampling Approach
- Source: [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- Uses sorting-free O(log v) rejection sampling
- Parallelized across multiple threads
- Not single-threaded like our implementation

### 4. vLLM Top-P/Top-K Sampler
- Source: [vLLM Documentation](https://docs.vllm.ai/en/v0.10.1/api/vllm/v1/sample/ops/topk_topp_sampler.html)
- Implements optional top-k and top-p filtering followed by weighted random sampling
- Uses GPU-optimized parallel algorithms

## Root Cause Analysis

### Current Kernel Issues
1. **Single-threaded execution**: Only thread 0 does work (line 58: `if (tid == 0)`)
2. **Large vocabulary**: Qwen2 has vocab_size=151936 tokens
3. **Sequential loops**: Even with single-pass algorithm, 151K iterations on one thread is too slow

### Estimated Execution Time
- Thread 0 must iterate through ~151,936 tokens
- GPU core speed: ~2-3 GHz
- Each iteration: ~10-20 cycles (memory access + float add + compare)
- Total: ~1.5-3 million cycles = ~0.5-1 ms
- BUT: Memory latency dominates! GPU memory access is ~100-200 cycles
- Actual: ~15-30 million cycles = **5-10 ms minimum**

### Why It Still Hangs
The watchdog timeout might be triggered by:
1. Memory access pattern issues (non-coalesced reads)
2. GPU scheduler starvation (single thread keeping wavefront busy)
3. Driver-level timeout for "stalled" wavefronts

## Potential Solutions

### Option 1: Parallelize Prefix Sum (RECOMMENDED)
Use CUB-style block-wide scan to parallelize the work:
- Divide vocabulary across all threads in block
- Use shared memory for parallel prefix sum
- Thread 0 only performs final sampling

### Option 2: Use CPU Fallback
- Accept that GPU sampling isn't viable for large vocabularies without complex kernels
- Use GPU only for softmax + temperature
- Use CPU for sampling (already implemented)

### Option 3: Smaller Batch Strategy
- Process tokens in chunks
- Multi-pass kernel with intermediate results
- More complex but potentially viable

## References
- HIP Error Codes: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/error_codes.html
- FlashInfer Sampling: https://flashinfer.ai/2025/03/10/sampling.html
- vLLM Sampler: https://docs.vllm.ai/en/v0.10.1/api/vllm/v1/sample/ops/topk_topp_sampler.html

## Next Steps
1. Implement Option 1: Parallelize prefix sum using CUB-style scan
2. If Option 1 too complex, implement Option 2: CPU fallback for sampling
