# Phase 6 Plan 1 Summary: Flash Attention Research

**Completed:** 2026-01-18
**Duration:** ~30 minutes
**Type:** Research / Documentation

---

## Accomplishments

### 1. Comprehensive Kernel Documentation

Documented all three existing flash attention kernels:

| Kernel | File | Lines | Purpose |
|--------|------|-------|---------|
| `flash_attention_kernel` | `kernels/flash_attention.hip` | 252 | Generic fused attention with optional mask |
| `flash_attention_causal_kernel` | `kernels/flash_attention_causal.hip` | 172 | Fused causal attention |
| `flash_attention_nocausal_kernel` | `kernels/flash_attention_nocausal.hip` | 156 | Fused non-causal attention |

**Key Findings:**
- All kernels already compiled via `build.rs`
- Rust wrapper functions exist in `src/attention/kernels.rs`
- Kernels loaded via global `Mutex<Option<KernelCache>>` pattern
- Not currently used in `GpuAttentionBackend::forward()` path

### 2. Backend Registry Analysis

**BackendImplementation Trait:**
- Operates on host slices (`&[f32]`), not device tensors
- Returns owned `Vec<f32>` (output allocated on host)
- Configuration via `AttentionConfig` struct
- `supports()` method for capability checking

**Existing Backends:**
- `CpuAttentionBackend`: Always supports everything (fallback)
- `GpuAttentionBackend`: Has `use_flash_attention` flag (currently always false)

**Selection Logic:**
- Registry auto-selects first backend that supports config
- Can set default backend via `set_default()`
- Returns error if no suitable backend found

### 3. Flash Attention Algorithm Documentation

**What is Flash Attention:**
- IO-aware exact attention algorithm
- Fuses operations: QK^T + scale + mask + softmax + weighted sum
- Reduces memory bandwidth (no attention matrix materialization)
- Uses tiling to fit in SRAM/registers

**Speedup Mechanisms:**
- Memory bandwidth: Eliminates O(seq^2) scores matrix
- Kernel fusion: 1 kernel vs 5+ separate operations
- No CPU-GPU sync: Computation stays on GPU
- Better cache utilization: Intermediates in registers/shared memory

**Limitations:**
- Head dimension: <= 128 (register limit)
- Sequence length: <= 2048 (shared memory)
- Causal masking: Supported via dedicated kernel
- Custom masks: Requires generic flash_attention kernel

### 4. Integration Strategy Design

**Evaluated 3 Options:**

| Option | Description | Complexity | Recommendation |
|--------|-------------|------------|----------------|
| 1 | Separate FlashAttention backend | Medium | No - code duplication |
| 2 | Optimization variant of GPU backend | Medium | **Yes** - best UX |
| 3 | Automatic detection with fallback | High | No - too complex |

**Recommended: Option 2 (Optimization Variant)**

**Rationale:**
- Flash attention is an optimization, not a fundamentally different backend
- Users shouldn't need to know about it
- Extends existing `GpuAttentionBackend` without new types
- Can enable/disable via `use_flash_attention` flag
- Always has fallback path

**Implementation Plan:**
```
06-02: Add flash support detection (can_use_flash_attention)
06-03: Implement flash forward path with kernel launches
06-04: Benchmark and validate vs traditional path
```

### 5. ROCm/AMDGPU Specific Considerations

**Target GPU:** AMD Radeon RX 7900 XT (gfx1100, RDNA3)

**Key Characteristics:**
- Wave32: 32 threads per wavefront
- Shared memory: 64KB per compute unit
- L2 cache: Large (helps with memory patterns)

**Kernel Optimization Patterns:**
- Wave32 reduction for dot products
- Register blocking (Q row stored in registers)
- Block size: 256 threads (generic) or 32 threads (causal/nocausal)

---

## Files Created

| File | Purpose |
|------|---------|
| `.planning/phases/06-attention-optimization/RESEARCH.md` | Comprehensive research documentation (779 lines) |

---

## Files Analyzed

| File | Purpose |
|------|---------|
| `kernels/flash_attention.hip` | Generic flash attention kernel |
| `kernels/flash_attention_causal.hip` | Causal flash attention kernel |
| `kernels/flash_attention_nocausal.hip` | Non-causal flash attention kernel |
| `src/attention/backend_registry.rs` | Backend implementation registry |
| `src/attention/gpu.rs` | Current GPU backend implementation |
| `src/attention/kernels.rs` | Kernel wrapper functions |
| `src/attention/cpu.rs` | CPU backend for comparison |
| `build.rs` | Build system configuration |

---

## Key Decisions

### Decision 1: Integration Approach

**Choice:** Extend `GpuAttentionBackend` with automatic flash detection

**Alternatives Considered:**
- Separate `FlashAttentionBackend` struct
- Manual backend selection

**Rationale:**
- Flash attention is an internal optimization
- Users benefit from automatic selection
- Follows existing pattern in codebase
- Allows gradual rollout via feature flag

### Decision 2: Detection Criteria

**Criteria for using flash attention:**
```rust
fn can_use_flash_attention(&self, config: &AttentionConfig, mask: Option<&[f32]>) -> bool {
    self.use_flash_attention
        && config.head_dim <= 128
        && config.max_sequence_length <= 2048
        && (mask.is_none() || config.is_causal)
}
```

**Rationale:**
- Head dimension limit from kernel docs
- Sequence length practical limit for shared memory
- Custom masks not yet supported in flash kernels

### Decision 3: Kernel Selection

**Decision tree:**
```
if config.is_causal:
    use flash_attention_causal_kernel
else if mask.is_none():
    use flash_attention_nocausal_kernel
else:
    use flash_attention_kernel (generic with mask)
```

**Rationale:**
- Causal variant has built-in masking (faster)
- Non-causal variant is simplest (no mask logic)
- Generic kernel handles custom masks

---

## Next Steps

### 06-02: Flash Attention Backend Registration

**Tasks:**
1. Add `can_use_flash_attention()` to `GpuAttentionBackend`
2. Add `select_flash_kernel()` for kernel type selection
3. Add `forward_flash()` stub
4. Update `forward()` to call flash path when conditions met
5. Add tests for detection logic

**Expected Outcome:**
- Detection logic implemented and tested
- Flash path called when conditions met
- Falls back to traditional path otherwise

### 06-03: Flash Attention Kernel Integration

**Tasks:**
1. Implement buffer allocation and GPU memory management
2. Implement kernel launch functions (causal, non-causal, generic)
3. Add synchronization and error handling
4. Integrate with existing `GpuAttentionBackend::forward()`
5. Add correctness tests vs traditional path

**Expected Outcome:**
- Flash attention kernels fully integrated
- Correctness validated against traditional path
- Ready for benchmarking

### 06-04: Benchmark and Optimize

**Tasks:**
1. Benchmark suite for different seq_len, head_dim, batch sizes
2. Profile memory bandwidth usage
3. Identify and fix bottlenecks
4. Document performance characteristics

**Expected Outcome:**
- Performance baseline established
- 2-4x speedup documented for typical workloads
- Optimization targets identified

---

## Performance Expectations

**Expected Speedup:** 2-4x for typical inference workloads

**Workload Characteristics:**
- Sequence length: 512-2048 tokens
- Head dimension: 64-128
- Batch size: 1-4

**Primary Benefit:** Reduced memory bandwidth
- Traditional: Read/write attention matrix (O(seq^2))
- Flash: No attention matrix materialization

**Secondary Benefit:** Kernel fusion
- Traditional: 5+ kernel launches with sync
- Flash: 1 kernel launch, no sync

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Incorrect output from flash kernels | High | Validate against CPU implementation |
| Performance regression | Medium | Benchmark before/after |
| Limited applicability | Low | Always have fallback path |
| Shared memory issues | Medium | Test at seq_len boundaries |

---

## Notes

- Flash attention kernels already exist and are built (no kernel development needed)
- Rust wrapper functions already available in `src/attention/kernels.rs`
- Main work is integration, not kernel implementation
- Existing `GpuAttentionBackend::forward()` uses multi-kernel approach (QK^T matmul + scale + mask + softmax + weighted matmul)
- Flash attention replaces all 5 operations with single kernel launch

---

## References

- Flash Attention paper: "Flash Attention: Fast and Memory-Efficient Exact Attention"
- AMD RDNA3 Architecture: https://www.amd.com/en/products/graphics/amd-radeon-rx-7900-xt
- ROCm Documentation: https://rocm.docs.amd.com/

---

*End of Summary*
