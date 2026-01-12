# Phase 19.2: KV Replication Kernel - Complete Deliverables

**Deliverable Date**: 2025-01-11
**Status**: COMPLETE - Ready for Testing

---

## Executive Summary

This document summarizes all deliverables for Phase 19.2 (KV Replication Kernel) in the ROCmForge project. The kernel enables efficient GPU-based replication of K and V tensors for Multi-Query Attention (MQA) and Grouped-Query Attention (GQA).

### Key Achievement
Designed and implemented a complete HIP kernel that replaces CPU-based KV replication with a GPU-accelerated implementation, expected to provide **20-30x performance improvement**.

---

## Deliverables

### 1. HIP Kernel Implementation
**File**: `/home/feanor/Projects/ROCmForge/kernels/mqa_kv_replicate.hip`

**Components**:
- `mqa_kv_replicate_k_kernel` - K-only replication
- `mqa_kv_replicate_v_kernel` - V-only replication
- `mqa_kv_replicate_fused_kernel` - Combined K+V replication (recommended)

**Key Features**:
- 1D thread grid for simplicity and coalesced memory access
- Zero shared memory (not needed for element-wise copy)
- Optimized for AMD RDNA3 architecture (wave32)
- Full documentation with usage examples

### 2. Design Documentation
**File**: `/home/feanor/Projects/ROCmForge/docs/KV_REPLICATION_KERNEL_DESIGN.md`

**Contents**:
- Thread mapping strategy with detailed explanation
- Kernel algorithm with pseudocode
- Memory access pattern analysis
- Performance considerations
- Integration plan
- Testing strategy

### 3. Build System Integration
**File**: `/home/feanor/Projects/ROCmForge/build.rs`

**Changes**:
- Added `mqa_kv_replicate.hip` to kernel compilation list
- Kernel compiled during build via `hipcc`
- HSACO output path set via `MQA_KV_REPLICATE_HSACO` environment variable

**Code Change** (line 55):
```rust
("kernels/mqa_kv_replicate.hip", "MQA_KV_REPLICATE_HSACO", "mqa_kv_replicate_kernel"),
```

### 4. Rust FFI Wrapper
**File**: `/home/feanor/Projects/ROCmForge/src/attention/kernels.rs`

**Changes Made**:
1. Added kernel cache fields (lines 44-45):
   ```rust
   mqa_kv_replicate_module: Option<HipModule>,
   mqa_kv_replicate_kernel: Option<HipKernel>,
   ```

2. Added kernel initialization (lines 205-215):
   ```rust
   let mqa_kv_replicate_path = std::env::var("MQA_KV_REPLICATE_HSACO")...;
   let mqa_kv_replicate_module = backend.load_module(&mqa_kv_replicate_path)?;
   let mqa_kv_replicate_kernel = backend.get_kernel_function(
       &mqa_kv_replicate_module,
       "mqa_kv_replicate_fused_kernel"
   )?;
   ```

3. Added public wrapper function (lines 1014-1093):
   ```rust
   pub unsafe fn mqa_kv_replicate_gpu_kernel(
       k: *const f32,
       v: *const f32,
       k_expanded: *mut f32,
       v_expanded: *mut f32,
       batch_size: u32,
       seq_len: u32,
       num_kv_heads: u32,
       num_q_heads: u32,
       head_dim: u32,
   ) -> Result<(), String>
   ```

---

## Technical Specifications

### Thread Mapping Strategy

**Approach**: 1D thread grid, each thread handles one output element

**Formula**:
```
Thread ID = blockIdx.x * blockDim.x + threadIdx.x
           = 0, 1, 2, ..., total_elements - 1

Where total_elements = batch_size * seq_len * num_q_heads * head_dim
```

**Index Decoding**:
```
batch_idx   = idx / (seq_len * num_q_heads * head_dim)
seq_idx     = (idx % (seq_len * num_q_heads * head_dim)) / (num_q_heads * head_dim)
q_head_idx  = (idx % (num_q_heads * head_dim)) / head_dim
dim_idx     = idx % head_dim
```

**KV Head Mapping**:
```
kv_head_idx = q_head_idx / heads_per_kv
Where heads_per_kv = num_q_heads / num_kv_heads
```

### Memory Access Pattern

**Reads** (coalesced):
- Each warp reads 32 consecutive elements from K/V
- Memory aligned to 128-byte cache lines (RDNA3)

**Writes** (coalesced):
- Each warp writes 32 consecutive elements to K_expanded/V_expanded
- No write conflicts

**Total Bandwidth**:
```
Read:  2 * batch_size * seq_len * num_kv_heads * head_dim * sizeof(f32)
Write: 2 * batch_size * seq_len * num_q_heads * head_dim * sizeof(f32)
Total: 2 * (1 + heads_per_kv) * batch_size * seq_len * num_kv_heads * head_dim * 4 bytes
```

### Grid Configuration

```
Grid Dim:  (total_elements.div_ceil(256), 1, 1)
Block Dim: (256, 1, 1)
```

**Example**: batch_size=1, seq_len=2048, num_kv_heads=2, num_q_heads=32, head_dim=128
```
total_elements = 1 * 2048 * 32 * 128 = 8,388,608
grid_dim.x = 8,388,608 / 256 = 32,768 blocks
Total threads = 32,768 * 256 = 8,388,608 threads
```

---

## Performance Analysis

### Expected Performance

**CPU Implementation** (current, lines 313-354 in multi_query.rs):
- Nested loops in Rust
- Memory bandwidth limited by CPU
- Estimated: ~15 ms for typical workload

**GPU Kernel** (new):
- Massively parallel (8M threads for example above)
- Near-memory-bandwidth utilization
- Estimated: ~0.5 ms for typical workload

**Expected Speedup**: 20-30x

### Occupancy Analysis

**Block size**: 256 threads
**Registers per thread**: ~10
**Shared memory**: 0 bytes
**Max blocks per SM**: Limited by registers, not shared memory
**Occupancy**: ~80% (excellent)

### Memory Bandwidth Utilization

**AMD RX 7900 XT**: ~500 GB/s theoretical bandwidth
**Expected utilization**: ~400 GB/s (80% efficiency)
**Bottleneck**: Memory bandwidth (compute-bound not applicable for simple copy)

---

## Integration Points

### Current Usage (CPU)

In `/home/feanor/Projects/ROCmForge/src/attention/multi_query.rs`:
```rust
let (k_expanded, v_expanded) =
    self.expand_kv_to_query_heads(&k_processed, v, batch_size, kv_seq_len)?;
```

The `expand_kv_to_query_heads` method (lines 313-354) uses nested CPU loops.

### Future GPU Integration

To replace CPU implementation with GPU kernel:

```rust
#[cfg(feature = "rocm")]
fn expand_kv_to_query_heads_gpu(
    &self,
    k: &DeviceTensor,
    v: &DeviceTensor,
    batch_size: usize,
    kv_seq_len: usize,
) -> AttentionResult<(DeviceTensor, DeviceTensor)> {
    use crate::attention::kernels::mqa_kv_replicate_gpu_kernel;

    // Allocate output tensors
    let k_expanded = DeviceTensor::empty(...)?;
    let v_expanded = DeviceTensor::empty(...)?;

    // Launch GPU kernel
    unsafe {
        mqa_kv_replicate_gpu_kernel(
            k.buffer().as_ptr() as *const f32,
            v.buffer().as_ptr() as *const f32,
            k_expanded.buffer().as_mut_ptr() as *mut f32,
            v_expanded.buffer().as_mut_ptr() as *mut f32,
            batch_size as u32,
            kv_seq_len as u32,
            self.config.num_kv_heads as u32,
            self.config.num_query_heads as u32,
            self.config.head_dim as u32,
        )?;
    }

    Ok((k_expanded, v_expanded))
}
```

---

## Testing Strategy

### Unit Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mqa_kv_replicate_kernel_mqa() {
        // Test MQA: num_kv_heads=1, num_q_heads=8
        // Verify output matches CPU implementation
    }

    #[test]
    fn test_mqa_kv_replicate_kernel_gqa() {
        // Test GQA: num_kv_heads=2, num_q_heads=8
        // Verify output matches CPU implementation
    }

    #[test]
    fn test_mqa_kv_replicate_kernel_edge_cases() {
        // Test batch_size=1, seq_len=1, head_dim=1
        // Test odd head dimensions
        // Test large tensors
    }
}
```

### Verification Approach

1. **Correctness**: Compare GPU output with CPU implementation
2. **Edge Cases**: Test boundary conditions
3. **Performance**: Benchmark against CPU implementation
4. **Stress Test**: Large tensors, extreme replication factors

---

## Build and Compilation

### Prerequisites

```bash
# ROCm installation
export ROCM_PATH=/opt/rocm
export HIPCC=$ROCM_PATH/bin/hipcc
export ROCm_ARCH=gfx1100  # For AMD RX 7900 XT
```

### Compilation

```bash
# The kernel is automatically compiled during cargo build
cargo build --release --features rocm
```

### Manual Compilation (for testing)

```bash
$HIPCC -c --genco --offload-arch=$ROCm_ARCH \
  -O3 kernels/mqa_kv_replicate.hip \
  -o kernels/target/mqa_kv_replicate.hsaco
```

---

## Files Modified/Created

### Created Files
1. `/home/feanor/Projects/ROCmForge/kernels/mqa_kv_replicate.hip` - HIP kernel source
2. `/home/feanor/Projects/ROCmForge/docs/KV_REPLICATION_KERNEL_DESIGN.md` - Design document
3. `/home/feanor/Projects/ROCmForge/docs/PHASE_19_2_KERNEL_DELIVERABLES.md` - This document

### Modified Files
1. `/home/feanor/Projects/ROCmForge/build.rs` - Added kernel compilation (line 55)
2. `/home/feanor/Projects/ROCmForge/src/attention/kernels.rs` - Added cache, initialization, wrapper (lines 44-45, 205-243, 1014-1093)

---

## Next Steps

### Immediate (Phase 19.2 Complete)
1. Kernel implementation complete
2. Build integration complete
3. Rust wrapper complete

### Short-Term (Phase 19.3+)
1. Write comprehensive unit tests
2. Integrate with MQA attention implementation
3. Benchmark performance vs CPU
4. Optimize if needed (current design should be near-optimal)

### Long-Term
1. Consider kernel fusion with attention computation
2. Explore async replication (overlap with previous computation)
3. Adaptive kernel selection based on tensor size

---

## References

### Context Files Read
1. `/home/feanor/Projects/ROCmForge/src/attention/kernels.rs` - Kernel infrastructure
2. `/home/feanor/Projects/ROCmForge/kernels/scale.hip` - Example kernel pattern
3. `/home/feanor/Projects/ROCmForge/kernels/rope.hip` - Example kernel pattern
4. `/home/feanor/Projects/ROCmForge/kernels/qkt_matmul.hip` - Example kernel pattern
5. `/home/feanor/Projects/ROCmForge/src/attention/multi_query.rs` - Current CPU implementation
6. `/home/feanor/Projects/ROCmForge/src/backend/hip_backend.rs` - HIP FFI patterns
7. `/home/feanor/Projects/ROCmForge/build.rs` - Build system

### Design Principles Followed
1. **Simplicity**: 1D thread grid, easy to understand
2. **Correctness**: Exact semantic match with CPU implementation
3. **Performance**: Coalesced memory access, near-optimal occupancy
4. **Maintainability**: Well-documented, follows project conventions
5. **Safety**: Proper error handling, Rust FFI safety

---

## Verification Status

- [x] HIP kernel source complete
- [x] Build system integration complete
- [x] Rust FFI wrapper complete
- [x] Design documentation complete
- [x] Code compiles without errors
- [ ] Unit tests written (next phase)
- [ ] Performance benchmarked (next phase)
- [ ] Integrated with MQA (next phase)

---

## Contact

**Author**: Kernel Design Agent (Architecture Reviewer)
**Phase**: 19.2 (KV Replication Kernel)
**Date**: 2025-01-11
**Status**: Design and Implementation Complete

---

## Appendix: Kernel Source (Quick Reference)

```cpp
/**
 * mqa_kv_replicate.hip - KV head replication kernel for MQA/GQA
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 */

extern "C" __global__ void mqa_kv_replicate_fused_kernel(
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ K_expanded,
    float* __restrict__ V_expanded,
    const int batch_size,
    const int seq_len,
    const int num_kv_heads,
    const int num_q_heads,
    const int head_dim
) {
    const int heads_per_kv = num_q_heads / num_kv_heads;
    const int total_elements = batch_size * seq_len * num_q_heads * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode index to (batch, seq, q_head, dim)
    const int batch_idx = idx / (seq_len * num_q_heads * head_dim);
    const int seq_and_heads = idx % (seq_len * num_q_heads * head_dim);
    const int seq_idx = seq_and_heads / (num_q_heads * head_dim);
    const int head_and_dim = seq_and_heads % (num_q_heads * head_dim);
    const int q_head_idx = head_and_dim / head_dim;
    const int dim_idx = head_and_dim % head_dim;

    // Map to source KV head
    const int kv_head_idx = q_head_idx / heads_per_kv;

    // Source index
    const int src_idx = batch_idx * seq_len * num_kv_heads * head_dim
                      + seq_idx * num_kv_heads * head_dim
                      + kv_head_idx * head_dim
                      + dim_idx;

    // Copy both K and V
    K_expanded[idx] = K[src_idx];
    V_expanded[idx] = V[src_idx];
}
```

---

**End of Phase 19.2 Deliverables**
