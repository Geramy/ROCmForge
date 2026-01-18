# SIMD Strategy Decision

**Date:** 2026-01-18
**Phase:** 04-cpu-simd-backend
**Plan:** 04-01
**Status:** APPROVED

---

## Decision

Use **`std::simd`** (portable SIMD in Rust standard library) for CPU SIMD operations.

---

## Rationale

### 1. Standard Library Availability
- `std::simd` was stabilized in Rust 1.82.0 (November 2024)
- No external dependency required
- Part of the standard library - long-term support guaranteed

### 2. API Stability
- Stable API contract - won't break between compiler versions
- Unlike `packed_simd` (deprecated) or `wide` (third-party), the stdlib API is the future
- Aligned with Rust's evolution - portable SIMD is now a first-class citizen

### 3. Platform Support
- x86/x86_64: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2
- ARM64 (aarch64): NEON
- WebAssembly: SIMD128
- Automatic code generation based on target

### 4. Performance Benefits
Expected speedups for CPU operations:
- **Matmul (f32):** 4-8x with AVX2, 3-4x with SSE
- **Softmax:** 4-6x (horizontal operations benefit from SIMD)
- **Attention compute:** 4-6x overall

### 5. Compiler Support
- LLVM generates optimal SIMD instructions
- Zero-cost abstraction - no runtime overhead
- Compile-time optimization with target-feature attributes

---

## Implementation Strategy

### Phase 1: Core SIMD Primitives (Plan 04-02)

Create `src/backend/cpu/simd.rs` module with:

1. **SIMD-Enabled Matmul**
   ```rust
   use std::simd::{f32x4, f32x8, Simd, SimdFloat};

   pub fn simd_matmul_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32>
   ```

2. **Feature Detection**
   - Compile-time via `#[cfg(target_arch)]`
   - Runtime detection via `std::is_x86_feature_detected!`

3. **Type Aliases per Architecture**
   ```rust
   #[cfg(target_arch = "x86_64")]
   type SimdF32 = f32x8;  // AVX2: 8 floats per vector

   #[cfg(target_arch = "aarch64")]
   type SimdF32 = f32x4;  // NEON: 4 floats per vector
   ```

### Phase 2: Attention Operations (Plan 04-03)

1. **SIMD Softmax**
   - Vectorized exponential
   - Horizontal sum for normalization
   - Mask support for causal attention

2. **SIMD QK^T Matmul**
   - Batched matrix multiplication
   - Transpose optimization
   - Strided access patterns

3. **SIMD Weighted Sum**
   - scores * V multiplication
   - Accumulation in registers

### Phase 3: Integration (Plan 04-04)

1. **CPU Backend Trait**
   ```rust
   pub trait CpuBackend {
       fn matmul(&mut self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>>;
       fn softmax(&mut self, x: &mut [f32], batch_size: usize, seq_len: usize) -> Result<()>;
   }
   ```

2. **SIMD Implementation**
   ```rust
   pub struct SimdCpuBackend;

   impl CpuBackend for SimdCpuBackend {
       // Use SIMD-optimized implementations
   }
   ```

3. **Scalar Fallback**
   ```rust
   pub struct ScalarCpuBackend;

   impl CpuBackend for ScalarCpuBackend {
       // Use existing scalar implementations
   }
   ```

---

## Feature Detection Strategy

### Compile-Time (Initial Implementation)

Use `#[cfg(target_arch)]` for architecture-specific code:

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn simd_matmul_avx2(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // f32x8 implementation
}

#[cfg(target_arch = "aarch64")]
fn simd_matmul_neon(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // f32x4 implementation
}
```

### Runtime Detection (Future Enhancement)

Use `std::is_x86_feature_detected!` for runtime capability detection:

```rust
#[cfg(target_arch = "x86_64")]
fn dispatch_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            simd_matmul_avx2(a, b, m, n, k)
        } else if is_x86_feature_detected!("sse4.1") {
            simd_matmul_sse41(a, b, m, n, k)
        } else {
            scalar_matmul(a, b, m, n, k)
        }
    }
}
```

---

## Performance Expectations

### Matrix Multiplication (C = A * B)

| Operation | Scalar (baseline) | SSE (f32x4) | AVX2 (f32x8) |
|-----------|-------------------|-------------|--------------|
| 1024x1024 matmul | 1.0x | 3.5x | 6.5x |
| 4096x4096 matmul | 1.0x | 3.8x | 7.2x |

### Attention Layer (single head)

| Operation | Scalar (baseline) | SIMD (AVX2) |
|-----------|-------------------|-------------|
| QK^T + scale | 1.0x | 6.5x |
| Softmax | 1.0x | 4.5x |
| Weighted sum | 1.0x | 6.5x |
| **Total attention** | **1.0x** | **5.5x** |

### Real-World Impact

For a 7B parameter model inference:
- **Scalar CPU:** ~5-10 tokens/second
- **SIMD CPU (AVX2):** ~25-50 tokens/second
- **GPU (ROCm):** ~50-100+ tokens/second

SIMD CPU backend makes CPU fallback viable for:
- Development/testing without GPU
- Systems without AMD GPU
- Emergency fallback when GPU unavailable

---

## Dependencies and MSRV

### Current Project Rust Version
To be verified in Cargo.toml (not explicitly specified).

### Required MSRV
- **Minimum:** Rust 1.82.0 (for stable `std::simd`)

### Cargo.toml Changes

No new dependencies required. Add optional feature:

```toml
[features]
default = ["cpu-simd"]
cpu-simd = []  # Enable SIMD optimizations (always available on Rust 1.82+)
```

For users on older Rust versions (unlikely, but possible):
```toml
[features]
default = []
cpu-simd = []  # Will fail compile on Rust < 1.82
```

---

## Alternatives Considered

### packed_simd crate
**Rejected:** Deprecated and unmaintained. Superseded by std::simd.

### wide crate
**Rejected:** External dependency providing functionality already in stdlib. API differs from std::simd, potential migration pain.

### std::arch (platform intrinsics)
**Rejected for primary implementation:** Platform-specific, requires multiple code paths. May be used for future targeted optimizations.

### Hand-written assembly
**Rejected:** Not maintainable. Compiler generates optimal code from std::simd.

---

## Risks and Mitigation

### Risk 1: MSRV Increase to 1.82
**Impact:** Users on older Rust versions cannot compile.

**Mitigation:**
- Rust 1.82 released November 2024 - widely available
- Most active Rust projects update within 6-12 months
- Document MSRV clearly in README

### Risk 2: Performance Regression on Unsupported Architectures
**Impact:** SIMD code may be slower than scalar on some platforms.

**Mitigation:**
- Provide scalar fallback
- Benchmark on target platforms
- Use compile-time feature detection

### Risk 3: std::simd API Changes
**Impact:** Future Rust versions may change the API.

**Mitigation:**
- API is now stable (1.82+)
- Changes would be major version bumps
- Standard library maintains backward compatibility

---

## Implementation Order

1. **Plan 04-02:** Create SIMD matmul primitives
   - Add `src/backend/cpu/simd.rs`
   - Implement `simd_matmul_f32` with f32x4/f32x8
   - Add tests for correctness

2. **Plan 04-03:** SIMD attention operations
   - Implement `simd_softmax`
   - Optimize QK^T and weighted sum
   - Benchmark vs scalar

3. **Plan 04-04:** Backend integration
   - Create `CpuBackend` trait
   - Implement SIMD and scalar backends
   - Update engine to use CPU backend when GPU unavailable

---

## Success Criteria

- [ ] `std::simd` compiles on target platforms (x86_64, aarch64)
- [ ] SIMD matmul passes correctness tests (matches scalar output)
- [ ] SIMD matmul achieves 4x+ speedup over scalar (AVX2)
- [ ] No new external dependencies added
- [ ] CPU backend integrates cleanly with existing GPU backend
- [ ] Documentation updated with SIMD capabilities

---

## References

- Research document: `.planning/phases/04-cpu-simd-backend/RESEARCH.md`
- [Rust std::simd documentation](https://doc.rust-lang.org/stable/std/simd/)
- [Rust 1.82.0 Release Announcement](https://blog.rust-lang.org/2024/11/28/Rust-1.82.0.html)
