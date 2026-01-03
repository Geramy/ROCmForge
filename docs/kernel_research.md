# ROCmForge GPU Kernel Research

> Generated: 2026-01-03
> Purpose: Code snippets and patterns for implementing missing HIP kernels
> Target: AMD CDNA3/CDNA4 architectures (MI200/MI300 series)
>
> **NOTE:** For exact contracts (signatures, layouts, correctness harness), see `implementation_roadmap.md`.
> This document contains reference patterns; the roadmap contains implementation contracts.

---

## Table of Contents

1. [FlashAttention Implementation](#1-flashattention-implementation)
2. [Softmax Kernel with LDS](#2-softmax-kernel-with-lds)
3. [SwiGLU Activation](#3-swiglu-activation)
4. [RoPE GPU Kernel](#4-rope-gpu-kernel)
5. [Matrix Multiplication with MFMA](#5-matrix-multiplication-with-mfma)
6. [LayerNorm Kernel](#6-layernorm-kernel)
7. [Build System Patterns](#7-build-system-patterns)
8. [Reference Resources](#8-reference-resources)

---

## 1. FlashAttention Implementation

### Core Concept from FlashAttention-2

FlashAttention fuses the entire attention computation into a single kernel:

```
QK^T → scale → causal mask → softmax → (softmax × V)
```

Key optimizations:
- **Tiling**: Load blocks of Q, K, V into shared memory (LDS)
- **Recomputation**: Don't store full attention matrix; compute softmax on-the-fly
- **Online softmax**: Use running statistics to compute softmax in one pass
- **Wave64 alignment**: AMD GPUs use 64 threads per wave (vs NVIDIA's 32)

### AMD-Specific Adaptations

From AMD Matrix Core Programming research:

```cpp
// AMD CDNA3/CDNA4 specific block sizes
constexpr int BLOCK_M = 64;   // M dimension tile
constexpr int BLOCK_N = 64;   // N dimension tile
constexpr int BLOCK_DMODEL = 64;  // Head dim tile

// Wave64: 64 threads per workgroup
constexpr int WAVE_SIZE = 64;
constexpr int BLOCK_SIZE = 256;  // 4 waves per block
```

### Reference Implementation Pattern

```cpp
// Based on a-hamdi/GPU FlashAttention implementation
// Adapted for AMD HIP from CUDA

template <typename T, int HEAD_DIM>
__global__ void flash_attention_fwd_kernel(
    const T* __restrict__ Q,     // [batch, seq_len, num_heads, head_dim]
    const T* __restrict__ K,     // [batch, seq_len, num_heads, head_dim]
    const T* __restrict__ V,     // [batch, seq_len, num_heads, head_dim]
    T* __restrict__ output,      // [batch, seq_len, num_heads, head_dim]
    const float scale,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float* __restrict__ causal_mask  // Optional: [seq_len, seq_len]
) {
    // LDS (Local Data Share) for shared memory
    __shared__ float Q_tile[BLOCK_M][BLOCK_DMODEL];
    __shared__ float K_tile[BLOCK_N][BLOCK_DMODEL];
    __shared__ float V_tile[BLOCK_N][BLOCK_DMODEL];

    // Online softmax statistics
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM] = {0};

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int row_idx = blockIdx.x * BLOCK_M + threadIdx.y;

    const int q_offset = ((batch_idx * seq_len + row_idx) * num_heads + head_idx) * head_dim;

    // Loop over K blocks in sequence dimension
    for (int tile_idx = 0; tile_idx * BLOCK_N < seq_len; ++tile_idx) {
        // Load K tile into LDS
        const int k_base = ((batch_idx * seq_len + tile_idx * BLOCK_N) * num_heads + head_idx) * head_dim;
        load_tile_to_lds(K + k_base, K_tile, seq_len, tile_idx);

        // Load V tile into LDS
        const int v_base = ((batch_idx * seq_len + tile_idx * BLOCK_N) * num_heads + head_idx) * head_dim;
        load_tile_to_lds(V + v_base, V_tile, seq_len, tile_idx);

        __syncthreads();

        // Load Q row into registers
        float q_row[HEAD_DIM];
        if (row_idx < seq_len) {
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; ++i) {
                q_row[i] = __half2float(Q[q_offset + i]);
            }
        }

        // Compute QK^T for this tile
        float scores[BLOCK_N];
        #pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
            scores[j] = 0.0f;
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; ++i) {
                scores[j] += q_row[i] * K_tile[threadIdx.y][i];
            }
            scores[j] *= scale;
        }

        // Apply causal mask (for decode, only attend to current and past tokens)
        #pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
            int col_idx = tile_idx * BLOCK_N + j;
            if (col_idx > row_idx) {
                scores[j] = -INFINITY;
            }
        }

        // Online softmax update
        float tile_max = -INFINITY;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
            tile_max = fmaxf(tile_max, scores[j]);
        }

        float old_max = max_score;
        max_score = fmaxf(max_score, tile_max);

        float exp_scale = expf(old_max - max_score);
        sum_exp *= exp_scale;

        #pragma unroll
        for (int j = 0; j < BLOCK_N; ++j) {
            float exp_val = expf(scores[j] - max_score);
            sum_exp += exp_val;

            // Accumulate output
            #pragma unroll
            for (int i = 0; i < HEAD_DIM; ++i) {
                acc[i] = acc[i] * exp_scale + exp_val * V_tile[j][i];
            }
        }

        __syncthreads();
    }

    // Write output with final softmax normalization
    if (row_idx < seq_len) {
        const int out_offset = ((batch_idx * seq_len + row_idx) * num_heads + head_idx) * head_dim;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM; ++i) {
            output[out_offset + i] = __float2half(acc[i] / sum_exp);
        }
    }
}
```

---

## 2. Softmax Kernel with LDS

### Row-wise Softmax with LDS Reduction

```cpp
// Optimized softmax for AMD GPUs using LDS for reduction
// From ROCm optimization patterns

template <int BLOCK_SIZE>
__global__ void softmax_fwd_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols
) {
    __shared__ float sdata[BLOCK_SIZE * 2];  // For max and sum reduction

    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_idx >= rows) return;

    const float* row = input + row_idx * cols;

    // Step 1: Find max per row (reduce across cols)
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, row[i]);
    }

    // Reduction in LDS
    sdata[tid] = max_val;
    __syncthreads();

    // Parallel reduction (wave-aware)
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    max_val = sdata[0];

    // Step 2: Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = expf(row[i] - max_val);
        sum += val;
        output[row_idx * cols + i] = val;  // Store unnormalized exp
    }

    // Reduction for sum
    sdata[tid + BLOCK_SIZE] = sum;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid + BLOCK_SIZE] += sdata[tid + stride + BLOCK_SIZE];
        }
        __syncthreads();
    }
    sum = sdata[BLOCK_SIZE];

    // Step 3: Normalize
    const float inv_sum = 1.0f / sum;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        output[row_idx * cols + i] *= inv_sum;
    }
}
```

### Wave-Aware Reduction Optimization

```cpp
// AMD CDNA3 has built-in wave reduction functions
// More efficient than manual LDS reduction

__device__ __forceinline__ float wave_reduce_max(float val) {
    // Use AMD's built-in wave reduction
    val = __builtin_amdgcn_ds_bpermute(val, 0x0001);  // Exchange within wave
    // ... more wave shuffle operations
    return val;
}

// Simpler version using rocPRIM-style patterns
__device__ __forceinline__ float warp_reduce_max(float val) {
    const int warpSize = 64;  // AMD wave64
    int mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}
```

---

## 3. SwiGLU Activation

### SwiGLU Formula

```
SwiGLU(x) = Swish(x @ W_gate) ⊙ (x @ W_up)
where Swish(x) = x * sigmoid(x)
```

### Element-wise SwiGLU Kernel

```cpp
// SwiGLU activation: gate * swish(up)
// gate and up are intermediate activations from MLP

template <typename T>
__global__ void swiglu_kernel(
    const T* __restrict__ gate,  // [rows, cols]
    const T* __restrict__ up,    // [rows, cols]
    T* __restrict__ output,      // [rows, cols]
    const int rows,
    const int cols
) {
    const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx >= rows || col_idx >= cols) return;

    const int idx = row_idx * cols + col_idx;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // Swish activation: u * sigmoid(u)
    float swish = u * (1.0f / (1.0f + expf(-u)));

    // SwiGLU: gate * swish(up)
    output[idx] = __float2half(g * swish);
}
```

### Fused GEMM + SwiGLU Pattern

```cpp
// When calling after hipBLAS GEMM, you can fuse the activation
// This avoids a separate kernel launch

// Pattern:
// 1. hipBLAS gemm for gate = x @ W_gate
// 2. hipBLAS gemm for up = x @ W_up
// 3. swiglu_kernel for final output
// 4. hipBLAS gemm for output @ W_down

// For maximum efficiency, implement custom GEMM with fused activation
```

---

## 4. RoPE GPU Kernel

### Rotary Positional Embedding

```cpp
// RoPE rotates pairs of dimensions based on position
// For head_dim dimensions, we rotate (dim[0], dim[1]), (dim[2], dim[3]), etc.

template <typename T, int HEAD_DIM>
__global__ void rope_kernel(
    T* __restrict__ input,        // [batch, seq_len, num_heads, head_dim]
    const float* __restrict__ cos,  // [max_seq_len, head_dim/2] precomputed
    const float* __restrict__ sin,  // [max_seq_len, head_dim/2] precomputed
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int token_idx = blockIdx.x;

    const int half_dim = head_dim / 2;
    const int pos = token_idx + position_offset;

    // Each thread processes one pair of dimensions
    const int dim_pair = threadIdx.x;  // 0 to half_dim-1

    if (token_idx >= seq_len || dim_pair >= half_dim) return;

    const int base_offset = ((batch_idx * seq_len + token_idx) * num_heads + head_idx) * head_dim;
    const int cos_offset = pos * half_dim + dim_pair;

    const float c = cos[cos_offset];
    const float s = sin[cos_offset];

    // Load the pair of values
    const int i0 = base_offset + dim_pair;
    const int i1 = base_offset + dim_pair + half_dim;

    float x0 = __half2float(input[i0]);
    float x1 = __half2float(input[i1]);

    // Apply rotation: [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
    input[i0] = __float2half(x0 * c - x1 * s);
    input[i1] = __float2half(x0 * s + x1 * c);
}
```

### Fused RoPE + KV Append Kernel

```cpp
// Apply RoPE and immediately append to KV cache
// Eliminates GPU→CPU round-trip during decode

template <typename T>
__global__ void rope_and_append_kernel(
    const T* __restrict__ input,       // [batch, 1, num_kv_heads, head_dim] - single token
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    T* __restrict__ kv_cache,          // [batch, max_seq_len, num_kv_heads, head_dim]
    const int batch_idx,
    const int num_heads,
    const int head_dim,
    const int seq_len,                 // Current sequence length (write position)
    const int max_seq_len,
    const int position_offset
) {
    const int head_idx = blockIdx.y;
    const int dim_pair = threadIdx.x;

    const int half_dim = head_dim / 2;
    if (dim_pair >= half_dim) return;

    const int pos = seq_len + position_offset;
    const int cos_offset = pos * half_dim + dim_pair;

    const float c = cos_cache[cos_offset];
    const float s = sin_cache[cos_offset];

    // Process input (single token)
    const int in_base = (batch_idx * num_heads + head_idx) * head_dim;
    const int out_base = (batch_idx * max_seq_len + seq_len) * num_heads + head_idx;

    const int i0_in = in_base + dim_pair;
    const int i1_in = in_base + dim_pair + half_dim;
    const int i0_out = out_base * head_dim + dim_pair;
    const int i1_out = out_base * head_dim + dim_pair + half_dim;

    float x0 = __half2float(input[i0_in]);
    float x1 = __half2float(input[i1_in]);

    // Apply RoPE rotation
    kv_cache[i0_out] = __float2half(x0 * c - x1 * s);
    kv_cache[i1_out] = __float2half(x0 * s + x1 * c);
}
```

---

## 5. Matrix Multiplication with MFMA

### AMD CDNA3 MFMA Instructions

AMD CDNA3/CDNA4 provides Matrix Fused Multiply-Add (MFMA) instructions:

```cpp
// MFMA for matrix operations
// Syntax: mfma_<out_type>_<in_type>_<a_shape>_<b_shape>_<c_shape>

// Examples for MI300 (CDNA3):
// mfma_f32_16x16x16_f32 - 16x16x16 matrix multiply, FP32 in/out
// mfma_f32_32x32x4_f16  - 32x32x4 matrix multiply, FP16 in, FP32 out

__device__ __forceinline__ void mfma_f32_16x16x16_f32(
    const float* a, const float* b, float* c,
    const int lda, const int ldb
) {
    // Use inline assembly for MFMA
    // This is typically handled by the compiler with proper intrinsics

    // HIP provides rocBLAS for optimized GEMM
    // For custom kernels, use Composable Kernel (CK) library
}
```

### Using rocBLAS for GEMM (Recommended)

```cpp
// ROCmForge already wraps hipBLAS - use it for QKV projection
// Custom MFMA is only needed for specialized operations

#include <hipblas/hipblas.h>

// QKV projection: [seq_len, hidden] @ [hidden, 3*num_heads*head_dim]
// Split into 3 GEMMs for Q, K, V

hipblasHandle_t handle;
hipblasCreate(&handle);

const float alpha = 1.0f;
const float beta = 0.0f;

// Q projection
hipblasSgemm(handle,
    HIPBLAS_OP_N, HIPBLAS_OP_T,  // No transpose on x, transpose on W_q
    seq_len, num_heads * head_dim, hidden,
    &alpha,
    x, hidden,                    // [seq_len, hidden]
    W_q, hidden,                  // [num_heads*head_dim, hidden]
    &beta,
    Q, num_heads * head_dim       // [seq_len, num_heads*head_dim]
);
```

### Small GEMM Optimization for Attention

For small matrix sizes (common in attention), use custom kernel:

```cpp
// Thread-block GEMM for small attention matrices
// Optimized for shapes like [seq_len, head_dim] x [head_dim, seq_len]

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void small_gemm_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    const int M, const int N, const int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    for (int tile = 0; tile < (K + TILE_K - 1) / TILE_K; ++tile) {
        // Load tile to LDS
        if (row < M && threadIdx.x < TILE_K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_K + threadIdx.x];
        }
        if (threadIdx.y < TILE_K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_K + threadIdx.y) * N + col];
        }
        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
```

---

## 6. LayerNorm Kernel

### RMSNorm (Used in LLaMA-style models)

```cpp
// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
// Simpler than LayerNorm (no centering/bias subtraction)

template <typename T>
__global__ void rms_norm_kernel(
    const T* __restrict__ input,    // [rows, hidden_dim]
    const T* __restrict__ weight,   // [hidden_dim]
    T* __restrict__ output,         // [rows, hidden_dim]
    const float eps,
    const int rows,
    const int hidden_dim
) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockDim = 256;  // Adjust based on warp size

    __shared__ float s_data[256];

    if (row_idx >= rows) return;

    const float* row = (const float*)(input + row_idx * hidden_dim);

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim) {
        float val = row[i];
        sum_sq += val * val;
    }

    // Reduce to get mean
    s_data[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    float mean_sq = s_data[0] / (float)hidden_dim;
    float rms = rsqrtf(mean_sq + eps);

    // Apply normalization
    for (int i = tid; i < hidden_dim; i += blockDim) {
        float w = __half2float(weight[i]);
        output[row_idx * hidden_dim + i] = __float2half(row[i] * rms * w);
    }
}
```

---

## 7. Build System Patterns

### build.rs for HIP Kernel Compilation

```rust
// build.rs - Compile .hip files to .hsaco (HIP Shader Archive)

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=kernels");

    // Find hipcc compiler
    let hipcc = match env::var("HIPCC") {
        Ok(path) => PathBuf::from(path),
        Err(_) => PathBuf::from("hipcc"),
    };

    // List of HIP kernels to compile
    let kernels = vec![
        "kernels/softmax.hip",
        "kernels/mask.hip",
        "kernels/scale.hip",
        "kernels/flash_attention.hip",
        "kernels/rope.hip",
        "kernels/swiglu.hip",
        "kernels/rms_norm.hip",
    ];

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    for kernel in &kernels {
        let kernel_path = PathBuf::from(kernel);
        if !kernel_path.exists() {
            println!("cargo:warning=Kernel file not found: {}", kernel);
            continue;
        }

        let output_name = kernel_path
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();

        let hsaco_path = out_dir.join(format!("{}.hsaco", output_name));

        // Compile with hipcc
        let status = std::process::Command::new(&hipcc)
            .arg(kernel)
            .arg("-o")
            .arg(&hsaco_path)
            .arg("-O3")
            .arg("--amdgpu-target=amdgcn-amd-amdhsa")  // Generic AMDGPU target
            // Or specific: "--amdgpu-target=gfx942" for MI300X
            .arg("-ffast-math")
            .status()
            .expect("Failed to execute hipcc");

        if !status.success() {
            panic!("Failed to compile kernel: {}", kernel);
        }

        println!("cargo:rustc-env={}={}", output_name.to_uppercase(), hsaco_path.display());
    }

    // Link with amdhip64
    println!("cargo:rustc-link-lib=amdhip64");

    // Add ROCm library search path if available
    if let Ok(rocm_path) = env::var("ROCM_PATH") {
        let lib_path = format!("{}/lib", rocm_path);
        println!("cargo:rustc-link-search={}", lib_path);
    }
}
```

### Loading Compiled Kernels in Rust

```rust
// src/backend/kernel_loader.rs

use std::ffi::CString;
use std::ptr;

pub struct HipKernel {
    module: *mut hipModule_t,
    function: *mut hipFunction_t,
}

impl HipKernel {
    pub unsafe fn load_from_hsaco(hsaco_path: &str, kernel_name: &str) -> Result<Self, HipError> {
        let mut module: *mut hipModule_t = ptr::null_mut();
        let mut function: *mut hipFunction_t = ptr::null_mut();

        let path_cstr = CString::new(hsaco_path)?;
        let name_cstr = CString::new(kernel_name)?;

        // Load HSACO file
        hip_check(hipModuleLoad(&mut module, path_cstr.as_ptr()))?;

        // Get kernel function
        hip_check(hipModuleGetFunction(&mut function, module, name_cstr.as_ptr()))?;

        Ok(Self { module, function })
    }

    pub unsafe fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: hipStream_t,
        args: &[*mut libc::c_void],
    ) -> Result<(), HipError> {
        hip_check(hipModuleLaunchKernel(
            self.function,
            grid_dim.0, grid_dim.1, grid_dim.2,
            block_dim.0, block_dim.1, block_dim.2,
            shared_mem_bytes,
            stream,
            args.as_ptr() as *mut _,
            ptr::null_mut(),
        ))?;

        Ok(())
    }
}

// Example usage in hip_backend.rs
impl HipBackend {
    pub unsafe fn load_kernels(&mut self) -> Result<(), HipError> {
        let softmax_hsaco = env!("SOFTMAX_HSACO");
        self.softmax_kernel = HipKernel::load_from_hsaco(softmax_hsaco, "softmax_fwd_kernel")?;

        let flash_attn_hsaco = env!("FLASH_ATTENTION_HSACO");
        self.flash_attention_kernel = HipKernel::load_from_hsaco(flash_attn_hsaco, "flash_attention_fwd_kernel")?;

        Ok(())
    }
}
```

---

## 8. Reference Resources

### Official AMD Resources

1. **ROCm Documentation**
   - https://rocm.docs.amd.com/
   - HIP Programming Guide, optimization guides

2. **AMD Matrix Core Programming**
   - Blog post on MFMA instructions
   - CDNA3/CDNA4 architecture guides

3. **Composable Kernel (CK) Library**
   - AMD's performance-oriented library
   - https://github.com/AMD/CK
   - Contains production-ready kernel implementations

### Open Source Implementations

1. **a-hamdi/GPU**
   - 100 days of GPU kernel implementations
   - FlashAttention, RoPE, SwiGLU, softmax implementations
   - Both CUDA and HIP versions

2. **llama.cpp HIP Backend**
   - Working HIP implementations for LLM inference
   - Reference for attention, RoPE, RMSNorm

3. **FlashAttention Official**
   - Original CUDA implementation
   - Translate CUDA→HIP patterns (mostly syntax changes)

### Key Implementation Notes

1. **Wave64 vs Wave32**: AMD uses 64 threads per wave, NVIDIA uses 32
   - Adjust block sizes accordingly
   - Wave reduction functions differ

2. **LDS vs Shared Memory**: Same concept, different naming
   - `__shared__` in CUDA/HIP both work
   - AMD has larger LDS per CU (typically 64KB-128KB)

3. **MFMA vs Tensor Cores**:
   - AMD MFMA is instruction-based
   - Can be used via inline assembly or rocBLAS/CK

4. **hipBLAS vs cuBLAS**:
   - API-compatible in most cases
   - Performance characteristics differ slightly

5. **Compilation Targets**:
   - `gfx942` = MI300X (CDNA3)
   - `gfx940` = MI200 (CDNA2)
   - Use `--amdgpu-target` flag in hipcc

---

## Implementation Priority

Based on ROCmForge's current state (infrastructure complete, kernels missing):

1. **First: Implement softmax, mask, scale kernels**
   - Replace no-op stubs in `src/attention/kernels.rs`
   - These are simple element-wise operations

2. **Second: Fused attention kernel**
   - Single biggest performance unlock
   - Use FlashAttention-2 pattern

3. **Third: RoPE GPU kernel**
   - Eliminates CPU fallback in `src/attention/rope.rs:233`

4. **Fourth: SwiGLU + RMSNorm**
   - Complete GPU MLP path

5. **Fifth: GPU sampler**
   - Final optimization for decode latency

---

## Code Snippets for Immediate Use

### softmax.hip

```cpp
#include <hip/hip_fp16.h>

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void softmax_fwd(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int rows,
    const int cols
) {
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const T* row_ptr = input + row * cols;

    // Find max
    float max_val = -1e20f;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, __half2float(row_ptr[i]));
    }

    s_max[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    max_val = s_max[0];

    // Compute exp sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = expf(__half2float(row_ptr[i]) - max_val);
        sum += val;
        output[row * cols + i] = __float2half(val);
    }

    s_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / s_sum[0];

    // Normalize
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        output[row * cols + i] = __float2half(__half2float(output[row * cols + i]) * inv_sum);
    }
}

extern "C" {
    __global__ void softmax_fwd_f16(const half* input, half* output, int rows, int cols) {
        softmax_fwd<half>(input, output, rows, cols);
    }

    __global__ void softmax_fwd_f32(const float* input, float* output, int rows, int cols) {
        softmax_fwd<float>(input, output, rows, cols);
    }
}
```

### scale.hip

```cpp
#include <hip/hip_fp16.h>

template <typename T>
__global__ void scale_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float scale,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = __float2half(__half2float(input[idx]) * scale);
}

extern "C" {
    __global__ void scale_f16(const half* input, half* output, float scale, int size) {
        scale_kernel<half>(input, output, scale, size);
    }

    __global__ void scale_f32(const float* input, float* output, float scale, int size) {
        scale_kernel<float>(input, output, scale, size);
    }
}
```

### mask.hip

```cpp
#include <hip/hip_fp16.h>

template <typename T>
__global__ void causal_mask_kernel(
    T* __restrict__ data,
    const int rows,
    const int cols,
    const float mask_value
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    // Causal mask: only attend to current and previous positions
    if (col > row) {
        data[row * cols + col] = __float2half(mask_value);
    }
}

extern "C" {
    __global__ void causal_mask_f16(half* data, int rows, int cols, float mask_value) {
        causal_mask_kernel<half>(data, rows, cols, mask_value);
    }

    __global__ void causal_mask_f32(float* data, int rows, int cols, float mask_value) {
        causal_mask_kernel<float>(data, rows, cols, mask_value);
    }
}
```

---

> **Next Step**: Use these patterns and code snippets to create a detailed implementation plan in plan mode.
