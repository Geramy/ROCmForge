# Quantization Formats and Dequantization Research

**Phase:** 05-quantized-operations
**Plan:** 05-01 (Research-only)
**Created:** 2026-01-18
**Purpose:** Technical foundation for implementing HIP dequantization kernels

---

## Table of Contents

1. [Q-Format Specifications](#q-format-specifications)
2. [CPU Dequantization Algorithms](#cpu-dequantization-algorithms)
3. [HIP Kernel Patterns](#hip-kernel-patterns)
4. [Kernel Implementation Strategy](#kernel-implementation-strategy)
5. [Build System Integration](#build-system-integration)
6. [References](#references)

---

## Q-Format Specifications

### Overview

GGUF (GPT-Generated Unified Format) supports multiple quantization formats for compressing neural network weights. These formats trade precision for memory efficiency, enabling larger models to fit in GPU memory.

### Format Comparison Table

| Format | Bits/Element | Block Size | Block Bytes | Compression vs FP16 | Usage |
|--------|--------------|------------|-------------|---------------------|-------|
| F32 | 32 | 1 | 4 | 1.0x (baseline) | Reference, not used for weights |
| F16 | 16 | 1 | 2 | 0.5x | Activations, intermediate tensors |
| Q4_0 | 4 | 32 | 20 | 4.5x | **Most common** for weights |
| Q4_1 | 4 | 32 | 24 | 4.0x | Less common, slightly better quality |
| Q5_0 | 5 | 32 | 24 | 3.4x | Higher quality than Q4 |
| Q5_1 | 5 | 32 | 28 | 2.9x | Best Q5 quality |
| Q8_0 | 8 | 32 | 36 | 1.8x | Activations, temporary tensors |
| Q2_K | 2.5 | 256 | 208 | 7.3x | Extreme compression |
| Q3_K | 3 | 256 | 208 | 6.1x | Aggressive compression |
| Q4_K | 4.5 | 256 | 256 | 5.1x | **Modern K-quants** (better quality) |
| Q5_K | 5 | 256 | 320 | 4.1x | High quality K-quant |
| Q6_K | 6 | 256 | 416 | 3.4x | Near-FP16 quality |
| MXFP4 | 4 | 32 | 17 | 4.7x | Block-scaled FP (OCP MX Spec) |
| MXFP6 | 6 | 32 | 25 | 3.2x | Block-scaled FP (OCP MX Spec) |

### Detailed Block Structures

#### Q4_0 (Most Common)

```
Block size: 32 elements
Block bytes: 20 bytes
┌─────────────────────────────────────────────────────┐
│ Scale (f32, 4 bytes) │ Quantized values (16 bytes)  │
├─────────────────────────────────────────────────────┤
│ 0x00 0x00 0x80 3F │ [packed 4-bit values x32]     │
│   scale=1.0       │                                  │
└─────────────────────────────────────────────────────┘
```

**Dequantization formula:**
```
value = scale * ((packed & 0x0F) - 8)
```

**Bit packing:**
- Each byte contains two 4-bit values
- Low nibble: `packed & 0x0F`
- High nibble: `(packed >> 4) & 0x0F`
- Values are unsigned 0-15, interpreted as signed -8 to +7

---

#### Q4_1

```
Block size: 32 elements
Block bytes: 24 bytes
┌──────────────────────────────────────────────────────────────────────┐
│ Scale (f32, 4) │ Min (f32, 4) │ Quantized values (16 bytes)         │
├──────────────────────────────────────────────────────────────────────┤
│ scale          │ minimum      │ [packed 4-bit values x32]          │
└──────────────────────────────────────────────────────────────────────┘
```

**Dequantization formula:**
```
value = min + (quant * scale)
```

---

#### Q5_0

```
Block size: 32 elements
Block bytes: 24 bytes
┌──────────────────────────────────────────────────────────────────────────┐
│ Scale (f32, 4) │ qh (u32, 4) │ Quantized values (20 bytes)             │
├──────────────────────────────────────────────────────────────────────────┤
│ scale          │ high bits  │ [packed 4-bit values x32 + high bit]    │
└──────────────────────────────────────────────────────────────────────────┘
```

**Dequantization formula:**
```
low_bits = (packed & 0x0F)  // or (packed >> 4) & 0x0F for high nibble
high_bit = (qh >> bit_idx) & 1
quant = (low_bits | (high_bit << 4))  // 5-bit value
value = (quant - 16) * scale
```

---

#### Q5_1

```
Block size: 32 elements
Block bytes: 28 bytes
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Scale (f32, 4) │ Min (f32, 4) │ qh (u32, 4) │ Quantized values (20 bytes)    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ scale          │ minimum      │ high bits  │ [4-bit packed + high bit]      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Dequantization formula:**
```
value = min + (quant * scale)  // quant is 5-bit from packed + qh
```

---

#### Q8_0 (Activations)

```
Block size: 32 elements
Block bytes: 36 bytes
┌─────────────────────────────────────────────────────────┐
│ Scale (f32, 4 bytes) │ Quantized values (32 bytes)     │
├─────────────────────────────────────────────────────────┤
│ scale                │ [int8 values x32]               │
└─────────────────────────────────────────────────────────┘
```

**Dequantization formula:**
```
value = scale * int8_value
```

**Note:** Values are already signed int8, no unpacking needed.

---

#### Q4_K (Super-Block Structure)

```
Super-block size: 256 elements (8 sub-blocks of 32)
Super-block bytes: 256 bytes
┌─────────────────────────────────────────────────────────────────────────────┐
│ Half-precision scales (16 bytes) │ int8 mins (16 bytes) │ Quantized (160)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 8 x f16 scales (2 bytes each)   │ 8 x int8 mins       │ 8 x 20-byte sub-  │
│                                 │                     │ blocks (4-bit x32) │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Structure:**
- 8 sub-blocks per super-block
- Each sub-block: 32 elements, 20 bytes quantized + 2 bytes scale + 1 byte min
- Total: 8 * (20 + 2 + 1) = 184 bytes for data + 72 bytes overhead

**Dequantization formula (per sub-block):**
```
sub_block_idx = element_idx / 32
element_in_subblock = element_idx % 32

scale = f16_scales[sub_block_idx]
min = int8_mins[sub_block_idx]
quant = extract_4bit(quant_data[sub_block_idx], element_in_subblock)

value = min + (quant * scale)
```

---

#### Q6_K

```
Super-block size: 256 elements
Super-block bytes: 256 bytes
┌─────────────────────────────────────────────────────────┐
│ Half-precision scales (32 bytes) │ Quantized (224)      │
├─────────────────────────────────────────────────────────┤
│ 16 x f16 scales (2 bytes each)   │ [6-bit packed x256] │
│ - One scale per 16 elements      │                      │
└─────────────────────────────────────────────────────────┘
```

**Structure:**
- 16 groups of 16 elements each
- Each group shares a scale
- 6-bit values packed across byte boundaries

**Dequantization formula:**
```
group_idx = element_idx / 16
scale = f16_scales[group_idx]
quant_val = extract_6bit(quant_data, element_idx)

// Convert to signed
signed_val = quant_val >= 32 ? (quant_val - 64) : quant_val
value = signed_val * scale
```

**6-bit unpacking:**
```
bit_offset = (element_idx * 6) % 8
byte_idx = (element_idx * 6) / 8
combined = (data[byte_idx + 1] << 8) | data[byte_idx]
quant_val = (combined >> bit_offset) & 0x3F
```

---

#### MXFP4 (OCP MX Specification v1.0)

```
Block size: 32 elements
Block bytes: 17 bytes
┌─────────────────────────────────────────────────────────┐
│ Scale (E8M0, 1 byte) │ Packed E2M1 values (16 bytes)  │
├─────────────────────────────────────────────────────────┤
│ exponent (int8)      │ [4-bit E2M1 x32]               │
└─────────────────────────────────────────────────────────┘
```

**E8M0 scale:**
```
scale = 2^exponent  // exponent is signed int8
```

**E2M1 format (4-bit):**
```
Bits: [sign(1) | exp(2) | mant(1)]
value = (-1)^sign * 2^(exp-1) * (1 + mant)
Range: [-8, 8], Special: 0 = zero
```

**Dequantization:**
```
scale = 2^scale_exp
e2m1_bits = extract_4bit(data, element_idx)
decoded = decode_e2m1(e2m1_bits)
value = clamp(scale * decoded, -8.0, 8.0)
```

---

#### MXFP6 (OCP MX Specification v1.0)

```
Block size: 32 elements
Block bytes: 25 bytes
┌─────────────────────────────────────────────────────────┐
│ Scale (E8M0, 1 byte) │ Packed E2M3 values (24 bytes)  │
├─────────────────────────────────────────────────────────┤
│ exponent (int8)      │ [6-bit E2M3 x32]               │
└─────────────────────────────────────────────────────────┘
```

**E2M3 format (6-bit):**
```
Bits: [sign(1) | exp(2) | mant(3)]
value = (-1)^sign * 2^(exp-1) * (1 + mant/8)
Range: [-7.5, 7.5], Special: 0 = zero
```

**Dequantization:**
```
scale = 2^scale_exp
bit_offset = (element_idx * 6) % 8
byte_idx = (element_idx * 6) / 8
combined = (data[byte_idx + 1] << 8) | data[byte_idx]
e2m3_bits = (combined >> (10 - bit_offset)) & 0x3F
decoded = decode_e2m3(e2m3_bits)
value = clamp(scale * decoded, -7.5, 7.5)
```

---

## CPU Dequantization Algorithms

### Source: `/home/feanor/Projects/ROCmForge/src/loader/dequant.rs`

### Algorithm Patterns

#### 1. Block Processing Pattern

All Q-formats follow this structure:

```rust
let total_elements = tensor.total_elements();
let blocks = total_elements.div_ceil(block_size);
let mut result = vec![0.0f32; total_elements];

for block_idx in 0..blocks {
    let block_start = block_idx * block_size;

    // 1. Read scale (and min if applicable)
    // 2. Read quantized data
    // 3. Unpack and dequantize each element

    // Write to result
    result[element_idx] = dequantized_value;
}
```

#### 2. Scale Reading Pattern

```rust
// FP32 scale (Q4_0, Q5_0, Q8_0)
let scale = f32::from_le_bytes([
    data[0], data[1], data[2], data[3]
]);

// F16 scale (Q4_K, Q6_K)
let scale_bits = u16::from_le_bytes([data[0], data[1]]);
let scale = half::f16::from_bits(scale_bits).to_f32();

// E8M0 scale (MXFP)
let scale_exp = data[0] as i8;
let scale = 2.0_f32.powi(scale_exp as i32);
```

#### 3. 4-Bit Unpacking Pattern

```rust
for (i, &packed) in packed_quants.iter().enumerate() {
    for j in 0..2 {
        let element_idx = block_idx * 32 + i * 2 + j;
        let quant = if j == 0 {
            packed & 0x0F        // Low nibble
        } else {
            (packed >> 4) & 0x0F // High nibble
        };
        // Dequantize using quant
    }
}
```

#### 4. 5-Bit Unpacking (Q5_0, Q5_1)

```rust
// Read high bits (qh)
let qh = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

for (i, &packed) in packed_quants.iter().enumerate() {
    for j in 0..2 {
        let bit_idx = i * 2 + j;
        let low_bits = if j == 0 {
            packed & 0x0F
        } else {
            (packed >> 4) & 0x0F
        };
        let high_bit = (qh >> bit_idx) & 1;
        let quant = (low_bits as u32 | (high_bit << 4)) as u8;
        // Dequantize using 5-bit quant
    }
}
```

#### 5. 6-Bit Unpacking (Q6_K, MXFP6)

```rust
for i in 0..32 {
    let bit_offset = (i * 6) % 8;
    let byte_idx = (i * 6) / 8;

    let combined = ((data[byte_idx + 1] as u16) << 8) |
                   (data[byte_idx] as u16);
    let quant = ((combined >> bit_offset) & 0x3F) as u8;
    // Dequantize using 6-bit quant
}
```

#### 6. Parallel Processing with Rayon

Q4_0 and Q8_0 use Rayon for parallel CPU processing:

```rust
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

let result = vec![0.0f32; total_elements];
let result_lock = Arc::new(RwLock::new(result));

(0..blocks).into_par_iter().for_each(|block_idx| {
    // Process block independently
    // Write to shared result via RwLock
    if let Ok(mut result) = result_lock.write() {
        result[element_idx] = dequantized_value;
    }
});
```

**Performance notes:**
- ~4x speedup on multi-core CPUs
- Each block is independent - perfect for data parallelism
- RwLock contention is minimal (each thread writes different elements)

### CPU Implementation Reference

The following table shows the reference implementations for each format:

| Format | Function | Lines of Code | Parallel |
|--------|----------|---------------|----------|
| Q4_0 | `dequant_q4_0()` | 55 | Yes (Rayon) |
| Q4_1 | `dequant_q4_1()` | 48 | No |
| Q5_0 | `dequant_q5_0()` | 53 | No |
| Q5_1 | `dequant_q5_1()` | 56 | No |
| Q8_0 | `dequant_q8_0()` | 53 | Yes (Rayon) |
| Q4_K | `dequant_q4_k()` | 77 | No |
| Q6_K | `dequant_q6_k()` | 65 | No |
| MXFP4 | `dequant_mxfp4()` | 42 | No |
| MXFP6 | `dequant_mxfp6()` | 48 | No |

---

## HIP Kernel Patterns

### Source: `/home/feanor/Projects/ROCmForge/kernels/mxfp_dequant.hip`

### Architecture Constants

```cpp
// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;    // RDNA3 wavefront size
```

### Kernel Launch Pattern

```cpp
// Grid: One block per quantized block
// Block: BLOCK_SIZE threads (256 threads = 8 waves)

dim3 grid(num_blocks, 1, 1);
dim3 block(BLOCK_SIZE, 1, 1);

kernel<<<grid, block, 0, stream>>>(input, output, num_blocks);
```

### Device Function Patterns

#### 1. Decode Function (MXFP E2M1)

```cpp
__device__ __forceinline__ float decode_e2m1(uint8_t bits) {
    if (bits == 0) {
        return 0.0f;
    }

    int sign = (bits & 0x08) ? -1 : 1;
    int exp = ((bits >> 1) & 0x03) - 1;
    int mant = bits & 0x01;

    return sign * __int2float_rn(1 + mant) * exp2f(exp);
}
```

**Pattern notes:**
- `__device__ __forceinline__` for all helper functions
- Early return for zero (special encoding)
- Bit extraction with masks and shifts
- `exp2f()` for power-of-2 calculation

#### 2. Scale Conversion (E8M0)

```cpp
__device__ __forceinline__ float e8m0_to_f32(int8_t exponent) {
    return exp2f(static_cast<float>(exponent));
}
```

#### 3. Kernel Template

```cpp
extern "C" __global__ void format_to_fp32_kernel(
    const uint8_t* __restrict__ input,  // Packed quantized data
    float* __restrict__ output,          // FP32 output
    const int num_blocks                 // Number of blocks
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Boundary check
    if (block_idx >= num_blocks) {
        return;
    }

    // Each block processes up to 32 elements
    const int element_idx = block_idx * 32 + tid;

    if (tid >= 32) {
        return;  // Extra threads do nothing
    }

    // Read block scale
    const int block_offset = block_idx * BLOCK_SIZE;
    float scale = read_scale(input, block_offset);

    // Unpack element
    uint8_t packed_bits = unpack_element(input, block_offset, tid);

    // Decode and apply scale
    float decoded = decode(packed_bits);
    float val = scale * decoded;

    // Clamp to format range
    val = fmaxf(min_val, fminf(max_val, val));

    // Store
    output[element_idx] = val;
}
```

### Memory Access Patterns

#### Coalesced Reads

```cpp
// Read scale (single transaction per block)
int8_t scale_exp = reinterpret_cast<const int8_t*>(input)[block_offset];

// Read packed data (coalesced across warp)
const uint8_t* data = input + block_offset + 1;
uint8_t packed = data[byte_idx];  // Adjacent threads read adjacent bytes
```

#### Boundary Checking

```cpp
// Always check block boundary
if (block_idx >= num_blocks) {
    return;
}

// Check thread boundary for block size < BLOCK_SIZE
if (tid >= elements_per_block) {
    return;
}

// For partial blocks at end of tensor
if (element_idx >= total_elements) {
    return;
}
```

### Bit Unpacking Patterns

#### 4-Bit Nibble Unpacking

```cpp
int byte_idx = tid / 2;
int nibble_idx = tid % 2;

uint8_t bits;
if (nibble_idx == 0) {
    bits = (data[byte_idx] >> 4) & 0x0F;  // High nibble
} else {
    bits = data[byte_idx] & 0x0F;         // Low nibble
}
```

#### 6-Bit Cross-Byte Unpacking

```cpp
int bit_offset = (tid * 6) % 8;
int byte_idx = (tid * 6) / 8;

uint16_t combined = (static_cast<uint16_t>(data[byte_idx + 1]) << 8) |
                    static_cast<uint16_t>(data[byte_idx]);
uint8_t bits = (combined >> (10 - bit_offset)) & 0x3F;
```

### Optimization Notes

1. **Block-level parallelism:** One GPU block per quantized block maximizes independence
2. **Wave utilization:** 256 threads = 8 waves on RDNA3 (good utilization)
3. **No shared memory:** Direct global memory access is sufficient
4. **Restrict pointers:** `__restrict__` hints compiler about aliasing
5. **Early returns:** Boundary checks before memory access

---

## Kernel Implementation Strategy

### Priority Order

Based on model usage and importance:

1. **Q4_0** - Most common format for transformer weights
2. **Q8_0** - Used for activations and temporary tensors
3. **Q4_K, Q6_K** - Modern K-quants (better quality than Q4_0)
4. **Q5_0, Q5_1** - Less common but higher quality
5. **Q4_1** - Rarely used

### Kernel Design Per Format

#### Standard Formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)

**Input:** Packed quantized data (uint8_t*)
**Output:** FP32 values (float*)
**Block layout:** 32 elements per block
**Thread assignment:** One thread per element in block (32 threads active)

```cpp
// Template signature
extern "C" __global__ void q4_0_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
);
```

**Kernel structure:**
```cpp
const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;
const int block_idx = element_idx / 32;
const int element_in_block = element_idx % 32;

if (element_idx >= num_elements) return;

// Read scale at start of block
int block_offset = block_idx * BLOCK_SIZE;
float scale = *reinterpret_cast<const float*>(input + block_offset);

// Unpack 4-bit value
// Apply scale
// Store to output
```

#### K-Quant Formats (Q4_K, Q6_K)

**Input:** Super-block structure
**Block layout:** 256 elements per super-block
**Thread assignment:** One thread per element (256 threads)

```cpp
extern "C" __global__ void q4_k_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
);
```

**Kernel structure:**
```cpp
const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;
const int superblock_idx = element_idx / 256;
const int element_in_superblock = element_idx % 256;

if (element_idx >= num_elements) return;

// Calculate sub-block index
const int sub_block_idx = element_in_superblock / 32;
const int element_in_subblock = element_in_superblock % 32;

// Read sub-block scale (from first 16 bytes of superblock)
// Read sub-block min (from bytes 16-31)
// Unpack 4-bit value from sub-block data (bytes 32+)
// Apply dequantization: min + quant * scale
```

### File Naming Convention

```
kernels/q4_0_dequant.hip   -> Q4_0_TO_FP32_HSACO
kernels/q8_0_dequant.hip   -> Q8_0_TO_FP32_HSACO
kernels/q4_k_dequant.hip   -> Q4_K_TO_FP32_HSACO
kernels/q6_k_dequant.hip   -> Q6_K_TO_FP32_HSACO
```

### Build System Integration

#### 1. Add Kernel Entry to build.rs

```rust
// In build.rs kernels array
(
    "kernels/q4_0_dequant.hip",
    "Q4_0_DEQUANT_HSACO",
    "q4_0_to_fp32_kernel",
),
(
    "kernels/q8_0_dequant.hip",
    "Q8_0_DEQUANT_HSACO",
    "q8_0_to_fp32_kernel",
),
```

#### 2. Create Environment Variable

```rust
println!("cargo:rustc-env=Q4_0_DEQUANT_HSACO={}", hsaco_path.display());
```

#### 3. Create Rust Wrapper

```rust
// In src/ggml/hip_backend/ops/q_dequant.rs

use crate::backend::HipBackend;

pub fn dequantize_q4_0_gpu(
    backend: &HipBackend,
    quantized_data: &HipBuffer,
    output: &HipBuffer,
    num_elements: usize,
) -> Result<(), String> {
    // Load HSACO
    let hsaco_path = env!("Q4_0_DEQUANT_HSACO");
    let module = backend.load_module(hsaco_path)?;

    // Get kernel
    let kernel = module.get_function("q4_0_to_fp32_kernel")?;

    // Calculate grid size
    let num_blocks = (num_elements + 255) / 256; // Round up

    // Launch kernel
    unsafe {
        hipLaunchKernel(
            kernel,
            dim3(num_blocks, 1, 1),
            dim3(256, 1, 1),
            0,
            backend.stream(),
            [&quantized_data, &output, num_elements],
        );
    }

    Ok(())
}
```

### Testing Strategy

1. **CPU-GPU equivalence test:**
   - Generate random test data
   - Dequantize on CPU (reference)
   - Dequantize on GPU
   - Compare results (allow small floating-point differences)

2. **Format-specific tests:**
   - Test edge cases (all zeros, all max values)
   - Test partial blocks (non-multiple of block size)
   - Test negative values

3. **Integration test:**
   - Load real GGUF model
   - Dequantize weights
   - Run inference
   - Compare outputs with CPU dequantization

---

## Build System Integration

### Source: `/home/feanor/Projects/ROCmForge/build.rs`

### Current Kernel List

The build system currently compiles these kernels:

| Kernel File | Env Var | Kernel Function | Purpose |
|-------------|---------|-----------------|---------|
| scale.hip | SCALE_HSACO | scale_kernel | Element-wise scaling |
| mask.hip | MASK_HSACO | mask_kernel | Attention mask application |
| softmax.hip | SOFTMAX_HSACO | softmax_kernel | Softmax activation |
| rope.hip | ROPE_HSACO | rope_kernel | Rotary position encoding |
| position_embeddings.hip | POSITION_EMBEDDINGS_HSACO | position_embeddings_kernel | Position embeddings |
| qkt_matmul.hip | QKT_MATMUL_HSACO | qkt_matmul_kernel | Query-Key transpose matmul |
| weighted_matmul.hip | WEIGHTED_MATMUL_HSACO | weighted_matmul_kernel | Attention-weighted values |
| flash_attention_*.hip | FLASH_ATTENTION_*_HSACO | flash_attention_*_kernel | Flash attention variants |
| swiglu.hip | SWIGLU_HSACO | swiglu_kernel | SwiGLU activation |
| rms_norm.hip | RMS_NORM_HSACO | rms_norm_kernel | RMS normalization |
| mxfp_dequant.hip | MXFP_DEQUANT_HSACO | mxfp4_to_fp32_kernel | MXFP4 dequantization |
| mqa_kv_replicate.hip | MQA_KV_REPLICATE_HSACO | mqa_kv_replicate_kernel | Multi-query attention KV replication |

### Adding New Dequantization Kernels

To add Q-format dequantization kernels:

1. **Create kernel file** in `kernels/` directory:
   ```
   kernels/q4_0_dequant.hip
   ```

2. **Add entry to kernels array** in `build.rs`:
   ```rust
   let kernels = [
       // ... existing kernels ...
       (
           "kernels/q4_0_dequant.hip",
           "Q4_0_DEQUANT_HSACO",
           "q4_0_to_fp32_kernel",
       ),
   ];
   ```

3. **Build system will:**
   - Compile `.hip` file to `.hsaco` using `hipcc`
   - Set environment variable with HSACO path
   - Make path available at compile time via `env!()`

4. **In Rust code**, load and launch:
   ```rust
   let hsaco_path = env!("Q4_0_DEQUANT_HSACO");
   let module = backend.load_module(hsaco_path)?;
   ```

### Kernel Compilation Command

```bash
hipcc -c --genco --offload-arch=gfx1100 -O3 \
    kernels/q4_0_dequant.hip \
    -o /path/to/out/dir/q4_0_to_fp32_kernel.hsaco
```

**Flags:**
- `-c`: Compile to object file (HSACO)
- `--genco`: Generate code object
- `--offload-arch=gfx1100`: Target AMD Radeon RX 7900 XT (RDNA3)
- `-O3`: Maximum optimization

---

## References

### Source Files Analyzed

1. `/home/feanor/Projects/ROCmForge/src/loader/dequant.rs` - CPU dequantization implementations
2. `/home/feanor/Projects/ROCmForge/src/loader/tensor_type.rs` - Tensor type definitions and block sizes
3. `/home/feanor/Projects/ROCmForge/src/loader/mxfp.rs` - MXFP format support (E2M1, E2M3, E8M0)
4. `/home/feanor/Projects/ROCmForge/src/ggml/hip_backend/ops/quantized_matmul.rs` - Quantized matmul operations
5. `/home/feanor/Projects/ROCmForge/kernels/mxfp_dequant.hip` - MXFP HIP kernels (reference implementation)
6. `/home/feanor/Projects/ROCmForge/build.rs` - Build system integration

### External Specifications

1. **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
2. **OCP MX Specification v1.0**: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
3. **ROCm HIP API**: https://rocm.docs.amd.com/projects/HIP/en/latest/
4. **RDNA3 Architecture**: gfx1100 instruction set, wave32 frontend

### Key Insights

1. **Block-based quantization:** All formats use blocks (32 or 256 elements) with per-block scaling
2. **Bit packing is format-specific:** 4-bit (nibbles), 5-bit (nibble + high bit), 6-bit (cross-byte), 8-bit (direct)
3. **CPU patterns translate to GPU:** CPU dequantization loops map directly to GPU thread blocks
4. **MXFP kernels are reference:** `mxfp_dequant.hip` shows the correct patterns for all Q-format kernels
5. **Build system is extensible:** Adding new kernels follows the same pattern as existing ones

---

**End of Research Document**

This document provides the complete technical foundation for implementing Q-format dequantization kernels in Phase 5 (Plans 05-02, 05-03, 05-04).
