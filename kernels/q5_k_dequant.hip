/**
 * q5_k_dequant.hip - Q5_K Dequantization Kernels
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Implements dequantization of GGUF Q5_K format.
 * Q5_K is a "K-quant" format with 5-bit precision.
 *
 * Q5_K format specification:
 * - Super-block size: 256 elements (16 sub-blocks of 16 elements each)
 * - Per super-block (256 bytes):
 *   - 32 bytes: 16 half-precision scales (2 bytes each) for 16 sub-blocks
 *   - 16 bytes: 16 int8 mins (1 byte each) for 16 sub-blocks
 *   - 160 bytes: 16 sub-blocks of 5-bit quantized values (10 bytes each)
 *   - 48 bytes: additional data
 * - Each sub-block (16 elements): scale (f16) + min (int8) + 10 bytes packed 5-bit values
 * - Dequantization: value = min + (quant * scale)
 *
 * Reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

// Q5_K super-block size
constexpr int Q5_K_SUPER_BLOCK_SIZE = 256;  // Total bytes per super-block
constexpr int Q5_K_ELEMENTS_PER_BLOCK = 256;  // Elements per super-block
constexpr int Q5_K_SUB_BLOCKS = 16;  // Sub-blocks per super-block
constexpr int Q5_K_ELEMENTS_PER_SUB_BLOCK = 16;  // Elements per sub-block

// Offsets within super-block
constexpr int Q5_K_SCALES_OFFSET = 0;      // Start of scales
constexpr int Q5_K_MINS_OFFSET = 32;       // Start of mins (after 32 bytes of scales)
constexpr int Q5_K_QUANTS_OFFSET = 48;     // Start of quants (after 48 bytes of scales+mins)

/**
 * Convert half-precision float (FP16) to single-precision float (FP32)
 */
__device__ __forceinline__ float f16_to_f32(uint16_t f16_bits) {
    uint32_t f32_bits;

    if ((f16_bits & 0x7FFF) == 0) {
        f32_bits = (f16_bits & 0x8000) << 16;
    } else {
        uint32_t sign = (f16_bits & 0x8000) << 16;
        uint32_t mant = (f16_bits & 0x03FF) << 13;
        int32_t exp = (f16_bits & 0x7C00) >> 10;
        exp = exp - 15 + 127;
        f32_bits = sign | (exp << 23) | mant;
    }

    return *reinterpret_cast<float*>(&f32_bits);
}

/**
 * Q5_K dequantization kernel
 *
 * Dequantizes Q5_K super-blocks to FP32
 *
 * Grid: (num_super_blocks, 1, 1)
 * Block: BLOCK_SIZE threads
 */
extern "C" __global__ void q5_k_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_super_blocks
) {
    const int super_block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (super_block_idx >= num_super_blocks) {
        return;
    }

    const int element_idx = super_block_idx * Q5_K_ELEMENTS_PER_BLOCK + tid;

    if (tid >= Q5_K_ELEMENTS_PER_BLOCK) {
        return;
    }

    // Calculate which sub-block and position within sub-block
    const int sub_block_idx = tid / Q5_K_ELEMENTS_PER_SUB_BLOCK;
    const int sub_block_elem = tid % Q5_K_ELEMENTS_PER_SUB_BLOCK;

    // Base pointer for this super-block
    const uint8_t* super_block = &input[super_block_idx * Q5_K_SUPER_BLOCK_SIZE];

    // Read scale for this sub-block
    const int scale_offset = Q5_K_SCALES_OFFSET + sub_block_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(&super_block[scale_offset]);
    const float scale = f16_to_f32(scale_bits);

    // Read min for this sub-block
    const int min_offset = Q5_K_MINS_OFFSET + sub_block_idx;
    const float min = static_cast<float>(static_cast<int8_t>(super_block[min_offset]));

    // Extract 5-bit quantized value
    // 16 values * 5 bits = 80 bits = 10 bytes per sub-block
    const int quants_offset = Q5_K_QUANTS_OFFSET + sub_block_idx * 10;
    const int bit_pos = sub_block_elem * 5;
    const int byte_idx = bit_pos / 8;
    const int bit_offset = bit_pos % 8;

    uint16_t combined = *reinterpret_cast<const uint16_t*>(&super_block[quants_offset + byte_idx]);
    const uint8_t quant = (combined >> bit_offset) & 0x1F;

    // Dequantize
    output[element_idx] = min + quant * scale;
}

/**
 * Q5_K batch dequantization kernel
 *
 * Element-based grid for better load balancing
 */
extern "C" __global__ void q5_k_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx >= num_elements) {
        return;
    }

    const int super_block_idx = element_idx / Q5_K_ELEMENTS_PER_BLOCK;
    const int tid = element_idx % Q5_K_ELEMENTS_PER_BLOCK;

    const int sub_block_idx = tid / Q5_K_ELEMENTS_PER_SUB_BLOCK;
    const int sub_block_elem = tid % Q5_K_ELEMENTS_PER_SUB_BLOCK;

    const uint8_t* super_block = &input[super_block_idx * Q5_K_SUPER_BLOCK_SIZE];

    const int scale_offset = Q5_K_SCALES_OFFSET + sub_block_idx * 2;
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(&super_block[scale_offset]);
    const float scale = f16_to_f32(scale_bits);

    const int min_offset = Q5_K_MINS_OFFSET + sub_block_idx;
    const float min = static_cast<float>(static_cast<int8_t>(super_block[min_offset]));

    const int quants_offset = Q5_K_QUANTS_OFFSET + sub_block_idx * 10;
    const int bit_pos = sub_block_elem * 5;
    const int byte_idx = bit_pos / 8;
    const int bit_offset = bit_pos % 8;

    uint16_t combined = *reinterpret_cast<const uint16_t*>(&super_block[quants_offset + byte_idx]);
    const uint8_t quant = (combined >> bit_offset) & 0x1F;

    output[element_idx] = min + quant * scale;
}
