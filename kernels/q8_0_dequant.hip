/**
 * q8_0_dequant.hip - Q8_0 Dequantization Kernels
 *
 * GPU: AMD Radeon RX 7900 XT (gfx1100, RDNA3, wave32)
 * Block size: 256 threads (8 waves of 32)
 *
 * Implements dequantization of GGUF Q8_0 format.
 * Q8_0 is commonly used for activations in quantized models.
 *
 * Q8_0 format specification:
 * - Block size: 32 elements
 * - Per block: scale (f32, 4 bytes) + 32 bytes int8 values = 36 bytes
 * - Dequantization: value = (int8_value - 128) * scale
 *
 * Reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h
 */

#include <hip/hip_runtime.h>

// RDNA3 tuning constants
constexpr int BLOCK_SIZE = 256;  // 8 waves of 32 threads
constexpr int WARP_SIZE = 32;     // RDNA3 wavefront size

// Q8_0 block size
constexpr int Q8_0_BLOCK_SIZE = 36;  // scale (4) + quants (32)

/**
 * Q8_0 dequantization kernel
 *
 * Dequantizes Q8_0 blocks to FP32 (full precision float)
 * Each block: 32 elements packed into 36 bytes (1 scale + 32 int8 values)
 *
 * Grid: (num_blocks, 1, 1) - one block per Q8_0 block
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed Q8_0 data
 * @param output     Output FP32 data
 * @param num_blocks Number of Q8_0 blocks to process
 */
extern "C" __global__ void q8_0_to_fp32_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_blocks
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    // Each block processes 32 elements
    const int element_idx = block_idx * 32 + tid;

    if (tid >= 32) {
        return;
    }

    // Read scale (f32)
    const int block_offset = block_idx * Q8_0_BLOCK_SIZE;
    const float scale = *reinterpret_cast<const float*>(input + block_offset);

    // Read int8 value (stored as uint8_t, need to convert to signed range)
    const int8_t quant = reinterpret_cast<const int8_t*>(input + block_offset + 4)[tid];

    // Dequantize: value = (int8 - 128) * scale
    // Note: Q8_0 stores values in [0, 255] representing signed range [-128, 127]
    const float val = (static_cast<float>(quant) - 128.0f) * scale;

    // Store as FP32
    output[element_idx] = val;
}

/**
 * Batched Q8_0 dequantization kernel
 *
 * Optimized version for processing multiple blocks.
 * Each thread processes one element directly using element index calculation.
 *
 * Grid: (num_blocks, 1, 1) or ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
 * Block: BLOCK_SIZE threads
 *
 * @param input      Packed Q8_0 data
 * @param output     Output FP32 data
 * @param num_elements Total number of output elements
 */
extern "C" __global__ void q8_0_to_fp32_batch_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const int num_elements
) {
    const int element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx >= num_elements) {
        return;
    }

    // Calculate which block and element within block
    const int block_idx = element_idx / 32;
    const int element_in_block = element_idx % 32;

    // Read scale (f32)
    const int block_offset = block_idx * Q8_0_BLOCK_SIZE;
    const float scale = *reinterpret_cast<const float*>(input + block_offset);

    // Read int8 value (stored as uint8_t, need to convert to signed range)
    const int8_t quant = reinterpret_cast<const int8_t*>(input + block_offset + 4)[element_in_block];

    // Dequantize: value = (int8 - 128) * scale
    const float val = (static_cast<float>(quant) - 128.0f) * scale;

    // Store as FP32
    output[element_idx] = val;
}
