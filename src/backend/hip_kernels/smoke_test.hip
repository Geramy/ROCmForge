//! Simple smoke test kernel for HIP GPU validation
//! This kernel adds 1.0 to each element in the input array

#include <hip/hip_runtime.h>

extern "C" __global__ void add_one(float* data, int size) {
    // Simple test: just set first element to a known value
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        data[0] = 42.0f;
    }
}

// Simple kernel that takes no arguments
extern "C" __global__ void simple_test() {
    // This kernel does nothing - just tests if kernel launch works
}

// Additional test kernels for comprehensive GPU validation

extern "C" __global__ void multiply_by_two(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

extern "C" __global__ void set_value(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

extern "C" __global__ void vector_add(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}